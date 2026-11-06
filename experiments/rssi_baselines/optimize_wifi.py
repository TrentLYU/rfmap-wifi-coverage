"""Compare six fixed-router RSSI estimators using nested cross-validation."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import warnings

import joblib
import train_wifi
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, Delaunay
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, WhiteKernel
from sklearn.model_selection import KFold, GroupKFold
from sklearn.exceptions import ConvergenceWarning

from train_wifi import (fit_ensemble, neural_predict, fit_formula,
                       formula_predict, idw_predict, portable_model)

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'optimized'
FAMILIES=['formula','idw','mlp','physics_mlp','physics_gp','physics_idw']
NAMES={'formula':'传播公式','idw':'距离加权插值','mlp':'坐标神经网络',
       'physics_mlp':'传播公式＋残差神经网络','physics_gp':'传播公式＋GP修正','physics_idw':'传播公式＋残差插值'}
COMPLEXITY=['formula','idw','physics_idw','physics_gp','mlp','physics_mlp']
TIE_MARGIN=.2
ROOM=[12.,12.]
GP_WARNINGS=[]


def configurations():
    c={'formula':[{}],'idw':[{'power':p} for p in [1,2,3]],
       'mlp':[{'layers':[8],'alpha':.1},{'layers':[8],'alpha':1.}, {'layers':[16,8],'alpha':1.}]}
    c['physics_mlp']=[{'layers':h,'alpha':a,'weight':w}
                      for h,a in [([8],1.),([8],10.),([8,4],1.)] for w in [.25,.5,1.]]
    c['physics_gp']=[{'kernel':k,'weight':w} for k in ['rbf','matern25'] for w in [.5,1.]]
    c['physics_idw']=[{'power':p,'weight':w} for p in [1,2,3] for w in [.25,.5,1.]]
    return c


def base_key(family,config):
    return (family,json.dumps({k:v for k,v in config.items() if k!='weight'},sort_keys=True))


def fit(family,config,X,target):
    model={'family':family,'configuration':config,'formula':fit_formula(X,target),'fit_positions':len(X)}
    model['convergence_warnings']=0
    if family=='formula': return model
    if family in ['idw','physics_idw']:
        model['X']=np.array(X)
        model['target']=np.array(target) if family=='idw' else target-formula_predict(model['formula'],X)
    elif family in ['mlp','physics_mlp']:
        response=target if family=='mlp' else target-formula_predict(model['formula'],X)
        model['networks'],model['convergence_warnings']=fit_ensemble(X,response,config)
    elif family=='physics_gp':
        residual=target-formula_predict(model['formula'],X)
        mean=float(residual.mean());scale=max(float(residual.std()),1e-8)
        smooth=RBF(2.,(.5,8.)) if config['kernel']=='rbf' else Matern(2.,(.5,8.),nu=2.5)
        kernel=ConstantKernel(1.,(.05,20.))*smooth+WhiteKernel(.1,(.03,1.))
        gp=GaussianProcessRegressor(kernel=kernel,alpha=1e-6,normalize_y=False,
                                   n_restarts_optimizer=0,random_state=42)
        with warnings.catch_warnings(record=True) as log:
            warnings.simplefilter('always',ConvergenceWarning)
            gp.fit(X,(residual-mean)/scale)
        model.update({'gp':gp,'residual_mean':mean,'residual_scale':scale})
        # Kernel-bound warnings are logged, not automatically treated as solver failures.
        GP_WARNINGS.extend(str(a.message) for a in log)
        model['kernel_bound_warnings']=len(log)
    return model


def predict(model,X,weight=None):
    family=model['family'];config=model['configuration']
    baseline=formula_predict(model['formula'],X)
    if family=='formula': return baseline
    if family in ['idw','physics_idw']:
        correction=idw_predict(model['X'],model['target'],X,config['power'])
    elif family in ['mlp','physics_mlp']:
        correction=neural_predict(model['networks'],X)
    else:
        correction=model['residual_mean']+model['residual_scale']*model['gp'].predict(X)
    if family in ['idw','mlp']: return correction
    return baseline+(config['weight'] if weight is None else weight)*correction


def groups(X):
    blocks = np.minimum((3 * X / np.asarray(ROOM)).astype(int), 2)
    return blocks[:, 0]*3 + blocks[:, 1]


def select_configs(X,target,configs,tag):
    folds=list(GroupKFold(n_splits=3).split(X,target,groups(X)))
    scores={f:[[] for _ in configs[f]] for f in FAMILIES}
    warning_count=0
    for train,valid in folds:
        cache={}
        for family in FAMILIES:
            for j,config in enumerate(configs[family]):
                key=base_key(family,config)
                if key not in cache:
                    model=fit(family,config,X[train],target[train])
                    warning_count+=model['convergence_warnings']
                    # Predict a residual with weight one, then reuse it for shrinkage candidates.
                    full=predict(model,X[valid],weight=1.)
                    baseline=formula_predict(model['formula'],X[valid])
                    cache[key]=(full,baseline)
                full,baseline=cache[key]
                pred=baseline+config['weight']*(full-baseline) if 'weight' in config else full
                scores[family][j].extend(np.abs(pred-target[valid]).tolist())
    selected={};records=[]
    for family in FAMILIES:
        values=[float(np.mean(e)) for e in scores[family]]
        chosen=int(np.argmin(values));selected[family]=configs[family][chosen]
        for j,c in enumerate(configs[family]):
            records.append({'outer_fold':tag,'family':family,'configuration':json.dumps(c,sort_keys=True),
                            'inner_mae_db':values[j],'selected':j==chosen})
    return selected,records,warning_count


def stats(actual,prediction):
    error=np.array(prediction)-np.array(actual)
    return {'mae_db':float(np.abs(error).mean()),'rmse_db':float(np.sqrt((error**2).mean())),
            'p90_abs_error_db':float(np.quantile(np.abs(error),.9)),
            'max_abs_error_db':float(np.abs(error).max()),'within_5db_fraction':float((np.abs(error)<=5).mean()),
            'n':len(error)}


def export(model,X):
    family=model['family']
    result={'family':family,'configuration':model['configuration'],'formula':model['formula'],
            'router_xy':train_wifi.ROUTER.tolist(),'coordinate_bounds':[[0,ROOM[0]],[0,ROOM[1]]],
            'support_polygon_xy':X[ConvexHull(X).vertices].tolist(),
            'fit_positions':len(X),'convergence_warnings':model['convergence_warnings']}
    if family in ['idw','physics_idw']:
        result.update({'training_xy':model['X'].tolist(),'targets':model['target'].tolist()})
    elif family in ['mlp','physics_mlp']:
        result['neural']=portable_model(model['networks'])
    elif family=='physics_gp':
        gp=model['gp']
        result['gp']={'training_xy':gp.X_train_.tolist(),'alpha':gp.alpha_.tolist(),
                      'amplitude':float(gp.kernel_.k1.k1.constant_value),
                      'length_scale':float(gp.kernel_.k1.k2.length_scale),
                      'kernel':model['configuration']['kernel'],
                      'residual_mean':model['residual_mean'],'residual_scale':model['residual_scale'],
                      'white_noise':float(gp.kernel_.k2.noise_level)}
    return result


def portable_predict(model,X):
    X=np.atleast_2d(np.asarray(X,dtype=float));f=model['formula']
    d=np.maximum(np.linalg.norm(X-np.array(model['router_xy']),axis=1),1.)
    baseline=f['p0_dbm']-10*f['n']*np.log10(d)
    family=model['family'];config=model['configuration']
    if family=='formula':return baseline
    if family in ['idw','physics_idw']:
        correction=idw_predict(np.array(model['training_xy']),np.array(model['targets']),X,config['power'])
    elif family in ['mlp','physics_mlp']:
        values=[]
        for m in model['neural']['models']:
            a=(X-np.array(m['input_mean']))/np.array(m['input_scale'])
            for i,(w,b) in enumerate(zip(m['coefs'],m['intercepts'])):
                a=a@np.array(w)+np.array(b)
                if i<len(m['coefs'])-1:a=np.tanh(a)
            values.append(a.ravel()*m['target_scale'][0]+m['target_mean'][0])
        correction=np.mean(values,axis=0)
    else:
        g=model['gp'];distance=np.linalg.norm(X[:,None,:]-np.array(g['training_xy'])[None,:,:],axis=-1)/g['length_scale']
        if g['kernel']=='rbf':kernel=np.exp(-.5*distance**2)
        else:
            z=np.sqrt(5)*distance;kernel=(1+z+z*z/3)*np.exp(-z)
        correction=g['residual_mean']+g['residual_scale']*g['amplitude']*(kernel@np.array(g['alpha']))
    return correction if family in ['idw','mlp'] else baseline+config['weight']*correction


def main():
    parser = argparse.ArgumentParser(description='Evaluate fixed-router RSSI models.')
    parser.add_argument('--input', type=Path, default=ROOT/'examples'/'synthetic_rssi.csv')
    parser.add_argument('--room', type=float, nargs=2, default=[12.,12.], metavar=('X','Y'))
    parser.add_argument('--router', type=float, nargs=2, default=[1.,6.], metavar=('X','Y'))
    args = parser.parse_args()
    ROOM[:] = args.room
    train_wifi.ROUTER = np.asarray(args.router)
    if not np.isfinite([*ROOM, *args.router]).all() or min(ROOM) <= 0 or not np.all((train_wifi.ROUTER >= 0) & (train_wifi.ROUTER <= ROOM)):
        parser.error('Room dimensions must be positive; router must be inside the room.')
    data = pd.read_csv(args.input, usecols=['x_m', 'y_m', 'rssi_dbm'])
    data = data.apply(pd.to_numeric, errors='raise')
    X=data[['x_m','y_m']].to_numpy();target=data.rssi_dbm.to_numpy()
    if (len(X) < 10 or data.duplicated(['x_m','y_m']).any() or not np.isfinite(data).all().all()
        or np.any((X < 0) | (X > ROOM)) or np.any((target < -120) | (target > 0))
        or np.linalg.matrix_rank(X-X.mean(axis=0)) < 2 or len(np.unique(groups(X))) < 5):
        parser.error('Supply at least 10 distinct, non-collinear in-room positions across 5 or more spatial blocks, with RSSI between -120 and 0 dBm.')
    data.insert(0, 'point_id', [f'P{i+1:03d}' for i in range(len(X))])
    OUT.mkdir(exist_ok=True)
    configs=configurations()
    plan={'n_original_positions':len(X),'families':FAMILIES,'configs':configs,
          'outer_random':{'folds':5,'seed':20261021},'outer_spatial':'leave out each occupied spatial block in a 3-by-3 partition',
          'inner':'3-fold GroupKFold on spatial blocks; all preprocessing and calibration use inner training only',
          'selection_score':'0.5*random_oof_MAE + 0.5*spatial_oof_MAE',
          'tie_policy':f'Prefer simpler model within {TIE_MARGIN} dB of lowest selection score; engineering preference, not a significance test',
          'complexity_order':COMPLEXITY,
          'limitation':'Exploratory model selection; new prospective labels are needed for independent validation.'}
    (OUT/'experiment_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
    outer=[]
    for i,(train,test) in enumerate(KFold(5,shuffle=True,random_state=20261021).split(X)):
        outer.append(('random',f'random_{i+1}',train,test))
    g=groups(X)
    for block in sorted(set(g)):
        outer.append(('spatial',f'block_{block}',np.flatnonzero(g!=block),np.flatnonzero(g==block)))
    point_rows=[];fold_rows=[];tuning=[];splits=[];warnings_total=0
    for task,tag,train,test in outer:
        selected,rows,count=select_configs(X[train],target[train],configs,tag)
        warnings_total+=count;tuning.extend(rows)
        split={'task':task,'fold':tag,'train_point_ids':data.iloc[train].point_id.tolist(),
               'test_point_ids':data.iloc[test].point_id.tolist()}
        splits.append(split)
        for family in FAMILIES:
            fitted=fit(family,selected[family],X[train],target[train]);warnings_total+=fitted['convergence_warnings']
            pred=predict(fitted,X[test])
            fold_rows.append({'task':task,'fold':tag,'family':family,'train_n':len(train),
                              'configuration':json.dumps(selected[family],sort_keys=True),
                              'convergence_warnings':fitted['convergence_warnings'],**stats(target[test],pred)})
            for i,value in zip(test,pred):
                point_rows.append({'task':task,'fold':tag,'family':family,'point_id':data.iloc[i].point_id,
                                   'x_m':X[i,0],'y_m':X[i,1],'actual_dbm':target[i],
                                   'predicted_dbm':float(value),'error_db':float(value-target[i])})
        # Incremental outputs make progress and failures reviewable without a second training run.
        pd.DataFrame(point_rows).to_csv(OUT/'oof_predictions.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame(fold_rows).to_csv(OUT/'fold_metrics.csv',index=False,encoding='utf-8-sig')
        print(json.dumps({'completed_fold':tag,'test_n':len(test),
                          'mae':{r['family']:round(r['mae_db'],3) for r in fold_rows[-6:]}},ensure_ascii=True),flush=True)
    predictions=pd.DataFrame(point_rows);summary_rows=[]
    for family in FAMILIES:
        pieces={}
        for task in ['random','spatial']:
            part=predictions[(predictions.family==family)&(predictions.task==task)]
            assert len(part)==len(X) and part.point_id.nunique()==len(X)
            pieces[task]=stats(part.actual_dbm,part.predicted_dbm)
        all_points=predictions[predictions.family==family]
        spatial_folds=[r for r in fold_rows if r['family']==family and r['task']=='spatial']
        summary_rows.append({'family':family,'name':NAMES[family],
                             'selection_mae_db':.5*(pieces['random']['mae_db']+pieces['spatial']['mae_db']),
                             'random_mae_db':pieces['random']['mae_db'],'random_rmse_db':pieces['random']['rmse_db'],
                             'spatial_mae_db':pieces['spatial']['mae_db'],'spatial_rmse_db':pieces['spatial']['rmse_db'],
                             'worst_block_mae_db':max(r['mae_db'] for r in spatial_folds),
                             **{'combined_'+k:v for k,v in stats(all_points.actual_dbm,all_points.predicted_dbm).items()}})
    ranking=pd.DataFrame(summary_rows).sort_values('selection_mae_db')
    empirical_best=ranking.iloc[0].family
    eligible=set(ranking[ranking.selection_mae_db<=ranking.selection_mae_db.min()+TIE_MARGIN].family)
    recommended=next(f for f in COMPLEXITY if f in eligible)
    best_neural=ranking[ranking.family.isin(['mlp','physics_mlp'])].iloc[0].family
    ranking['recommended']=ranking.family==recommended
    ranking.to_csv(OUT/'model_ranking.csv',index=False,encoding='utf-8-sig')
    full_selected,rows,count=select_configs(X,target,configs,'full_data_selection')
    warnings_total+=count;tuning.extend(rows)
    pd.DataFrame(tuning).to_csv(OUT/'inner_selection.csv',index=False,encoding='utf-8-sig')
    final_models={};portable_models={}
    axis_x=np.linspace(0,ROOM[0],121);axis_y=np.linspace(0,ROOM[1],128)
    gx,gy=np.meshgrid(axis_x,axis_y,indexing='ij');grid=np.column_stack([gx.ravel(),gy.ravel()])
    support=Delaunay(X).find_simplex(grid)>=0
    grid_frame=pd.DataFrame({'x_m':grid[:,0],'y_m':grid[:,1],'inside_sample_hull':support})
    for family in FAMILIES:
        model=fit(family,full_selected[family],X,target)
        exported=export(model,X)
        test_queries=np.vstack([X,train_wifi.ROUTER[None, :]])
        np.testing.assert_allclose(predict(model,test_queries),portable_predict(exported,test_queries),atol=1e-8)
        final_models[family]=model;portable_models[family]=exported
        grid_frame[family+'_dbm']=predict(model,grid)
    grid_frame.to_csv(OUT/'heatmap_grid.csv',index=False,encoding='utf-8-sig')
    joblib.dump(final_models,OUT/'all_models.joblib')
    joblib.dump(final_models[recommended],OUT/'recommended_model.joblib')
    joblib.dump(final_models[best_neural],OUT/'best_neural_model.joblib')
    (OUT/'portable_models.json').write_text(json.dumps(portable_models,ensure_ascii=False),encoding='utf-8')
    (OUT/'recommended_model.json').write_text(json.dumps(portable_models[recommended],ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'best_neural_model.json').write_text(json.dumps(portable_models[best_neural],ensure_ascii=False,indent=2),encoding='utf-8')
    report={'empirical_best':empirical_best,'recommended':recommended,'best_neural':best_neural,
            'names':NAMES,'ranking':json.loads(ranking.to_json(orient='records')),
            'room_dimensions_m':{'x_m':ROOM[0],'y_m':ROOM[1]},'router_xy':train_wifi.ROUTER.tolist(),
            'n_original_positions':len(X),'random_folds':5,'spatial_folds':len(np.unique(g)),'inner_folds':3,
            'n_oof_predictions_per_family':2*len(X),'selection_tie_margin_db':TIE_MARGIN,
            'full_data_configurations':full_selected,
            'training_convergence_warnings':warnings_total,
            'final_model_convergence_warnings':{k:v['convergence_warnings'] for k,v in final_models.items()},
            'gp_kernel_bound_warning_count':len(GP_WARNINGS),
            'gp_kernel_bound_warning_messages':sorted(set(GP_WARNINGS)),
            'source_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),
            'grid_shape':list(gx.shape),'evaluation_note':'Exploratory nested CV on repeatedly examined single-session data; prospective validation required.'}
    (OUT/'splits.json').write_text(json.dumps(splits,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'finished':True,'ranking':report['ranking'],'recommended':recommended,'best_neural':best_neural},ensure_ascii=True),flush=True)


if __name__=='__main__':
    main()
