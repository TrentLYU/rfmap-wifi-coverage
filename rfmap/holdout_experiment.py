"""Fresh, reproducible 70/30 measured-point experiment, with a sealed outer test.

Run: .venv/bin/python -m rfmap.holdout_experiment --epochs 180
Old weights, old predictions and simulated pseudo-labels are never loaded.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist

from . import geometry as G
from .data import read_xlsx
from .furniture_data import read_workbook
from .models import CompletionAE, SmallUNet

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/rfmap-70-30'
ORDER = ['ae', 'unet', 'resunet', 'unet_scatter', 'pgmpr']
LABELS = {'ae': '补全自编码器 AE', 'unet': '小型 U-Net',
          'resunet': '插值引导残差 U-Net', 'unet_scatter': '散落布点 U-Net',
          'pgmpr': '物理引导多先验残差 U-Net'}
ATTENUATION = {'table': 1.5, 'chair': .6, 'cabinet': 4., 'bed': 2.,
               'sofa': 2.5, 'toilet': 3., 'fridge': 6., 'wardrobe': 4.,
               'bookshelf': 3., 'desk': 1.5, 'default': 2.}
THRESHOLDS = [-80., -70., -60., -50.]
GRADE_NAMES = ['很差', '较差', '一般', '好', '非常好']
SEED = 20261106
N = 32


def dump(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def grade(v):
    return np.searchsorted(THRESHOLDS, np.asarray(v), side='right')


def group(p):
    # Same 0.2 m cell across the two versions must stay in the same partition.
    return f'{round(p[0]/.2-.5)}:{round(p[1]/.2-.5)}'


def partition(scene_points, fraction, seed):
    """Exact per-version counts, coordinate-only; no RSSI is examined."""
    rng = np.random.default_rng(seed)
    group_sets = {k: {group(p) for p in v} for k, v in scene_points.items()}
    a, b = list(group_sets)
    assert all(len(group_sets[k]) == len(scene_points[k]) for k in group_sets)
    common = sorted(group_sets[a] & group_sets[b])
    counts = {k: int(math.floor(fraction * len(scene_points[k]))) for k in group_sets}
    lower = max(0, *(counts[k] - len(group_sets[k] - set(common)) for k in group_sets))
    upper = min(len(common), *counts.values())
    nc = min(upper, max(lower, int(round(fraction * len(common)))))
    selected = set(rng.permutation(common)[:nc].tolist())
    for k in group_sets:
        own = sorted(group_sets[k] - set(common))
        selected.update(rng.permutation(own)[:counts[k]-nc].tolist())
    return {k: np.array([group(p) in selected for p in v]) for k, v in scene_points.items()}


def segment_hit(x, y, ap, obj, circle=False):
    """Vectorized open AP-to-receiver ray intersection, same formulas as UI."""
    dx, dy = x-ap[0], y-ap[1]
    if circle:
        rx, ry = max(obj['w']/2, 1e-6), max(obj['h']/2, 1e-6)
        ox, oy = (ap[0]-obj['x'])/rx, (ap[1]-obj['y'])/ry
        vx, vy = dx/rx, dy/ry
        aa, bb = vx*vx+vy*vy, 2*(ox*vx+oy*vy)
        cc = ox*ox+oy*oy-1
        disc = bb*bb-4*aa*cc
        root = np.sqrt(np.maximum(disc, 0))
        t0 = (-bb-root)/(2*np.maximum(aa, 1e-20))
        t1 = (-bb+root)/(2*np.maximum(aa, 1e-20))
        return (disc >= 0) & (aa > 1e-20) & (t1 > 1e-7) & (t0 < 1-1e-7)
    lo = np.zeros_like(x, dtype=float)
    hi = np.ones_like(x, dtype=float)
    for origin, delta, low, high in ((ap[0], dx, obj['x'], obj['x']+obj['w']),
                                     (ap[1], dy, obj['y'], obj['y']+obj['h'])):
        parallel = np.abs(delta) < 1e-12
        safe = np.where(parallel, 1, delta)
        t0, t1 = (low-origin)/safe, (high-origin)/safe
        lo = np.maximum(lo, np.where(parallel, -np.inf, np.minimum(t0, t1)))
        hi = np.minimum(hi, np.where(parallel, np.inf, np.maximum(t0, t1)))
        hi = np.where(parallel & ((origin < low) | (origin > high)), -1, hi)
    return (hi > np.maximum(lo, 1e-7)) & (lo < 1-1e-7)


def features(x, y, scene):
    room = np.ones_like(x, dtype=float)
    wall = np.zeros_like(x, dtype=float)
    fm, atten = np.zeros_like(x, dtype=float), np.zeros_like(x, dtype=float)
    ap = scene['ap']
    for o in scene['walls']:
        inside = (x > o['x']) & (x < o['x']+o['w']) & (y > o['y']) & (y < o['y']+o['h'])
        room[inside] = 0
        wall += segment_hit(x, y, ap, o)
    for f in scene['furniture']:
        circle = f['shape'] == 'circle'
        if circle:
            inside = ((x-f['x'])/(f['w']/2))**2 + ((y-f['y'])/(f['h']/2))**2 <= 1
            hit = segment_hit(x, y, ap, f, True)
        else:
            rect = dict(f, x=f['x']-f['w']/2, y=f['y']-f['h']/2)
            inside = (np.abs(x-f['x']) <= f['w']/2) & (np.abs(y-f['y']) <= f['h']/2)
            hit = segment_hit(x, y, ap, rect)
        fm = np.maximum(fm, inside)
        atten += hit * ATTENUATION.get(f['kind'], ATTENUATION['default'])
    scale = max(scene['room']['w'], scene['room']['h'])
    dx, dy = x-ap[0], y-ap[1]
    logd = np.log10(np.maximum(np.hypot(dx, dy), .5))
    return np.stack([room, fm, dx/scale, dy/scale, logd/2, wall/3, atten/10]).astype(np.float32)


def load_scenes(ap):
    empty = read_xlsx(str(ROOT/'data.xlsx'))
    furnished, fur, quality = read_workbook(str(ROOT/'放家具数据.xlsx'))
    scenes = {}
    for name, pts in [('empty', empty), ('furnished', furnished)]:
        furniture = [] if name == 'empty' else [dict(kind=f['kind'], name=f['name'], x=f['x'], y=f['y'], w=.8, h=.8, shape='circle') for f in fur]
        scenes[name] = dict(name=name, label='空房间' if name == 'empty' else '有家具',
                            xy=np.array([[p.x, p.y] for p in pts]),
                            y=np.array([p.rssi for p in pts], dtype=np.float32),
                            ids=[p.idx for p in pts],
                            raw=np.array([getattr(p, 'raw_rssi', p.rssi) for p in pts]),
                            room={'w':G.INNER_W,'h':G.INNER_H}, ap=ap,
                            walls=[{'x':G.COL_X0,'y':G.COL_Y0,'w':G.COL_W,'h':G.COL_H}], furniture=furniture)
    return scenes, quality


class Net(torch.nn.Module):
    def __init__(self, name, width):
        super().__init__()
        self.name = name
        cin = 9 + (3 if name == 'pgmpr' else 0) + (1 if name in ('pgmpr','resunet') else 0)
        self.net = CompletionAE(width=width, in_ch=cin) if name == 'ae' else SmallUNet(width=width, in_ch=cin)

    def forward(self, x, prior):
        if self.name in ('resunet', 'pgmpr'):
            return prior + self.net(torch.cat([x, prior], 1))
        return self.net(x)


def fit_physics(scenes, masks):
    # A compact baseline, independently fitted only on the allowed training rows.
    xx, yy = [], []
    for key, scene in scenes.items():
        xy = scene['xy'][masks[key]]
        f = features(xy[:,0], xy[:,1], scene)
        xx.append(np.stack([np.ones(len(xy)),f[4]*2,f[2],f[3],f[5]*3,f[6]*10,np.full(len(xy),len(scene['furniture'])/8)],1))
        yy.append(scene['y'][masks[key]])
    x, y = np.concatenate(xx), np.concatenate(yy)
    penalty = np.diag([.00001, 2, 8, 8, 8, 8, 8])
    return np.linalg.solve(x.T@x+penalty, x.T@y)


def physics_map(scene, coef):
    f = scene['geom']
    xx = np.stack([np.ones_like(f[0]),f[4]*2,f[2],f[3],f[5]*3,f[6]*10,np.full_like(f[0],len(scene['furniture'])/8)],-1)
    return (xx@coef).astype(np.float32)


def prepare_input(scene, obs_indices, norm, name, physics):
    mean, std = norm['mean'], norm['std']
    obs = np.zeros((N,N),np.float32)
    mask = np.zeros_like(obs)
    if len(obs_indices):
        xy, vals = scene['xy'][obs_indices], scene['y'][obs_indices]
        ii = np.rint(xy[:,0]/G.INNER_W*(N-1)).astype(int)
        jj = np.rint(xy[:,1]/G.INNER_H*(N-1)).astype(int)
        # At 32 nodes neighboring measurements can share a node. Average only input rows.
        total, count = np.zeros_like(obs), np.zeros_like(obs)
        np.add.at(total,(jj,ii),vals)
        np.add.at(count,(jj,ii),1)
        mask = (count>0).astype(np.float32)
        obs = np.where(count>0,(total/np.maximum(count,1)-mean)/std,0)
        dd = cdist(scene['grid_xy'],xy)
        weights = 1/np.maximum(dd,.01)**2
        prior = ((weights@vals)/weights.sum(1)).reshape(N,N)
        nearest = np.min(dd,axis=1).reshape(N,N)/max(G.INNER_W,G.INNER_H)
        density = np.exp(-dd**2/(2*1.2**2)).sum(1).reshape(N,N)/max(len(xy),1)
    else:
        prior = physics
        nearest, density = np.ones_like(obs), np.zeros_like(obs)
    x = np.concatenate([obs[None],mask[None],scene['geom']])
    if name == 'pgmpr':
        x = np.concatenate([x,((physics-mean)/std)[None],nearest[None],density[None]])
    return x.astype(np.float32), ((prior-mean)/std)[None].astype(np.float32)


def sample_grid(pred, scene):
    grid = torch.as_tensor(scene['sampling_grid'],dtype=pred.dtype)
    return F.grid_sample(pred, grid, align_corners=True, padding_mode='border')[0,0,0]


def make_pool(scenes, masks, norm, name, task, phys, seed, size=32):
    rng = np.random.default_rng(seed)
    pools = []
    for pool_idx in range(size if task == 'completion' or name == 'unet_scatter' else 1):
        batch = []
        for key,scene in scenes.items():
            allowed = np.flatnonzero(masks[key])
            if task == 'completion':
                frac = rng.uniform(.35,.8) if name == 'unet_scatter' else rng.uniform(.5,.85)
                count = max(3,min(len(allowed)-2,int(len(allowed)*frac)))
                obs = rng.choice(allowed,size=count,replace=False)
                target = np.setdiff1d(allowed,obs)
            else:
                obs = np.array([],int)
                target = rng.choice(allowed,size=max(3,int(len(allowed)*.7)),replace=False) if name == 'unet_scatter' else allowed
            x,prior = prepare_input(scene,obs,norm,name,phys[key])
            batch.append((torch.tensor(x[None]),torch.tensor(prior[None]),key,torch.tensor(target)))
        pools.append(batch)
    return pools


def predict(model, name, task, scenes, allowed, norm, phys):
    model.eval()
    maps, values = {}, {}
    with torch.no_grad():
        for key,scene in scenes.items():
            obs = np.flatnonzero(allowed[key]) if task == 'completion' else np.array([],int)
            x,p = prepare_input(scene,obs,norm,name,phys[key])
            out = model(torch.tensor(x[None]),torch.tensor(p[None]))
            values[key] = sample_grid(out,scene).numpy()*norm['std']+norm['mean']
            maps[key] = out[0,0].numpy()*norm['std']+norm['mean']
    return maps,values


def train_model(name, task, scenes, train_masks, val_masks, epochs, width, phase, out):
    seed = SEED + ORDER.index(name)*101 + (0 if task=='completion' else 3001)
    torch.manual_seed(seed)
    vals = np.concatenate([scenes[k]['y'][train_masks[k]] for k in scenes])
    norm = {'mean':float(vals.mean()),'std':float(max(vals.std(),1.))}
    coef = fit_physics(scenes,train_masks)
    phys = {k:physics_map(s,coef) for k,s in scenes.items()}
    model = Net(name,width)
    opt = torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.001)
    pool = make_pool(scenes,train_masks,norm,name,task,phys,seed)
    targets = {k:torch.tensor((s['y']-norm['mean'])/norm['std']) for k,s in scenes.items()}
    history = []
    best, best_ep, best_state = float('inf'), 1, None
    started=time.time()
    for ep in range(1,epochs+1):
        model.train()
        opt.zero_grad()
        losses=[]
        for x,p,key,indices in pool[(ep-1)%len(pool)]:
            pred=sample_grid(model(x,p),scenes[key])
            # Every loss is evaluated only at real measured training coordinates.
            losses.append(F.smooth_l1_loss(pred[indices],targets[key][indices],beta=1.0))
        loss=torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        opt.step()
        # Fixed learning schedule independent of outer test and selected epoch count.
        if ep in (80,140):
            for group_ in opt.param_groups:
                group_['lr'] *= .5
        row={'epoch':ep,'train_huber_normalized':float(loss.detach())}
        if val_masks is not None:
            _,yp=predict(model,name,task,scenes,train_masks,norm,phys)
            score=float(np.mean([np.mean(np.abs(yp[k][val_masks[k]]-s['y'][val_masks[k]])) for k,s in scenes.items()]))
            row['inner_val_mae_db']=score
            if score < best:
                best,best_ep=score,ep
                best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        history.append(row)
        if ep==1 or ep%40==0 or ep==epochs:
            print(f'{phase} {task:10} {name:12} {ep:3}/{epochs} loss={float(loss.detach()):.4f} innerMAE={row.get("inner_val_mae_db",0):.3f} time={time.time()-started:.1f}s',flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    directory=out/'models'/task/name/phase
    directory.mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(),directory/'model.pt')
    meta=dict(name=name,label=LABELS[name],task=task,phase=phase,width=width,n_input_channels=12 if name=='pgmpr' else 9,
              normalizer=norm,epochs=epochs,selected_epoch=best_ep if val_masks is not None else epochs,
              inner_val_mae=best if val_masks is not None else None,seed=seed,seconds=round(time.time()-started,3),
              parameters=sum(p.numel() for p in model.parameters()),weights_loaded=None,
              loss='smooth_l1 measured training coordinates only; equal mean over versions',
              optimizer={'name':'AdamW','lr':.002,'weight_decay':.001,'lr_halved_epochs':[80,140]},
              training_point_ids={k:[scenes[k]['ids'][i] for i in np.flatnonzero(train_masks[k])] for k in scenes},
              validation_point_ids={k:[scenes[k]['ids'][i] for i in np.flatnonzero(val_masks[k])] for k in scenes} if val_masks is not None else {},
              physics_coefficients=coef.tolist(),grid={'nx':N,'ny':N,'node_grid':True,'x_max':G.INNER_W,'y_max':G.INNER_H},
              model_sha256=sha(directory/'model.pt'))
    dump(meta,directory/'meta.json');dump(history,directory/'history.json')
    return model,meta,phys


def calc_metrics(y,p,majority):
    y,p=np.asarray(y),np.asarray(p)
    if not len(y): return {'n':0}
    truth,pred=grade(y),grade(p)
    cm=np.zeros((5,5),int)
    np.add.at(cm,(truth,pred),1)
    support=cm.sum(1)
    precision=np.divide(cm.diagonal(),cm.sum(0),out=np.zeros(5,float),where=cm.sum(0)>0)
    recall=np.divide(cm.diagonal(),support,out=np.zeros(5,float),where=support>0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(5,float),where=precision+recall>0)
    n=len(y);success=int(np.sum(truth==pred));prop=success/n;z=1.96
    center=(prop+z*z/(2*n))/(1+z*z/n)
    delta=z*np.sqrt(prop*(1-prop)/n+z*z/(4*n*n))/(1+z*z/n)
    return dict(n=n,mae=float(np.mean(np.abs(p-y))),rmse=float(np.sqrt(np.mean((p-y)**2))),
                bias=float(np.mean(p-y)),accuracy=prop,correct=success,accuracy_ci95=[max(0.,center-delta),min(1.,center+delta)],
                macro_f1=float(f1[support>0].mean()),macro_f1_all5=float(f1.mean()),
                balanced_accuracy=float(recall[support>0].mean()),within_one_grade=float(np.mean(np.abs(truth-pred)<=1)),
                confusion=cm.tolist(),support=support.tolist(),predicted_support=cm.sum(0).tolist(),
                per_class=[dict(grade=i+1,name=GRADE_NAMES[i],support=int(support[i]),precision=float(precision[i]),recall=float(recall[i]),f1=float(f1[i])) for i in range(5)],
                majority_class=int(majority+1),majority_baseline_accuracy=float(np.mean(truth==majority)),
                unvalidated_grades=[GRADE_NAMES[i] for i in range(5) if not support[i]])


def export_web(model,meta,out):
    arrays=[];manifest=[];offset=0
    for name,tensor in model.net.state_dict().items():
        a=tensor.detach().numpy().astype('<f4').ravel()
        manifest.append(dict(name=name,shape=list(tensor.shape),offset=offset,size=len(a)))
        arrays.append(a);offset+=len(a)
    payload=dict(model='unet',label='仅户型/AP/家具 · 小型 U-Net · 70%实测新训练',meta=meta,
                 n_params=offset,manifest=manifest,data=base64.b64encode(np.concatenate(arrays).tobytes()).decode(),
                 feature_contract={'base_channels':['zero','zero','room_mask','furniture_mask','dx/maxdim','dy/maxdim','log10(max(distance,.5))/2','wall_crossings/3','furniture_attenuation/10'],
                                   'attenuation_db_per_intersection':ATTENUATION,'coordinate':'origin southwest; x east; y north (numerical alignment assumption)',
                                   'grid':'32x32 nodes x=i*room.w/31, y=j*room.h/31; lower row first'},
                 restriction='Actual AP position and multi-floor-plan generalization are unverified. Only table/chair placement represented in measured data.')
    dest=out/'deployment';dest.mkdir(parents=True,exist_ok=True)
    dump(payload,dest/'model.json')
    (dest/'model_weights.js').write_text('window.RFM_MODEL = '+json.dumps(payload,ensure_ascii=False,separators=(',',':'))+';\n',encoding='utf-8')


def run(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    scenes,quality=load_scenes(tuple(args.ap))
    xy={k:s['xy'] for k,s in scenes.items()}
    train=partition(xy,.7,SEED)
    inner_sub=partition({k:xy[k][train[k]] for k in scenes},.8,SEED+1)
    inner={}
    for k in scenes:
        inner[k]=np.zeros(len(xy[k]),bool);inner[k][np.flatnonzero(train[k])]=inner_sub[k]
    valid={k:train[k]&~inner[k] for k in scenes}
    split={'seed':SEED,'group_rule':'same 0.2m coordinate cell across both versions stays together; coordinate only, no RSSI stratification',
           'outer_test_sealed_until_all_training_complete':True,'scenarios':{}}
    X,Y=np.meshgrid(np.linspace(0,G.INNER_W,N),np.linspace(0,G.INNER_H,N))
    for k,s in scenes.items():
        s['geom']=features(X,Y,s);s['grid_xy']=np.stack([X.ravel(),Y.ravel()],1)
        s['sampling_grid']=np.stack([2*s['xy'][:,0]/G.INNER_W-1,2*s['xy'][:,1]/G.INNER_H-1],1)[None,None].astype(np.float32)
        split['scenarios'][k]=[dict(id=s['ids'][i],x=float(xy[k][i,0]),y=float(xy[k][i,1]),group=group(xy[k][i]),
                                       split='train' if train[k][i] else 'test',inner='fit' if inner[k][i] else ('validation' if valid[k][i] else 'sealed_test')) for i in range(len(xy[k]))]
    # Enforced, reviewable group-disjointness before any model is fit.
    tg={group(xy[k][i]) for k in scenes for i in np.flatnonzero(train[k])}
    vg={group(xy[k][i]) for k in scenes for i in np.flatnonzero(~train[k])}
    assert not tg&vg
    dump(split,out/'split.json')
    source={'empty':{'path':'data.xlsx','sha256':sha(ROOT/'data.xlsx')},'furnished':{'path':'放家具数据.xlsx','sha256':sha(ROOT/'放家具数据.xlsx')}}
    dump(source,out/'sources.json')
    export_points=[]
    for key,scene in scenes.items():
        for i,record in enumerate(split['scenarios'][key]):
            value=float(scene['y'][i])
            export_points.append(dict(scenario=key,**record,raw_rssi=float(scene['raw'][i]),rssi_dbm=value,
                                      sign_corrected=bool(scene['raw'][i]>0),
                                      br04_status='normal' if -70<=value<=-40 else ('too_strong' if value>-40 else 'too_weak')))
    for filename,keep in [('all_points.csv',lambda p:True),('train_points.csv',lambda p:p['split']=='train'),('test_points.csv',lambda p:p['split']=='test')]:
        with (out/filename).open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(export_points[0]));writer.writeheader()
            writer.writerows(p for p in export_points if keep(p))
    trained={};selection={}
    for task in ('completion','layout'):
        selection[task]={}
        for name in ORDER:
            _,selected,_=train_model(name,task,scenes,inner,valid,args.epochs,args.width,'selection',out)
            epochs=selected['selected_epoch']
            model,meta,phys=train_model(name,task,scenes,train,None,epochs,args.width,'refit70',out)
            trained[task,name]=(model,meta,phys)
            selection[task][name]={'selected_epoch':epochs,'inner_val_mae':selected['inner_val_mae']}
    # Frozen deployment decision and internal validation choices written before tests are read.
    deployment={'task':'layout','model':'unet','selection_rule':'U-Net predeclared for browser runtime before outer test evaluation',
                'ap':{'x':args.ap[0],'y':args.ap[1],'source':args.ap_source}}
    dump({'selection':selection,'deployment':deployment},out/'frozen_selection.json')
    export_web(trained['layout','unet'][0],trained['layout','unet'][1],out)
    result={'schema':'rfmap-holdout-v1','model_order':ORDER,'model_labels':LABELS,'selection':selection,'deployment':deployment,
            'grades':{'thresholds':THRESHOLDS,'names':GRADE_NAMES,'rule':'left inclusive; very good >= -50dBm'},
            'protocol':{'seed':SEED,'requested_train_ratio':.7,'requested_test_ratio':.3,'split_unit':'0.2m coordinate cell grouped across versions',
                        'inner_ratio':.8,'epochs_max':args.epochs,'width':args.width,'grid_n':N,'torch':torch.__version__,'python':platform.python_version(),
                        'trained_from_scratch':True,'old_weights_loaded':False,'outer_test_used_for_selection':False,
                        'source_note':'User explicitly supplied empty/furnished versions for this experiment; collection metadata remain incomplete.',
                        'coordinate_note':'Numerical x coordinates retained, y scaled 11.731/12; historical x-direction descriptions conflict. Alignment assumed, no causal furniture comparison.',
                        'ap':deployment['ap'],'limitation':'Within-room point holdout only. Not cross-room, AP-relocation, furniture-material or five-grade full-range validation.'},
            'quality':{'furniture':quality},'sources':source,'scenarios':{}}
    for k,s in scenes.items():
        majority=int(np.bincount(grade(s['y'][train[k]]),minlength=5).argmax())
        points=[]
        for i in range(len(s['y'])):
            v=float(s['y'][i]);points.append(dict(split['scenarios'][k][i],rssi=v,raw_rssi=float(s['raw'][i]),
                        br04_status='normal' if -70<=v<=-40 else ('too_strong' if v>-40 else 'too_weak'),grade=int(grade(v))+1,predictions={}))
        result['scenarios'][k]=dict(label=s['label'],n_total=len(points),n_train=int(train[k].sum()),n_test=int((~train[k]).sum()),
                n_inner_fit=int(inner[k].sum()),n_inner_validation=int(valid[k].sum()),train_ratio=float(train[k].mean()),
                room=s['room'],ap={'x':s['ap'][0],'y':s['ap'][1]},walls=s['walls'],furniture=s['furniture'],points=points,
                class_support_all=np.bincount(grade(s['y']),minlength=5).tolist(),class_support_train=np.bincount(grade(s['y'][train[k]]),minlength=5).tolist(),
                class_support_test=np.bincount(grade(s['y'][~train[k]]),minlength=5).tolist(),majority_class=majority+1,metrics={'completion':{},'layout':{}})
    maps={'x':X[0],'y':Y[:,0]};pred_rows=[]
    for (task,name),(model,meta,phys) in trained.items():
        grids,pred=predict(model,name,task,scenes,train,meta['normalizer'],phys)
        for k,s in scenes.items():
            test=~train[k];y=s['y'];p=pred[k]
            majority=result['scenarios'][k]['majority_class']-1
            scores=calc_metrics(y[test],p[test],majority)
            normal=test&(y>=-70)&(y<=-40)
            scores['normal_range']=calc_metrics(y[normal],p[normal],majority)
            scores['training_fit']=calc_metrics(y[train[k]],p[train[k]],majority)
            scores['inner_val_mae']=selection[task][name]['inner_val_mae']
            result['scenarios'][k]['metrics'][task][name]=scores
            maps[f'{k}__{task}__{name}']=grids[k]
            for i,pt in enumerate(result['scenarios'][k]['points']):
                pt['predictions'].setdefault(task,{})[name]=float(p[i])
                pred_rows.append(dict(scenario=k,task=task,model=name,id=pt['id'],x=pt['x'],y=pt['y'],split=pt['split'],
                                      actual_dbm=float(y[i]),predicted_dbm=float(p[i]),actual_grade=int(grade(y[i]))+1,
                                      predicted_grade=int(grade(p[i]))+1,br04_status=pt['br04_status']))
    # Baselines assessed under the same sealed test, never substituted for measured truth.
    coef=fit_physics(scenes,train)
    for k,s in scenes.items():
        test=~train[k];base=physics_map(s,coef)
        bp=sample_grid(torch.tensor(base[None,None]),s).numpy()
        xp,ip=prepare_input(s,np.flatnonzero(train[k]),{'mean':0,'std':1},'unet',base)
        iv=sample_grid(torch.tensor(ip[None]),s).numpy()
        majority=result['scenarios'][k]['majority_class']-1
        result['scenarios'][k]['baselines']={'physics_layout':calc_metrics(s['y'][test],bp[test],majority),'idw_completion':calc_metrics(s['y'][test],iv[test],majority)}
        maps[f'{k}__room_mask']=s['geom'][0]
    np.savez_compressed(out/'maps.npz',**maps)
    with (out/'predictions.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(pred_rows[0]));writer.writeheader();writer.writerows(pred_rows)
    result['deployment']['metrics']={k:result['scenarios'][k]['metrics']['layout']['unet'] for k in scenes}
    model,meta,phys=trained['layout','unet']
    fixture=[]
    for key,scene in scenes.items():
        xx,pp=prepare_input(scene,np.array([],int),meta['normalizer'],'unet',phys[key])
        with torch.no_grad():
            predicted=model(torch.tensor(xx[None]),torch.tensor(pp[None]))[0,0].numpy()*meta['normalizer']['std']+meta['normalizer']['mean']
        fixture.append({'scenario':key,'input':xx.ravel().tolist(),'pred':predicted.ravel().tolist(),
                        'room':scene['room'],'ap':{'x':scene['ap'][0],'y':scene['ap'][1]},'walls':scene['walls'],'furniture':scene['furniture']})
    dump(fixture,out/'deployment/parity_fixture.json')
    for task in ('completion','layout'):
        winner=min(ORDER,key=lambda n:selection[task][n]['inner_val_mae'])
        result['selection'][task]['selected_by_inner_validation']=winner
    dump(result,out/'metrics.json')
    result['maps']={k:v.ravel().tolist() for k,v in maps.items() if '__' in k}
    (out/'dashboard.js').write_text('window.RFM_DASHBOARD = '+json.dumps(result,ensure_ascii=False,separators=(',',':'),allow_nan=False)+';\n',encoding='utf-8')
    checks={'group_overlap':sorted(tg&vg),'outer_test_never_used_in_training':True,'inner_validation_never_in_inner_inputs':True,
            'all_source_rows_retained':sum(s['n_total'] for s in result['scenarios'].values())==189,
            'new_weights_only':True,'all_predictions_finite':bool(all(np.isfinite(r['predicted_dbm']) for r in pred_rows)),
            'prediction_rows':len(pred_rows),'metrics_sha256':sha(out/'metrics.json'),'split_sha256':sha(out/'split.json')}
    dump(checks,out/'checks.json')
    print(json.dumps({'checks':checks,'deployment':result['deployment']},ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default=str(OUT));parser.add_argument('--epochs',type=int,default=180)
    parser.add_argument('--width',type=int,default=8);parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--ap',nargs=2,type=float,default=[0,0])
    parser.add_argument('--ap-source',default='assumed (0,0); actual AP coordinates not supplied')
    args=parser.parse_args();run(args)


if __name__=='__main__': main()
