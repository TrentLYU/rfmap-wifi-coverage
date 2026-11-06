"""Portable trained RSSI model query; only NumPy is required."""
import argparse
import json
from pathlib import Path
import numpy as np


def predict(model,points):
    X=np.atleast_2d(np.asarray(points,dtype=float))
    if X.shape[1]!=2 or not np.isfinite(X).all():raise ValueError('Supply finite x,y coordinates in meters.')
    bounds=np.asarray(model['coordinate_bounds'])
    if np.any(X<bounds[:,0]) or np.any(X>bounds[:,1]):raise ValueError('Outside the room bounds.')
    polygon=np.asarray(model['support_polygon_xy'])
    for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
        edge=b-a;v=X-a
        if np.any(edge[0]*v[:,1]-edge[1]*v[:,0]<-1e-8):
            raise ValueError('Outside measured spatial support; additional measurements needed.')
    f=model['formula'];d=np.maximum(np.linalg.norm(X-np.asarray(model['router_xy']),axis=1),1.)
    baseline=f['p0_dbm']-10*f['n']*np.log10(d)
    family=model['family'];config=model['configuration']
    if family=='formula':return baseline
    if family in ['idw','physics_idw']:
        distance=np.maximum(np.linalg.norm(X[:,None,:]-np.asarray(model['training_xy'])[None,:,:],axis=-1),1e-9)
        w=distance**(-config['power']);w/=w.sum(axis=1,keepdims=True)
        correction=w@np.asarray(model['targets'])
    elif family in ['mlp','physics_mlp']:
        values=[]
        for m in model['neural']['models']:
            a=(X-np.asarray(m['input_mean']))/np.asarray(m['input_scale'])
            for i,(w,b) in enumerate(zip(m['coefs'],m['intercepts'])):
                a=a@np.asarray(w)+np.asarray(b)
                if i<len(m['coefs'])-1:a=np.tanh(a)
            values.append(a.ravel()*m['target_scale'][0]+m['target_mean'][0])
        correction=np.mean(values,axis=0)
    else:
        g=model['gp'];distance=np.linalg.norm(X[:,None,:]-np.asarray(g['training_xy'])[None,:,:],axis=-1)/g['length_scale']
        if g['kernel']=='rbf':kernel=np.exp(-.5*distance**2)
        else:
            z=np.sqrt(5)*distance;kernel=(1+z+z*z/3)*np.exp(-z)
        correction=g['residual_mean']+g['residual_scale']*g['amplitude']*(kernel@np.asarray(g['alpha']))
    return correction if family in ['mlp','idw'] else baseline+config['weight']*correction


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Fixed-router RSSI query; x up, y left.')
    parser.add_argument('--x',required=True,type=float);parser.add_argument('--y',required=True,type=float)
    parser.add_argument('--model',choices=['recommended','neural'],default='recommended')
    args=parser.parse_args();root=Path(__file__).resolve().parent/'optimized'
    filename='recommended_model.json' if args.model=='recommended' else 'best_neural_model.json'
    m=json.loads((root/filename).read_text(encoding='utf-8'));p=predict(m,[[args.x,args.y]])[0]
    print(json.dumps({'family':m['family'],'x_m':args.x,'y_m':args.y,'predicted_rssi_dbm':round(float(p),3),
                      'router_xy':m['router_xy']}))
