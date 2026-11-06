"""Deploy the trained layout U-Net using geometry and router position only.

Example:
    .venv/bin/python -m rfmap.holdout_infer \
      --project outputs/rfmap-70-30/studio/sample_room.json \
      --out outputs/rfmap-70-30/inference-sample

The only input files opened by this module are the project JSON and the exported
deployment model JSON. Workbooks, experiment metrics and held-out values are not
loaded. Geometry construction is shared with the audited training implementation.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .holdout_experiment import features
from .models import SmallUNet

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / 'outputs/rfmap-70-30/deployment/model.json'
N = 32
THRESHOLDS = [-80., -70., -60., -50.]
GRADE_NAMES = ['很差', '较差', '一般', '好', '非常好']


def finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{label} must be a finite number')
    return float(value)


def normalize_project(project):
    """Whitelist geometry fields; no RSSI values enter the model input."""
    if project.get('points'):
        raise ValueError('Layout-only inference does not accept measured points. Remove the points field.')
    room = {key: finite_number(project['room'][key], f'room.{key}') for key in ('w', 'h')}
    if not all(2 <= v <= 50 for v in room.values()):
        raise ValueError('room.w and room.h must be between 2 and 50 metres')
    ap = tuple(finite_number(project['ap'][key], f'ap.{key}') for key in ('x', 'y'))
    if not (0 <= ap[0] <= room['w'] and 0 <= ap[1] <= room['h']):
        raise ValueError('Router position must be inside the rectangular room boundary')
    scene = {'room':room, 'ap':ap, 'walls':[], 'furniture':[]}
    for category in ('walls', 'furniture'):
        for index, obj in enumerate(project.get(category, [])):
            item = {key: finite_number(obj[key], f'{category}[{index}].{key}') for key in ('x','y','w','h')}
            if item['w'] <= 0 or item['h'] <= 0:
                raise ValueError(f'{category}[{index}] dimensions must be positive')
            if category == 'furniture':
                item['kind'] = str(obj.get('kind', 'default'))
                item['shape'] = obj.get('shape', 'rect')
                if item['shape'] not in ('rect', 'circle'):
                    raise ValueError(f'{category}[{index}].shape must be rect or circle')
            scene[category].append(item)
    return scene


def load_network(payload):
    if payload.get('model') != 'unet' or payload['meta'].get('task') != 'layout':
        raise ValueError('Expected the exported layout U-Net model')
    meta = payload['meta']
    if meta.get('n_input_channels') != 9:
        raise ValueError('This predictor requires the 9-channel layout model')
    raw = np.frombuffer(base64.b64decode(payload['data'], validate=True), dtype='<f4')
    state = {}
    for item in payload['manifest']:
        offset, size = int(item['offset']), int(item['size'])
        shape = tuple(int(v) for v in item['shape'])
        if offset < 0 or size != math.prod(shape) or offset+size > len(raw):
            raise ValueError('Invalid deployment weight manifest')
        array = raw[offset:offset+size].reshape(shape).copy()
        if not np.isfinite(array).all():
            raise ValueError('Non-finite model weights')
        state[item['name']] = torch.from_numpy(array)
    model = SmallUNet(width=int(meta['width']), in_ch=9)
    model.load_state_dict(state, strict=True)
    return model.eval()


def predict_layout(project, payload):
    """Pure geometry + trained weights → 32×32 RSSI grid; no file access."""
    scene = normalize_project(project)
    x, y = np.linspace(0, scene['room']['w'], N), np.linspace(0, scene['room']['h'], N)
    X, Y = np.meshgrid(x, y)
    geometry = features(X, Y, scene)
    channels = np.concatenate([np.zeros((2,N,N), np.float32), geometry])
    model = load_network(payload)
    norm = payload['meta']['normalizer']
    with torch.no_grad():
        raw = model(torch.from_numpy(channels[None]))[0,0].numpy()*float(norm['std'])+float(norm['mean'])
    if not np.isfinite(raw).all():
        raise RuntimeError('Prediction contains non-finite RSSI')
    valid = geometry[0] > .5
    # -120 is the same sentinel as the browser; excluded rows receive grade zero.
    pred = np.where(valid, raw, -120.).astype(np.float32)
    grades = np.where(valid, np.searchsorted(THRESHOLDS, pred, side='right')+1, 0).astype(np.int8)
    return {'x':x, 'y':y, 'rssi_dbm':pred, 'raw_rssi_dbm':raw,
            'grade':grades, 'inside_room':valid, 'channels':channels}


def main():
    parser = argparse.ArgumentParser(description='Predict Wi-Fi RSSI from room, AP, walls and furniture only.')
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    project_bytes, model_bytes = args.project.read_bytes(), args.model.read_bytes()
    project, payload = json.loads(project_bytes), json.loads(model_bytes)
    grid = predict_layout(project, payload)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out/'grid.npz', **grid)
    with (args.out/'grid.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=['row','column','x_m','y_m','inside_room','rssi_dbm','grade','grade_label'])
        writer.writeheader()
        for j,y in enumerate(grid['y']):
            for i,x in enumerate(grid['x']):
                valid = bool(grid['inside_room'][j,i]); level=int(grid['grade'][j,i])
                writer.writerow({'row':j,'column':i,'x_m':float(x),'y_m':float(y),'inside_room':valid,
                                 'rssi_dbm':float(grid['rssi_dbm'][j,i]) if valid else '',
                                 'grade':level,'grade_label':GRADE_NAMES[level-1] if valid else '墙体/无效区域'})
    valid_values=grid['rssi_dbm'][grid['inside_room']]
    metadata = {'schema':'rfmap-layout-inference-v1','input_files':[str(args.project.resolve()),str(args.model.resolve())],
                'project_sha256':hashlib.sha256(project_bytes).hexdigest(),
                'model_sha256':hashlib.sha256(model_bytes).hexdigest(),
                'input_channels':'9 geometry-only channels; RSSI and observation-mask channels are identically zero',
                'measured_data_read':False,'hidden_test_data_read':False,'nx':N,'ny':N,
                'room':project['room'],'ap':project['ap'],
                'grid_convention':'nodes inclusive of room boundaries; first row y=0; x=i*w/31, y=j*h/31',
                'grades':{'thresholds':THRESHOLDS,'names':GRADE_NAMES,'invalid_grade':0},
                'valid_nodes':int(grid['inside_room'].sum()),
                'predicted_rssi_range':[float(valid_values.min()),float(valid_values.max())] if len(valid_values) else None,
                'limitations':['Trained on one numerical room geometry and assumed AP position; changed layouts/AP positions are extrapolation.',
                               'Furniture attenuation defaults are assumptions, not measured per-material calibration.',
                               'The training data do not validate all five signal grades.']}
    (args.out/'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'grid_csv':str(args.out/'grid.csv'),'grid_npz':str(args.out/'grid.npz'),
                      'valid_nodes':metadata['valid_nodes'],'rssi_range':metadata['predicted_rssi_range'],
                      'measured_data_read':False}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
