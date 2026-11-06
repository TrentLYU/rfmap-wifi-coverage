"""Independent audit of the 70/30 experiment, saved predictions and browser model.

Run after training: .venv/bin/python -m rfmap.holdout_selfcheck
This never changes training data, model weights or reported experiment metrics.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import subprocess
import tempfile
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch

from . import holdout_experiment as E


def independent_metrics(actual, predicted):
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    # Deliberately independent of the experiment's classification helper.
    ag = np.sum(actual[:, None] >= np.array([-80, -70, -60, -50]), axis=1)
    pg = np.sum(predicted[:, None] >= np.array([-80, -70, -60, -50]), axis=1)
    confusion = np.zeros((5, 5), int)
    for a, p in zip(ag, pg):
        confusion[a, p] += 1
    error = predicted - actual
    recall = np.divide(np.diag(confusion), confusion.sum(1), out=np.zeros(5), where=confusion.sum(1) > 0)
    f1 = np.divide(2 * np.diag(confusion), confusion.sum(0) + confusion.sum(1),
                   out=np.zeros(5), where=confusion.sum(0) + confusion.sum(1) > 0)
    return {'mae': float(np.abs(error).mean()),
            'rmse': float(np.sqrt(np.mean(error ** 2))),
            'bias': float(error.mean()), 'accuracy': float(np.mean(ag == pg)),
            'confusion': confusion.tolist(), 'support': confusion.sum(1).tolist(),
            'correct': int(np.sum(ag == pg)), 'n': len(actual),
            'macro_f1': float(f1[confusion.sum(1) > 0].mean()),
            'macro_f1_all5': float(f1.mean()),
            'balanced_accuracy': float(recall[confusion.sum(1) > 0].mean()),
            'within_one_grade': float(np.mean(np.abs(ag - pg) <= 1))}


def initialize_geometry(scenes):
    x, y = np.meshgrid(np.linspace(0, E.G.INNER_W, E.N),
                       np.linspace(0, E.G.INNER_H, E.N))
    for scene in scenes.values():
        scene['geom'] = E.features(x, y, scene)
        scene['grid_xy'] = np.stack([x.ravel(), y.ravel()], 1)
        scene['sampling_grid'] = np.stack([
            2 * scene['xy'][:, 0] / E.G.INNER_W - 1,
            2 * scene['xy'][:, 1] / E.G.INNER_H - 1,
        ], 1)[None, None].astype(np.float32)


def browser_cases(scenes):
    cases = []
    for key, s in scenes.items():
        cases.append({'name': key, 'room': s['room'], 'ap': {'x': s['ap'][0], 'y': s['ap'][1]},
                      'walls': s['walls'], 'furniture': s['furniture']})
    # Exercise furniture types absent from measured data, moved AP, tangencies,
    # AP on furniture, and an AP exactly at a grid node.
    cases.append({'name': 'boundary', 'room': {'w': 12.4, 'h': 12.4},
                  'ap': {'x': 0, 'y': 0}, 'walls': [{'x': 2, 'y': 1, 'w': 1, 'h': 1}],
                  'furniture': [
                      {'kind': 'table', 'x': 0, 'y': 0, 'w': .8, 'h': .8, 'shape': 'circle'},
                      {'kind': 'cabinet', 'x': 4, 'y': 4, 'w': 1.2, 'h': .6, 'shape': 'rect'},
                      {'kind': 'toilet', 'x': 8, 'y': 8, 'w': .7, 'h': .8, 'shape': 'circle'},
                      {'kind': 'bed', 'x': 9, 'y': 3, 'w': 1.8, 'h': 2, 'shape': 'rect'},
                  ]})
    cases.append({'name': 'moved_ap', 'room': {'w': 9, 'h': 6},
                  'ap': {'x': 4.5, 'y': 3}, 'walls': [{'x': 3, 'y': 1, 'w': .2, 'h': 2}],
                  'furniture': [{'kind': 'sofa', 'x': 5, 'y': 4, 'w': 2, 'h': 1, 'shape': 'rect'}]})
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=E.OUT)
    parser.add_argument('--studio', type=Path, default=E.OUT / 'studio')
    parser.add_argument('--skip-tamper-training', action='store_true')
    args = parser.parse_args()
    out = args.out
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    result = json.loads((out / 'metrics.json').read_text())
    split = json.loads((out / 'split.json').read_text())
    checks = []

    def check(name, passed, details=None):
        checks.append({'name': name, 'passed': bool(passed), 'details': details})

    ap = result['protocol']['ap']
    scenes, _ = E.load_scenes((ap['x'], ap['y']))
    initialize_geometry(scenes)
    train, inner, validation = {}, {}, {}
    for key, scene in scenes.items():
        records = split['scenarios'][key]
        points = result['scenarios'][key]['points']
        train[key] = np.array([p['split'] == 'train' for p in records])
        inner[key] = np.array([p['inner'] == 'fit' for p in records])
        validation[key] = np.array([p['inner'] == 'validation' for p in records])
        expected = (34, 15) if key == 'empty' else (98, 42)
        check(f'{key}: exact 70/30 rounded counts', (sum(train[key]), sum(~train[key])) == expected)
        check(f'{key}: source rows, ids and truth retained',
              len(records) == len(scene['y']) == len(points)
              and [p['id'] for p in records] == scene['ids']
              and np.array_equal([p['rssi'] for p in points], scene['y'])
              and np.array_equal([p['raw_rssi'] for p in points], scene['raw']))
        check(f'{key}: inner fit/validation partition only the 70%',
              not np.any(inner[key] & validation[key])
              and np.array_equal(inner[key] | validation[key], train[key]))
        check(f'{key}: source coordinates match split',
              np.allclose([[p['x'], p['y']] for p in records], scene['xy'], atol=1e-10))
    for masks, title in [(train, 'outer'), (inner, 'inner')]:
        a = {E.group(s['xy'][i]) for k, s in scenes.items() for i in np.flatnonzero(masks[k])}
        b = {E.group(s['xy'][i]) for k, s in scenes.items() for i in np.flatnonzero(~masks[k])}
        check(f'{title}: no cross-version coordinate group overlap', not a & b, sorted(a & b))
    fresh_partition = E.partition({k: s['xy'] for k, s in scenes.items()}, .7, E.SEED)
    check('saved partition reproducible', all(np.array_equal(train[k], fresh_partition[k]) for k in scenes))
    nodes = defaultdict(list)
    for key, scene in scenes.items():
        for i, (x, y) in enumerate(scene['xy']):
            node = (round(x / E.G.INNER_W * (E.N - 1)), round(y / E.G.INNER_H * (E.N - 1)))
            nodes[node].append({'scenario': key, 'id': scene['ids'][i], 'x': float(x), 'y': float(y),
                                'split': 'train' if train[key][i] else 'test'})
    aliases = [{'node': list(node), 'points': pts} for node, pts in nodes.items()
               if len({p['split'] for p in pts}) > 1]
    for source in result['sources'].values():
        check(f'source SHA256: {source["path"]}', E.sha(E.ROOT / source['path']) == source['sha256'])

    # Threshold boundaries including values immediately below and at each cut.
    edge = np.array([-81, -80.00001, -80, -70.00001, -70, -60.00001, -60, -50.00001, -50, 0])
    check('five-grade inclusive boundary semantics',
          np.array_equal(E.grade(edge), [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]))

    maps = np.load(out / 'maps.npz')
    csv_rows = list(csv.DictReader((out / 'predictions.csv').open(encoding='utf-8-sig')))
    check('predictions CSV contains all 189 points x 2 tasks x 5 methods', len(csv_rows) == 1890)
    lookup = {(r['scenario'], r['task'], r['model'], int(r['id'])): r for r in csv_rows}
    check('predictions CSV has no repeated scenario/task/model/id', len(lookup) == len(csv_rows))
    deployed = None
    for task in ('completion', 'layout'):
        for name in E.ORDER:
            directory = out / 'models' / task / name / 'refit70'
            meta = json.loads((directory / 'meta.json').read_text())
            selected = json.loads((directory.parent / 'selection' / 'meta.json').read_text())
            history = json.loads((directory.parent / 'selection' / 'history.json').read_text())
            for phase_meta, masks, phase in [(meta, train, 'refit70'), (selected, inner, 'selection')]:
                values = np.concatenate([s['y'][masks[k]] for k, s in scenes.items()])
                check(f'{task}/{name}/{phase}: normalizer uses allowed training only',
                      abs(phase_meta['normalizer']['mean'] - float(values.mean())) < 1e-6
                      and abs(phase_meta['normalizer']['std'] - max(float(values.std()), 1)) < 1e-6)
                check(f'{task}/{name}/{phase}: physics uses allowed training only',
                      np.allclose(phase_meta['physics_coefficients'], E.fit_physics(scenes, masks), atol=1e-10))
                expected_train = {k: [s['ids'][i] for i in np.flatnonzero(masks[k])] for k, s in scenes.items()}
                expected_val = ({k: [s['ids'][i] for i in np.flatnonzero(validation[k])] for k, s in scenes.items()}
                                if phase == 'selection' else {})
                check(f'{task}/{name}/{phase}: stored training/validation ids match split',
                      phase_meta['training_point_ids'] == expected_train and phase_meta['validation_point_ids'] == expected_val)
            best = min(history, key=lambda row: row['inner_val_mae_db'])
            check(f'{task}/{name}: refit epochs chosen by inner validation',
                  best['epoch'] == meta['epochs'] == selected['selected_epoch'])
            check(f'{task}/{name}: model hash matches metadata', E.sha(directory / 'model.pt') == meta['model_sha256'])
            model = E.Net(name, meta['width'])
            model.load_state_dict(torch.load(directory / 'model.pt', map_location='cpu', weights_only=True))
            phys = {k: E.physics_map(s, np.array(meta['physics_coefficients'])) for k, s in scenes.items()}
            grids, predictions = E.predict(model, name, task, scenes, train, meta['normalizer'], phys)
            for key, scene in scenes.items():
                points = result['scenarios'][key]['points']
                saved = np.array([p['predictions'][task][name] for p in points])
                delta = float(np.max(np.abs(saved - predictions[key])))
                check(f'{key}/{task}/{name}: saved model reproduces point predictions', delta < 5e-4, delta)
                check(f'{key}/{task}/{name}: saved grid matches restored model',
                      np.allclose(maps[f'{key}__{task}__{name}'], grids[key], atol=5e-4))
                score = independent_metrics(scene['y'][~train[key]], saved[~train[key]])
                reported = result['scenarios'][key]['metrics'][task][name]
                check(f'{key}/{task}/{name}: metrics independently recomputed',
                      all(abs(score[k] - reported[k]) < 1e-5 for k in
                          ('mae', 'rmse', 'bias', 'accuracy', 'macro_f1', 'macro_f1_all5', 'balanced_accuracy', 'within_one_grade'))
                      and all(score[k] == reported[k] for k in ('confusion', 'support', 'n', 'correct')))
                check(f'{key}/{task}/{name}: CSV and report values agree',
                      all(abs(float(lookup[key, task, name, p['id']]['predicted_dbm']) - v) < 1e-10
                          for p, v in zip(points, saved)))
            if task == 'layout' and name == 'unet':
                deployed = (model, meta)

    # Metamorphic leakage check: change every sealed label dramatically while
    # preserving all coordinates, then verify features, physics and training.
    tampered = copy.deepcopy(scenes)
    for key in scenes:
        tampered[key]['y'][~train[key]] = np.linspace(-180, 30, sum(~train[key]))
    norm = deployed[1]['normalizer']
    coef = E.fit_physics(scenes, train)
    check('test-label tampering cannot alter fitted physics', np.array_equal(coef, E.fit_physics(tampered, train)))
    for key, scene in scenes.items():
        phys = E.physics_map(scene, coef)
        for name in E.ORDER:
            for task in ('completion', 'layout'):
                allowed = np.flatnonzero(train[key]) if task == 'completion' else np.array([], int)
                old = E.prepare_input(scene, allowed, norm, name, phys)
                new = E.prepare_input(tampered[key], allowed, norm, name, phys)
                check(f'{key}/{task}/{name}: sealed-label tampering cannot alter input',
                      all(np.array_equal(a, b) for a, b in zip(old, new)))
                if task == 'layout':
                    check(f'{key}/{name}: layout RSSI/mask channels empty', np.count_nonzero(old[0][:2]) == 0)
    if not args.skip_tamper_training:
        with tempfile.TemporaryDirectory(prefix='rfmap-tamper-') as tmp, redirect_stdout(io.StringIO()):
            for task in ('completion', 'layout'):
                a, _, _ = E.train_model('unet', task, scenes, train, None, 2, 4, 'original', Path(tmp))
                b, _, _ = E.train_model('unet', task, tampered, train, None, 2, 4, 'tampered', Path(tmp))
                check(f'{task}: test-label tampering leaves actual training weights identical',
                      all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items()))

    # Cross-runtime parity checks use the actual browser files and deployment.
    node = Path('/Users/lvzhaonan/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node')
    cases = browser_cases(scenes)
    script = r'''
const fs=require('fs'),vm=require('vm');global.window=global;
vm.runInThisContext(fs.readFileSync(process.argv[2],'utf8'));
vm.runInThisContext(fs.readFileSync(process.argv[3],'utf8'));
vm.runInThisContext(fs.readFileSync(process.argv[4],'utf8'));
const cases=JSON.parse(fs.readFileSync(process.argv[5],'utf8'));
const output=cases.map(p=>{const x=RFM_Geometry.buildInput(p);const q=RFM_LayoutInfer(x);
return {name:p.name,channels:Array.from(x.channels),room:Array.from(x.roomMask),prediction:Array.from(q.pred)};});
process.stdout.write(JSON.stringify(output));
'''
    with tempfile.TemporaryDirectory(prefix='rfmap-browser-') as tmp:
        tmp = Path(tmp)
        (tmp / 'parity.cjs').write_text(script)
        (tmp / 'cases.json').write_text(json.dumps(cases))
        response = subprocess.run([str(node), str(tmp / 'parity.cjs'), str(out / 'deployment/model_weights.js'),
                                   str(args.studio / 'geometry.js'), str(args.studio / 'nn_model.js'),
                                   str(tmp / 'cases.json')], check=True, capture_output=True, text=True)
        browser = json.loads(response.stdout)
    model, meta = deployed
    model.eval()
    for case, actual in zip(cases, browser):
        width, height = case['room']['w'], case['room']['h']
        x, y = np.meshgrid(np.linspace(0, width, E.N), np.linspace(0, height, E.N))
        scene = dict(case, ap=(case['ap']['x'], case['ap']['y']))
        feat = E.features(x, y, scene)
        inp = np.concatenate([np.zeros((2, E.N, E.N), np.float32), feat])
        feature_delta = float(np.max(np.abs(inp.ravel() - actual['channels'])))
        disagreement = np.flatnonzero(np.abs(inp.ravel() - actual['channels']) > 2e-6)
        details = {'max_delta': feature_delta, 'differences': [
            {'channel': int(index // E.N ** 2), 'j': int(index % E.N ** 2 // E.N),
             'i': int(index % E.N), 'python': float(inp.ravel()[index]),
             'javascript': actual['channels'][index]}
            for index in disagreement[:12]]}
        check(f'JS/Python {case["name"]}: nine geometry channels agree', feature_delta < 2e-6, details)
        with torch.no_grad():
            prediction = model(torch.tensor(inp[None]), torch.zeros(1, 1, E.N, E.N))[0, 0].numpy()
        prediction = prediction * meta['normalizer']['std'] + meta['normalizer']['mean']
        mask = feat[0].ravel() > .5
        delta = float(np.max(np.abs(prediction.ravel()[mask] - np.array(actual['prediction'])[mask])))
        check(f'JS/Python {case["name"]}: deployed network forward agrees', delta < .001, delta)
    report = {'passed': all(c['passed'] for c in checks), 'n_checks': len(checks),
              'n_passed': sum(c['passed'] for c in checks), 'n_failed': sum(not c['passed'] for c in checks),
              'metrics_sha256': hashlib.sha256((out / 'metrics.json').read_bytes()).hexdigest(),
              'nearby_cross_partition_node_aliases': aliases,
              'alias_interpretation': 'Different measured coordinates may round to the same 32-node cell. This is a spatial-resolution limitation, not repeated ground-truth labels.',
              'checks': checks}
    E.dump(report, out / 'independent_validation.json')
    print(json.dumps({k: report[k] for k in ('passed', 'n_checks', 'n_passed', 'n_failed')}, indent=2))
    for item in checks:
        if not item['passed']:
            print('FAIL', item)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
