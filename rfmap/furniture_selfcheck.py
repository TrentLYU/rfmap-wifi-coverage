"""放家具实验交付前自检；失败时返回非零退出码。"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import zipfile

import numpy as np

from .furniture_models import MODEL_ORDER, load_trained, n_parameters

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'outputs/rfmap-furniture')
WORK = os.path.join(ROOT, '.deepworks/tmp/rfmap-furniture')


def main():
    checks = []

    def check(name, condition, details=None):
        checks.append({'name': name, 'passed': bool(condition), 'details': details})

    def load(path):
        with open(path, encoding='utf-8') as f:
            return json.load(f)

    q = load(os.path.join(OUT, 'data/data_quality.json'))
    furniture = load(os.path.join(OUT, 'data/furniture.json'))
    check('实测点数=140', q['n_points'] == 140, q['n_points'])
    check('缺失点=4且未插值冒充实测', q['n_missing'] == 4 and len(q['missing_cells']) == 4)
    check('家具=2桌6椅', q['table_count'] == 2 and q['chair_count'] == 6 and len(furniture) == 8)
    check('所有测点在房间内', not q['outside_room_idx'], q['outside_room_idx'])
    check('RSSI符号修正数量', q['raw_positive_count'] == 120 and q['raw_negative_count'] == 20)
    with open(os.path.join(OUT, 'data/measured.csv'), encoding='utf-8-sig') as f:
        measured = list(csv.DictReader(f))
    check('规范化CSV=140行', len(measured) == 140, len(measured))
    check('RSSI=-abs(raw)逐行成立', all(abs(float(r['rssi_dbm']) + abs(float(r['raw_rssi']))) < 1e-9
                                      for r in measured))

    for split in ('train', 'val', 'test'):
        a = np.load(os.path.join(WORK, 'sim', split + '.npz'))
        b = np.load(os.path.join(WORK, 'sim_scatter', split + '.npz'))
        check(f'{split}: 常规/散落同户型', np.array_equal(a['room'], b['room']))
        check(f'{split}: 常规/散落同家具', np.array_equal(a['furniture'], b['furniture']))
        check(f'{split}: 常规/散落同真值', np.array_equal(a['target'], b['target']))
        check(f'{split}: 两类观测数量相同', np.array_equal(a['obs_mask'].sum((1, 2)),
                                                            b['obs_mask'].sum((1, 2))))
        obs = a['obs_mask'] > 0.5
        check(f'{split}: 观测值来自真值', np.allclose(a['obs_value'][obs], a['target'][obs]))
        room = a['room'] > 0.5
        check(f'{split}: 房间内真值有限', np.isfinite(a['target'][room]).all())
        check(f'{split}: 家具不改动房间有效区', np.all(a['furniture'][~room] == 0))
        check(f'{split}: 真值范围合法', float(a['target'][room].min()) >= -95.0 and
              float(a['target'][room].max()) <= -15.0,
              [float(a['target'][room].min()), float(a['target'][room].max())])

    train = load(os.path.join(OUT, 'training_summary.json'))
    expected_channels = {'ae': 4, 'unet': 4, 'resunet': 5, 'unet_scatter': 4, 'pgmpr': 8}
    for name in MODEL_ORDER:
        meta = train[name]
        hist = load(os.path.join(OUT, 'models', name, 'history.json'))
        check(f'{name}: 已脱离第0轮初始权重', meta['best_epoch'] > 0, meta['best_epoch'])
        check(f'{name}: 训练轮数一致', len(hist['epoch']) == meta['epochs_run'])
        check(f'{name}: 损失全部有限', np.isfinite(hist['train']).all() and np.isfinite(hist['val']).all())
        check(f'{name}: 最佳验证点一致', abs(min(hist['val']) - meta['best_val_loss']) < 1e-6)
        check(f'{name}: 输入通道正确', meta['n_input_channels'] == expected_channels[name])
        model, loaded_meta, _ = load_trained(os.path.join(OUT, 'models', name), 'cpu')
        check(f'{name}: 预训练权重可加载且参数量一致', n_parameters(model) == meta['params'])
        ft, _, _ = load_trained(os.path.join(OUT, 'models', name), 'cpu', finetuned=True)
        check(f'{name}: 全量微调权重可加载', n_parameters(ft) == meta['params'])

    real = load(os.path.join(OUT, 'results/metrics_real.json'))
    expected_hidden = {'corner': 6, 'center': 9, 'furniture': 13}
    for proto, entry in real['protocols'].items():
        check(f'{proto}: 输入+隐藏=140', entry['n_input'] + entry['n_hidden'] == 140)
        check(f'{proto}: 隐藏点数正确', entry['n_hidden'] == expected_hidden[proto])
        check(f'{proto}: 隐藏坐标与逐点记录一致',
              all([r[0], r[1], r[2]] == p[:3] for r, p in
                  zip(entry['hidden_points'], entry['methods']['ae']['per_point'])))
        max_diff = 0.0
        for method, m in entry['methods'].items():
            rows = m['per_point']
            err = np.array([r[3] - r[2] for r in rows], dtype=float)
            rmse = float(np.sqrt((err ** 2).mean()))
            max_diff = max(max_diff, abs(rmse - m['rmse']))
        check(f'{proto}: 全部逐点RMSE可重算', max_diff <= 0.002, max_diff)
    check('点数扫描每组20次', all(m['runs'] == 20 for entry in real['sweep'].values()
                                   for m in entry.values()))

    sim = load(os.path.join(OUT, 'results/metrics_sim.json'))
    for tag, entry in sim['testsets'].items():
        check(f'{tag}: 独立测试200场景', entry['n_samples'] == 200)
        check(f'{tag}: 每种方法均评估200次', all(m['runs'] == 200 for m in entry['methods'].values()))

    for name in MODEL_ORDER:
        p = os.path.join(OUT, 'predictions', name + '_full_map.npz')
        d = np.load(p)
        room = d['room'] > 0.5
        check(f'{name}: 整图64×64', d['rssi'].shape == (64, 64))
        check(f'{name}: 有效区预测全部有限', np.isfinite(d['rssi'][room]).all())
        with open(os.path.join(OUT, 'predictions', name + '_full_map.csv'), encoding='utf-8-sig') as f:
            n_csv = sum(1 for _ in f) - 1
        check(f'{name}: CSV覆盖所有有效格点', n_csv == int(room.sum()), [n_csv, int(room.sum())])

    deliverables = [
        '放家具环境_RF覆盖预测实验报告.html', '放家具环境_RF覆盖预测实验报告.md',
        '放家具环境_RF覆盖预测实验报告.docx', '实验结果汇总.xlsx',
        '复现与模型使用说明.md', 'run_manifest.json', 'training_summary.json',
    ]
    for filename in deliverables:
        path = os.path.join(OUT, filename)
        check(f'交付文件存在：{filename}', os.path.isfile(path) and os.path.getsize(path) > 0)
    with open(os.path.join(OUT, '放家具环境_RF覆盖预测实验报告.html'), encoding='utf-8') as f:
        n_report_figures = f.read().count('<figure><img ')
    with zipfile.ZipFile(os.path.join(OUT, '放家具环境_RF覆盖预测实验报告.docx')) as z:
        xml = z.read('word/document.xml')
        check(f'Word含表格和{n_report_figures}幅图（与HTML一致）',
              xml.count(b'<w:tbl>') >= 10 and xml.count(b'<w:drawing>') == n_report_figures,
              {'tables': xml.count(b'<w:tbl>'), 'drawings': xml.count(b'<w:drawing>'),
               'html_figures': n_report_figures})
    import openpyxl
    wb = openpyxl.load_workbook(os.path.join(OUT, '实验结果汇总.xlsx'), read_only=True)
    check('Excel含7个数据表', len(wb.sheetnames) == 7, wb.sheetnames)

    manifest = load(os.path.join(OUT, 'run_manifest.json'))
    h = hashlib.sha256()
    with open(os.path.join(ROOT, '放家具数据.xlsx'), 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    check('输入SHA-256与清单一致', h.hexdigest() == manifest['input_sha256'])

    result = {'passed': all(c['passed'] for c in checks),
              'n_checks': len(checks), 'n_passed': sum(c['passed'] for c in checks),
              'n_failed': sum(not c['passed'] for c in checks), 'checks': checks}
    out = os.path.join(OUT, 'results/validation.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: result[k] for k in ('passed', 'n_checks', 'n_passed', 'n_failed')},
                     ensure_ascii=False, indent=2))
    for c in checks:
        if not c['passed']:
            print('失败：', c)
    print('自检报告：', out)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
