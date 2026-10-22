"""统一重训五个方案，保证横向对比的公平性。

为什么必须"统一重训"
--------------------
方案四（散落布点）和方案二（常规布点）是**配对消融**：两者网络结构、超参、
随机种子完全一致，唯一差别是训练数据的布点方式。如果方案二沿用早期在 CPU 上
训练出来的权重、方案四却用 MPS 训练，差异里就混进了"设备/浮点路径"这个变量，
数据消融的结论就不成立了。

因此本脚本把**五个方案**放在同一设备、同一组超参、同一随机种子下重训：

======================  ====================  ==========================================
方案                    训练数据              与谁配对比较
======================  ====================  ==========================================
方案一 CompletionAE      data/sim              方案二/三/五（同数据）
方案二 SmallUNet         data/sim              方案四（配对的对照组）
方案三 InterpResidual    data/sim              ——
方案四 SmallUNet         data/sim_scatter      方案二（唯一变量 = 布点方式）
方案五 PGMPRUNet         data/sim + scatter    全部（模拟"数据更多样"的收益）
======================  ====================  ==========================================

用法::

    .venv/bin/python -m rfmap.train_stages                 # 全部
    .venv/bin/python -m rfmap.train_stages --only unet unet_scatter
"""

from __future__ import annotations

import argparse
import json
import os
import time

from rfmap.train import train_sim

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 统一超参：五个方案完全一致，只有"数据"和"结构"两个变量
COMMON = dict(epochs=120, width=24, batch_size=16, lr=2e-3, seed=7, patience=35)

STAGES = [
    dict(name='ae', data_dir='data/sim'),
    dict(name='unet', data_dir='data/sim'),
    dict(name='resunet', data_dir='data/sim'),
    dict(name='unet_scatter', data_dir='data/sim_scatter'),
    dict(name='pgmpr', data_dir='data/sim',
         train_files=['data/sim/train.npz', 'data/sim_scatter/train.npz']),
]


def main(argv=None):
    ap = argparse.ArgumentParser(description='统一重训五个方案')
    ap.add_argument('--only', nargs='*', default=None)
    ap.add_argument('--device', default='mps')
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--epochs', type=int, default=COMMON['epochs'])
    args = ap.parse_args(argv)

    os.chdir(args.root)
    todo = [s for s in STAGES if args.only is None or s['name'] in args.only]
    summary = {}
    t_all = time.time()

    for st in todo:
        name = st['name']
        print(f'\n{"=" * 70}\n== 训练 {name} ==\n{"=" * 70}', flush=True)
        meta = train_sim(name, device=args.device, root='.',
                         data_dir=st.get('data_dir'),
                         train_files=st.get('train_files'),
                         out_dir=os.path.join('models', name),
                         epochs=args.epochs, **{k: v for k, v in COMMON.items()
                                                if k != 'epochs'})
        summary[name] = {
            'params': meta['params'], 'train_size': meta['train_size'],
            'input_mode': meta['input_mode'], 'loss': meta['loss'],
            'best_val_loss': round(meta['best_val_loss'], 5),
            'best_epoch': meta['best_epoch'], 'epochs_run': meta['epochs_run'],
            'seconds': meta['seconds'], 'device': meta['device'],
        }
        print(f'  >> {name}: best_val={meta["best_val_loss"]:.5f} '
              f'@ep{meta["best_epoch"]}, {meta["seconds"]}s', flush=True)

    print(f'\n{"=" * 70}\n全部完成，总耗时 {time.time() - t_all:.0f}s\n{"=" * 70}')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    out = os.path.join('.deepworks/tmp', 'retrain_summary.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
