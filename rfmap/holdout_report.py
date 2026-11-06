"""Render the fixed 70/30 experiment from saved predictions; never fit a model."""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/rfmap-7030-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle, Circle
import numpy as np

from . import geometry as G

METHODS = ['ae', 'unet', 'resunet', 'unet_scatter', 'pgmpr']
NAMES = {'ae': '卷积 AE', 'unet': '小型 U-Net', 'resunet': '残差 U-Net',
         'unet_scatter': '散落布点 U-Net', 'pgmpr': '物理引导 PGMPR'}
SCENES = {'empty': '空房', 'furnished': '有家具'}
TASKS = {'completion': '70% 实测点辅助补全', 'layout': '仅户型 / AP / 家具'}
LEVELS = ['很差', '较差', '一般', '好', '非常好']
COLORS = ['#b33d42', '#d98848', '#d6bc60', '#76aa91', '#277d70']


def configure():
    plt.rcParams.update({'font.sans-serif': ['Arial Unicode MS', 'PingFang SC', 'DejaVu Sans'],
                         'axes.unicode_minus': False, 'figure.dpi': 150,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titlesize': 11, 'axes.labelsize': 9,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8})


def points(scene, split):
    return [p for p in scene['points'] if p['split'] == split]


def room(ax, furniture=None):
    ax.plot(*np.array(G.room_polygon() + [G.room_polygon()[0]]).T,
            color='#243649', lw=1.2, zorder=5)
    ax.add_patch(Rectangle((G.COL_X0, G.COL_Y0), G.COL_W, G.COL_H,
                           facecolor='#243649', zorder=5))
    for f in furniture or []:
        ax.add_patch(Circle((f['x'], f['y']), f.get('radius', .4),
                            facecolor='none', edgecolor='#384c60', lw=.8, zorder=6))
    ax.set(xlim=(-.4, 13.1), ylim=(-.4, 12.2), xlabel='x 坐标 (m)', ylabel='y 坐标 (m)')
    ax.set_aspect('equal')


def markers(ax, scene, labels=False):
    for split, marker, color, desc in [('train', 'o', '#203b60', '训练点（约70%）'),
                                      ('test', 'x', '#c1422c', '测试点（约30%）')]:
        ps = points(scene, split)
        if marker == 'o':
            ax.scatter([p['x'] for p in ps], [p['y'] for p in ps], s=17,
                       facecolors='none', edgecolors=color, marker=marker, lw=.9,
                       label=desc, zorder=9)
        else:
            ax.scatter([p['x'] for p in ps], [p['y'] for p in ps], s=27, c=color,
                       marker=marker, lw=1.1, label=desc, zorder=10)
        if labels:
            for p in ps:
                narrow = len(scene['points'])>100 and p['x']<1.0
                ax.annotate(str(p['id']), (p['x'], p['y']), xytext=(-3 if narrow else 3,3),
                            ha='right' if narrow else 'left',textcoords='offset points', fontsize=5.5, color='#26364a')


def save(fig, dest):
    fig.savefig(dest, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_splits(d, out):
    fig, axs = plt.subplots(1, 2, figsize=(13, 6.5))
    for ax, (key, label) in zip(axs, SCENES.items()):
        s = d['scenarios'][key]
        room(ax, s.get('furniture'))
        markers(ax, s, labels=True)
        ax.set_title(f"{label}：{s['n_train']} 训练点 / {s['n_test']} 测试点")
        ax.legend(loc='upper center', bbox_to_anchor=(.5,-.13), ncol=2, fontsize=9)
    fig.suptitle('固定 70/30 点位划分 · 两版相同坐标保持同一集合', fontsize=15)
    fig.text(.5, .015, '来源：data.xlsx / 放家具数据.xlsx · 2026-11-06 实验；按表格数值坐标绘制，y 按净高对齐；编号对应逐点明细。',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0,.11,1,.94))
    save(fig, out/'split_points.png')


def plot_maps(d, maps, out, task):
    fig, axs = plt.subplots(2, 5, figsize=(21, 9))
    extent = map_extent(maps)
    im = None
    for row, (key, label) in enumerate(SCENES.items()):
        for col, method in enumerate(METHODS):
            ax = axs[row, col]
            field = np.where(maps[f'{key}__room_mask'] > .5, maps[f'{key}__{task}__{method}'], np.nan)
            im = ax.imshow(field, origin='lower', extent=extent, vmin=-55, vmax=-20,
                           cmap='viridis', interpolation='nearest')
            room(ax, d['scenarios'][key].get('furniture'))
            markers(ax, d['scenarios'][key])
            m = d['scenarios'][key]['metrics'][task][method]
            ax.set_title(f"{label} · {NAMES[method]}\n测试 MAE {m['mae']:.2f} dB / RMSE {m['rmse']:.2f} dB")
    fig.subplots_adjust(left=.04, right=.93, bottom=.11, top=.88, wspace=.27, hspace=.35)
    cax = fig.add_axes([.945,.23,.012,.5])
    fig.colorbar(im, cax=cax, label='预测 RSSI (dBm)，统一色标')
    fig.suptitle(TASKS[task]+' · 五种方法覆盖图', fontsize=17)
    fig.text(.5,.045,'空心圆 = 训练点；红叉 = 留出的测试点。仅测试点用于图中误差。背景是预测，不是实测真值。',ha='center',fontsize=10)
    fig.text(.5,.018,'来源：本次保存权重的推理结果 · 同一场地两版记录 · 图中细节不代表具有相同密度的实测证据。',ha='center',fontsize=9)
    save(fig, out/f'heatmaps_{task}.png')


def plot_levels(d, maps, out):
    fig, axs = plt.subplots(1,2,figsize=(13,6.3))
    cmap = ListedColormap(COLORS)
    for ax, (key,label) in zip(axs,SCENES.items()):
        values=np.where(maps[f'{key}__room_mask']>.5,maps[f'{key}__layout__unet'],np.nan)
        levels=np.where(np.isfinite(values),np.digitize(values,[-80,-70,-60,-50]),np.nan)
        im=ax.imshow(levels,origin='lower',extent=map_extent(maps),cmap=cmap,vmin=-.5,vmax=4.5)
        room(ax,d['scenarios'][key].get('furniture'))
        markers(ax,d['scenarios'][key])
        ax.set_title(label+' · 部署 U-Net 的五档预测')
    fig.subplots_adjust(right=.86,bottom=.14,top=.86,wspace=.25)
    cax=fig.add_axes([.89,.25,.016,.5]); cb=fig.colorbar(im,cax=cax,ticks=range(5)); cb.ax.set_yticklabels(LEVELS)
    fig.suptitle('仅户型 / AP / 家具模型 · 固定 RSSI 五档',fontsize=15)
    fig.text(.5,.05,'分界：−80 / −70 / −60 / −50 dBm；深绿属于最高档。空心圆训练，红叉测试。',ha='center',fontsize=10)
    fig.text(.5,.015,'本数据主要覆盖强信号；较弱的三档无实测验证，不表示它们的识别已经准确。',ha='center',fontsize=9)
    save(fig,out/'five_levels.png')


def map_extent(maps):
    x,y=maps['x'],maps['y']; dx,dy=x[1]-x[0],y[1]-y[0]
    return [float(x[0]-dx/2),float(x[-1]+dx/2),float(y[0]-dy/2),float(y[-1]+dy/2)]


def plot_curves(d,out):
    root=out.parent
    for phase,metric,label in [('selection','inner_val_mae_db','70% 内部验证 MAE (dB)'),
                                ('refit70','train_huber_normalized','归一化 Huber 训练损失（无量纲）')]:
        fig,axs=plt.subplots(2,5,figsize=(20,7))
        for r,(task,title) in enumerate(TASKS.items()):
            for c,method in enumerate(METHODS):
                ax=axs[r,c]; hist=json.loads((root/'models'/task/method/phase/'history.json').read_text())
                ax.plot([a['epoch'] for a in hist],[a[metric] for a in hist],color='#256e86',lw=1.5)
                selected=d['selection'][task][method]['selected_epoch']
                if phase=='selection':ax.axvline(selected,color='#ba7738',ls='--',lw=1,label=f'选定第 {selected} 轮')
                ax.set(xlabel='训练轮次（epoch）',ylabel=label,title=title+'\n'+NAMES[method])
                ax.grid(alpha=.15)
                if phase=='selection':ax.legend(fontsize=8,frameon=False)
        fig.suptitle('内部选轮曲线（测试点完全封存）' if phase=='selection' else '全部 70% 重训曲线（轮数已冻结）',fontsize=16)
        fig.tight_layout(rect=(0,.03,1,.94));save(fig,out/f'curves_{phase}.png')


def plot_metrics(d,out):
    fig,axs=plt.subplots(2,2,figsize=(14,8))
    xs=np.arange(5); width=.36
    for row,(task,title) in enumerate(TASKS.items()):
        for sidx,(key,label) in enumerate(SCENES.items()):
            ms=d['scenarios'][key]['metrics'][task]
            axs[row,0].bar(xs+(sidx-.5)*width,[ms[n]['mae'] for n in METHODS],width,label=label,color=['#2e6c89','#c8914d'][sidx])
            axs[row,1].bar(xs+(sidx-.5)*width,[ms[n]['accuracy']*100 for n in METHODS],width,label=label,color=['#2e6c89','#c8914d'][sidx])
        axs[row,0].set(ylabel='测试 MAE (dB)，越低越好',title=title+' · 数值误差')
        axs[row,1].set(ylabel='测试五档准确率 (%)',ylim=(0,105),title=title+' · 五档准确率')
        for ax in axs[row]:
            ax.set_xticks(xs,[NAMES[n] for n in METHODS],rotation=12)
            ax.set_xlabel('方法'); ax.legend(frameon=False); ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('统一留出测试 · 五种网络 / 两版数据 / 两种输入',fontsize=16)
    fig.text(.5,.01,'准确率需结合报告中的多数类基线及每档样本数理解；当前数据几乎全部处于“非常好”档。',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.05,1,.95));save(fig,out/'metric_comparison.png')


def plot_confusion(d,out):
    fig,axs=plt.subplots(2,2,figsize=(11,9))
    for r,(task,title) in enumerate(TASKS.items()):
        for c,(key,label) in enumerate(SCENES.items()):
            ax=axs[r,c]; m=d['scenarios'][key]['metrics'][task]['unet']; a=np.array(m['confusion'])
            ax.imshow(a,cmap='Blues',vmin=0,vmax=max(1,a.max()))
            for i in range(5):
                for j in range(5):ax.text(j,i,str(a[i,j]),ha='center',va='center',color='white' if a[i,j]>a.max()/2 else '#213547',fontsize=11)
            ax.set_xticks(range(5),LEVELS);ax.set_yticks(range(5),LEVELS)
            ax.set(xlabel='预测等级',ylabel='实测等级',title=label+' · '+title+'\nU-Net 留出测试点数')
    fig.suptitle('混淆矩阵 · 零样本等级不能视为已验证',fontsize=15)
    fig.tight_layout(rect=(0,0,1,.95));save(fig,out/'confusion_unet.png')


def metric_rows(d,task):
    rows=[]
    for key,label in SCENES.items():
        s=d['scenarios'][key]
        metrics=dict(s['metrics'][task])
        baseline='idw_completion' if task=='completion' else 'physics_layout'
        metrics[baseline]=s['baselines'][baseline]
        for method,m in metrics.items():
            if not isinstance(m,dict) or 'mae' not in m:continue
            normal=m.get('normal_range',{})
            nmae=normal.get('mae')
            names={**NAMES,'idw_completion':'基线 · IDW 插值','physics_layout':'基线 · 物理趋势回归'}
            row=[label,names.get(method,method),f"{m['mae']:.3f}",f"{m['rmse']:.3f}",
                 f"{m['accuracy']*100:.2f}%",f"{m.get('macro_f1',0):.3f}",
                 '—' if nmae is None else f'{nmae:.3f}']
            rows.append('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in row)+'</tr>')
    return ''.join(rows)


def write_html(d,out):
    body=[]
    for task,title in TASKS.items():
        body.append(f'<section id="{task}"><h2>{html.escape(title)}</h2><p>两版共用固定测试点。MAE / RMSE 越低越好。正常范围 MAE 仅统计实测位于 −70～−40 dBm 的测试点。</p><div class="scroll"><table><thead><tr><th>数据</th><th>方法</th><th>MAE (dB)</th><th>RMSE (dB)</th><th>五档准确率</th><th>Macro F1</th><th>正常范围 MAE</th></tr></thead><tbody>{metric_rows(d,task)}</tbody></table></div><a href="figures/heatmaps_{task}.png"><img src="figures/heatmaps_{task}.png" alt="{title} 五方法预测热力图" /></a></section>')
    summaries=[]
    for key,label in SCENES.items():
        s=d['scenarios'][key];m=s['metrics']['layout']['unet']
        support=m['support']; n=sum(support);major=max(support)/n if n else 0
        ci=m['accuracy_ci95']
        summaries.append(f'<article><span>{label} · 仅户型 U-Net</span><strong>{m["mae"]:.2f}<small> dB MAE</small></strong><p>RMSE {m["rmse"]:.2f} dB · 五档准确率 {m["accuracy"]*100:.1f}%<br>准确率 95% Wilson 区间 {ci[0]*100:.1f}%～{ci[1]*100:.1f}%</p><p>{s["n_train"]} 训练 / {s["n_test"]} 测试<br>测试档位人数（很差→非常好）：{html.escape(str(support))}<br>始终猜训练集多数档的准确率：{m["majority_baseline_accuracy"]*100:.1f}%</p></article>')
    content='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Wi-Fi 覆盖实验 · 70/30 验证</title><style>
    :root{color-scheme:light;font-family:-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;color:#193346;background:#f4f6f7}body{max-width:1240px;margin:auto;padding:36px 28px 70px;line-height:1.7}h1{font-size:32px;line-height:1.3;margin:10px 0}h2{font-size:23px;margin:0 0 12px}p{margin:8px 0 16px;color:#526473}a{color:#126b76}nav{display:flex;gap:22px;flex-wrap:wrap;margin:20px 0 28px}.eyebrow{font-size:12px;letter-spacing:2px;color:#637581}.lead{max-width:900px}.notice{background:#fff5de;border-left:4px solid #c68c34;padding:18px 22px;margin:24px 0}.summary{display:grid;grid-template-columns:1fr 1fr;gap:30px;margin:30px 0}.summary article{border-top:2px solid #267984;padding:17px 0}.summary span{font-size:14px}.summary strong{display:block;font-size:34px;font-weight:600}.summary small{font-size:15px}section{padding:30px 0;border-top:1px solid #d9e0e4}img{display:block;width:100%;height:auto;background:#fff;margin:24px 0}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;white-space:nowrap;background:#fff}th,td{text-align:left;padding:12px 14px;border-bottom:1px solid #e0e6ea;font-size:14px}th{background:#eaf0f2;font-weight:600}li{margin:9px 0}details{margin:15px 0}pre{white-space:pre-wrap;font-size:12px;max-height:450px;overflow:auto;background:#fff;padding:18px}footer{font-size:13px;color:#637581}.button{display:inline-block;background:#156f78;color:#fff;text-decoration:none;padding:10px 18px;border-radius:5px}@media(max-width:700px){body{padding:20px 16px}.summary{grid-template-columns:1fr}h1{font-size:27px}}
    </style><div class="eyebrow">RF MAP / 本地实测实验 / 2026-11-06</div><h1>Wi-Fi 覆盖预测与独立留出验证</h1><p class="lead">空房 49 点、有家具 140 点。五种方法使用约 70% 的数据训练，封存约 30% 的点检查误差。用户可打开预测编辑器，设置户型、Wi-Fi 位置和家具。</p><nav><a class="button" href="studio/index.html">打开户型与家具预测器</a><a href="#splits">查看训练 / 测试点</a><a href="#completion">实测点补全</a><a href="#layout">仅户型预测</a><a href="predictions.csv" download>逐点预测 CSV</a></nav><div class="notice"><b>当前准确率不能代表五档都识别准确。</b> 189 个实测点中 187 个为“非常好”，2 个为“好”。较弱的三个等级没有样本。必须同时看 dB 误差与多数类基线；本实验也没有独立新户型测试。</div>'''
    content+='<div class="summary">'+''.join(summaries)+'</div>'
    content+='<section id="splits"><h2>训练点与测试点</h2><p>空心圆用于训练，红叉用于最终测试。测试点从未用于梯度、归一化、训练轮数选择或部署模型选择。五种方法使用相同划分；部署界面预先选用小型 U-Net。</p><img src="figures/split_points.png" alt="两版数据训练与测试点位图"></section>'
    content+=''.join(body)
    content+='<p>本次家具版 IDW 基线 MAE 为 1.791 dB，低于五种补全网络。仅户型物理趋势基线 MAE 为 2.186 dB，也低于部署 U-Net 的 2.558 dB。当前证据不支持“神经网络必然更准”。Macro F1 只对测试中出现的等级平均；全五档 F1 及每档 precision/recall 可在完整指标 JSON 复查。</p>'
    content+='<section><h2>训练与选择</h2><p>两种任务各训练五个网络。先在 70% 内部留出验证点，最多训练 180 轮选出各网络轮数，再从零使用全部 70% 重训。内部验证更偏好 AE；编辑器事先固定用 U-Net 以复用浏览器推理，不根据最终测试成绩改选。32 × 32 网格、宽度 8；基础输入扩为 9 通道，PGMPR 为 12 通道并拼接残差先验。损失只在真实测点上计算。</p><p>补全任务的残差先验为训练输入点 IDW；仅户型任务的先验是只由 70% 拟合的物理趋势。散落布点方法在补全训练中改变输入抽样，在仅户型训练中改变目标点子集。没有使用旧全量微调权重，也没有将插值充当实测答案。</p><img src="figures/curves_selection.png" alt="五方法内部验证曲线"><img src="figures/curves_refit70.png" alt="五方法70%重训损失曲线"></section>'
    content+='<section><h2>五档与类别覆盖</h2><p>五档分界为 −80 / −70 / −60 / −50 dBm。该分类描述 RSSI 强弱，不保证网速，也不把偏强测量的饱和风险视为合格采集。</p><img src="figures/five_levels.png" alt="部署模型五档图"><img src="figures/confusion_unet.png" alt="U-Net 两任务混淆矩阵"><img src="figures/metric_comparison.png" alt="五方法准确率和MAE比较"></section>'
    content+='''<section><h2>实验边界与可复查文件</h2><ul><li>只有同一场地的两版记录；随机邻近点留出不能证明新户型或新 AP 部署的泛化精度。</li><li>两版的路由器位置缺少现场记录；本次采用的 AP 坐标与来源见下面配置。若为假设，成绩仅对应这一假设。</li><li>家具表 120 个正数沿用已有的负值修正规则；原始读数保留。空房 8 点、家具 32 点强于 −40 dBm，保留并标注为偏强；这些数据仍有采集质量风险。</li><li>历史文档的 x 轴方向有冲突，本次按表格数值坐标统一绘制，y 按房间净高对齐。两版不作家具因果衰减结论。</li><li>家具实测仅有桌与椅；柜子、床、沙发、马桶等可添加对象使用几何及假设遮挡强度。图片需标定尺寸并人工描绘墙体；自动识别户型不是本次实现。</li><li>训练在本机完成，编辑器离线推理。所有曲线、权重、拆分和逐点预测随交付保存。</li></ul><p><a href="metrics.json">完整指标 JSON</a> · <a href="split.json">固定划分</a> · <a href="predictions.csv">逐点预测</a> · <a href="README.md">复现与使用说明</a></p>'''
    content+='<p>空间相关性限制：两版有两组不同原始位置相距约 0.21 m，在 32 节点输出网格落入同一节点附近且分属训练/测试。它们不是重复标签，但说明本次属于邻近点预测；未采用隔离整块区域的空间外推检验。95% 区间也未修正这种空间相关，不能作为新户型保证。</p>'
    for key in ('protocol','quality','selection'):
        if key in d:content+='<details><summary>'+html.escape(key)+' · 配置与记录</summary><pre>'+html.escape(json.dumps(d[key],ensure_ascii=False,indent=2))+'</pre></details>'
    content+='</section><footer>报告由保存的本次结果生成。预测网格没有密集实测答案；评分只使用封存的测点。来源：本项目两份原始工作簿及本次训练输出。</footer></html>'
    (out/'report.html').write_text(content,encoding='utf-8')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='outputs/rfmap-70-30');args=ap.parse_args()
    out=Path(args.out); figs=out/'figures';figs.mkdir(parents=True,exist_ok=True)
    d=json.loads((out/'metrics.json').read_text()); maps=np.load(out/'maps.npz')
    configure();plot_splits(d,figs)
    for task in TASKS:plot_maps(d,maps,figs,task)
    plot_levels(d,maps,figs);plot_metrics(d,figs);plot_confusion(d,figs);plot_curves(d,figs);write_html(d,out)
    print('Saved report.html and eight figures in',out)


if __name__=='__main__':main()
