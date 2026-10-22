"""把 PyTorch 权重导出成浏览器可直接加载的格式。

动机
----
演示软件要求**双击 index.html 就能用**，没有后端、没有构建步骤。而 ``file://``
协议下浏览器会拦截对本地 ``.bin`` / ``.json`` 的 fetch，所以权重必须内联进
一个 ``.js`` 文件：

* ``model_weights.js`` —— 定义 ``window.RFM_MODEL``，含
  ``meta``（标准化参数、网格、结构）、``manifest``（每个张量的名字/形状/偏移）
  和 ``data``（全部 float32 按顺序拼接后的 base64）。
* ``nn_model.js`` —— 由本工程手写的纯 JS 前向实现（见
  :func:`write_js_inference` 的说明），按 manifest 的顺序取权重。

这样做的代价是文件比较大（约 1.4 MB 的权重 -> 约 1.9 MB base64），
好处是完全离线、零依赖、可审计：任何一层权重都能被单独读出来核对。
"""

from __future__ import annotations

import argparse
import base64
import json
import os

import numpy as np

from .train import load_trained
from .models import MODEL_LABELS


def export_for_web(model_name: str = 'unet', root: str = '.',
                   out_dir: str = 'outputs/rfmap-studio') -> dict:
    """导出权重为 ``model_weights.js``，返回导出摘要。"""
    model, meta, norm = load_trained(model_name, root=root, device='cpu')
    sd = model.state_dict()

    manifest, arrays, off = [], [], 0
    for k, v in sd.items():
        a = v.detach().cpu().numpy().astype('<f4').ravel()
        manifest.append({'name': k, 'shape': list(v.shape),
                         'offset': int(off), 'size': int(a.size)})
        arrays.append(a)
        off += a.size

    blob = np.concatenate(arrays).tobytes()
    b64 = base64.b64encode(blob).decode('ascii')

    payload = {
        'model': model_name,
        'label': MODEL_LABELS.get(model_name, model_name),
        'meta': meta,
        'n_params': int(off),
        'weight_bytes': len(blob),
        'manifest': manifest,
        'data': b64,
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'model_weights.js')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('// 由 rfmap.export 自动生成，请勿手工编辑。\n')
        f.write('// 模型: %s (%d 参数, %.1f KB)\n' % (
            payload['label'], payload['n_params'], len(blob) / 1024))
        f.write('window.RFM_MODEL = ')
        json.dump(payload, f, ensure_ascii=False, separators=(',', ':'))
        f.write(';\n')

    summary = {
        'path': path,
        'file_kb': round(os.path.getsize(path) / 1024, 1),
        'n_params': int(off),
        'weight_kb': round(len(blob) / 1024, 1),
        'n_tensors': len(manifest),
        'model': model_name,
        'normalizer': norm.to_dict(),
    }
    return summary


def _fmt_js_conv(kind: str) -> str:
    return kind


def write_js_inference(out_dir: str = 'outputs/rfmap-studio') -> str:
    """写出纯 JS 前向实现 ``nn_model.js``。

    实现的是**方案二 小型 U-Net**，与 :class:`rfmap.models.SmallUNet` 逐层对应：

    * ``conv2d``：3x3、padding=1、stride=1，LeakyReLU(0.2) 作为单独步骤；
    * ``maxpool2``：2x2、stride=2；
    * ``conv_transpose2``：k=2、stride=2 —— 用"像素重排"等价实现
      （PyTorch 的 ConvTranspose2d 在 stride == kernel 时不会重叠相加，
      可直接写成每个输入像素向 2x2 输出块散射）；
    * 输入 3 通道（稀疏观测 / 观测掩码 / 房间掩码），输出 1 通道。

    重要：这里**不依赖任何第三方库**，也不使用 WebGL —— 纯 CPU 数组运算。
    64x64 的图，实测一次前向在 M 系列芯片上约 30~60 ms。
    """
    js = r'''// 由 rfmap.export 自动生成：纯 JS 前向推理（方案二 小型 U-Net）。
// 与 rfmap/models.py 中 SmallUNet 层序严格一致；权重来自 model_weights.js。
(function () {
  'use strict';

  var MODEL = window.RFM_MODEL;
  if (!MODEL) { console.error('未找到 window.RFM_MODEL，先加载 model_weights.js'); return; }

  // ---------- 权重解包 ----------
  var raw = atob(MODEL.data);
  var bytes = new Uint8Array(raw.length);
  for (var i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  var all = new Float32Array(bytes.buffer);

  var W = {};
  MODEL.manifest.forEach(function (it) {
    W[it.name] = { shape: it.shape, data: all.subarray(it.offset, it.offset + it.size) };
  });

  var NORM = MODEL.meta.normalizer;

  // ---------- 基础算子 ----------
  // 通用 conv2d：支持任意奇数核（本项目用到 3x3 与 1x1），same padding
  // 输入 inp 布局 [c][h][w]，权重 [out][in][kh][kw]
  function conv2d(inp, cin, hin, win, wname, bname) {
    var w = W[wname], b = W[bname];
    var cout = w.shape[0], kh = w.shape[2], kw = w.shape[3];
    var ph = (kh - 1) >> 1, pw = (kw - 1) >> 1;
    var out = new Float32Array(cout * hin * win);
    var wd = w.data;
    for (var oc = 0; oc < cout; oc++) {
      var bv = b.data[oc];
      for (var y = 0; y < hin; y++) {
        for (var x = 0; x < win; x++) {
          var acc = bv;
          for (var ic = 0; ic < cin; ic++) {
            var wbase = ((oc * cin + ic) * kh) * kw;
            for (var ky = 0; ky < kh; ky++) {
              var yy = y + ky - ph;
              if (yy < 0 || yy >= hin) continue;
              for (var kx = 0; kx < kw; kx++) {
                var xx = x + kx - pw;
                if (xx < 0 || xx >= win) continue;
                acc += inp[(ic * hin + yy) * win + xx] * wd[wbase + ky * kw + kx];
              }
            }
          }
          out[(oc * hin + y) * win + x] = acc;
        }
      }
    }
    return out;
  }

  function leaky(inp, slope) {
    var out = new Float32Array(inp.length);
    for (var i = 0; i < inp.length; i++) {
      var v = inp[i];
      out[i] = v > 0 ? v : v * slope;
    }
    return out;
  }

  function maxpool2(inp, c, h, w) {
    var oh = h >> 1, ow = w >> 1;
    var out = new Float32Array(c * oh * ow);
    for (var k = 0; k < c; k++) {
      for (var y = 0; y < oh; y++) {
        for (var x = 0; x < ow; x++) {
          var m = -Infinity;
          for (var dy = 0; dy < 2; dy++) {
            for (var dx = 0; dx < 2; dx++) {
              var v = inp[(k * h + (y * 2 + dy)) * w + (x * 2 + dx)];
              if (v > m) m = v;
            }
          }
          out[(k * oh + y) * ow + x] = m;
        }
      }
    }
    return out;
  }

  // ConvTranspose2d(k=2, stride=2)：等价于每个输入像素散射到 2x2 输出块
  // 权重布局 [cin][cout][2][2]
  function convT2(inp, cin, hin, win, wname, bname) {
    var w = W[wname], b = W[bname];
    var cout = w.shape[1];
    var oh = hin * 2, ow = win * 2;
    var out = new Float32Array(cout * oh * ow);
    for (var oc = 0; oc < cout; oc++) {
      var bv = b.data[oc];
      for (var y = 0; y < oh; y++) {
        for (var x = 0; x < ow; x++) out[(oc * oh + y) * ow + x] = bv;
      }
    }
    var wd = w.data;
    for (var ic = 0; ic < cin; ic++) {
      for (var y = 0; y < hin; y++) {
        for (var x = 0; x < win; x++) {
          var v = inp[(ic * hin + y) * win + x];
          if (v === 0) continue;
          for (var oc2 = 0; oc2 < cout; oc2++) {
            var wb = ((ic * cout + oc2) * 2) * 2;
            for (var ky = 0; ky < 2; ky++) {
              for (var kx = 0; kx < 2; kx++) {
                out[(oc2 * oh + (y * 2 + ky)) * ow + (x * 2 + kx)] += v * wd[wb + ky * 2 + kx];
              }
            }
          }
        }
      }
    }
    return out;
  }

  function cat(a, ca, b, cb, h, w) {
    var out = new Float32Array((ca + cb) * h * w);
    out.set(a, 0);
    out.set(b, ca * h * w);
    return out;
  }

  function doubleConv(inp, cin, cout, prefix, h, w) {
    // 注意：Python 侧 DoubleConv 内部是 Sequential，state_dict 的键多一层 .block
    var p = prefix + '.block';
    var t = leaky(conv2d(inp, cin, h, w, p + '.0.weight', p + '.0.bias'), 0.2);
    return leaky(conv2d(t, cout, h, w, p + '.2.weight', p + '.2.bias'), 0.2);
  }

  // ---------- 前向 ----------
  function forward(x, cin, h, w) {
    var e1 = doubleConv(x, cin, 24, 'enc1', h, w);       // 24
    var p1 = maxpool2(e1, 24, h, w);
    var e2 = doubleConv(p1, 24, 48, 'enc2', h / 2, w / 2); // 48
    var p2 = maxpool2(e2, 48, h / 2, w / 2);
    var e3 = doubleConv(p2, 48, 96, 'enc3', h / 4, w / 4); // 96
    var p3 = maxpool2(e3, 96, h / 4, w / 4);
    var bn = doubleConv(p3, 96, 192, 'bottleneck', h / 8, w / 8);

    var d3 = convT2(bn, 192, h / 8, w / 8, 'up3.weight', 'up3.bias');
    d3 = doubleConv(cat(d3, 96, e3, 96, h / 4, w / 4), 192, 96, 'dec3', h / 4, w / 4);
    var d2 = convT2(d3, 96, h / 4, w / 4, 'up2.weight', 'up2.bias');
    d2 = doubleConv(cat(d2, 48, e2, 48, h / 2, w / 2), 96, 48, 'dec2', h / 2, w / 2);
    var d1 = convT2(d2, 48, h / 2, w / 2, 'up1.weight', 'up1.bias');
    d1 = doubleConv(cat(d1, 24, e1, 24, h, w), 48, 24, 'dec1', h, w);
    return conv2d(d1, 24, h, w, 'out.weight', 'out.bias'); // 1 通道
  }

  // ---------- 对外接口：与 app.js 约定一致 ----------
  function nnInfer(inputObj) {
    var nx = inputObj.nx, ny = inputObj.ny;
    var X = new Float32Array(3 * nx * ny);
    var mean = NORM.mean, std = NORM.std;
    var obs = inputObj.obs, om = inputObj.obsMask, rm = inputObj.roomMask;
    for (var i = 0; i < nx * ny; i++) {
      X[i] = om[i] > 0.5 ? (obs[i] - mean) / std : 0;
      X[nx * ny + i] = om[i];
      X[2 * nx * ny + i] = rm[i];
    }
    var z = forward(X, 3, ny, nx);
    var pred = new Float32Array(nx * ny);
    for (var k = 0; k < nx * ny; k++) {
      pred[k] = rm[k] > 0.5 ? z[k] * std + mean : -120;
    }
    return { pred: pred };
  }

  window.RFM_Infer = nnInfer;
  window.RFM_InferKind = 'nn';
  console.log('已加载神经网络推理（SmallUNet, ' + MODEL.n_params + ' 参数）');
})();
'''
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'nn_model.js')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(js)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description='导出 Web 端权重与 JS 推理')
    ap.add_argument('--model', default='unet')
    ap.add_argument('--root', default='.')
    ap.add_argument('--out-dir', default='outputs/rfmap-studio')
    args = ap.parse_args(argv)
    s = export_for_web(args.model, root=args.root, out_dir=args.out_dir)
    p = write_js_inference(args.out_dir)
    print(json.dumps(s, indent=2, ensure_ascii=False))
    print('js ->', p)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
