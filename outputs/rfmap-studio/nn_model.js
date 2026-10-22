// 由 rfmap.export 自动生成：纯 JS 前向推理（方案二 小型 U-Net）。
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
