// 纯 JS 小型 U-Net 前向，派生自原 Studio；仅加载本次 70/30 部署权重。
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
    var c = MODEL.meta.width || 8;
    var e1 = doubleConv(x, cin, c, 'enc1', h, w);
    var p1 = maxpool2(e1, c, h, w);
    var e2 = doubleConv(p1, c, 2*c, 'enc2', h/2, w/2);
    var p2 = maxpool2(e2, 2*c, h/2, w/2);
    var e3 = doubleConv(p2, 2*c, 4*c, 'enc3', h/4, w/4);
    var p3 = maxpool2(e3, 4*c, h/4, w/4);
    var bn = doubleConv(p3, 4*c, 8*c, 'bottleneck', h/8, w/8);
    var d3 = convT2(bn, 8*c, h/8, w/8, 'up3.weight', 'up3.bias');
    d3 = doubleConv(cat(d3, 4*c, e3, 4*c, h/4, w/4), 8*c, 4*c, 'dec3', h/4, w/4);
    var d2 = convT2(d3, 4*c, h/4, w/4, 'up2.weight', 'up2.bias');
    d2 = doubleConv(cat(d2, 2*c, e2, 2*c, h/2, w/2), 4*c, 2*c, 'dec2', h/2, w/2);
    var d1 = convT2(d2, 2*c, h/2, w/2, 'up1.weight', 'up1.bias');
    d1 = doubleConv(cat(d1, c, e1, c, h, w), 2*c, c, 'dec1', h, w);
    return conv2d(d1, c, h, w, 'out.weight', 'out.bias');
  }

  function nnInfer(inputObj) {
    if (inputObj.channels.length !== 9 * inputObj.nx * inputObj.ny) throw new Error('Expected 9 geometry-only channels');
    var z = forward(inputObj.channels, 9, inputObj.ny, inputObj.nx);
    var pred = new Float32Array(z.length);
    for (var k=0;k<z.length;k++) pred[k] = inputObj.roomMask[k] > .5 ? z[k]*NORM.std + NORM.mean : -120;
    return { pred:pred };
  }
  window.RFM_LayoutInfer = nnInfer;
  window.RFM_LayoutModel = MODEL;
})();
