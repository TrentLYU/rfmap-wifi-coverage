/* =============================================================================
 * RF Map Studio - 纯前端 Wi-Fi RSSI 热力图工作台
 *
 * 本文件分两部分：
 *   1. 推理接口 window.RFM_Infer(inputObj)  ——  当前为占位实现（IDW 反距离加权）
 *   2. UI 逻辑（画布绘制 / 户型编辑 / 测点编辑 / 导出）
 *
 * 坐标约定（与任务书一致）：
 *   - 房间内部矩形宽 W（x 方向 12.643 m）、高 H（y 方向 11.731 m）。
 *   - 坐标系原点在房间内部【右下角】，x 轴正向指向【左】，y 轴正向指向【上】。
 *   - 因此把房间坐标 (rx, ry) 画到 Canvas 时必须把 x 反向：
 *       sx(rx) = 原点屏幕 x - rx * scale      （rx 越大，屏幕 x 越小）
 *       sy(ry) = 原点屏幕 y - ry * scale      （ry 越大，屏幕 y 越小）
 *
 * 网格数据布局（与 RFM_Infer 接口一致）：
 *   - nx = 64 列（x 方向，i 为列号），ny = 64 行（y 方向，j 为行号）。
 *   - 一维线性下标 k = j * nx + i ，即 row-major，行 j 对应 y 索引。
 *   - 格点(i,j)的中心物理坐标：x = x0 + (i + 0.5) * cell，y = y0 + (j + 0.5) * cell。
 *   - j = 0 是 y 最小的一行。
 *   - 画到 Canvas 时：图像行/列都要翻转，使 row 0 落在 y 最小侧、列 0 落在 x 最大侧。
 *
 * 无外部依赖、无构建步骤，双击 index.html 即可运行。
 * ============================================================================= */

/* ===========================================================================
 * 一、推理接口（占位实现）
 * =========================================================================== */

/**
 * 占位推理：反距离加权插值（IDW，权重 1/d²）。
 *
 * 与 UI 完全解耦。后续接入神经网络时，只需在加载模型后把
 * window.RFM_Infer 指向新的实现即可（本文件末尾用
 * `window.RFM_Infer = window.RFM_Infer || placeholderInfer;` 保证可被覆盖）。
 *
 * @param {Object} inputObj
 * @param {Float32Array} inputObj.obs       长度 nx*ny 的稀疏观测值，未观测处为 0
 * @param {Float32Array} inputObj.obsMask   长度 nx*ny，1 表示该格点有观测
 * @param {Float32Array} inputObj.roomMask  长度 nx*ny，1 表示该格点在房间内部
 * @param {number} inputObj.nx              网格列数（x 方向）
 * @param {number} inputObj.ny              网格行数（y 方向）
 * @param {number} inputObj.x0              网格原点 x（米）
 * @param {number} inputObj.y0              网格原点 y（米）
 * @param {number} inputObj.cell            网格边长（米）
 * @returns {{pred: Float32Array}} pred 长度 nx*ny，单位 dBm，房间外填 -120
 */
function placeholderInfer(inputObj) {
  var nx = inputObj.nx;
  var ny = inputObj.ny;
  var x0 = inputObj.x0;
  var y0 = inputObj.y0;
  var cell = inputObj.cell;
  var obs = inputObj.obs;
  var obsMask = inputObj.obsMask;
  var roomMask = inputObj.roomMask;

  var N = nx * ny;
  var OUTSIDE = -120.0;      // 房间外填充值
  var NO_OBS = -60.0;        // 完全没有观测点时的默认值
  var RADIUS = 6.0;          // 超过该距离（米）回退为全局均值
  var EPS2 = 1e-12;          // 判定“格点与观测点重合”的平方距离阈值

  var pred = new Float32Array(N);

  /* ---- 1. 收集观测点（物理坐标 + 值），并求全局均值 ---- */
  var ox = [];
  var oy = [];
  var ov = [];
  var sum = 0;
  for (var k = 0; k < N; k++) {
    if (obsMask[k] > 0.5) {
      var ii = k % nx;
      var jj = (k - ii) / nx;
      ox.push(x0 + (ii + 0.5) * cell);
      oy.push(y0 + (jj + 0.5) * cell);
      ov.push(obs[k]);
      sum += obs[k];
    }
  }
  var nObs = ov.length;
  var globalMean = nObs > 0 ? sum / nObs : NO_OBS;

  /* ---- 2. 逐格点做 IDW ---- */
  for (var idx = 0; idx < N; idx++) {
    // 房间外：直接填 -120
    if (roomMask[idx] <= 0.5) {
      pred[idx] = OUTSIDE;
      continue;
    }
    // 无任何观测：回退默认值
    if (nObs === 0) {
      pred[idx] = NO_OBS;
      continue;
    }

    var i = idx % nx;
    var j = (idx - i) / nx;
    var px = x0 + (i + 0.5) * cell;
    var py = y0 + (j + 0.5) * cell;

    var wsum = 0;
    var vsum = 0;
    var dmin2 = Infinity;
    var exact = null;

    for (var m = 0; m < nObs; m++) {
      var dx = px - ox[m];
      var dy = py - oy[m];
      var d2 = dx * dx + dy * dy;
      if (d2 < dmin2) { dmin2 = d2; }
      if (d2 < EPS2) { exact = ov[m]; break; }   // 与观测点重合：直接取该值
      var w = 1.0 / d2;                          // 1/d² 权重
      wsum += w;
      vsum += w * ov[m];
    }

    if (exact !== null) {
      pred[idx] = exact;
    } else if (Math.sqrt(dmin2) > RADIUS) {
      pred[idx] = globalMean;                    // 离所有观测点都太远
    } else {
      pred[idx] = vsum / wsum;
    }
  }

  return { pred: pred };
}

/* 允许外部（例如未来的神经网络实现）先行覆盖；否则使用占位实现。 */
window.RFM_Infer = window.RFM_Infer || placeholderInfer;

/* ===========================================================================
 * 二、UI 逻辑
 * =========================================================================== */

(function () {
  'use strict';

  /* --------------------------- 常量 --------------------------- */

  var DEFAULT_ROOM = { w: 12.643, h: 11.731 };
  var DEFAULT_DOOR = { w: 2.330 };                                   // 底部门，居中
  var DEFAULT_OBSTACLES = [
    { x: 11.813, y: 5.761, w: 0.830, h: 1.010, name: '结构柱' }      // 贴 x=12.643 那面墙
  ];
  var DEFAULT_GRID = { x0: 0, y0: 0, cell: 0.2, nx: 64, ny: 64 };

  var COLOR_MIN = -100;      // 色标下限（深蓝）
  var COLOR_MAX = -40;       // 色标上限（深红）
  var HIT_R = 10;            // 命中测点的屏幕像素半径
  var HANDLE_R = 7;          // 户型缩放手柄半径

  /* --------------------------- 状态 --------------------------- */

  var nextId = 1;

  var state = {
    room: { w: DEFAULT_ROOM.w, h: DEFAULT_ROOM.h },
    door: { w: DEFAULT_DOOR.w },
    obstacles: [],
    points: [],
    grid: { x0: DEFAULT_GRID.x0, y0: DEFAULT_GRID.y0, cell: DEFAULT_GRID.cell, nx: DEFAULT_GRID.nx, ny: DEFAULT_GRID.ny },
    heat: null,              // Float32Array(nx*ny)，房间外为 -120
    inferMs: null,
    selectedPointId: null,
    selectedObsId: null,
    drag: null               // {kind:'point'|'obs'|'room', ...}
  };

  /* --------------------------- DOM --------------------------- */

  var canvas = document.getElementById('scene');
  var ctx = canvas.getContext('2d');
  var tooltip = document.getElementById('tooltip');

  var el = {
    infoArea: document.getElementById('infoArea'),
    infoPoints: document.getElementById('infoPoints'),
    infoTime: document.getElementById('infoTime'),
    infoRange: document.getElementById('infoRange'),
    roomWNum: document.getElementById('roomWNum'),
    roomWRange: document.getElementById('roomWRange'),
    roomHNum: document.getElementById('roomHNum'),
    roomHRange: document.getElementById('roomHRange'),
    doorWNum: document.getElementById('doorWNum'),
    obsList: document.getElementById('obsList'),
    obsEditor: document.getElementById('obsEditor'),
    obsX: document.getElementById('obsX'),
    obsY: document.getElementById('obsY'),
    obsW: document.getElementById('obsW'),
    obsH: document.getElementById('obsH'),
    pointList: document.getElementById('pointList'),
    pointCount: document.getElementById('pointCount'),
    rssiModal: document.getElementById('rssiModal'),
    rssiModalPos: document.getElementById('rssiModalPos'),
    rssiInput: document.getElementById('rssiInput'),
    rssiErr: document.getElementById('rssiErr')
  };

  /* --------------------------- 视图变换 ---------------------------
   * 采用“固定世界尺寸 + 居中”的策略：原点(房间右下角)在屏幕上的位置固定，
   * 房间变大时 scale 自动缩小以容纳，拖动手柄时能看到房间真实缩放。
   * ---------------------------------------------------------------- */

  var view = { cssW: 0, cssH: 0, scale: 1, ox: 0, oy: 0, padL: 64, padR: 104, padT: 40, padB: 60 };

  function computeView() {
    var dpr = window.devicePixelRatio || 1;
    var cssW = canvas.clientWidth || 800;
    var cssH = canvas.clientHeight || 600;

    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    view.cssW = cssW;
    view.cssH = cssH;

    var padL = view.padL, padR = view.padR, padT = view.padT, padB = view.padB;
    var plotW = Math.max(40, cssW - padL - padR);
    var plotH = Math.max(40, cssH - padT - padB);

    // 世界边长：至少 14.5 m，且始终比房间大 0.6 m，保证有边距
    var world = Math.max(14.5, state.room.w + 0.6, state.room.h + 0.6);
    var s = Math.min(plotW / world, plotH / world);
    view.scale = s;

    var contentW = world * s;
    var contentH = world * s;

    // 原点 = 房间内部右下角 = 世界坐标 (0,0)
    view.ox = padL + (plotW - contentW) / 2 + contentW;
    view.oy = padT + (plotH - contentH) / 2 + contentH;
  }

  // 房间坐标 -> 屏幕坐标（x 反向）
  function sx(rx) { return view.ox - rx * view.scale; }
  function sy(ry) { return view.oy - ry * view.scale; }
  // 屏幕坐标 -> 房间坐标
  function rxOf(mx) { return (view.ox - mx) / view.scale; }
  function ryOf(my) { return (view.oy - my) / view.scale; }

  /* --------------------------- 颜色映射 --------------------------- */

  var STOPS = [
    [0.00, [16, 30, 138]],    // 深蓝
    [0.25, [22, 132, 214]],   // 蓝青
    [0.50, [42, 186, 106]],   // 绿
    [0.75, [246, 214, 48]],   // 黄
    [1.00, [186, 22, 32]]     // 深红
  ];

  function colorFor(v) {
    var t = (v - COLOR_MIN) / (COLOR_MAX - COLOR_MIN);
    if (t < 0) { t = 0; }
    if (t > 1) { t = 1; }
    var a = STOPS[0], b = STOPS[STOPS.length - 1];
    for (var i = 0; i < STOPS.length - 1; i++) {
      if (t >= STOPS[i][0] && t <= STOPS[i + 1][0]) { a = STOPS[i]; b = STOPS[i + 1]; break; }
    }
    var span = b[0] - a[0] || 1;
    var f = (t - a[0]) / span;
    return [
      Math.round(a[1][0] + (b[1][0] - a[1][0]) * f),
      Math.round(a[1][1] + (b[1][1] - a[1][1]) * f),
      Math.round(a[1][2] + (b[1][2] - a[1][2]) * f)
    ];
  }

  function rgbaFor(v, alpha) {
    var c = colorFor(v);
    return 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',' + alpha + ')';
  }

  /* --------------------------- 几何辅助 --------------------------- */

  function doorRect() {
    var w = state.door.w;
    return { x: (state.room.w - w) / 2, y: 0, w: w, h: 0 };   // 居中于底边
  }

  function insideObstacle(rx, ry) {
    for (var i = 0; i < state.obstacles.length; i++) {
      var o = state.obstacles[i];
      if (rx >= o.x && rx <= o.x + o.w && ry >= o.y && ry <= o.y + o.h) { return o; }
    }
    return null;
  }

  function insideRoom(rx, ry) {
    return rx >= 0 && rx <= state.room.w && ry >= 0 && ry <= state.room.h;
  }

  function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

  /* --------------------------- 绘制 --------------------------- */

  function render() {
    computeView();
    ctx.clearRect(0, 0, view.cssW, view.cssH);

    drawRoomFloor();
    if (state.heat) { drawHeat(); }
    drawWalls();
    drawObstacles();
    drawDoor();
    drawAxes();
    drawPoints();
    if (state.heat) { drawColorbar(); }
    drawRoomHandle();
    drawLegend();
    drawEmptyNotice();
  }

  // 房间内部浅色底
  function drawRoomFloor() {
    var W = state.room.w, H = state.room.h;
    ctx.fillStyle = '#f4f7fb';
    ctx.fillRect(sx(W), sy(H), W * view.scale, H * view.scale);
  }

  // 热力图（半透明叠加，裁剪到“房间内部 - 障碍物”）
  function drawHeat() {
    var g = state.grid;
    var nx = g.nx, ny = g.ny, cell = g.cell;
    var heatW = nx * cell;    // 网格覆盖的物理宽度（0.2*64 = 12.8 m）
    var heatH = ny * cell;
    var W = state.room.w, H = state.room.h;

    // 1. 先生成一张“渲染用”的值表 rv。
    //    房间内直接用预测值；房间外先置 NaN，然后向外做两轮“最近内部值”扩散。
    //    这么做的原因：房间外的格点是 -120 dBm，如果直接参与平滑放大，
    //    贴墙那一圈会把 -120（深蓝）混进房间边缘，画出一圈假的深蓝描边。
    var rv = new Float32Array(nx * ny);
    for (var jj = 0; jj < ny; jj++) {
      for (var ii = 0; ii < nx; ii++) {
        var kk = jj * nx + ii;
        var ccx = g.x0 + (ii + 0.5) * cell;
        var ccy = g.y0 + (jj + 0.5) * cell;
        if (ccx <= W && ccy <= H && !insideObstacle(ccx, ccy)) {
          rv[kk] = state.heat[kk];
        } else {
          rv[kk] = NaN;
        }
      }
    }
    for (var pass = 0; pass < 2; pass++) {
      var next = Float32Array.from(rv);
      for (var j2 = 0; j2 < ny; j2++) {
        for (var i2 = 0; i2 < nx; i2++) {
          var k2 = j2 * nx + i2;
          if (!isNaN(rv[k2])) { continue; }
          var acc = 0, cnt = 0;
          if (i2 > 0 && !isNaN(rv[k2 - 1])) { acc += rv[k2 - 1]; cnt++; }
          if (i2 < nx - 1 && !isNaN(rv[k2 + 1])) { acc += rv[k2 + 1]; cnt++; }
          if (j2 > 0 && !isNaN(rv[k2 - nx])) { acc += rv[k2 - nx]; cnt++; }
          if (j2 < ny - 1 && !isNaN(rv[k2 + nx])) { acc += rv[k2 + nx]; cnt++; }
          if (cnt > 0) { next[k2] = acc / cnt; }
        }
      }
      rv = next;
    }

    // 2. 生成 nx*ny 的离屏像素图
    var off = document.createElement('canvas');
    off.width = nx;
    off.height = ny;
    var octx = off.getContext('2d');
    var img = octx.createImageData(nx, ny);
    var d = img.data;

    for (var j = 0; j < ny; j++) {
      for (var i = 0; i < nx; i++) {
        var raw = rv[j * nx + i];
        var v = isNaN(raw) ? -120 : raw;   // 仍未被扩散到的最外围格点按房间外处理
        var c = colorFor(v);
        // 图像行列均翻转：
        //   图像列 c 对应网格列 nx-1-i（x 反向）
        //   图像行 r 对应网格行 ny-1-j（row 0 在 y 最小处）
        var r = ny - 1 - j;
        var col = nx - 1 - i;
        var p = (r * nx + col) * 4;
        d[p] = c[0];
        d[p + 1] = c[1];
        d[p + 2] = c[2];
        d[p + 3] = 255;
      }
    }
    octx.putImageData(img, 0, 0);

    // 3. 裁剪到房间内部（挖掉障碍物），再平滑放大
    ctx.save();
    ctx.beginPath();
    ctx.rect(sx(W), sy(H), W * view.scale, H * view.scale);
    for (var oi = 0; oi < state.obstacles.length; oi++) {
      var o = state.obstacles[oi];
      ctx.rect(sx(o.x + o.w), sy(o.y + o.h), o.w * view.scale, o.h * view.scale);
    }
    ctx.clip('evenodd');

    ctx.globalAlpha = 0.72;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    // 图像左上角对应物理点 (heatW, heatH)，向右下覆盖 heatW*scale
    ctx.drawImage(off, sx(heatW), sy(heatH), heatW * view.scale, heatH * view.scale);
    ctx.globalAlpha = 1;
    ctx.restore();
  }

  // 墙体深色描边
  function drawWalls() {
    var W = state.room.w, H = state.room.h;
    var s = view.scale;
    var x = sx(W), y = sy(H), w = W * s, h = H * s;

    ctx.lineWidth = 5;
    ctx.strokeStyle = '#22303f';
    ctx.lineJoin = 'miter';
    ctx.strokeRect(x, y, w, h);

    ctx.lineWidth = 2;
    ctx.strokeStyle = '#4a5c6e';
    ctx.strokeRect(x + 3, y + 3, w - 6, h - 6);
  }

  // 障碍物
  function drawObstacles() {
    var s = view.scale;
    for (var i = 0; i < state.obstacles.length; i++) {
      var o = state.obstacles[i];
      var x = sx(o.x + o.w);
      var y = sy(o.y + o.h);
      var w = o.w * s;
      var h = o.h * s;

      ctx.fillStyle = '#59656f';
      ctx.fillRect(x, y, w, h);

      ctx.lineWidth = 1.5;
      ctx.strokeStyle = state.selectedObsId === o.id ? '#1a6fd4' : '#39434c';
      ctx.strokeRect(x, y, w, h);

      if (w > 44 && h > 18) {
        ctx.fillStyle = '#ffffff';
        ctx.font = '11px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(o.name || '障碍物', x + w / 2, y + h / 2);
      }
    }
    ctx.textAlign = 'start';
    ctx.textBaseline = 'alphabetic';
  }

  // 底部门（在底墙上留缺口 + 门扇弧线）
  function drawDoor() {
    var dr = doorRect();
    if (dr.w <= 0) { return; }
    var s = view.scale;
    var y = view.oy;                       // 底墙屏幕 y
    var xa = sx(dr.x + dr.w);              // 门洞左端（屏幕左）
    var xb = sx(dr.x);                     // 门洞右端（屏幕右）

    // 用底色抹掉这一段墙
    ctx.fillStyle = '#f4f7fb';
    ctx.fillRect(xa - 2, y - 4, (xb - xa) + 4, 8);

    // 双开门：两个四分之一圆弧
    var r = (dr.w / 2) * s;
    ctx.strokeStyle = '#b8862b';
    ctx.lineWidth = 1.6;

    // 右半扇（铰点 xb）
    ctx.beginPath();
    ctx.arc(xb, y, r, Math.PI, Math.PI * 1.5);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(xb, y);
    ctx.lineTo(xb, y - r);
    ctx.stroke();

    // 左半扇（铰点 xa）
    ctx.beginPath();
    ctx.arc(xa, y, r, Math.PI * 1.5, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(xa, y);
    ctx.lineTo(xa, y - r);
    ctx.stroke();
  }

  // 坐标轴与刻度（标注 x 反向 / y 向上）
  function drawAxes() {
    var W = state.room.w, H = state.room.h;
    var s = view.scale;
    var ox = view.ox, oy = view.oy;

    ctx.font = '11px sans-serif';
    ctx.fillStyle = '#5a6775';
    ctx.strokeStyle = '#9aa8b6';
    ctx.lineWidth = 1;

    // x 轴刻度（沿底边，0 在右下角）
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (var xv = 0; xv <= Math.floor(W); xv += 2) {
      var px = sx(xv);
      ctx.beginPath();
      ctx.moveTo(px, oy);
      ctx.lineTo(px, oy + 6);
      ctx.stroke();
      ctx.fillText(String(xv), px, oy + 9);
    }

    // y 轴刻度（沿右边，0 在右下角）
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    for (var yv = 0; yv <= Math.floor(H); yv += 2) {
      var py = sy(yv);
      ctx.beginPath();
      ctx.moveTo(ox, py);
      ctx.lineTo(ox + 6, py);
      ctx.stroke();
      ctx.fillText(String(yv), ox + 9, py);
    }

    // x 方向箭头（指向左）
    var ax0 = sx(Math.min(W, 3.2));
    ctx.strokeStyle = '#c0392b';
    ctx.fillStyle = '#c0392b';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(sx(Math.min(W, 5.6)), oy + 30);
    ctx.lineTo(ax0, oy + 30);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(ax0, oy + 30);
    ctx.lineTo(ax0 + 9, oy + 25);
    ctx.lineTo(ax0 + 9, oy + 35);
    ctx.closePath();
    ctx.fill();
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText('x (m) 向左为正', ax0 + 16, oy + 30);

    // y 方向箭头（指向上）
    var ay0 = sy(Math.min(H, 3.0));
    ctx.beginPath();
    ctx.moveTo(ox + 40, sy(Math.min(H, 5.4)));
    ctx.lineTo(ox + 40, ay0);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(ox + 40, ay0);
    ctx.lineTo(ox + 35, ay0 + 9);
    ctx.lineTo(ox + 45, ay0 + 9);
    ctx.closePath();
    ctx.fill();
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText('y (m) 向上为正', ox + 40, ay0 - 6);

    // 原点标注
    ctx.fillStyle = '#22303f';
    ctx.beginPath();
    ctx.arc(ox, oy, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'top';
    ctx.fillText('原点 (0, 0) 房间右下角', ox - 8, oy + 6);

    ctx.textAlign = 'start';
    ctx.textBaseline = 'alphabetic';
  }

  // 测点
  function drawPoints() {
    for (var i = 0; i < state.points.length; i++) {
      var p = state.points[i];
      var px = sx(p.x);
      var py = sy(p.y);
      var selected = state.selectedPointId === p.id;

      // 十字准星
      ctx.strokeStyle = 'rgba(30,40,52,.55)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(px - 8, py);
      ctx.lineTo(px + 8, py);
      ctx.moveTo(px, py - 8);
      ctx.lineTo(px, py + 8);
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(px, py, 6, 0, Math.PI * 2);
      ctx.fillStyle = selected ? '#1a6fd4' : '#ff7a18';
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = selected ? '#0d4a94' : '#ffffff';
      ctx.stroke();

      ctx.font = '10px sans-serif';
      ctx.fillStyle = '#26313c';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      ctx.fillText(String(p.rssi), px, py - 9);
    }
    ctx.textAlign = 'start';
    ctx.textBaseline = 'alphabetic';
  }

  // 户型缩放手柄（房间左上角，对应物理点 (W, H)）
  function drawRoomHandle() {
    var hx = sx(state.room.w);
    var hy = sy(state.room.h);
    ctx.beginPath();
    ctx.arc(hx, hy, HANDLE_R, 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff';
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#1a6fd4';
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(hx, hy, 2.5, 0, Math.PI * 2);
    ctx.fillStyle = '#1a6fd4';
    ctx.fill();

    ctx.font = '11px sans-serif';
    ctx.fillStyle = '#1a6fd4';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'bottom';
    ctx.fillText('拖动调整宽高', hx - 10, hy - 4);
    ctx.textAlign = 'start';
    ctx.textBaseline = 'alphabetic';
  }

  // 色标条（绘制在画布右侧，导出 PNG 时会一并包含）
  function drawColorbar() {
    var barX = view.cssW - view.padR + 34;
    var barY = view.padT;
    var barW = 22;
    var barH = view.cssH - view.padT - view.padB;

    var off = document.createElement('canvas');
    off.width = 1;
    off.height = 256;
    var octx = off.getContext('2d');
    for (var r = 0; r < 256; r++) {
      // 顶部 = -40（深红），底部 = -100（深蓝）
      var v = COLOR_MAX - (r / 255) * (COLOR_MAX - COLOR_MIN);
      var c = colorFor(v);
      octx.fillStyle = 'rgb(' + c[0] + ',' + c[1] + ',' + c[2] + ')';
      octx.fillRect(0, r, 1, 1);
    }
    ctx.drawImage(off, barX, barY, barW, barH);

    ctx.lineWidth = 1;
    ctx.strokeStyle = '#39434c';
    ctx.strokeRect(barX, barY, barW, barH);

    ctx.font = '11px sans-serif';
    ctx.fillStyle = '#39434c';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    for (var v2 = COLOR_MIN; v2 <= COLOR_MAX; v2 += 10) {
      var t = (COLOR_MAX - v2) / (COLOR_MAX - COLOR_MIN);
      var y = barY + t * barH;
      ctx.beginPath();
      ctx.moveTo(barX + barW, y);
      ctx.lineTo(barX + barW + 5, y);
      ctx.stroke();
      ctx.fillText(String(v2), barX + barW + 8, y);
    }

    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.font = '11px sans-serif';
    ctx.fillText('RSSI (dBm)', barX + barW / 2, barY - 6);

    ctx.textAlign = 'start';
    ctx.textBaseline = 'alphabetic';
  }

  // 图例
  function drawLegend() {
    var x = view.padL;
    var y = view.padT;
    var w = 168;
    var h = 100;

    ctx.fillStyle = 'rgba(255,255,255,.94)';
    ctx.fillRect(x, y, w, h);
    ctx.lineWidth = 1;
    ctx.strokeStyle = '#d3dae3';
    ctx.strokeRect(x, y, w, h);

    var items = [
      ['#f4f7fb', '房间内部', '#8b98a6'],
      ['#22303f', '墙体', '#22303f'],
      ['#59656f', '障碍物（结构柱）', '#39434c'],
      ['#b8862b', '门（底边居中）', '#b8862b'],
      ['#ff7a18', '实测测点', '#ffffff']
    ];

    ctx.font = '11px sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    for (var i = 0; i < items.length; i++) {
      var iy = y + 14 + i * 17;
      ctx.fillStyle = items[i][0];
      ctx.fillRect(x + 10, iy - 6, 14, 12);
      ctx.lineWidth = 1;
      ctx.strokeStyle = items[i][2];
      ctx.strokeRect(x + 10, iy - 6, 14, 12);
      ctx.fillStyle = '#3a4756';
      ctx.fillText(items[i][1], x + 32, iy);
    }

    ctx.textAlign = 'start';
    ctx.textBaseline = 'alphabetic';
  }

  // 空状态提示
  function drawEmptyNotice() {
    var msg = null;
    if (state.points.length === 0) {
      msg = '在房间内部单击即可添加测点，或点击“载入示例数据（49 点）”';
    } else if (!state.heat) {
      msg = '已就绪：点击“生成热力图”查看整间房 RSSI 覆盖预测';
    }
    if (!msg) { return; }

    ctx.font = '13px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    var tw = ctx.measureText(msg).width;
    var cx = view.cssW / 2;
    var cy = view.cssH - 22;

    ctx.fillStyle = 'rgba(22,33,44,.86)';
    var pad = 12;
    ctx.fillRect(cx - tw / 2 - pad, cy - 13, tw + pad * 2, 26);
    ctx.fillStyle = '#ffffff';
    ctx.fillText(msg, cx, cy);

    ctx.textAlign = 'start';
    ctx.textBaseline = 'alphabetic';
  }

  /* --------------------------- 推理 --------------------------- */

  // 把当前测点栅格化为 RFM_Infer 的输入
  function buildInput() {
    var g = state.grid;
    var nx = g.nx, ny = g.ny, cell = g.cell, x0 = g.x0, y0 = g.y0;
    var N = nx * ny;

    var obs = new Float32Array(N);
    var obsMask = new Float32Array(N);
    var roomMask = new Float32Array(N);
    var accSum = new Float32Array(N);
    var accCnt = new Float32Array(N);

    // 房间内部掩膜（挖掉障碍物）
    for (var j = 0; j < ny; j++) {
      for (var i = 0; i < nx; i++) {
        var cx = x0 + (i + 0.5) * cell;
        var cy = y0 + (j + 0.5) * cell;
        var inside = (cx <= state.room.w && cy <= state.room.h && !insideObstacle(cx, cy));
        roomMask[j * nx + i] = inside ? 1 : 0;
      }
    }

    // 测点填入观测格点（同一格点多点取均值）
    for (var p = 0; p < state.points.length; p++) {
      var pt = state.points[p];
      var ii = clamp(Math.floor((pt.x - x0) / cell), 0, nx - 1);
      var jj = clamp(Math.floor((pt.y - y0) / cell), 0, ny - 1);
      var k = jj * nx + ii;
      accSum[k] += pt.rssi;
      accCnt[k] += 1;
    }
    for (var q = 0; q < N; q++) {
      if (accCnt[q] > 0) {
        obs[q] = accSum[q] / accCnt[q];
        obsMask[q] = 1;
      }
    }

    return { obs: obs, obsMask: obsMask, roomMask: roomMask, nx: nx, ny: ny, x0: x0, y0: y0, cell: cell };
  }

  function runInfer() {
    if (state.points.length === 0) {
      window.alert('请先添加测点，或点击“载入示例数据（49 点）”。');
      return;
    }
    var input = buildInput();
    var t0 = (window.performance && performance.now) ? performance.now() : Date.now();
    var res = window.RFM_Infer(input);
    var t1 = (window.performance && performance.now) ? performance.now() : Date.now();

    state.heat = res.pred;
    state.inferMs = t1 - t0;
    updateInfo();
    render();
  }

  /* --------------------------- 信息栏 --------------------------- */

  function updateInfo() {
    var gross = state.room.w * state.room.h;
    var obsArea = 0;
    for (var i = 0; i < state.obstacles.length; i++) {
      obsArea += state.obstacles[i].w * state.obstacles[i].h;
    }
    el.infoArea.textContent = gross.toFixed(2) + ' m²（净 ' + (gross - obsArea).toFixed(2) + '）';
    el.infoPoints.textContent = String(state.points.length);
    el.infoTime.textContent = state.inferMs === null ? '-' : state.inferMs.toFixed(1) + ' ms';
    el.infoRange.textContent = COLOR_MIN + ' ~ ' + COLOR_MAX + ' dBm';
    el.pointCount.textContent = String(state.points.length);
  }

  /* --------------------------- 列表渲染 --------------------------- */

  function renderPointList() {
    el.pointList.innerHTML = '';
    for (var i = 0; i < state.points.length; i++) {
      var p = state.points[i];
      var li = document.createElement('li');
      if (state.selectedPointId === p.id) { li.className = 'active'; }
      li.dataset.id = String(p.id);

      var sw = document.createElement('span');
      sw.className = 'swatch';
      sw.style.background = rgbaFor(p.rssi, 1);
      sw.style.border = '1px solid #ffffff';

      var nm = document.createElement('span');
      nm.className = 'name';
      nm.textContent = '#' + (i + 1);

      var meta = document.createElement('span');
      meta.className = 'meta';
      meta.textContent = '(' + p.x.toFixed(2) + ', ' + p.y.toFixed(2) + ')  ' + p.rssi + ' dBm';

      var del = document.createElement('button');
      del.className = 'btn danger';
      del.textContent = '删除';
      del.dataset.del = String(p.id);

      li.appendChild(sw);
      li.appendChild(nm);
      li.appendChild(meta);
      li.appendChild(del);
      el.pointList.appendChild(li);
    }
  }

  function renderObsList() {
    el.obsList.innerHTML = '';
    for (var i = 0; i < state.obstacles.length; i++) {
      var o = state.obstacles[i];
      var li = document.createElement('li');
      if (state.selectedObsId === o.id) { li.className = 'active'; }
      li.dataset.id = String(o.id);

      var sw = document.createElement('span');
      sw.className = 'swatch';
      sw.style.background = '#59656f';

      var nm = document.createElement('span');
      nm.className = 'name';
      nm.textContent = o.name || ('障碍' + (i + 1));

      var meta = document.createElement('span');
      meta.className = 'meta';
      meta.textContent = '(' + o.x.toFixed(2) + ', ' + o.y.toFixed(2) + ')  ' +
        o.w.toFixed(2) + '×' + o.h.toFixed(2) + ' m';

      var del = document.createElement('button');
      del.className = 'btn danger';
      del.textContent = '删除';
      del.dataset.del = String(o.id);

      li.appendChild(sw);
      li.appendChild(nm);
      li.appendChild(meta);
      li.appendChild(del);
      el.obsList.appendChild(li);
    }
  }

  function syncObsEditor() {
    var o = null;
    for (var i = 0; i < state.obstacles.length; i++) {
      if (state.obstacles[i].id === state.selectedObsId) { o = state.obstacles[i]; }
    }
    if (!o) {
      el.obsEditor.classList.add('hidden');
      return;
    }
    el.obsEditor.classList.remove('hidden');
    el.obsX.value = o.x;
    el.obsY.value = o.y;
    el.obsW.value = o.w;
    el.obsH.value = o.h;
  }

  /* --------------------------- 户型 / 障碍物操作 --------------------------- */

  function setRoomSize(w, h, syncInputs) {
    state.room.w = clamp(w, 2, 15);
    state.room.h = clamp(h, 2, 15);
    if (state.door.w > state.room.w) { state.door.w = state.room.w; }
    if (syncInputs !== false) {
      el.roomWNum.value = state.room.w.toFixed(3);
      el.roomWRange.value = state.room.w;
      el.roomHNum.value = state.room.h.toFixed(3);
      el.roomHRange.value = state.room.h;
      el.doorWNum.value = state.door.w.toFixed(3);
    }
    updateInfo();
    render();
  }

  function resetRoom() {
    state.room.w = DEFAULT_ROOM.w;
    state.room.h = DEFAULT_ROOM.h;
    state.door.w = DEFAULT_DOOR.w;
    state.obstacles = [];
    for (var i = 0; i < DEFAULT_OBSTACLES.length; i++) {
      var d = DEFAULT_OBSTACLES[i];
      state.obstacles.push({ id: nextId++, x: d.x, y: d.y, w: d.w, h: d.h, name: d.name });
    }
    state.selectedObsId = null;
    setRoomSize(state.room.w, state.room.h, true);
    renderObsList();
    syncObsEditor();
  }

  function addObstacle() {
    var w = 2.0, h = 2.0;
    var o = {
      id: nextId++,
      x: clamp(state.room.w / 2 - w / 2, 0, Math.max(0, state.room.w - w)),
      y: clamp(state.room.h / 2 - h / 2, 0, Math.max(0, state.room.h - h)),
      w: w, h: h, name: '障碍物'
    };
    state.obstacles.push(o);
    state.selectedObsId = o.id;
    renderObsList();
    syncObsEditor();
    updateInfo();
    render();
  }

  function removeObstacle(id) {
    state.obstacles = state.obstacles.filter(function (o) { return o.id !== id; });
    if (state.selectedObsId === id) { state.selectedObsId = null; }
    renderObsList();
    syncObsEditor();
    if (state.heat) { runInfer(); } else { updateInfo(); render(); }
  }

  /* --------------------------- 测点操作 --------------------------- */

  function addPoint(rx, ry, rssi) {
    var p = { id: nextId++, x: rx, y: ry, rssi: rssi };
    state.points.push(p);
    state.selectedPointId = p.id;
    renderPointList();
    updateInfo();
    render();
  }

  function removePoint(id) {
    state.points = state.points.filter(function (p) { return p.id !== id; });
    if (state.selectedPointId === id) { state.selectedPointId = null; }
    renderPointList();
    updateInfo();
    render();
  }

  function clearPoints() {
    state.points = [];
    state.selectedPointId = null;
    state.heat = null;
    state.inferMs = null;
    renderPointList();
    updateInfo();
    render();
  }

  /* --------------------------- RSSI 输入弹窗 --------------------------- */

  var pendingPoint = null;

  function openRssiModal(rx, ry) {
    pendingPoint = { x: rx, y: ry };
    el.rssiModalPos.textContent = '坐标：(' + rx.toFixed(2) + ', ' + ry.toFixed(2) + ') m';
    el.rssiErr.classList.add('hidden');
    el.rssiModal.classList.remove('hidden');
    el.rssiInput.focus();
    el.rssiInput.select();
  }

  function closeRssiModal() {
    pendingPoint = null;
    el.rssiModal.classList.add('hidden');
  }

  function confirmRssi() {
    if (!pendingPoint) { return; }
    var v = parseFloat(el.rssiInput.value);
    if (!isFinite(v) || v < -100 || v > 0) {
      el.rssiErr.classList.remove('hidden');
      return;
    }
    addPoint(pendingPoint.x, pendingPoint.y, v);
    closeRssiModal();
  }

  /* --------------------------- 载入示例数据 --------------------------- */

  // 优先读文档内联 JSON（file:// 双击场景下 fetch 会被浏览器拦截）；
  // 内联块不存在时再回退到 fetch('sample_room.json')。
  function readInlineSample() {
    var tag = document.getElementById('sampleRoomData');
    if (!tag) { return null; }
    var txt = (tag.textContent || '').trim();
    if (!txt || txt.indexOf('__SAMPLE_ROOM_JSON__') === 0) { return null; }
    try { return JSON.parse(txt); } catch (e) { return null; }
  }

  function applySample(data) {
    if (data.room) {
      state.room.w = data.room.w;
      state.room.h = data.room.h;
      if (data.room.door && typeof data.room.door.w === 'number') {
        state.door.w = data.room.door.w;
      }
      state.obstacles = [];
      var obs = data.room.obstacles || [];
      for (var i = 0; i < obs.length; i++) {
        var o = obs[i];
        state.obstacles.push({
          id: nextId++, x: o.x, y: o.y, w: o.w, h: o.h, name: o.name || '障碍物'
        });
      }
    }
    if (data.grid) {
      state.grid = {
        x0: data.grid.x0, y0: data.grid.y0,
        cell: data.grid.cell, nx: data.grid.nx, ny: data.grid.ny
      };
    }
    state.points = [];
    var pts = data.points || [];
    for (var j = 0; j < pts.length; j++) {
      state.points.push({ id: nextId++, x: pts[j].x, y: pts[j].y, rssi: pts[j].rssi });
    }
    state.heat = null;
    state.inferMs = null;
    state.selectedPointId = null;
    state.selectedObsId = null;

    setRoomSize(state.room.w, state.room.h, true);
    renderObsList();
    syncObsEditor();
    renderPointList();
    updateInfo();
    render();
  }

  function loadSample() {
    var data = readInlineSample();
    if (data) { applySample(data); return; }
    fetch('sample_room.json')
      .then(function (r) { return r.json(); })
      .then(applySample)
      .catch(function () {
        window.alert('无法读取示例数据：内联数据缺失且 sample_room.json 加载失败。');
      });
  }

  /* --------------------------- 导出 --------------------------- */

  function download(name, href) {
    var a = document.createElement('a');
    a.href = href;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }

  function exportPng() {
    if (!state.heat) {
      window.alert('请先生成热力图。');
      return;
    }
    render();
    var url = canvas.toDataURL('image/png');
    download('rfmap_heatmap.png', url);
  }

  function exportCsv() {
    if (state.points.length === 0) {
      window.alert('当前没有测点可以导出。');
      return;
    }
    var lines = ['idx,x_room,y_room,rssi_dbm'];
    for (var i = 0; i < state.points.length; i++) {
      var p = state.points[i];
      lines.push((i + 1) + ',' + p.x.toFixed(4) + ',' + p.y.toFixed(4) + ',' + p.rssi);
    }
    var blob = new Blob([lines.join('\n') + '\n'], { type: 'text/csv;charset=utf-8' });
    var url = URL.createObjectURL(blob);
    download('rfmap_points.csv', url);
    setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
  }

  /* --------------------------- 命中测试 --------------------------- */

  function hitPoint(mx, my) {
    for (var i = state.points.length - 1; i >= 0; i--) {
      var p = state.points[i];
      var dx = sx(p.x) - mx;
      var dy = sy(p.y) - my;
      if (dx * dx + dy * dy <= HIT_R * HIT_R) { return p; }
    }
    return null;
  }

  function hitObstacle(mx, my) {
    for (var i = state.obstacles.length - 1; i >= 0; i--) {
      var o = state.obstacles[i];
      var x0 = sx(o.x + o.w), y0 = sy(o.y + o.h);
      var w = o.w * view.scale, h = o.h * view.scale;
      if (mx >= x0 && mx <= x0 + w && my >= y0 && my <= y0 + h) { return o; }
    }
    return null;
  }

  function hitRoomHandle(mx, my) {
    var dx = sx(state.room.w) - mx;
    var dy = sy(state.room.h) - my;
    return (dx * dx + dy * dy) <= (HANDLE_R + 3) * (HANDLE_R + 3);
  }

  /* --------------------------- 鼠标 / 指针交互 --------------------------- */

  function localPos(ev) {
    var r = canvas.getBoundingClientRect();
    return { x: ev.clientX - r.left, y: ev.clientY - r.top };
  }

  canvas.addEventListener('pointerdown', function (ev) {
    var m = localPos(ev);

    // 户型手柄
    if (hitRoomHandle(m.x, m.y)) {
      state.drag = { kind: 'room' };
      canvas.setPointerCapture(ev.pointerId);
      return;
    }

    // 测点
    var p = hitPoint(m.x, m.y);
    if (p) {
      if (ev.button === 2) { removePoint(p.id); return; }
      state.selectedPointId = p.id;
      state.selectedObsId = null;
      state.drag = { kind: 'point', id: p.id };
      renderPointList();
      syncObsEditor();
      canvas.setPointerCapture(ev.pointerId);
      render();
      return;
    }

    // 障碍物
    var o = hitObstacle(m.x, m.y);
    if (o) {
      if (ev.button === 2) { removeObstacle(o.id); return; }
      state.selectedObsId = o.id;
      state.selectedPointId = null;
      state.drag = { kind: 'obs', id: o.id, dx: rxOf(m.x) - o.x, dy: ryOf(m.y) - o.y };
      renderObsList();
      renderPointList();
      syncObsEditor();
      canvas.setPointerCapture(ev.pointerId);
      render();
      return;
    }

    // 空白处：左键添加测点（必须在房间内部、且不在障碍物上）
    if (ev.button === 0) {
      var rx = rxOf(m.x);
      var ry = ryOf(m.y);
      if (insideRoom(rx, ry) && !insideObstacle(rx, ry)) {
        openRssiModal(rx, ry);
      }
    }
  });

  canvas.addEventListener('pointermove', function (ev) {
    var m = localPos(ev);

    // 拖拽中
    if (state.drag) {
      if (state.drag.kind === 'room') {
        var nw = rxOf(m.x);
        var nh = ryOf(m.y);
        setRoomSize(nw, nh, true);
        return;
      }
      if (state.drag.kind === 'point') {
        for (var i = 0; i < state.points.length; i++) {
          if (state.points[i].id === state.drag.id) {
            state.points[i].x = clamp(rxOf(m.x), 0, state.room.w);
            state.points[i].y = clamp(ryOf(m.y), 0, state.room.h);
          }
        }
        renderPointList();
        render();
        return;
      }
      if (state.drag.kind === 'obs') {
        for (var j = 0; j < state.obstacles.length; j++) {
          if (state.obstacles[j].id === state.drag.id) {
            var o = state.obstacles[j];
            o.x = clamp(rxOf(m.x) - state.drag.dx, 0, Math.max(0, state.room.w - o.w));
            o.y = clamp(ryOf(m.y) - state.drag.dy, 0, Math.max(0, state.room.h - o.h));
          }
        }
        renderObsList();
        syncObsEditor();
        render();
        return;
      }
    }

    // 悬停读数
    var rx2 = rxOf(m.x);
    var ry2 = ryOf(m.y);
    if (state.heat && insideRoom(rx2, ry2)) {
      var g = state.grid;
      var ii = Math.floor((rx2 - g.x0) / g.cell);
      var jj = Math.floor((ry2 - g.y0) / g.cell);
      if (ii >= 0 && ii < g.nx && jj >= 0 && jj < g.ny) {
        var v = state.heat[jj * g.nx + ii];
        if (v > -119) {
          tooltip.textContent = 'x=' + rx2.toFixed(2) + ' m  y=' + ry2.toFixed(2) +
            ' m\n预测 ' + v.toFixed(1) + ' dBm';
          tooltip.classList.remove('hidden');
          tooltip.style.left = (m.x + 16) + 'px';
          tooltip.style.top = (m.y + 12) + 'px';
          return;
        }
      }
    }
    tooltip.classList.add('hidden');
  });

  function endDrag(ev) {
    if (state.drag) {
      var wasObs = state.drag.kind === 'obs';
      state.drag = null;
      if (ev && ev.pointerId !== undefined && canvas.hasPointerCapture && canvas.hasPointerCapture(ev.pointerId)) {
        canvas.releasePointerCapture(ev.pointerId);
      }
      if (wasObs && state.heat) { runInfer(); }
    }
  }

  canvas.addEventListener('pointerup', endDrag);
  canvas.addEventListener('pointercancel', endDrag);
  canvas.addEventListener('pointerleave', function () { tooltip.classList.add('hidden'); });

  // 屏蔽右键菜单（右键用于删除测点 / 障碍物）
  canvas.addEventListener('contextmenu', function (ev) { ev.preventDefault(); });

  /* --------------------------- 面板事件绑定 --------------------------- */

  el.roomWNum.addEventListener('input', function () {
    setRoomSize(parseFloat(el.roomWNum.value) || state.room.w, state.room.h, true);
  });
  el.roomWRange.addEventListener('input', function () {
    setRoomSize(parseFloat(el.roomWRange.value), state.room.h, true);
  });
  el.roomHNum.addEventListener('input', function () {
    setRoomSize(state.room.w, parseFloat(el.roomHNum.value) || state.room.h, true);
  });
  el.roomHRange.addEventListener('input', function () {
    setRoomSize(state.room.w, parseFloat(el.roomHRange.value), true);
  });
  el.doorWNum.addEventListener('input', function () {
    var v = parseFloat(el.doorWNum.value);
    if (isFinite(v) && v > 0) {
      state.door.w = clamp(v, 0.3, state.room.w);
      render();
    }
  });

  document.getElementById('btnResetRoom').addEventListener('click', resetRoom);
  document.getElementById('btnAddObs').addEventListener('click', addObstacle);
  document.getElementById('btnLoadSample').addEventListener('click', loadSample);
  document.getElementById('btnClearPoints').addEventListener('click', clearPoints);
  document.getElementById('btnInfer').addEventListener('click', runInfer);
  document.getElementById('btnClearHeat').addEventListener('click', function () {
    state.heat = null;
    state.inferMs = null;
    updateInfo();
    render();
  });
  document.getElementById('btnExportPng').addEventListener('click', exportPng);
  document.getElementById('btnExportCsv').addEventListener('click', exportCsv);

  el.obsList.addEventListener('click', function (ev) {
    var delId = ev.target.dataset ? ev.target.dataset.del : null;
    if (delId) { removeObstacle(parseInt(delId, 10)); return; }
    var li = ev.target.closest('li');
    if (li && li.dataset.id) {
      state.selectedObsId = parseInt(li.dataset.id, 10);
      state.selectedPointId = null;
      renderObsList();
      renderPointList();
      syncObsEditor();
      render();
    }
  });

  el.pointList.addEventListener('click', function (ev) {
    var delId = ev.target.dataset ? ev.target.dataset.del : null;
    if (delId) { removePoint(parseInt(delId, 10)); return; }
    var li = ev.target.closest('li');
    if (li && li.dataset.id) {
      state.selectedPointId = parseInt(li.dataset.id, 10);
      state.selectedObsId = null;
      renderPointList();
      syncObsEditor();
      render();
    }
  });

  function bindObsField(input, key) {
    input.addEventListener('input', function () {
      var o = null;
      for (var i = 0; i < state.obstacles.length; i++) {
        if (state.obstacles[i].id === state.selectedObsId) { o = state.obstacles[i]; }
      }
      if (!o) { return; }
      var v = parseFloat(input.value);
      if (!isFinite(v)) { return; }
      o[key] = Math.max(key === 'w' || key === 'h' ? 0.1 : 0, v);
      renderObsList();
      render();
    });
  }
  bindObsField(el.obsX, 'x');
  bindObsField(el.obsY, 'y');
  bindObsField(el.obsW, 'w');
  bindObsField(el.obsH, 'h');

  document.getElementById('rssiOk').addEventListener('click', confirmRssi);
  document.getElementById('rssiCancel').addEventListener('click', closeRssiModal);
  el.rssiInput.addEventListener('keydown', function (ev) {
    if (ev.key === 'Enter') { confirmRssi(); }
    else if (ev.key === 'Escape') { closeRssiModal(); }
  });
  el.rssiModal.addEventListener('click', function (ev) {
    if (ev.target === el.rssiModal) { closeRssiModal(); }
  });

  document.addEventListener('keydown', function (ev) {
    if (ev.key !== 'Delete' && ev.key !== 'Backspace') { return; }
    if (ev.target && (ev.target.tagName === 'INPUT' || ev.target.tagName === 'TEXTAREA')) { return; }
    if (state.selectedPointId !== null) { removePoint(state.selectedPointId); }
    else if (state.selectedObsId !== null) { removeObstacle(state.selectedObsId); }
  });

  window.addEventListener('resize', function () { render(); });

  /* --------------------------- 启动 --------------------------- */

  resetRoom();
  updateInfo();
  render();
})();
