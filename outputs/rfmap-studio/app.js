/* RF Map Studio · 家具感知版
 * 纯前端、离线、无第三方依赖。神经网络由 nn_model.js 提供；加载失败时回退 IDW。
 * 坐标：原点在房间西南角，x 向东（画布向右），y 向北（画布向上）。
 */
(function () {
  'use strict';

  var DEFAULT_ROOM = { w: 12.643, h: 11.731 };
  var DEFAULT_DOOR = { w: 2.330 };
  var DEFAULT_OBSTACLES = [{ x: 11.813, y: 5.761, w: 0.830, h: 1.010, name: '结构柱' }];
  var DEFAULT_GRID = { x0: 0, y0: 0, cell: 0.2, nx: 64, ny: 64 };
  var COLOR_MIN = -70, COLOR_MAX = -15;
  var LEVEL_COLORS = ['#d6e6f5', '#9ec9e8', '#62a9d7', '#2c7fb8', '#084d8c'];
  var LEVEL_NAMES = ['⑤极弱', '④弱', '③中', '②强', '①极强'];
  var FURNITURE = {
    table: { label: '桌子', short: '桌', color: '#d9a441', w: 0.8, h: 0.8, shape: 'circle' },
    chair: { label: '椅子', short: '椅', color: '#5b9bd5', w: 0.8, h: 0.8, shape: 'circle' },
    sofa: { label: '沙发', short: '沙', color: '#9b7ede', w: 1.8, h: 0.8, shape: 'rect' },
    bed: { label: '床', short: '床', color: '#e89aa8', w: 1.8, h: 2.0, shape: 'rect' },
    cabinet: { label: '柜子', short: '柜', color: '#8c6d4f', w: 1.2, h: 0.6, shape: 'rect' }
  };
  var STOPS = [
    [0.00, [16, 30, 138]], [0.25, [22, 132, 214]], [0.50, [42, 186, 106]],
    [0.75, [246, 214, 48]], [1.00, [186, 22, 32]]
  ];
  var nextId = 1, inferTimer = null, pendingPoint = null;

  var state = {
    room: { w: DEFAULT_ROOM.w, h: DEFAULT_ROOM.h }, door: { w: DEFAULT_DOOR.w },
    obstacles: [], furniture: [], points: [], grid: Object.assign({}, DEFAULT_GRID),
    heat: null, comparisonHeat: null, inferMs: null, displayMode: 'continuous',
    selectedPointId: null, selectedObsId: null, selectedFurnitureId: null, drag: null
  };

  var canvas = document.getElementById('scene'), ctx = canvas.getContext('2d');
  var tooltip = document.getElementById('tooltip');
  var ids = ['infoArea','infoPoints','infoFurniture','infoTime','infoRange','roomWNum','roomWRange',
    'roomHNum','roomHRange','doorWNum','obsList','obsEditor','obsX','obsY','obsW','obsH',
    'furnitureType','furnitureList','furnitureEditor','furnitureX','furnitureY','furnitureW','furnitureH',
    'pointList','pointCount','comparisonResult','displayMode','modelInfo','pointsCsvInput','projectInput',
    'rssiModal','rssiModalPos','rssiInput','rssiErr'];
  var el = {};
  ids.forEach(function (id) { el[id] = document.getElementById(id); });

  function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }
  function finite(v) { return typeof v === 'number' && isFinite(v); }
  function levelIndex(v) { return v < -48 ? 0 : (v < -44 ? 1 : (v < -40 ? 2 : (v < -36 ? 3 : 4))); }
  function insideRoom(x, y) { return x >= 0 && x <= state.room.w && y >= 0 && y <= state.room.h; }
  function insideObstacle(x, y) {
    for (var i = 0; i < state.obstacles.length; i++) {
      var o = state.obstacles[i];
      if (x >= o.x && x <= o.x + o.w && y >= o.y && y <= o.y + o.h) return o;
    }
    return null;
  }
  function insideFurniture(x, y, f) {
    var dx = Math.abs(x - f.x), dy = Math.abs(y - f.y);
    if (f.shape === 'circle') {
      var rx = f.w / 2, ry = f.h / 2;
      return (dx * dx) / (rx * rx) + (dy * dy) / (ry * ry) <= 1;
    }
    return dx <= f.w / 2 && dy <= f.h / 2;
  }

  /* ---------------- 视图与颜色 ---------------- */
  var view = { cssW: 0, cssH: 0, scale: 1, ox: 0, oy: 0, padL: 66, padR: 116, padT: 42, padB: 64 };
  function computeView() {
    var dpr = window.devicePixelRatio || 1;
    var cssW = canvas.clientWidth || 850, cssH = canvas.clientHeight || 650;
    canvas.width = Math.round(cssW * dpr); canvas.height = Math.round(cssH * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    view.cssW = cssW; view.cssH = cssH;
    var plotW = Math.max(40, cssW - view.padL - view.padR);
    var plotH = Math.max(40, cssH - view.padT - view.padB);
    var world = Math.max(14.5, state.room.w + 0.7, state.room.h + 0.7);
    view.scale = Math.min(plotW / world, plotH / world);
    view.ox = view.padL + (plotW - world * view.scale) / 2;
    view.oy = view.padT + (plotH - world * view.scale) / 2 + world * view.scale;
  }
  function sx(x) { return view.ox + x * view.scale; }
  function sy(y) { return view.oy - y * view.scale; }
  function rxOf(x) { return (x - view.ox) / view.scale; }
  function ryOf(y) { return (view.oy - y) / view.scale; }
  function colorFor(v) {
    if (state.displayMode === 'levels') {
      var c = LEVEL_COLORS[levelIndex(v)].replace('#', '');
      return [parseInt(c.slice(0,2),16), parseInt(c.slice(2,4),16), parseInt(c.slice(4,6),16)];
    }
    var t = clamp((v - COLOR_MIN) / (COLOR_MAX - COLOR_MIN), 0, 1);
    var a = STOPS[0], b = STOPS[STOPS.length - 1];
    for (var i = 0; i < STOPS.length - 1; i++) {
      if (t >= STOPS[i][0] && t <= STOPS[i + 1][0]) { a = STOPS[i]; b = STOPS[i + 1]; break; }
    }
    var f = (t - a[0]) / ((b[0] - a[0]) || 1);
    return [0,1,2].map(function (k) { return Math.round(a[1][k] + (b[1][k] - a[1][k]) * f); });
  }
  function cssColor(v) { var c = colorFor(v); return 'rgb(' + c.join(',') + ')'; }

  /* ---------------- 绘制 ---------------- */
  function render() {
    computeView(); ctx.clearRect(0, 0, view.cssW, view.cssH);
    drawRoomFloor(); if (state.heat) drawHeat();
    drawWalls(); drawObstacles(); drawFurniture(); drawDoor(); drawAxes(); drawPoints();
    if (state.heat) drawColorbar(); drawRoomHandle(); drawLegend(); drawEmptyNotice();
  }
  function drawRoomFloor() {
    ctx.fillStyle = '#f4f7fb';
    ctx.fillRect(sx(0), sy(state.room.h), state.room.w * view.scale, state.room.h * view.scale);
  }
  function drawHeat() {
    var g = state.grid, nx = g.nx, ny = g.ny, N = nx * ny;
    var off = document.createElement('canvas'); off.width = nx; off.height = ny;
    var octx = off.getContext('2d'), img = octx.createImageData(nx, ny), d = img.data;
    for (var j = 0; j < ny; j++) for (var i = 0; i < nx; i++) {
      var k = j * nx + i, v = state.heat[k], c = colorFor(v), p = ((ny - 1 - j) * nx + i) * 4;
      d[p] = c[0]; d[p+1] = c[1]; d[p+2] = c[2]; d[p+3] = v > -119 ? 255 : 0;
    }
    octx.putImageData(img, 0, 0);
    ctx.save(); ctx.beginPath();
    ctx.rect(sx(0), sy(state.room.h), state.room.w * view.scale, state.room.h * view.scale);
    state.obstacles.forEach(function (o) {
      ctx.rect(sx(o.x), sy(o.y + o.h), o.w * view.scale, o.h * view.scale);
    });
    ctx.clip('evenodd'); ctx.globalAlpha = 0.78;
    ctx.imageSmoothingEnabled = state.displayMode === 'continuous';
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(off, sx(g.x0), sy(g.y0 + ny * g.cell), nx * g.cell * view.scale, ny * g.cell * view.scale);
    ctx.restore();
  }
  function drawWalls() {
    var x = sx(0), y = sy(state.room.h), w = state.room.w * view.scale, h = state.room.h * view.scale;
    ctx.lineWidth = 5; ctx.strokeStyle = '#22303f'; ctx.strokeRect(x, y, w, h);
    ctx.lineWidth = 1.5; ctx.strokeStyle = '#4a5c6e'; ctx.strokeRect(x+3, y+3, w-6, h-6);
  }
  function drawObstacles() {
    state.obstacles.forEach(function (o) {
      var x = sx(o.x), y = sy(o.y + o.h), w = o.w * view.scale, h = o.h * view.scale;
      ctx.fillStyle = '#59656f'; ctx.fillRect(x,y,w,h);
      ctx.lineWidth = state.selectedObsId === o.id ? 3 : 1.5;
      ctx.strokeStyle = state.selectedObsId === o.id ? '#1a6fd4' : '#39434c'; ctx.strokeRect(x,y,w,h);
      if (w > 42 && h > 18) { ctx.fillStyle='#fff'; ctx.font='11px sans-serif'; ctx.textAlign='center'; ctx.textBaseline='middle'; ctx.fillText(o.name || '固定结构',x+w/2,y+h/2); }
    });
    ctx.textAlign='start'; ctx.textBaseline='alphabetic';
  }
  function drawFurniture() {
    state.furniture.forEach(function (f) {
      var def = FURNITURE[f.kind] || FURNITURE.table, selected = state.selectedFurnitureId === f.id;
      ctx.fillStyle = def.color; ctx.strokeStyle = selected ? '#102f52' : '#3e4650'; ctx.lineWidth = selected ? 3 : 1.2;
      ctx.globalAlpha = 0.86; ctx.beginPath();
      if (f.shape === 'circle') ctx.ellipse(sx(f.x),sy(f.y),f.w*view.scale/2,f.h*view.scale/2,0,0,Math.PI*2);
      else ctx.rect(sx(f.x-f.w/2),sy(f.y+f.h/2),f.w*view.scale,f.h*view.scale);
      ctx.fill(); ctx.stroke(); ctx.globalAlpha = 1;
      ctx.fillStyle = '#17242f'; ctx.font = 'bold 10px sans-serif'; ctx.textAlign='center'; ctx.textBaseline='middle';
      ctx.fillText(def.short,sx(f.x),sy(f.y));
    });
    ctx.textAlign='start'; ctx.textBaseline='alphabetic';
  }
  function drawDoor() {
    var w = Math.min(state.door.w,state.room.w), x0=(state.room.w-w)/2, y=sy(0), r=w*view.scale/2;
    ctx.fillStyle='#f4f7fb'; ctx.fillRect(sx(x0)-1,y-4,w*view.scale+2,8);
    ctx.strokeStyle='#b8862b'; ctx.lineWidth=1.6;
    ctx.beginPath(); ctx.arc(sx(x0),y,r,-Math.PI/2,0); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(sx(x0),y); ctx.lineTo(sx(x0),y-r); ctx.stroke();
    ctx.beginPath(); ctx.arc(sx(x0+w),y,r,Math.PI,Math.PI*1.5); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(sx(x0+w),y); ctx.lineTo(sx(x0+w),y-r); ctx.stroke();
  }
  function drawAxes() {
    var s=view.scale; ctx.font='11px sans-serif'; ctx.fillStyle='#5a6775'; ctx.strokeStyle='#9aa8b6'; ctx.lineWidth=1;
    ctx.textAlign='center'; ctx.textBaseline='top';
    for(var x=0;x<=Math.floor(state.room.w);x+=2){ctx.beginPath();ctx.moveTo(sx(x),view.oy);ctx.lineTo(sx(x),view.oy+6);ctx.stroke();ctx.fillText(String(x),sx(x),view.oy+9);}
    ctx.textAlign='right';ctx.textBaseline='middle';
    for(var y=0;y<=Math.floor(state.room.h);y+=2){ctx.beginPath();ctx.moveTo(view.ox-6,sy(y));ctx.lineTo(view.ox,sy(y));ctx.stroke();ctx.fillText(String(y),view.ox-9,sy(y));}
    ctx.fillStyle='#22303f';ctx.beginPath();ctx.arc(view.ox,view.oy,4,0,Math.PI*2);ctx.fill();
    ctx.textAlign='left';ctx.textBaseline='top';ctx.fillText('原点 (0,0) 西南角',view.ox+8,view.oy+6);
    ctx.fillStyle='#5a6775';ctx.textAlign='center';ctx.fillText('x (m) 向东 →',sx(state.room.w/2),view.oy+34);
    ctx.save();ctx.translate(view.ox-42,sy(state.room.h/2));ctx.rotate(-Math.PI/2);ctx.textBaseline='middle';ctx.fillText('y (m) 向北 →',0,0);ctx.restore();
    ctx.textAlign='start';ctx.textBaseline='alphabetic';
  }
  function drawPoints() {
    state.points.forEach(function(p,index){
      var selected=state.selectedPointId===p.id, px=sx(p.x),py=sy(p.y);
      ctx.beginPath();ctx.arc(px,py,selected?7:5,0,Math.PI*2);ctx.fillStyle=selected?'#ff7a18':'#ffd166';ctx.fill();ctx.lineWidth=1.4;ctx.strokeStyle='#222';ctx.stroke();
      if(selected||state.points.length<=20){ctx.font='10px sans-serif';ctx.fillStyle='#26313c';ctx.textAlign='center';ctx.textBaseline='bottom';ctx.fillText(String(p.rssi),px,py-8);}
    });ctx.textAlign='start';ctx.textBaseline='alphabetic';
  }
  function drawRoomHandle() {
    var x=sx(state.room.w),y=sy(state.room.h);ctx.beginPath();ctx.arc(x,y,7,0,Math.PI*2);ctx.fillStyle='#fff';ctx.fill();ctx.lineWidth=2;ctx.strokeStyle='#1a6fd4';ctx.stroke();
    ctx.font='11px sans-serif';ctx.fillStyle='#1a6fd4';ctx.textAlign='right';ctx.textBaseline='bottom';ctx.fillText('拖动调整宽高',x-10,y-5);ctx.textAlign='start';ctx.textBaseline='alphabetic';
  }
  function drawColorbar() {
    var x=view.cssW-view.padR+38,y=view.padT,w=24,h=view.cssH-view.padT-view.padB;
    if(state.displayMode==='levels'){
      var bh=h/5;for(var i=0;i<5;i++){var level=4-i;ctx.fillStyle=LEVEL_COLORS[level];ctx.fillRect(x,y+i*bh,w,bh);ctx.strokeStyle='#526170';ctx.strokeRect(x,y+i*bh,w,bh);ctx.fillStyle=level>=2?'#fff':'#17354d';ctx.font='bold 11px sans-serif';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(String(level+1),x+w/2,y+(i+0.5)*bh);ctx.fillStyle='#39434c';ctx.textAlign='left';ctx.font='10px sans-serif';ctx.fillText(LEVEL_NAMES[level],x+w+7,y+(i+0.5)*bh);}
    }else{
      var off=document.createElement('canvas');off.width=1;off.height=256;var oc=off.getContext('2d');
      for(var r=0;r<256;r++){oc.fillStyle=cssColor(COLOR_MAX-r/255*(COLOR_MAX-COLOR_MIN));oc.fillRect(0,r,1,1);}ctx.drawImage(off,x,y,w,h);ctx.strokeStyle='#39434c';ctx.strokeRect(x,y,w,h);
      ctx.font='10px sans-serif';ctx.fillStyle='#39434c';ctx.textAlign='left';ctx.textBaseline='middle';for(var v=COLOR_MIN;v<=COLOR_MAX;v+=10){var yy=y+(COLOR_MAX-v)/(COLOR_MAX-COLOR_MIN)*h;ctx.fillText(String(v),x+w+7,yy);}
    }
    ctx.fillStyle='#39434c';ctx.textAlign='center';ctx.textBaseline='bottom';ctx.fillText(state.displayMode==='levels'?'等级':'dBm',x+w/2,y-6);ctx.textAlign='start';ctx.textBaseline='alphabetic';
  }
  function drawLegend() {
    var x=view.padL+8,y=view.padT+8,w=174,h=76;ctx.fillStyle='rgba(255,255,255,.92)';ctx.fillRect(x,y,w,h);ctx.strokeStyle='#d3dae3';ctx.strokeRect(x,y,w,h);
    ctx.font='11px sans-serif';ctx.textAlign='left';ctx.textBaseline='middle';ctx.fillStyle='#3a4756';
    ctx.fillText('黄色点：实测 RSSI',x+10,y+15);ctx.fillText('彩色图形：虚拟家具（参与模型）',x+10,y+34);ctx.fillText('灰色矩形：固定墙体/柱体',x+10,y+53);ctx.fillText('悬停：读取预测值与家具对照差',x+10,y+68);ctx.textAlign='start';ctx.textBaseline='alphabetic';
  }
  function drawEmptyNotice() {
    var msg=null;if(!state.points.length)msg='单击室内添加实测点，或载入 12 点家具版示例';else if(!state.heat)msg='点击“生成整屋预测”查看结果';if(!msg)return;
    ctx.font='13px sans-serif';ctx.textAlign='center';ctx.textBaseline='middle';var cx=view.cssW/2,cy=view.cssH-22,tw=ctx.measureText(msg).width;ctx.fillStyle='rgba(22,33,44,.86)';ctx.fillRect(cx-tw/2-12,cy-13,tw+24,26);ctx.fillStyle='#fff';ctx.fillText(msg,cx,cy);ctx.textAlign='start';ctx.textBaseline='alphabetic';
  }

  /* ---------------- 神经网络输入与推理 ---------------- */
  // 权重缺失时仍可查看 IDW 插值，但界面会明确提示，不把回退结果称作神经网络。
  if (!window.RFM_Infer) {
    window.RFM_Infer = function (input) {
      var n=input.nx*input.ny,pred=new Float32Array(n),obs=[];
      for(var k=0;k<n;k++)if(input.obsMask[k]>0.5){var i=k%input.nx,j=Math.floor(k/input.nx);obs.push({x:input.x0+(i+0.5)*input.cell,y:input.y0+(j+0.5)*input.cell,v:input.obs[k]});}
      for(var q=0;q<n;q++){if(input.roomMask[q]<0.5){pred[q]=-120;continue;}var ii=q%input.nx,jj=Math.floor(q/input.nx),x=input.x0+(ii+0.5)*input.cell,y=input.y0+(jj+0.5)*input.cell,ws=0,vs=0,mean=0;for(var o=0;o<obs.length;o++){var dx=x-obs[o].x,dy=y-obs[o].y,w=1/Math.max(dx*dx+dy*dy,1e-6);ws+=w;vs+=w*obs[o].v;mean+=obs[o].v;}pred[q]=obs.length?vs/ws:-60;}
      return{pred:pred};
    };
    window.RFM_InferKind='idw-fallback';
  }
  function buildInput(furnitureOverride) {
    var g=state.grid,nx=g.nx,ny=g.ny,N=nx*ny,obs=new Float32Array(N),obsMask=new Float32Array(N),roomMask=new Float32Array(N),furnitureMask=new Float32Array(N),sum=new Float32Array(N),cnt=new Float32Array(N);
    var furniture=furnitureOverride||state.furniture;
    for(var j=0;j<ny;j++)for(var i=0;i<nx;i++){var k=j*nx+i,cx=g.x0+(i+0.5)*g.cell,cy=g.y0+(j+0.5)*g.cell,inside=cx<=state.room.w&&cy<=state.room.h&&!insideObstacle(cx,cy);roomMask[k]=inside?1:0;if(inside){for(var f=0;f<furniture.length;f++){if(insideFurniture(cx,cy,furniture[f])){furnitureMask[k]=1;break;}}}}
    state.points.forEach(function(p){var i=Math.floor((p.x-g.x0)/g.cell),j=Math.floor((p.y-g.y0)/g.cell);if(i<0||i>=nx||j<0||j>=ny)return;var k=j*nx+i;if(roomMask[k]>0.5){sum[k]+=p.rssi;cnt[k]+=1;}});
    for(var q=0;q<N;q++)if(cnt[q]>0){obs[q]=sum[q]/cnt[q];obsMask[q]=1;}
    return {obs:obs,obsMask:obsMask,roomMask:roomMask,furnitureMask:furnitureMask,nx:nx,ny:ny,x0:g.x0,y0:g.y0,cell:g.cell};
  }
  function inferGrid(furnitureOverride) {
    if(!window.RFM_Infer)throw new Error('推理引擎未加载');return window.RFM_Infer(buildInput(furnitureOverride)).pred;
  }
  function setBusy(busy){document.getElementById('btnInfer').disabled=busy;document.getElementById('btnCompareFurniture').disabled=busy;}
  function runInfer() {
    if(state.points.length<3){window.alert('请至少输入 3 个实测点；建议 6 个以上并分散布置。');return;}
    setBusy(true);document.getElementById('btnInfer').textContent='计算中…';
    setTimeout(function(){try{var t0=performance.now();state.heat=inferGrid();state.inferMs=performance.now()-t0;state.comparisonHeat=null;el.comparisonResult.textContent='已更新当前家具布局预测。可继续对比“撤掉全部家具”。';}catch(e){window.alert('推理失败：'+e.message);console.error(e);}finally{setBusy(false);document.getElementById('btnInfer').textContent='生成整屋预测';updateInfo();render();}},30);
  }
  function queueInfer() { if(!state.heat)return;if(state.points.length<3){state.heat=null;state.comparisonHeat=null;updateInfo();render();return;}clearTimeout(inferTimer);inferTimer=setTimeout(runInfer,160); }
  function compareFurniture() {
    if(state.points.length<3){window.alert('请至少输入 3 个实测点。');return;}setBusy(true);document.getElementById('btnCompareFurniture').textContent='正在双次推理…';
    setTimeout(function(){try{var t0=performance.now(),withF=inferGrid(),withoutF=inferGrid([]);state.heat=withF;state.comparisonHeat=withoutF;state.inferMs=performance.now()-t0;var sum=0,abs=0,max=0,n=0,g=state.grid;for(var k=0;k<withF.length;k++)if(withF[k]>-119&&withoutF[k]>-119){var d=withF[k]-withoutF[k];sum+=d;abs+=Math.abs(d);max=Math.max(max,Math.abs(d));n++;}el.comparisonResult.textContent=n?('有效格点 '+n+' 个：有家具−无家具平均 '+ (sum/n>=0?'+':'')+(sum/n).toFixed(2)+' dB；平均绝对差 '+(abs/n).toFixed(2)+' dB；最大绝对差 '+max.toFixed(2)+' dB。仅为模型敏感性推演。'):'没有可比较的有效格点。';}catch(e){window.alert('对照失败：'+e.message);console.error(e);}finally{setBusy(false);document.getElementById('btnCompareFurniture').textContent='对比“撤掉全部家具”';updateInfo();render();}},30);
  }

  /* ---------------- 状态与列表 ---------------- */
  function updateInfo() {
    var fixed=state.obstacles.reduce(function(a,o){return a+o.w*o.h;},0),gross=state.room.w*state.room.h;
    el.infoArea.textContent=gross.toFixed(1)+' m²（净 '+(gross-fixed).toFixed(1)+'）';el.infoPoints.textContent=String(state.points.length);el.infoFurniture.textContent=String(state.furniture.length);el.infoTime.textContent=state.inferMs===null?'-':state.inferMs.toFixed(0)+' ms';el.infoRange.textContent=state.displayMode==='levels'?'五级色阶':'连续色标';el.pointCount.textContent=String(state.points.length);
  }
  function makeListItem(container,obj,selected,name,meta,color){var li=document.createElement('li');li.dataset.id=String(obj.id);if(selected)li.className='active';var sw=document.createElement('span');sw.className='swatch';sw.style.background=color;var nm=document.createElement('span');nm.className='name';nm.textContent=name;var md=document.createElement('span');md.className='meta';md.textContent=meta;var del=document.createElement('button');del.className='btn danger';del.dataset.del=String(obj.id);del.textContent='删除';li.appendChild(sw);li.appendChild(nm);li.appendChild(md);li.appendChild(del);container.appendChild(li);}
  function renderPointList(){el.pointList.innerHTML='';state.points.forEach(function(p,i){makeListItem(el.pointList,p,state.selectedPointId===p.id,'#'+(i+1),'('+p.x.toFixed(2)+', '+p.y.toFixed(2)+')  '+p.rssi+' dBm',cssColor(p.rssi));});}
  function renderObsList(){el.obsList.innerHTML='';state.obstacles.forEach(function(o,i){makeListItem(el.obsList,o,state.selectedObsId===o.id,o.name||'结构'+(i+1),'('+o.x.toFixed(2)+', '+o.y.toFixed(2)+')  '+o.w.toFixed(2)+'×'+o.h.toFixed(2)+' m','#59656f');});}
  function renderFurnitureList(){el.furnitureList.innerHTML='';state.furniture.forEach(function(f){var d=FURNITURE[f.kind]||FURNITURE.table;makeListItem(el.furnitureList,f,state.selectedFurnitureId===f.id,f.name||d.label,'('+f.x.toFixed(2)+', '+f.y.toFixed(2)+')  '+f.w.toFixed(2)+'×'+f.h.toFixed(2)+' m',d.color);});}
  function selected(list,id){for(var i=0;i<list.length;i++)if(list[i].id===id)return list[i];return null;}
  function syncEditors(){var o=selected(state.obstacles,state.selectedObsId);el.obsEditor.classList.toggle('hidden',!o);if(o){el.obsX.value=o.x;el.obsY.value=o.y;el.obsW.value=o.w;el.obsH.value=o.h;}var f=selected(state.furniture,state.selectedFurnitureId);el.furnitureEditor.classList.toggle('hidden',!f);if(f){el.furnitureX.value=f.x;el.furnitureY.value=f.y;el.furnitureW.value=f.w;el.furnitureH.value=f.h;}}
  function refresh(){renderPointList();renderObsList();renderFurnitureList();syncEditors();updateInfo();render();}
  function edited(){refresh();queueInfer();}
  function setRoomSize(w,h){state.room.w=clamp(finite(w)?w:state.room.w,2,12.8);state.room.h=clamp(finite(h)?h:state.room.h,2,12.8);state.door.w=Math.min(state.door.w,state.room.w);state.furniture.forEach(function(f){f.x=clamp(f.x,f.w/2,state.room.w-f.w/2);f.y=clamp(f.y,f.h/2,state.room.h-f.h/2);});state.obstacles.forEach(function(o){o.x=clamp(o.x,0,Math.max(0,state.room.w-o.w));o.y=clamp(o.y,0,Math.max(0,state.room.h-o.h));});state.points.forEach(function(p){p.x=clamp(p.x,0,state.room.w);p.y=clamp(p.y,0,state.room.h);});el.roomWNum.value=state.room.w.toFixed(3);el.roomWRange.value=state.room.w;el.roomHNum.value=state.room.h.toFixed(3);el.roomHRange.value=state.room.h;el.doorWNum.value=state.door.w.toFixed(3);refresh();}
  function addObstacle(){var w=1,h=1,o={id:nextId++,x:state.room.w/2-w/2,y:state.room.h/2-h/2,w:w,h:h,name:'固定结构'};state.obstacles.push(o);state.selectedObsId=o.id;state.selectedFurnitureId=null;state.selectedPointId=null;edited();}
  function addFurniture(kind){var d=FURNITURE[kind]||FURNITURE.table,n=state.furniture.filter(function(f){return f.kind===kind;}).length+1,f={id:nextId++,kind:kind,name:d.label+n,x:state.room.w/2,y:state.room.h/2,w:d.w,h:d.h,shape:d.shape};f.x=clamp(f.x,f.w/2,state.room.w-f.w/2);f.y=clamp(f.y,f.h/2,state.room.h-f.h/2);state.furniture.push(f);state.selectedFurnitureId=f.id;state.selectedObsId=null;state.selectedPointId=null;edited();}
  function removeById(listName,id){state[listName]=state[listName].filter(function(v){return v.id!==id;});if(listName==='points'&&state.selectedPointId===id)state.selectedPointId=null;if(listName==='obstacles'&&state.selectedObsId===id)state.selectedObsId=null;if(listName==='furniture'&&state.selectedFurnitureId===id)state.selectedFurnitureId=null;edited();}
  function addPoint(x,y,rssi){var p={id:nextId++,x:x,y:y,rssi:rssi};state.points.push(p);state.selectedPointId=p.id;state.selectedObsId=null;state.selectedFurnitureId=null;edited();}

  /* ---------------- 示例、导入与导出 ---------------- */
  function normalizeFurniture(f){var d=FURNITURE[f.kind]||FURNITURE.table;return{id:nextId++,kind:FURNITURE[f.kind]?f.kind:'table',name:f.name||d.label,x:Number(f.x),y:Number(f.y),w:Number(f.w)||d.w,h:Number(f.h)||d.h,shape:f.shape==='rect'?'rect':(f.shape==='circle'?'circle':d.shape)};}
  function applyProject(data){if(!data||!data.room)throw new Error('JSON 缺少 room');state.room={w:clamp(Number(data.room.w)||DEFAULT_ROOM.w,2,12.8),h:clamp(Number(data.room.h)||DEFAULT_ROOM.h,2,12.8)};state.door={w:Number(data.room.door&&data.room.door.w)||DEFAULT_DOOR.w};state.obstacles=(data.room.obstacles||data.obstacles||[]).map(function(o){return{id:nextId++,name:o.name||'固定结构',x:Number(o.x),y:Number(o.y),w:Number(o.w),h:Number(o.h)};}).filter(function(o){return finite(o.x)&&finite(o.y)&&o.w>0&&o.h>0;});state.furniture=(data.furniture||[]).map(normalizeFurniture).filter(function(f){return finite(f.x)&&finite(f.y)&&f.w>0&&f.h>0;});state.points=(data.points||[]).map(function(p){return{id:nextId++,x:Number(p.x),y:Number(p.y),rssi:Number(p.rssi)};}).filter(function(p){return finite(p.x)&&finite(p.y)&&finite(p.rssi)&&p.rssi>=-100&&p.rssi<=0;});state.heat=null;state.comparisonHeat=null;state.inferMs=null;state.selectedPointId=null;state.selectedObsId=null;state.selectedFurnitureId=null;setRoomSize(state.room.w,state.room.h);}
  function loadSample(){var tag=document.getElementById('sampleRoomData');try{applyProject(JSON.parse(tag.textContent));}catch(e){window.alert('示例载入失败：'+e.message);}}
  function download(name,href){var a=document.createElement('a');a.href=href;a.download=name;document.body.appendChild(a);a.click();a.remove();}
  function downloadText(name,text,type){var blob=new Blob([text],{type:type||'text/plain;charset=utf-8'}),url=URL.createObjectURL(blob);download(name,url);setTimeout(function(){URL.revokeObjectURL(url);},1000);}
  function exportPng(){if(!state.heat){window.alert('请先生成预测。');return;}render();download('rfmap_furniture_prediction.png',canvas.toDataURL('image/png'));}
  function exportPoints(){if(!state.points.length){window.alert('当前没有测点。');return;}var lines=['x_m,y_m,rssi_dbm'];state.points.forEach(function(p){lines.push(p.x.toFixed(4)+','+p.y.toFixed(4)+','+p.rssi);});downloadText('rfmap_measured_points.csv','\ufeff'+lines.join('\n')+'\n','text/csv;charset=utf-8');}
  function exportGrid(){if(!state.heat){window.alert('请先生成预测。');return;}var input=buildInput(),g=state.grid,lines=['x_m,y_m,rssi_pred_dbm,inside_room,furniture,no_furniture_rssi_dbm'];for(var j=0;j<g.ny;j++)for(var i=0;i<g.nx;i++){var k=j*g.nx+i;if(input.roomMask[k]>0.5){var x=g.x0+(i+0.5)*g.cell,y=g.y0+(j+0.5)*g.cell;lines.push(x.toFixed(3)+','+y.toFixed(3)+','+state.heat[k].toFixed(4)+',1,'+input.furnitureMask[k]+','+(state.comparisonHeat?state.comparisonHeat[k].toFixed(4):''));}}downloadText('rfmap_prediction_grid.csv','\ufeff'+lines.join('\n')+'\n','text/csv;charset=utf-8');}
  function exportProject(){var data={schema:'rfmap-studio-furniture-v1',coordinate:'origin=southwest;x=east;y=north;unit=m',model:(window.RFM_MODEL&&window.RFM_MODEL.label)||'unknown',room:{w:state.room.w,h:state.room.h,door:{w:state.door.w},obstacles:state.obstacles.map(function(o){return{x:o.x,y:o.y,w:o.w,h:o.h,name:o.name};})},furniture:state.furniture.map(function(f){return{kind:f.kind,name:f.name,x:f.x,y:f.y,w:f.w,h:f.h,shape:f.shape};}),points:state.points.map(function(p){return{x:p.x,y:p.y,rssi:p.rssi};}),grid:state.grid};downloadText('rfmap_furniture_project.json',JSON.stringify(data,null,2),'application/json;charset=utf-8');}
  function readFile(input,handler){var file=input.files&&input.files[0];if(!file)return;var reader=new FileReader();reader.onload=function(){handler(String(reader.result));input.value='';};reader.onerror=function(){window.alert('文件读取失败');input.value='';};reader.readAsText(file,'utf-8');}
  function importPoints(text){try{var lines=text.replace(/^\ufeff/,'').split(/\r?\n/).filter(function(s){return s.trim();});if(lines.length<2)throw new Error('CSV 至少需要表头和一行数据');var head=lines[0].split(',').map(function(s){return s.trim().toLowerCase();});function col(names){for(var i=0;i<names.length;i++){var k=head.indexOf(names[i]);if(k>=0)return k;}return-1;}var ix=col(['x_m','x_room','x']),iy=col(['y_m','y_room','y']),ir=col(['rssi_dbm','rssi']);if(ix<0||iy<0||ir<0)throw new Error('表头需包含 x_m,y_m,rssi_dbm（也支持 x,y,rssi）');var pts=[];for(var n=1;n<lines.length;n++){var c=lines[n].split(','),p={x:Number(c[ix]),y:Number(c[iy]),rssi:Number(c[ir])};if(finite(p.x)&&finite(p.y)&&finite(p.rssi)&&p.rssi>=-100&&p.rssi<=0&&insideRoom(p.x,p.y)&&!insideObstacle(p.x,p.y))pts.push(p);}if(!pts.length)throw new Error('没有有效的室内测点');state.points=pts.map(function(p){p.id=nextId++;return p;});state.heat=null;state.comparisonHeat=null;refresh();}catch(e){window.alert('测点导入失败：'+e.message);}}

  /* ---------------- 指针交互 ---------------- */
  function localPos(ev){var r=canvas.getBoundingClientRect();return{x:ev.clientX-r.left,y:ev.clientY-r.top};}
  function hitPoint(x,y){for(var i=state.points.length-1;i>=0;i--){var p=state.points[i],dx=sx(p.x)-x,dy=sy(p.y)-y;if(dx*dx+dy*dy<=100)return p;}return null;}
  function hitFurniture(x,y){for(var i=state.furniture.length-1;i>=0;i--){var f=state.furniture[i];if(insideFurniture(rxOf(x),ryOf(y),f))return f;}return null;}
  function hitObstacle(x,y){var rx=rxOf(x),ry=ryOf(y);return insideObstacle(rx,ry);}
  function hitRoomHandle(x,y){var dx=sx(state.room.w)-x,dy=sy(state.room.h)-y;return dx*dx+dy*dy<=100;}
  canvas.addEventListener('pointerdown',function(ev){var m=localPos(ev);if(hitRoomHandle(m.x,m.y)){state.drag={kind:'room'};canvas.setPointerCapture(ev.pointerId);return;}var p=hitPoint(m.x,m.y);if(p){if(ev.button===2){removeById('points',p.id);return;}state.selectedPointId=p.id;state.selectedObsId=null;state.selectedFurnitureId=null;state.drag={kind:'point',id:p.id,dx:rxOf(m.x)-p.x,dy:ryOf(m.y)-p.y};canvas.setPointerCapture(ev.pointerId);refresh();return;}var f=hitFurniture(m.x,m.y);if(f){if(ev.button===2){removeById('furniture',f.id);return;}state.selectedFurnitureId=f.id;state.selectedObsId=null;state.selectedPointId=null;state.drag={kind:'furniture',id:f.id,dx:rxOf(m.x)-f.x,dy:ryOf(m.y)-f.y};canvas.setPointerCapture(ev.pointerId);refresh();return;}var o=hitObstacle(m.x,m.y);if(o){if(ev.button===2){removeById('obstacles',o.id);return;}state.selectedObsId=o.id;state.selectedFurnitureId=null;state.selectedPointId=null;state.drag={kind:'obstacle',id:o.id,dx:rxOf(m.x)-o.x,dy:ryOf(m.y)-o.y};canvas.setPointerCapture(ev.pointerId);refresh();return;}if(ev.button===0){var rx=rxOf(m.x),ry=ryOf(m.y);if(insideRoom(rx,ry)&&!insideObstacle(rx,ry)){pendingPoint={x:rx,y:ry};el.rssiModalPos.textContent='坐标：('+rx.toFixed(2)+', '+ry.toFixed(2)+') m';el.rssiErr.classList.add('hidden');el.rssiModal.classList.remove('hidden');el.rssiInput.focus();el.rssiInput.select();}}});
  canvas.addEventListener('pointermove',function(ev){var m=localPos(ev);if(state.drag){var rx=rxOf(m.x),ry=ryOf(m.y),d=state.drag;if(d.kind==='room')setRoomSize(rx,ry);else if(d.kind==='point'){var p=selected(state.points,d.id),nx=clamp(rx-d.dx,0,state.room.w),ny=clamp(ry-d.dy,0,state.room.h);if(p&&!insideObstacle(nx,ny)){p.x=nx;p.y=ny;renderPointList();render();}}else if(d.kind==='furniture'){var f=selected(state.furniture,d.id);if(f){f.x=clamp(rx-d.dx,f.w/2,state.room.w-f.w/2);f.y=clamp(ry-d.dy,f.h/2,state.room.h-f.h/2);renderFurnitureList();syncEditors();render();}}else if(d.kind==='obstacle'){var o=selected(state.obstacles,d.id);if(o){o.x=clamp(rx-d.dx,0,Math.max(0,state.room.w-o.w));o.y=clamp(ry-d.dy,0,Math.max(0,state.room.h-o.h));renderObsList();syncEditors();render();}}return;}var x=rxOf(m.x),y=ryOf(m.y);if(state.heat&&insideRoom(x,y)&&!insideObstacle(x,y)){var g=state.grid,i=Math.floor((x-g.x0)/g.cell),j=Math.floor((y-g.y0)/g.cell);if(i>=0&&i<g.nx&&j>=0&&j<g.ny){var k=j*g.nx+i,v=state.heat[k];if(v>-119){var msg='x='+x.toFixed(2)+' m，y='+y.toFixed(2)+' m\n预测 '+v.toFixed(1)+' dBm · '+LEVEL_NAMES[levelIndex(v)];if(state.comparisonHeat)msg+='\n撤掉家具 '+state.comparisonHeat[k].toFixed(1)+' dBm（差 '+(v-state.comparisonHeat[k]>=0?'+':'')+(v-state.comparisonHeat[k]).toFixed(1)+' dB）';tooltip.textContent=msg;tooltip.classList.remove('hidden');tooltip.style.left=(m.x+16)+'px';tooltip.style.top=(m.y+12)+'px';return;}}}tooltip.classList.add('hidden');});
  function endDrag(ev){if(!state.drag)return;state.drag=null;if(ev&&ev.pointerId!==undefined&&canvas.hasPointerCapture&&canvas.hasPointerCapture(ev.pointerId))canvas.releasePointerCapture(ev.pointerId);queueInfer();}
  canvas.addEventListener('pointerup',endDrag);canvas.addEventListener('pointercancel',endDrag);canvas.addEventListener('pointerleave',function(){tooltip.classList.add('hidden');});canvas.addEventListener('contextmenu',function(ev){ev.preventDefault();});

  /* ---------------- 控件绑定 ---------------- */
  el.roomWNum.addEventListener('input',function(){setRoomSize(parseFloat(el.roomWNum.value),state.room.h);queueInfer();});el.roomWRange.addEventListener('input',function(){setRoomSize(parseFloat(el.roomWRange.value),state.room.h);queueInfer();});el.roomHNum.addEventListener('input',function(){setRoomSize(state.room.w,parseFloat(el.roomHNum.value));queueInfer();});el.roomHRange.addEventListener('input',function(){setRoomSize(state.room.w,parseFloat(el.roomHRange.value));queueInfer();});el.doorWNum.addEventListener('input',function(){var v=parseFloat(el.doorWNum.value);if(finite(v)){state.door.w=clamp(v,0.3,state.room.w);render();}});
  document.getElementById('btnResetRoom').addEventListener('click',function(){state.room={w:DEFAULT_ROOM.w,h:DEFAULT_ROOM.h};state.door={w:DEFAULT_DOOR.w};state.obstacles=DEFAULT_OBSTACLES.map(function(o){o.id=nextId++;return Object.assign({},o);});state.selectedObsId=null;setRoomSize(state.room.w,state.room.h);queueInfer();});
  document.getElementById('btnAddObs').addEventListener('click',addObstacle);document.getElementById('btnAddFurniture').addEventListener('click',function(){addFurniture(el.furnitureType.value);});document.getElementById('btnClearFurniture').addEventListener('click',function(){state.furniture=[];state.selectedFurnitureId=null;edited();});document.getElementById('btnLoadSample').addEventListener('click',loadSample);document.getElementById('btnClearPoints').addEventListener('click',function(){state.points=[];state.heat=null;state.comparisonHeat=null;state.inferMs=null;state.selectedPointId=null;refresh();});document.getElementById('btnInfer').addEventListener('click',runInfer);document.getElementById('btnCompareFurniture').addEventListener('click',compareFurniture);document.getElementById('btnClearHeat').addEventListener('click',function(){state.heat=null;state.comparisonHeat=null;state.inferMs=null;el.comparisonResult.textContent='对照会保持户型和测点不变，只把家具输入通道置零；结果是模型推演，不是家具因果实测。';refresh();});document.getElementById('btnExportPng').addEventListener('click',exportPng);document.getElementById('btnExportCsv').addEventListener('click',exportPoints);document.getElementById('btnExportGrid').addEventListener('click',exportGrid);document.getElementById('btnExportProject').addEventListener('click',exportProject);document.getElementById('btnImportProject').addEventListener('click',function(){el.projectInput.click();});document.getElementById('btnImportPoints').addEventListener('click',function(){el.pointsCsvInput.click();});el.projectInput.addEventListener('change',function(){readFile(el.projectInput,function(text){try{applyProject(JSON.parse(text));}catch(e){window.alert('工程导入失败：'+e.message);}});});el.pointsCsvInput.addEventListener('change',function(){readFile(el.pointsCsvInput,importPoints);});el.displayMode.addEventListener('change',function(){state.displayMode=el.displayMode.value;updateInfo();render();});
  function bindList(listEl,listName,selectionKey){listEl.addEventListener('click',function(ev){var del=ev.target.dataset&&ev.target.dataset.del;if(del){removeById(listName,parseInt(del,10));return;}var li=ev.target.closest('li');if(li&&li.dataset.id){state[selectionKey]=parseInt(li.dataset.id,10);if(selectionKey!=='selectedPointId')state.selectedPointId=null;if(selectionKey!=='selectedObsId')state.selectedObsId=null;if(selectionKey!=='selectedFurnitureId')state.selectedFurnitureId=null;refresh();}});}
  bindList(el.pointList,'points','selectedPointId');bindList(el.obsList,'obstacles','selectedObsId');bindList(el.furnitureList,'furniture','selectedFurnitureId');
  function bindObs(input,key){input.addEventListener('input',function(){var o=selected(state.obstacles,state.selectedObsId),v=parseFloat(input.value);if(!o||!finite(v))return;o[key]=key==='w'||key==='h'?Math.max(0.1,v):Math.max(0,v);o.x=clamp(o.x,0,Math.max(0,state.room.w-o.w));o.y=clamp(o.y,0,Math.max(0,state.room.h-o.h));edited();});}bindObs(el.obsX,'x');bindObs(el.obsY,'y');bindObs(el.obsW,'w');bindObs(el.obsH,'h');
  function bindFurniture(input,key){input.addEventListener('input',function(){var f=selected(state.furniture,state.selectedFurnitureId),v=parseFloat(input.value);if(!f||!finite(v))return;f[key]=key==='w'||key==='h'?clamp(v,0.2,5):Math.max(0,v);f.x=clamp(f.x,f.w/2,state.room.w-f.w/2);f.y=clamp(f.y,f.h/2,state.room.h-f.h/2);edited();});}bindFurniture(el.furnitureX,'x');bindFurniture(el.furnitureY,'y');bindFurniture(el.furnitureW,'w');bindFurniture(el.furnitureH,'h');
  function closeModal(){pendingPoint=null;el.rssiModal.classList.add('hidden');}function confirmModal(){if(!pendingPoint)return;var v=parseFloat(el.rssiInput.value);if(!finite(v)||v<-100||v>0){el.rssiErr.classList.remove('hidden');return;}var p=pendingPoint;closeModal();addPoint(p.x,p.y,v);}document.getElementById('rssiOk').addEventListener('click',confirmModal);document.getElementById('rssiCancel').addEventListener('click',closeModal);el.rssiInput.addEventListener('keydown',function(ev){if(ev.key==='Enter')confirmModal();if(ev.key==='Escape')closeModal();});el.rssiModal.addEventListener('click',function(ev){if(ev.target===el.rssiModal)closeModal();});
  document.addEventListener('keydown',function(ev){if(ev.key!=='Delete'&&ev.key!=='Backspace')return;if(ev.target&&(ev.target.tagName==='INPUT'||ev.target.tagName==='SELECT'))return;if(state.selectedPointId!==null)removeById('points',state.selectedPointId);else if(state.selectedFurnitureId!==null)removeById('furniture',state.selectedFurnitureId);else if(state.selectedObsId!==null)removeById('obstacles',state.selectedObsId);});window.addEventListener('resize',render);

  /* ---------------- 启动与测试接口 ---------------- */
  state.obstacles=DEFAULT_OBSTACLES.map(function(o){return Object.assign({id:nextId++},o);});
  if(window.RFM_MODEL){var m=window.RFM_MODEL,meta=m.meta||{};el.modelInfo.textContent=m.label+'：'+(meta.n_input_channels||3)+' 通道、'+m.n_params.toLocaleString()+' 参数、64×64 网格；权重来自 '+(meta.train_size||'?')+' 张家具仿真训练图。当前为跨房间预训练权重，未在网页内现场训练。';}
  else el.modelInfo.textContent='未检测到神经网络权重，将只能使用 IDW 回退。';
  setRoomSize(state.room.w,state.room.h);refresh();
  window.RFM_Studio={getState:function(){return state;},buildInput:buildInput,inferGrid:inferGrid,loadSample:loadSample,applyProject:applyProject};
})();
