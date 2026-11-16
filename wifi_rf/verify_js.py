"""在 quickjs 里跑构建好的单文件应用，把 JS 端数值导出来与 Python 对比。

用法: python -m wifi_rf.verify_js
输出: .deepworks/tmp/js_out.json
"""
from __future__ import annotations

import json
import os
import re
import time

import quickjs

from . import config as C

HTML = os.path.join(C.OUT_DIR, "wifi-coverage-studio.html")
OUT = os.path.join(C.ROOT, ".deepworks", "tmp", "js_out.json")

PRELUDE = r"""
const _els = {};
function mkCtx(){
  const store = {
    canvas: {width:480, height:480},
    measureText: () => ({width: 10}),
    createImageData: (w,h) => ({data: new Uint8ClampedArray(w*h*4), width:w, height:h}),
    getImageData: (x,y,w,h) => ({data: new Uint8ClampedArray(w*h*4), width:w, height:h}),
    createLinearGradient: () => ({addColorStop(){}}),
    createRadialGradient: () => ({addColorStop(){}}),
  };
  return new Proxy(store, {
    get(t,k){ if (k in t) return t[k]; return () => {}; },
    set(t,k,v){ t[k]=v; return true; }
  });
}
const __PARENT = {clientWidth: 820, clientHeight: 820, style: {}, children: [],
  appendChild(){}, getBoundingClientRect(){ return {left:0, top:0, width:820, height:820}; }};
function mkEl(id){
  const t = {
    parentElement: __PARENT,
    id: id, nodeName: id, style: {}, dataset: {}, children: [], value: '',
    textContent: '', innerHTML: '', width: 480, height: 480,
    clientWidth: 800, clientHeight: 800, offsetWidth: 800, offsetHeight: 800,
    checked: true, files: [],
    classList: {add(){}, remove(){}, toggle(){}, contains(){ return false; }},
    appendChild(c){ this.children.push(c); return c; },
    removeChild(){}, addEventListener(){}, removeEventListener(){},
    setAttribute(){}, getAttribute(){ return null; },
    getContext(){ return mkCtx(); },
    getBoundingClientRect(){ return {left:0, top:0, right:800, bottom:800, width:800, height:800}; },
    querySelectorAll(){ return []; }, querySelector(){ return mkEl('q'); },
    closest(){ return null; }, focus(){}, blur(){}, click(){},
  };
  return t;
}
const document = {
  getElementById(id){ if (!_els[id]) _els[id] = mkEl(id); return _els[id]; },
  createElement(tag){ return mkEl(tag); },
  querySelectorAll(){ return []; },
  querySelector(){ return mkEl('q'); },
  addEventListener(){}, removeEventListener(){},
  body: mkEl('body'), documentElement: mkEl('html'),
};
const window = globalThis;
window.addEventListener = () => {};
window.removeEventListener = () => {};
const performance = {now: () => Date.now()};
const requestAnimationFrame = (f) => { try { f(); } catch (e) { __ERR.push(String(e)); } };
const __ERR = [];
const setTimeout = (f) => { try { f(); } catch (e) { __ERR.push(String(e)); } };
const navigator = {userAgent: 'quickjs', clipboard: {writeText(){ return 1; }}};
function URL(){}
URL.createObjectURL = () => 'blob:0';
function Blob(){}
function atob(input){
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=';
  const str = String(input).replace(/=+$/, '');
  let output = '';
  let bc = 0, bs = 0, buffer = '', idx = 0;
  while ((buffer = str.charAt(idx++))) {
    buffer = chars.indexOf(buffer);
    if (~buffer) {
      bs = bc % 4 ? bs * 64 + buffer : buffer;
      if (bc++ % 4) output += String.fromCharCode(255 & (bs >> ((-2 * bc) & 6)));
    }
  }
  return output;
}
document.getElementById('dataset').textContent = __DATASET_TEXT;
"""

TAIL = r"""
(function(){
  const W = window.__WIFI;
  if (!W) return JSON.stringify({error: 'no __WIFI', errs: __ERR});
  const r4 = a => Array.from(a, v => Math.round(v * 1e4) / 1e4);
  const out = {n: W.N, cell: W.CELL, domain: W.DOMAIN, errs: __ERR,
               wkeys: Object.keys(W.WEIGHTS).sort(),
               wlen: Object.keys(W.WEIGHTS).reduce((o,k)=>{o[k]=W.WEIGHTS[k].length;return o;},{}),
               maskOn: Array.from(W.MASK).reduce((a,b)=>a+b,0)};
  W.state.model = 'deploy';  W.computeField(); out.deploy  = r4(W.state.field);
  out.physDeploy = r4(W.state.phys);
  W.state.model = 'physics'; W.computeField(); out.physics = r4(W.state.field);
  const t0 = Date.now();
  W.state.model = 'net';     W.computeField(); out.net = r4(W.state.field);
  out.netMs = Date.now() - t0;
  W.state.model = 'deploy';  W.computeField();
  out.thr = W.thresholds().map(v=>Math.round(v*1e3)/1e3);
  const cal = W.physField({x: AP_CAL[0], y: AP_CAL[1]}, [], PL0, NEXP);
  const ch = W.buildChannels(W.state.pts.filter(p => !p.test), cal);
  out.ch = [];
  for (let c=0; c<7; c++) out.ch.push(Array.from(ch.subarray(c*N*N, (c+1)*N*N), v=>Math.round(v*1e4)/1e4));
  out.sample = W.D.points.slice(0, 6).map(p => [p.x, p.y,
      Math.round(W.sampleBilinear(W.state.field, p.x, p.y) * 1e4) / 1e4]);
  // 贴墙/边界点：验证房外 -120 哨兵值不会污染读数
  out.sampleEdge = [[W.D.ap[0], W.D.ap[1]], [10.9, 12], [6.9, 12], [8.9, 12]]
      .map(([x, y]) => [x, y, Math.round(W.sampleBilinear(W.state.field, x, y) * 1e4) / 1e4]);
  // 无观测点
  W.state.usePoints = false; W.computeField(); out.deployNoPts = r4(W.state.field);
  W.state.usePoints = true;  W.computeField();
  // 像素级校验：确认"屏幕颜色"与"该位置场值"分级一致（防上下颠倒）
  let frame = null;
  globalThis.__ON_FRAME = (img, w, h) => { frame = {img: img, w: w, h: h}; };
  W.state.view = 'level'; W.state.scheme = 'rel';
  W.drawFieldLayer();
  out.pixels = [];
  if (frame){
    const probes = [[9, 11.5], [9, 10.4], [6, 10.7], [2, 2], [11.5, 11.5], [6.5, 1.0], [9, 0.5]];
    for (const pr of probes){
      const x = pr[0], y = pr[1];
      const px = Math.min(frame.w-1, Math.max(0, Math.floor(x / W.DOMAIN * frame.w)));
      const py = Math.min(frame.h-1, Math.max(0, Math.floor((1 - y / W.DOMAIN) * frame.h)));
      const i = (py*frame.w + px)*4;
      const cx = (px + 0.5) * W.DOMAIN / frame.w;
      const cy = (1 - (py + 0.5) / frame.h) * W.DOMAIN;
      const cv = W.sampleBilinear(W.state.field, cx, cy);
      out.pixels.push([x, y, Math.round(W.sampleBilinear(W.state.field, x, y)*100)/100,
                       [frame.img.data[i], frame.img.data[i+1], frame.img.data[i+2]],
                       frame.img.data[i+3],
                       Math.round(cv*1000)/1000,
                       W.levelOf(cv, W.thresholds())]);
    }
  }
  out.errs = __ERR;
  return JSON.stringify(out);
})()
"""


def main() -> None:
    with open(HTML, encoding="utf-8") as f:
        html = f.read()
    m = re.search(r'<script id="dataset" type="application/json">(.*?)</script>', html, re.S)
    payload = m.group(1)
    body = html[m.end():]
    body = body[body.index("<script>") + len("<script>"):]
    body = body[:body.rindex("</script>")]

    ctx = quickjs.Context()
    ctx.eval("var __DATASET_TEXT = " + json.dumps(payload) + ";")
    ctx.eval(PRELUDE)
    t0 = time.time()
    ctx.eval(body)
    print("script eval", round(time.time() - t0, 1), "s")
    res = ctx.eval(TAIL)
    data = json.loads(res)
    if "error" in data:
        print(data)
        raise SystemExit(1)
    print("net forward in js:", data.get("netMs"), "ms")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(data, f)
    print("->", OUT, "errs:", data["errs"][:3])


if __name__ == "__main__":
    main()
