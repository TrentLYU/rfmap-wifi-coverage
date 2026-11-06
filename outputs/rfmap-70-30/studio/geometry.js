/* Geometry-only nine-channel input. No measured RSSI is accepted or consumed here.
 * Coordinates: southwest origin; furniture center, wall lower-left corner.
 * Furniture loss values are modeling assumptions, not measured material constants.
 */
(function (root) {
  'use strict';
  var LOSS = { table:1.5, chair:0.6, cabinet:4, bed:2, sofa:2.5, toilet:3,
    fridge:6, wardrobe:4, bookshelf:3, desk:1.5, washer:2, bathtub:2, tv:2 };
  var EPS = 1e-7;
  function insideFurniture(x,y,f) {
    var dx=(x-f.x)/(f.w/2),dy=(y-f.y)/(f.h/2);
    return f.shape==='circle' ? dx*dx+dy*dy<=1 : Math.abs(dx)<=1 && Math.abs(dy)<=1;
  }
  function insideWall(x,y,o) { return x>o.x && x<o.x+o.w && y>o.y && y<o.y+o.h; }
  function rayRect(ax,ay,bx,by,r) {
    var t0=EPS,t1=1-EPS,dx=bx-ax,dy=by-ay;
    var axes=[[ax,dx,r.x,r.x+r.w],[ay,dy,r.y,r.y+r.h]];
    for(var i=0;i<2;i++) {
      var a=axes[i];
      if(Math.abs(a[1])<1e-12) { if(a[0]<a[2]||a[0]>a[3]) return false; }
      else { var tA=(a[2]-a[0])/a[1],tB=(a[3]-a[0])/a[1];
        t0=Math.max(t0,Math.min(tA,tB));t1=Math.min(t1,Math.max(tA,tB));if(t0>t1)return false; }
    }
    return t1>t0 && t1>EPS && t0<1-EPS;
  }
  function rayEllipse(ax,ay,bx,by,f) {
    var ox=(ax-f.x)/(f.w/2),oy=(ay-f.y)/(f.h/2);
    var dx=(bx-ax)/(f.w/2),dy=(by-ay)/(f.h/2),a=dx*dx+dy*dy;
    if(a<1e-16)return false;
    var b=2*(ox*dx+oy*dy),c=ox*ox+oy*oy-1,d=b*b-4*a*c;
    if(d<0)return false;
    var s=Math.sqrt(d),lo=(-b-s)/(2*a),hi=(-b+s)/(2*a);
    return hi>EPS && lo<1-EPS;
  }
  function rayFurniture(ax,ay,bx,by,f) {
    return f.shape==='circle' ? rayEllipse(ax,ay,bx,by,f) : rayRect(ax,ay,bx,by,{x:f.x-f.w/2,y:f.y-f.h/2,w:f.w,h:f.h});
  }
  function buildInput(project) {
    var nx=32,ny=32,n=nx*ny,channels=new Float32Array(n*9),roomMask=new Float32Array(n);
    var room=project.room,ap=project.ap,walls=project.walls||[],furniture=project.furniture||[];
    if(!room||!(room.w>0)||!(room.h>0)||!ap||!isFinite(ap.x)||!isFinite(ap.y))throw new Error('请输入有效的户型尺寸和路由器坐标');
    var den=Math.max(room.w,room.h);
    for(var j=0;j<ny;j++)for(var i=0;i<nx;i++) {
      // Match numpy.linspace's operation order and exact final endpoints. Boundary
      // rounding affects strict wall masks and tangent-ray occupancy decisions.
      var k=j*nx+i,x=i===nx-1?room.w:i*(room.w/(nx-1)),y=j===ny-1?room.h:j*(room.h/(ny-1)),dx=x-ap.x,dy=y-ap.y;
      var occupied=0,cross=0,loss=0,mask=1;
      walls.forEach(function(o){if(insideWall(x,y,o))mask=0;if(rayRect(ap.x,ap.y,x,y,o))cross++;});
      furniture.forEach(function(f){if(insideFurniture(x,y,f))occupied=1;if(rayFurniture(ap.x,ap.y,x,y,f))loss+=LOSS[f.kind]===undefined?2:LOSS[f.kind];});
      roomMask[k]=mask;
      channels[2*n+k]=mask;channels[3*n+k]=occupied;channels[4*n+k]=dx/den;channels[5*n+k]=dy/den;
      channels[6*n+k]=Math.log10(Math.max(Math.hypot(dx,dy),.5))/2;
      channels[7*n+k]=cross/3;channels[8*n+k]=loss/10;
    }
    return{channels:channels,roomMask:roomMask,nx:nx,ny:ny,w:room.w,h:room.h};
  }
  root.RFM_Geometry={buildInput:buildInput,loss:LOSS,insideWall:insideWall,insideFurniture:insideFurniture,rayRect:rayRect,rayFurniture:rayFurniture};
})(typeof window!=='undefined'?window:globalThis);
