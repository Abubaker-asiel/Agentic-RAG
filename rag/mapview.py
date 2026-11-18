"""The Terrain: a live map of the knowledge base where the agent's route is drawn as it thinks."""
import json

from .store import KnowledgeBase

HEIGHT = 440


def build_map_html(kb: KnowledgeBase, state: dict | None, height: int = HEIGHT) -> str:
    state = state or {}
    lay = kb.layout()
    data = {
        "points": lay["points"],
        "sources": lay["sources"],
        "snips": [c.text[:150].replace("\n", " ") for c in kb.chunks],
        "pages": [c.page for c in kb.chunks],
        "question": state.get("question", ""),
        "steps": state.get("steps", []),
        "ghost": state.get("ghost"),
        "animate_from": state.get("animate_from", 0),
        "animate_ghost": state.get("animate_ghost", False),
        "aspect": kb.ASPECT,
        "weak": 0.08 if kb.embedder.name == "hashing-fallback" else 0.45,
    }
    blob = json.dumps(data).replace("</", "<\\/")
    return TEMPLATE.replace("__DATA__", blob).replace("__H__", str(height))


TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&display=swap');
html,body{margin:0;background:transparent;overflow:hidden;font-family:'IBM Plex Mono',ui-monospace,Menlo,monospace}
#wrap{position:relative;width:100%;height:__H__px;border-radius:16px;overflow:hidden;background:#0B1824;
  box-shadow:inset 0 0 0 1px #223648,0 12px 30px -18px #0B1824}
canvas{position:absolute;inset:0;width:100%;height:100%}
.hud{position:absolute;color:#6f8a9e;font-size:10px;letter-spacing:.16em;text-transform:uppercase;pointer-events:none}
#tl{left:18px;top:14px}
#tr{right:18px;top:14px;text-align:right}
#bl{left:18px;bottom:14px;max-width:calc(100% - 130px);color:#D6E2EC;font-size:12px;letter-spacing:.01em;text-transform:none;line-height:1.5}
#bl i{font-style:normal;color:#FFC857}
#tip{position:absolute;pointer-events:none;max-width:300px;background:#0B1824f2;border:1px solid #2b4358;color:#D6E2EC;
  padding:8px 10px;border-radius:8px;font-size:11px;line-height:1.45;opacity:0;transition:opacity .12s;z-index:5}
#tip b{display:block;font-size:10px;letter-spacing:.12em;text-transform:uppercase;margin-bottom:3px}
</style></head><body>
<div id="wrap"><canvas id="c"></canvas>
<div class="hud" id="tl"></div><div class="hud" id="tr"></div><div class="hud" id="bl"></div><div id="tip"><b></b><span></span></div></div>
<script>
const D = __DATA__;
const PAL = ["#4FD1C5","#F6AD55","#B794F4","#68D391","#F687B3","#63B3ED","#ECC94B","#FC8181"];
const AMBER="#FFC857", CORAL="#FF6B5E", GHOST="#8FA8BC", WEAK=D.weak;
const wrap=document.getElementById('wrap'), cv=document.getElementById('c'), ctx=cv.getContext('2d');
const bg=document.createElement('canvas'), bctx=bg.getContext('2d');
const tip=document.getElementById('tip');
let W=0,H=0,DPR=1;
const PX=40, PY=50;
const X=u=>PX+u*(W-2*PX), Y=v=>PY+v*(H-2*PY);
const clamp=(v,a=0,b=1)=>Math.min(b,Math.max(a,v));
const ease=t=>1-Math.pow(1-clamp(t),3);
const col=i=>PAL[i%PAL.length];
const hexA=(h,a)=>{const n=parseInt(h.slice(1),16);return `rgba(${n>>16},${(n>>8)&255},${n&255},${a})`};
const N=D.points.length, S=D.sources, steps=D.steps;

document.getElementById('tl').textContent=`Terrain · ${N} passages · ${S.length} documents`;
document.getElementById('tr').textContent=`embedding map · PCA`;

// ---------- timeline ----------
const DUR={plan:900,search:2400,tool:1700,calc:1300,answer:2000};
const t0=performance.now()+250;
const starts=[], durs=[]; let acc=0;
steps.forEach((s,i)=>{const d=DUR[s.kind]||1300; durs.push(d);
  if(i<D.animate_from) starts.push(-1e9); else {starts.push(acc); acc+=d;}});
const ghostStart = D.animate_ghost ? acc+200 : -1e9, ghostDur=1800;
const P=(i,el)=>clamp((el-starts[i])/durs[i]);

// ---------- geometry helpers ----------
const srcByName={}; S.forEach((s,i)=>srcByName[s.name]=Object.assign({i},s));
function wpOf(s,prev){
  if(s.kind==='search') return s.xy;
  if(s.kind==='tool'){const r=srcByName[s.source]; return r?[r.cx,r.cy]:prev;}
  if(s.kind==='calc') return prev?[clamp(prev[0]+0.035,0.03,0.97),clamp(prev[1]-0.09,0.05,0.95)]:[0.5,0.5];
  if(s.kind==='answer'){
    const ids=s.cited||[]; if(!ids.length) return prev;
    let x=0,y=0; ids.forEach(id=>{x+=D.points[id][0];y+=D.points[id][1]}); return [x/ids.length,y/ids.length];}
  return null;
}
const QPOS=[0.045,0.13];
const route=[]; // {step, from, to}
(function(){let prev=QPOS;
  steps.forEach((s,i)=>{const w=wpOf(s,prev); if(!w) return; s._wp=w; route.push({i,from:prev,to:w}); prev=w;});})();

function bez(a,b,u){ // quadratic curve with a perpendicular bow
  const ax=X(a[0]),ay=Y(a[1]),bx=X(b[0]),by=Y(b[1]);
  const mx=(ax+bx)/2,my=(ay+by)/2,dx=bx-ax,dy=by-ay,len=Math.hypot(dx,dy)||1;
  const cx=mx-dy/len*len*0.16,cy=my+dx/len*len*0.16;
  return [(1-u)*(1-u)*ax+2*(1-u)*u*cx+u*u*bx,(1-u)*(1-u)*ay+2*(1-u)*u*cy+u*u*by];
}

// ---------- background (cached) ----------
function paintBg(){
  bg.width=W*DPR; bg.height=H*DPR; bctx.setTransform(DPR,0,0,DPR,0,0);
  const g=bctx.createRadialGradient(W*.5,H*.45,20,W*.5,H*.5,Math.max(W,H)*.75);
  g.addColorStop(0,'#10263A'); g.addColorStop(1,'#08121B'); bctx.fillStyle=g; bctx.fillRect(0,0,W,H);
  bctx.strokeStyle='rgba(120,160,190,.055)'; bctx.lineWidth=1; bctx.beginPath();
  for(let x=PX;x<=W-PX+1;x+=(W-2*PX)/20){bctx.moveTo(x,0);bctx.lineTo(x,H);}
  for(let y=PY;y<=H-PY+1;y+=(H-2*PY)/8){bctx.moveTo(0,y);bctx.lineTo(W,y);}
  bctx.stroke();
  S.forEach((s,i)=>{const cx=X(s.cx),cy=Y(s.cy),r=Math.max(s.r*(W-2*PX),34)*1.6;
    const rg=bctx.createRadialGradient(cx,cy,0,cx,cy,r*1.8);
    rg.addColorStop(0,hexA(col(i),.10)); rg.addColorStop(1,hexA(col(i),0)); bctx.fillStyle=rg;
    bctx.beginPath(); bctx.arc(cx,cy,r*1.8,0,7); bctx.fill();
    for(let k=1;k<=4;k++){bctx.strokeStyle=hexA(col(i),.10-k*.015); bctx.lineWidth=1;
      bctx.beginPath(); bctx.ellipse(cx,cy,r*k*.55,r*k*.42,-0.2*k,0,7); bctx.stroke();}});
  // ticks along edges
  bctx.fillStyle='rgba(140,170,195,.35)'; bctx.font='9px IBM Plex Mono,monospace';
  for(let i=0;i<=10;i++){const x=PX+(W-2*PX)*i/10; bctx.fillRect(x,H-PY+18,1,i%5?4:8);}
}

function resize(){DPR=window.devicePixelRatio||1; W=wrap.clientWidth; H=wrap.clientHeight;
  cv.width=W*DPR; cv.height=H*DPR; ctx.setTransform(DPR,0,0,DPR,0,0); paintBg();}
window.addEventListener('resize',resize); resize();

// ---------- hover ----------
let hover=-1;
cv.addEventListener('mousemove',e=>{const r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
  let best=-1,bd=14; for(let i=0;i<N;i++){const d=Math.hypot(X(D.points[i][0])-mx,Y(D.points[i][1])-my); if(d<bd){bd=d;best=i;}}
  hover=best;
  if(best>=0){const p=D.points[best]; tip.style.opacity=1;
    tip.firstChild.textContent=S[p[2]].name+(D.pages[best]?` · p.${D.pages[best]}`:''); tip.firstChild.style.color=col(p[2]);
    tip.lastChild.textContent=D.snips[best]+'…';
    tip.style.left=Math.min(mx+14,W-310)+'px'; tip.style.top=Math.min(my+14,H-90)+'px';}
  else tip.style.opacity=0;});
cv.addEventListener('mouseleave',()=>{hover=-1;tip.style.opacity=0;});

// ---------- drawing ----------
let placed=[];
function label(text,x,y,color,align='left',alpha=1){
  ctx.font='11px IBM Plex Mono,monospace'; const w=ctx.measureText(text).width+14;
  const lx=align==='right'?x-w:x; ctx.globalAlpha=alpha;
  for(const off of [0,24,-24,48,-48,72,-72]){const yy=y+off;
    if(!placed.some(r=>lx<r[0]+r[2]+4&&lx+w+4>r[0]&&yy-10<r[1]+r[3]+3&&yy+10+3>r[1]) && yy>14 && yy<H-14){y=yy;break;}}
  placed.push([lx,y-10,w,20]);
  ctx.fillStyle='rgba(8,18,27,.82)'; ctx.strokeStyle=hexA(color.startsWith('#')?color:'#FFC857',.55);
  ctx.beginPath(); ctx.roundRect(lx,y-10,w,20,10); ctx.fill(); ctx.stroke();
  ctx.fillStyle=color; ctx.textBaseline='middle'; ctx.fillText(text,lx+7,y); ctx.globalAlpha=1;}
const trunc=(s,n)=>s.length>n?s.slice(0,n-1)+'…':s;

let needle=0;
function frame(now){
  const el=now-t0;
  ctx.clearRect(0,0,W,H); ctx.drawImage(bg,0,0,W,H); placed=[];
  const lit=new Float32Array(N), litWeak=new Uint8Array(N), cited=new Float32Array(N), ghostLit=new Float32Array(N);

  // pass 1: compute lighting from steps
  steps.forEach((s,i)=>{const p=P(i,el); if(p<=0) return;
    if(s.kind==='search'){s.hits.forEach(([id,sc],k)=>{const f=ease((p-.35-.07*k)/.25); const v=f*(.55+.45*clamp(sc*1.2));
        if(v>lit[id]){lit[id]=v; litWeak[id]=s.best<WEAK?1:0;}});}
    if(s.kind==='tool'){D.points.forEach((pt,id)=>{if(S[pt[2]].name===s.source) lit[id]=Math.max(lit[id],.5*ease(p));});}
    if(s.kind==='answer'){(s.cited||[]).forEach((id,k)=>{cited[id]=Math.max(cited[id],ease((p-.1*k)/.5));});}});
  if(D.ghost){const g=clamp((el-ghostStart)/ghostDur); D.ghost.hits.forEach(([id],k)=>{ghostLit[id]=ease((g-.4-.06*k)/.3);});}

  // source labels
  ctx.textBaseline='middle'; ctx.font='500 10px IBM Plex Mono,monospace';
  S.forEach((s,i)=>{const x=X(s.cx),y=Math.max(Y(s.cy)-Math.max(s.r*(W-2*PX)*.9,26)-8,40); ctx.fillStyle=hexA(col(i),.75);
    ctx.textAlign='center'; ctx.fillText(s.name.toUpperCase(),x,y); });
  ctx.textAlign='left';

  // points
  const breathe=el*0.0016;
  for(let i=0;i<N;i++){const p=D.points[i],x=X(p[0]),y=Y(p[1]),c=col(p[2]),l=lit[i];
    const r=1.9+.5*Math.sin(breathe+i*1.7)*.5+(i===hover?2.5:0);
    ctx.fillStyle=hexA(c,.38+(i===hover?.5:0)); ctx.beginPath(); ctx.arc(x,y,r,0,7); ctx.fill();
    if(l>0){const lc=litWeak[i]?CORAL:c; ctx.save(); ctx.shadowColor=lc; ctx.shadowBlur=14*l;
      ctx.fillStyle=lc; ctx.globalAlpha=.35+.65*l; ctx.beginPath(); ctx.arc(x,y,2.6+3.4*l,0,7); ctx.fill(); ctx.restore();}
    if(ghostLit[i]>0){ctx.strokeStyle=hexA(GHOST,.9*ghostLit[i]); ctx.lineWidth=1.2; ctx.setLineDash([2,3]);
      ctx.beginPath(); ctx.arc(x,y,8*ghostLit[i]+2,0,7); ctx.stroke(); ctx.setLineDash([]);}
    if(cited[i]>0){ctx.save(); ctx.strokeStyle='#FFF6D6'; ctx.shadowColor=AMBER; ctx.shadowBlur=10; ctx.lineWidth=1.6;
      ctx.globalAlpha=cited[i]; ctx.beginPath(); ctx.arc(x,y,7+2*Math.sin(breathe*2+i),0,7); ctx.stroke(); ctx.restore();}}

  // route
  let head=null, lastWp=null;
  route.forEach((seg,k)=>{const s=steps[seg.i]; const p=P(seg.i,el); const u=ease(p*2.6); if(u<=0) return;
    ctx.save(); ctx.strokeStyle=AMBER; ctx.lineWidth=1.7; ctx.shadowColor=AMBER; ctx.shadowBlur=9; ctx.setLineDash([7,6]);
    ctx.lineDashOffset=-el*0.025; ctx.beginPath(); const n=28;
    for(let j=0;j<=n*u;j++){const q=bez(seg.from,seg.to,j/n); j?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1]);}
    ctx.stroke(); ctx.restore();
    const hp=bez(seg.from,seg.to,u); if(u<1) head=hp; lastWp={x:X(seg.to[0]),y:Y(seg.to[1]),u};});
  if(head){ctx.save(); ctx.fillStyle='#FFF'; ctx.shadowColor=AMBER; ctx.shadowBlur=16; ctx.beginPath(); ctx.arc(head[0],head[1],3.4,0,7); ctx.fill(); ctx.restore();}

  // step visuals
  steps.forEach((s,i)=>{const p=P(i,el); if(p<=0||!s._wp) return; const wx=X(s._wp[0]),wy=Y(s._wp[1]);
    if(s.kind==='search'){const weak=s.best<WEAK,c=weak?CORAL:AMBER, a=ease((p-.18)/.2);
      s.hits.forEach(([id,sc],k)=>{const f=ease((p-.35-.07*k)/.25); if(f<=0) return; const pt=D.points[id];
        ctx.strokeStyle=hexA(c,.15+.5*sc*f); ctx.lineWidth=.8+sc; ctx.beginPath(); ctx.moveTo(wx,wy);
        ctx.lineTo(wx+(X(pt[0])-wx)*f,wy+(Y(pt[1])-wy)*f); ctx.stroke();});
      for(let r=0;r<2;r++){const q=((p*2.2+r*.5)%1); if(p>.2){ctx.strokeStyle=hexA(c,.5*(1-q)*a); ctx.lineWidth=1.3;
        ctx.beginPath(); ctx.arc(wx,wy,8+q*(p<1?64:34),0,7); ctx.stroke();}}
      ctx.save(); ctx.globalAlpha=a; ctx.fillStyle='#08121B'; ctx.strokeStyle=c; ctx.lineWidth=2; ctx.shadowColor=c; ctx.shadowBlur=14;
      ctx.beginPath(); ctx.arc(wx,wy,7,0,7); ctx.fill(); ctx.stroke(); ctx.restore();
      ctx.fillStyle=c; ctx.globalAlpha=a; ctx.beginPath(); ctx.arc(wx,wy,2.5,0,7); ctx.fill(); ctx.globalAlpha=1;
      const tx=wx>W*.62?wx-14:wx+14; label(`“${trunc(s.q,30)}”`+(p>.7?`  ${s.best.toFixed(2)}${weak?' weak':''}`:''),tx,wy-18,weak?CORAL:'#FFE2A1',wx>W*.62?'right':'left',a);}
    if(s.kind==='tool'){const r=srcByName[s.source]; const a=ease(p/.3);
      if(r){const rad=Math.max(r.r*(W-2*PX),34)*1.35*ease(p); ctx.strokeStyle=hexA(AMBER,.55*(1-(p>.9?(p-.9)*4:0))); ctx.lineWidth=1.2; ctx.setLineDash([4,5]);
        ctx.beginPath(); ctx.arc(wx,wy,rad,0,7); ctx.stroke(); ctx.setLineDash([]);
        const ang=el*0.006; ctx.beginPath(); ctx.moveTo(wx,wy); ctx.lineTo(wx+Math.cos(ang)*rad,wy+Math.sin(ang)*rad); ctx.stroke();}
      ctx.save(); ctx.globalAlpha=a; ctx.translate(wx,wy); ctx.rotate(Math.PI/4); ctx.fillStyle='#08121B'; ctx.strokeStyle=AMBER; ctx.lineWidth=2;
      ctx.shadowColor=AMBER; ctx.shadowBlur=12; ctx.fillRect(-6,-6,12,12); ctx.strokeRect(-6,-6,12,12); ctx.restore();
      label((s.name==='analyze'?'▦ analyze · ':'¶ summarize · ')+trunc(s.source,22),wx>W*.62?wx-16:wx+16,wy+20,'#FFE2A1',wx>W*.62?'right':'left',a);}
    if(s.kind==='calc'){const a=ease(p/.3); ctx.save(); ctx.globalAlpha=a; ctx.strokeStyle=AMBER; ctx.lineWidth=1.6; ctx.fillStyle='#08121B';
      ctx.shadowColor=AMBER; ctx.shadowBlur=10; ctx.beginPath(); ctx.roundRect(wx-7,wy-7,14,14,4); ctx.fill(); ctx.stroke(); ctx.restore();
      ctx.fillStyle=AMBER; ctx.globalAlpha=a; ctx.font='600 11px IBM Plex Mono,monospace'; ctx.textAlign='center'; ctx.fillText('Σ',wx,wy+1); ctx.textAlign='left'; ctx.globalAlpha=1;
      label(trunc(s.label,26),wx+14,wy-14,'#FFE2A1','left',a);}
    if(s.kind==='answer'){const a=ease(p/.35);
      (s.cited||[]).forEach(id=>{const pt=D.points[id]; ctx.strokeStyle=hexA('#FFF6D6',.28*a); ctx.lineWidth=.8; ctx.beginPath(); ctx.moveTo(wx,wy);
        ctx.lineTo(X(pt[0]),Y(pt[1])); ctx.stroke();});
      ctx.save(); ctx.globalAlpha=a; ctx.shadowColor=AMBER; ctx.shadowBlur=22; ctx.fillStyle=AMBER; ctx.beginPath(); ctx.arc(wx,wy,8+1.5*Math.sin(el*.004),0,7); ctx.fill();
      ctx.fillStyle='#08121B'; ctx.beginPath(); ctx.arc(wx,wy,3,0,7); ctx.fill(); ctx.restore();
      label(`ANSWER · ${(s.cited||[]).length} sources`,wx>W*.62?wx-16:wx+16,wy+22,AMBER,wx>W*.62?'right':'left',a);}});

  // question flag
  if(steps.length){const x=X(QPOS[0]),y=Y(QPOS[1]); const a=ease((el)/500);
    ctx.save(); ctx.globalAlpha=a; ctx.strokeStyle='#FFF'; ctx.fillStyle=AMBER; ctx.lineWidth=1.5; ctx.shadowColor=AMBER; ctx.shadowBlur=10;
    ctx.beginPath(); ctx.moveTo(x,y+12); ctx.lineTo(x,y-14); ctx.stroke(); ctx.beginPath(); ctx.moveTo(x,y-14); ctx.lineTo(x+16,y-9); ctx.lineTo(x,y-4); ctx.fill(); ctx.restore();
    ctx.fillStyle='#D6E2EC'; ctx.globalAlpha=a; ctx.font='11px IBM Plex Mono,monospace'; ctx.textBaseline='middle'; ctx.fillText(trunc(D.question||'',58),x+24,y-8); ctx.globalAlpha=1;}

  // ghost (standard RAG)
  if(D.ghost){const g=clamp((el-ghostStart)/ghostDur), gx=X(D.ghost.xy[0]), gy=Y(D.ghost.xy[1]), a=ease(g/.3);
    D.ghost.hits.forEach(([id],k)=>{const f=ease((g-.4-.06*k)/.3); if(f<=0) return; const pt=D.points[id];
      ctx.strokeStyle=hexA(GHOST,.5*f); ctx.lineWidth=1; ctx.setLineDash([3,4]); ctx.beginPath(); ctx.moveTo(gx,gy);
      ctx.lineTo(gx+(X(pt[0])-gx)*f,gy+(Y(pt[1])-gy)*f); ctx.stroke(); ctx.setLineDash([]);});
    ctx.save(); ctx.globalAlpha=a; ctx.strokeStyle=GHOST; ctx.lineWidth=1.6; ctx.setLineDash([3,3]); ctx.beginPath(); ctx.arc(gx,gy,8,0,7); ctx.stroke(); ctx.restore();
    label('standard RAG · one shot',gx>W*.62?gx-14:gx+14,gy+20,GHOST,gx>W*.62?'right':'left',a);}

  // compass
  const cx=W-48,cy=H-48; ctx.strokeStyle='rgba(140,170,195,.45)'; ctx.lineWidth=1; ctx.beginPath(); ctx.arc(cx,cy,24,0,7); ctx.stroke();
  ctx.fillStyle='rgba(140,170,195,.6)'; ctx.font='9px IBM Plex Mono,monospace'; ctx.textAlign='center'; ctx.fillText('N',cx,cy-31); ctx.textAlign='left';
  let target=-Math.PI/2+Math.sin(el*.0007)*.25; if(lastWp) target=Math.atan2(lastWp.y-cy,lastWp.x-cx);
  let d=target-needle; d=Math.atan2(Math.sin(d),Math.cos(d)); needle+=d*.08;
  ctx.save(); ctx.translate(cx,cy); ctx.rotate(needle); ctx.fillStyle=AMBER; ctx.beginPath(); ctx.moveTo(19,0); ctx.lineTo(-4,-4); ctx.lineTo(-4,4); ctx.fill();
  ctx.fillStyle='rgba(140,170,195,.5)'; ctx.beginPath(); ctx.moveTo(-17,0); ctx.lineTo(-3,-3); ctx.lineTo(-3,3); ctx.fill(); ctx.restore();

  // ticker
  let tag='', body='';
  if(!N) body='Load documents and the terrain appears here.';
  else if(!steps.length) body='Ask a question. The agent’s route across your documents is drawn live.';
  else {let cur=0; steps.forEach((s,i)=>{if(P(i,el)>0) cur=i+1;}); const s=steps[Math.max(0,cur-1)];
    tag=`step ${Math.max(1,cur)}/${steps.length}  `; body=s.text||s.kind;}
  const bl=document.getElementById('bl'), key=tag+body;
  if(bl.dataset.t!==key){bl.dataset.t=key; bl.textContent=''; if(tag){const i=document.createElement('i'); i.textContent=tag; bl.appendChild(i);} bl.appendChild(document.createTextNode(body));}
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
</script></body></html>"""
