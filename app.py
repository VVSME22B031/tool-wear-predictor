import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, FancyBboxPatch
import matplotlib.patheffects as pe
import joblib, os

st.set_page_config(page_title="ToolSense",page_icon="⚙️",layout="wide",initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;500;600&display=swap');

html,body,[class*="css"]{font-family:'Inter',sans-serif}
.stApp{background:#0a1f0f}
.block-container{padding:0!important;max-width:100%!important}

/* HERO */
.hero{
  background:linear-gradient(135deg,#0d2613 0%,#122d18 50%,#0a1f0f 100%);
  border-bottom:2px solid #c9a84c;
  padding:36px 60px 30px;
  position:relative;overflow:hidden;
}
.hero::before{
  content:'';position:absolute;inset:0;
  background:repeating-linear-gradient(45deg,transparent,transparent 40px,rgba(201,168,76,0.015) 40px,rgba(201,168,76,0.015) 80px);
  pointer-events:none;
}
.h-badge{
  display:inline-block;background:rgba(201,168,76,0.12);
  border:1px solid rgba(201,168,76,0.4);color:#c9a84c;
  font-size:10px;font-weight:600;letter-spacing:.16em;text-transform:uppercase;
  padding:4px 16px;border-radius:2px;margin-bottom:14px;
  font-family:'JetBrains Mono',monospace;
}
.h-title{font-size:40px;font-weight:700;color:#f5f0e8;margin:0 0 6px;letter-spacing:-0.5px;font-family:'Playfair Display',serif}
.h-title span{color:#c9a84c}
.h-sub{font-size:13px;color:#5a7a60;margin:0 0 24px;font-family:'JetBrains Mono',monospace}
.h-stats{display:flex;gap:0;border:1px solid #c9a84c;border-radius:4px;overflow:hidden;width:fit-content;background:#071209}
.hs{padding:12px 24px;border-right:1px solid rgba(201,168,76,0.3)}
.hs:last-child{border-right:none}
.hs-v{font-size:18px;font-weight:700;color:#c9a84c;font-family:'JetBrains Mono',monospace;line-height:1}
.hs-l{font-size:9px;color:#3d6645;text-transform:uppercase;letter-spacing:.12em;margin-top:3px}

/* MAIN */
.mp{padding:32px 60px}
.sec{font-size:10px;font-weight:600;letter-spacing:.18em;text-transform:uppercase;color:#c9a84c;
     margin-bottom:14px;font-family:'JetBrains Mono',monospace;display:flex;align-items:center;gap:12px}
.sec::after{content:'';flex:1;height:1px;background:linear-gradient(90deg,rgba(201,168,76,0.4),transparent)}
.pnl{background:#071209;border:1px solid rgba(201,168,76,0.25);border-radius:6px;padding:22px;margin-bottom:16px}

/* INPUTS */
.stSlider>div>div>div>div{background:#c9a84c!important}
.stNumberInput input{
  background:#071209!important;border:1px solid rgba(201,168,76,0.3)!important;
  color:#f5f0e8!important;border-radius:4px!important;
  font-family:'JetBrains Mono',monospace!important;font-size:13px!important;
}
.stNumberInput input:focus{border-color:#c9a84c!important}
.stButton>button{
  background:linear-gradient(135deg,#1a5c2a,#155225)!important;
  color:#f5f0e8!important;border:2px solid #c9a84c!important;border-radius:4px!important;
  font-size:13px!important;font-weight:700!important;
  font-family:'Playfair Display',serif!important;
  letter-spacing:.12em!important;padding:14px 0!important;
  text-transform:uppercase!important;width:100%!important;
  box-shadow:0 0 20px rgba(201,168,76,0.15),inset 0 1px 0 rgba(255,255,255,0.05)!important;
}
.stButton>button:hover{
  background:linear-gradient(135deg,#1f7033,#1a5c2a)!important;
  box-shadow:0 0 32px rgba(201,168,76,0.3)!important;
}

/* STATUS */
.sc{border-radius:4px;padding:16px 22px;margin:16px 0;display:flex;align-items:center;gap:14px;border-left:4px solid}
.s-h{background:#071209;border-color:#22c55e;border:1px solid rgba(34,197,94,0.3);border-left:4px solid #22c55e}
.s-w{background:#071209;border-color:#f59e0b;border:1px solid rgba(245,158,11,0.3);border-left:4px solid #f59e0b}
.s-c{background:#071209;border-color:#ef4444;border:1px solid rgba(239,68,68,0.3);border-left:4px solid #ef4444}
.sl{font-size:18px;font-weight:700;line-height:1;margin-bottom:4px;font-family:'Playfair Display',serif}
.sd{font-size:12px;color:#5a7a60;font-family:'JetBrains Mono',monospace}

/* METRICS */
.mg{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:14px 0}
.mc{background:#071209;border:1px solid rgba(201,168,76,0.2);border-radius:4px;padding:16px 18px;position:relative;overflow:hidden}
.mc::before{content:'';position:absolute;bottom:0;left:0;right:0;height:2px}
.mc-g::before{background:linear-gradient(90deg,#22c55e,transparent)}
.mc-y::before{background:linear-gradient(90deg,#f59e0b,transparent)}
.mc-r::before{background:linear-gradient(90deg,#ef4444,transparent)}
.mc-b::before{background:linear-gradient(90deg,#c9a84c,transparent)}
.mv{font-size:28px;font-weight:700;font-family:'JetBrains Mono',monospace;line-height:1;margin-bottom:4px}
.ml{font-size:9px;color:#3d6645;text-transform:uppercase;letter-spacing:.12em}
.ms{font-size:11px;color:#1e3a22;margin-top:2px;font-family:'JetBrains Mono',monospace}

/* BREAKDOWN */
.bd{background:#071209;border:1px solid rgba(201,168,76,0.2);border-radius:4px;overflow:hidden;margin:12px 0}
.br{display:flex;justify-content:space-between;align-items:center;padding:10px 18px;border-bottom:1px solid rgba(201,168,76,0.08)}
.br:last-child{border-bottom:none}
.bk{font-size:11px;color:#5a7a60;font-family:'JetBrains Mono',monospace}
.bv{font-size:12px;color:#f5f0e8;font-weight:600;font-family:'JetBrains Mono',monospace}
.bg_{font-size:9px;font-weight:700;padding:2px 8px;border-radius:2px;margin-left:8px;font-family:'JetBrains Mono',monospace}
.bh{background:rgba(239,68,68,0.15);color:#fca5a5}
.bm{background:rgba(245,158,11,0.15);color:#fcd34d}
.bl{background:rgba(34,197,94,0.15);color:#86efac}

/* INFO TABLE */
.ic{background:#071209;border:1px solid rgba(201,168,76,0.2);border-radius:4px;overflow:hidden}
.ir{display:flex;justify-content:space-between;padding:10px 18px;border-bottom:1px solid rgba(201,168,76,0.08);font-size:12px}
.ir:last-child{border-bottom:none}
.ik{color:#5a7a60;font-family:'JetBrains Mono',monospace}
.iv{color:#c9a84c;font-family:'JetBrains Mono',monospace;font-weight:600}

label{color:#5a7a60!important;font-size:12px!important}
</style>
""", unsafe_allow_html=True)

THRESHOLD = 300.0

@st.cache_resource
def load_models():
    for wn, rn in [("best_wear_model.pkl","best_rul_model.pkl"),
                   ("best_wear_model (1).pkl","best_rul_model (1).pkl")]:
        if os.path.exists(wn) and os.path.exists(rn):
            return joblib.load(wn), joblib.load(rn), True
    return None, None, False

wear_model, rul_model, models_loaded = load_models()

def build_features(speed,feed,doc,cycle,vib_x,vib_y,vib_z,spindle,roughness):
    vm  = (vib_x**2+vib_y**2+vib_z**2)**0.5
    mrr = speed*feed*doc
    return np.array([[speed,feed,doc,vib_x,vib_y,vib_z,spindle,roughness,
                      vm,mrr,vm/(spindle+1e-6),np.log1p(cycle)]])

def predict(speed,feed,doc,cycle,vib_x,vib_y,vib_z,spindle,roughness):
    vm=( vib_x**2+vib_y**2+vib_z**2)**0.5
    mrr=speed*feed*doc
    wr=0.5*(speed/1200)+0.3*(feed/0.18)+0.2*(doc/1.0)
    sf=0.4*(spindle/70)+0.3*(vm/2.0)+0.3*(roughness/1.5)
    if models_loaded:
        X=build_features(speed,feed,doc,cycle,vib_x,vib_y,vib_z,spindle,roughness)
        wear=float(np.clip(wear_model.predict(X)[0],0,THRESHOLD-0.5))
        rul=max(0,int(rul_model.predict(X)[0]))
    else:
        wear=max(0.0,min(14*wr*sf*(cycle**1.28),THRESHOLD-0.5))
        rul=max(0,int((THRESHOLD-wear)*(cycle/max(wear,0.1))*0.85))
    return wear,rul,(wear/THRESHOLD)*100,vm,mrr,wr,sf

def bdg(val,ref):
    r=val/ref
    return ("HIGH","bh") if r>0.85 else ("MED","bm") if r>0.55 else ("LOW","bl")

def draw_roulette(pct, wear, rul):
    fig, ax = plt.subplots(figsize=(5.8,4.8), facecolor="#071209")
    ax.set_facecolor("#071209")
    ax.set_xlim(-1.55,1.55); ax.set_ylim(-1.15,1.65)
    ax.set_aspect('equal'); ax.axis('off')

    # Outer gold ring
    outer = plt.Circle((0,0),1.42,fill=False,edgecolor="#c9a84c",linewidth=2.5,alpha=0.8,zorder=2)
    ax.add_patch(outer)
    inner_ring = plt.Circle((0,0),1.27,fill=False,edgecolor="#c9a84c",linewidth=1,alpha=0.3,zorder=2)
    ax.add_patch(inner_ring)

    # Roulette segments with casino colors
    n=37
    seg_colors=[]
    for i in range(n):
        if i==0: seg_colors.append("#1a6b2e")        # zero = casino green
        elif i<int(n*0.60): seg_colors.append("#8b1a1a")   # red zone
        elif i<int(n*0.85): seg_colors.append("#ca8a04")   # amber zone
        else: seg_colors.append("#1a5c2a")            # dark green = critical

    for i,col in enumerate(seg_colors):
        t1=180-(i*180/n); t2=180-((i+1)*180/n)
        ax.add_patch(Wedge((0,0),1.38,t2,t1,width=0.13,
                           facecolor=col,edgecolor="#071209",linewidth=1.8,alpha=0.95))

    # Gold dots on rim
    for i in range(n):
        angle=np.radians(180-(i+0.5)*180/n)
        rx,ry=1.32*np.cos(angle),1.32*np.sin(angle)
        dot = plt.Circle((rx,ry),0.022,color="#c9a84c",zorder=6,alpha=0.9)
        ax.add_patch(dot)

    # Number labels on segments (every 6th)
    for i in range(0,n,6):
        angle=np.radians(180-(i+0.5)*180/n)
        tx,ty=1.22*np.cos(angle),1.22*np.sin(angle)
        ax.text(tx,ty,str(i),ha='center',va='center',fontsize=6,
                color="#f5f0e8",fontfamily='monospace',fontweight='bold',alpha=0.7)

    # Felt background for gauge
    felt = Wedge((0,0),1.10,0,180,width=0.48,
                 facecolor="#0a2410",edgecolor="#c9a84c",linewidth=1,alpha=0.95)
    ax.add_patch(felt)

    # Zone coloring on felt
    zones=[(0,0.60,"#22c55e",0.08),(0.60,0.85,"#f59e0b",0.08),(0.85,1.0,"#ef4444",0.10)]
    for z0,z1,zcol,za in zones:
        a0,a1=180-z0*180,180-z1*180
        ax.add_patch(Wedge((0,0),1.09,a1,a0,width=0.46,facecolor=zcol,edgecolor="none",alpha=za))

    # Fill arc
    bc="#22c55e" if pct<60 else "#f59e0b" if pct<85 else "#ef4444"
    if pct>0:
        fill=Wedge((0,0),1.08,180-pct*1.8,180,width=0.44,
                   facecolor=bc,edgecolor="none",alpha=0.2)
        ax.add_patch(fill)

    # Tick marks with labels
    for pm,lbl,col in [(0,"0","#3d6645"),(60,"60%","#22c55e"),(85,"85%","#f59e0b"),(100,"100%","#ef4444")]:
        angle=np.radians(180-pm*1.8)
        x1,y1=0.63*np.cos(angle),0.63*np.sin(angle)
        x2,y2=0.72*np.cos(angle),0.72*np.sin(angle)
        ax.plot([x1,x2],[y1,y2],color=col,lw=1.5,alpha=0.9,zorder=7)
        xl,yl=0.78*np.cos(angle),0.78*np.sin(angle)
        ax.text(xl,yl,lbl,ha='center',va='center',fontsize=7,
                color=col,fontfamily='monospace',alpha=0.95,fontweight='bold')

    # Needle
    na=np.radians(180-pct*1.8)
    nx,ny=0.55*np.cos(na),0.55*np.sin(na)
    ax.plot([0,nx],[0,ny],color="#f5f0e8",lw=2.5,zorder=9,
            solid_capstyle='round',
            path_effects=[pe.withStroke(linewidth=4,foreground="#071209")])
    tip=plt.Circle((nx,ny),0.018,color=bc,zorder=10)
    ax.add_patch(tip)
    ax.add_patch(plt.Circle((0,0),0.055,color="#c9a84c",zorder=11,alpha=0.9))
    ax.add_patch(plt.Circle((0,0),0.030,color="#f5f0e8",zorder=12))

    # Center readout
    ax.text(0,0.25,f"{pct:.1f}%",ha='center',va='center',
            fontsize=22,fontweight='bold',color=bc,fontfamily='monospace',
            path_effects=[pe.withStroke(linewidth=3,foreground="#0a2410")])
    ax.text(0,0.09,"LIFE CONSUMED",ha='center',va='center',
            fontsize=7,color="#5a7a60",fontfamily='monospace',fontweight='600',
            letter_spacing=2)

    # Dividers
    ax.plot([-0.55,0.55],[-0.16,-0.16],color="#c9a84c",lw=0.7,alpha=0.4)
    ax.plot([0,0],[-0.16,-0.70],color="#c9a84c",lw=0.7,alpha=0.4)

    # Bottom stats
    ax.text(-0.55,-0.33,f"{wear:.1f}µm",ha='center',va='center',
            fontsize=14,fontweight='700',color="#c9a84c",fontfamily='monospace')
    ax.text(-0.55,-0.50,"FLANK WEAR",ha='center',va='center',
            fontsize=6.5,color="#3d6645",fontfamily='monospace',fontweight='600')

    rul_col="#22c55e" if rul>10 else "#f59e0b" if rul>3 else "#ef4444"
    ax.text(0.55,-0.33,f"{rul}",ha='center',va='center',
            fontsize=14,fontweight='700',color=rul_col,fontfamily='monospace')
    ax.text(0.55,-0.50,"RUL CYCLES",ha='center',va='center',
            fontsize=6.5,color="#3d6645",fontfamily='monospace',fontweight='600')

    # Gold corner decorations
    for x,y in [(-1.45,1.45),(1.45,1.45)]:
        ax.text(x,y,"✦",ha='center',va='center',fontsize=12,color="#c9a84c",alpha=0.5)

    ax.set_title("Tool Health Monitor",color="#5a7a60",fontsize=9,
                 fontfamily='monospace',pad=6,fontweight='600',letter_spacing=2)
    plt.tight_layout(pad=0.5)
    return fig

def draw_wear_chart(cycle,wear,rul,wr,sf):
    fut=[range(1,cycle+rul+8)]
    fut=list(range(1,cycle+rul+8))
    proj=[min(14*wr*sf*(c**1.28),THRESHOLD) for c in fut]
    bc="#22c55e" if (wear/THRESHOLD)*100<60 else "#f59e0b" if (wear/THRESHOLD)*100<85 else "#ef4444"

    fig,ax=plt.subplots(figsize=(5,3.6),facecolor="#071209")
    ax.set_facecolor("#071209")
    # Zone bands
    ax.axhspan(0,180,alpha=0.06,color="#22c55e")
    ax.axhspan(180,255,alpha=0.06,color="#f59e0b")
    ax.axhspan(255,320,alpha=0.06,color="#ef4444")
    # Projection
    ax.fill_between(fut,proj,alpha=0.1,color=bc)
    ax.plot(fut,proj,color=bc,lw=2.5,zorder=3)
    # Lines
    ax.axhline(THRESHOLD,color="#ef4444",ls="--",lw=1.2,alpha=0.7,label="Failure 300µm")
    ax.axhline(THRESHOLD*0.85,color="#f59e0b",ls=":",lw=1,alpha=0.6,label="Warning 255µm")
    ax.axhline(180,color="#22c55e",ls=":",lw=0.8,alpha=0.5,label="Caution 180µm")
    # Current
    ax.axvline(cycle,color="#c9a84c",ls="--",lw=1,alpha=0.5)
    ax.scatter([cycle],[wear],color=bc,s=70,zorder=6,edgecolors="#071209",lw=2)
    ax.text(cycle+0.3,wear+14,f"{wear:.0f}µm",color=bc,fontsize=8.5,
            fontfamily="monospace",fontweight='bold')
    if rul>0:
        fc=cycle+rul
        ax.scatter([fc],[THRESHOLD],color="#ef4444",s=50,zorder=6,marker="x",lw=2)
        ax.text(fc+0.2,THRESHOLD-22,"FAIL",color="#ef4444",fontsize=7.5,
                fontfamily="monospace",alpha=0.8,fontweight='bold')

    ax.set_xlabel("Cycle",color="#5a7a60",fontsize=9)
    ax.set_ylabel("Wear (µm)",color="#5a7a60",fontsize=9)
    ax.tick_params(colors="#3d6645",labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor("#1a3d22")
    legend=ax.legend(fontsize=7,facecolor="#071209",labelcolor="#5a7a60",
                     edgecolor="#1a3d22",loc="upper left")
    ax.set_title("Wear Trajectory",color="#5a7a60",fontsize=9,
                 fontfamily='monospace',pad=6,fontweight='600')
    ax.set_ylim(0,THRESHOLD+35)
    plt.tight_layout(pad=1.2)
    return fig

def draw_influence(speed,feed,doc,vm,spindle,roughness):
    fac={"Speed":0.5*(speed/1200),"Feed":0.3*(feed/0.18),"DoC":0.2*(doc/1.0),
         "Vibration":0.3*(vm/2),"Spindle":0.4*(spindle/70),"Roughness":0.3*(roughness/1.5)}
    vals=list(fac.values()); total=sum(vals)
    pcts=[v/total*100 for v in vals]
    colors=["#22c55e","#c9a84c","#a855f7","#f97316","#3b82f6","#ec4899"]

    fig,ax=plt.subplots(figsize=(5,3.6),facecolor="#071209")
    ax.set_facecolor("#071209")
    bars=ax.barh(list(fac.keys()),vals,color=colors,alpha=0.85,height=0.52,edgecolor="none")
    for bar,v,p,col in zip(bars,vals,pcts,colors):
        ax.text(v+0.005,bar.get_y()+bar.get_height()/2,
                f"{v:.2f}  ({p:.0f}%)",va="center",color=col,fontsize=8.5,
                fontfamily="monospace",fontweight='bold')
    ax.set_xlabel("Influence Score",color="#5a7a60",fontsize=9)
    ax.tick_params(colors="#5a7a60",labelsize=9.5)
    for sp in ax.spines.values(): sp.set_edgecolor("#1a3d22")
    ax.set_title("Parameter Influence on Wear",color="#5a7a60",fontsize=9,
                 fontfamily='monospace',pad=6,fontweight='600')
    ax.set_xlim(0,max(vals)*1.5)
    plt.tight_layout(pad=1.2)
    return fig

# ── HERO
st.markdown(f"""
<div class="hero">
<div class="h-badge">⚙ BTP · Micro-Turning · {'Real ML Model · Gradient Boosting + XGBoost' if models_loaded else 'Physics Fallback'} · Predictive Maintenance</div>
<div class="h-title">Tool<span>Sense</span></div>
<div class="h-sub">real-time tool wear & remaining useful life prediction system</div>
<div class="h-stats">
<div class="hs"><div class="hs-v">0.98</div><div class="hs-l">Wear R²</div></div>
<div class="hs"><div class="hs-v">0.95</div><div class="hs-l">RUL R²</div></div>
<div class="hs"><div class="hs-v">K=10</div><div class="hs-l">GroupKFold CV</div></div>
<div class="hs"><div class="hs-v">300µm</div><div class="hs-l">Threshold</div></div>
<div class="hs"><div class="hs-v">{'✓ ML' if models_loaded else '⚠ FB'}</div><div class="hs-l">{'Model Active' if models_loaded else 'Fallback'}</div></div>
</div></div>
""", unsafe_allow_html=True)

st.markdown('<div class="mp">', unsafe_allow_html=True)
c1,_,c2=st.columns([4,0.3,6])

with c1:
    st.markdown('<div class="sec">Cutting Parameters</div>', unsafe_allow_html=True)
    st.markdown('<div class="pnl">', unsafe_allow_html=True)
    speed=st.slider("Spindle Speed (rpm)",500,2000,1000,step=50)
    feed=st.slider("Feed Rate (mm/rev)",0.05,0.30,0.13,step=0.01)
    doc=st.slider("Depth of Cut (mm)",0.1,2.0,0.8,step=0.1)
    cycle=st.slider("Cycle Number",1,100,8)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec">Sensor Readings</div>', unsafe_allow_html=True)
    st.markdown('<div class="pnl">', unsafe_allow_html=True)
    s1,s2=st.columns(2)
    with s1:
        vib_x=st.number_input("Vibration X (mm/s²)",value=1.35,step=0.05,format="%.2f")
        vib_z=st.number_input("Vibration Z (mm/s²)",value=0.75,step=0.05,format="%.2f")
        roughness=st.number_input("Roughness (µm)",value=1.4,step=0.1,format="%.1f")
    with s2:
        vib_y=st.number_input("Vibration Y (mm/s²)",value=1.20,step=0.05,format="%.2f")
        spindle=st.slider("Spindle Load (%)",10,100,61)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    clicked=st.button("⚡  PREDICT TOOL CONDITION",use_container_width=True)

with c2:
    if clicked:
        wear,rul,pct,vm,mrr,wr,sf=predict(speed,feed,doc,cycle,vib_x,vib_y,vib_z,spindle,roughness)
        bc="#22c55e" if pct<60 else "#f59e0b" if pct<85 else "#ef4444"
        rc="#22c55e" if rul>10 else "#f59e0b" if rul>3 else "#ef4444"
        mc="mc-g" if pct<60 else "mc-y" if pct<85 else "mc-r"
        if pct<60:   sc,icon,lbl="s-h","✅","HEALTHY"; desc=f"Safe operating range — {rul} cycles remaining before replacement."
        elif pct<85: sc,icon,lbl="s-w","⚠️","WARNING — PLAN REPLACEMENT"; desc=f"Elevated wear — schedule replacement within {rul} cycles."
        else:        sc,icon,lbl="s-c","🔴","CRITICAL — REPLACE NOW"; desc=f"Near failure. Only {rul} cycles remaining. Stop immediately."

        st.markdown(f"""
        <div class="sc {sc}">
        <div style="font-size:28px">{icon}</div>
        <div><div class="sl" style="color:{bc}">{lbl}</div><div class="sd">{desc}</div></div>
        </div>
        <div class="mg">
        <div class="mc mc-b"><div class="mv" style="color:#c9a84c">{wear:.1f}</div><div class="ml">Flank Wear (µm)</div><div class="ms">threshold · 300µm</div></div>
        <div class="mc {mc}"><div class="mv" style="color:{rc}">{rul}</div><div class="ml">Remaining Cycles</div><div class="ms">until failure</div></div>
        <div class="mc {mc}"><div class="mv" style="color:{bc}">{pct:.1f}%</div><div class="ml">Life Consumed</div><div class="ms">{100-pct:.1f}% remaining</div></div>
        </div>""", unsafe_allow_html=True)

        g1,g2=st.columns(2)
        with g1: st.pyplot(draw_roulette(pct,wear,rul),use_container_width=True)
        with g2: st.pyplot(draw_wear_chart(cycle,wear,rul,wr,sf),use_container_width=True)

        st.markdown('<div class="sec" style="margin-top:18px">Parameter Influence</div>', unsafe_allow_html=True)
        st.pyplot(draw_influence(speed,feed,doc,vm,spindle,roughness),use_container_width=True)

        rows=[("speed_rpm",f"{speed} rpm",*bdg(speed,1200)),
              ("feed_mm_rev",f"{feed:.2f} mm/rev",*bdg(feed,0.18)),
              ("depth_of_cut",f"{doc:.1f} mm",*bdg(doc,1.0)),
              ("vib_magnitude",f"{vm:.3f} mm/s²",*bdg(vm,2.0)),
              ("spindle_load",f"{spindle}%",*bdg(spindle,70)),
              ("surface_roughness",f"{roughness:.1f} µm",*bdg(roughness,1.5)),
              ("mrr",f"{mrr:.1f} mm³/min",*bdg(mrr,150))]
        html="".join([f'<div class="br"><span class="bk">{r[0]}</span><div><span class="bv">{r[1]}</span><span class="bg_ {r[3]}">{r[2]}</span></div></div>' for r in rows])
        st.markdown(f'<div class="sec" style="margin-top:14px">Feature Breakdown</div><div class="bd">{html}</div>',unsafe_allow_html=True)

        rec="Continue operation. Inspect after 10 cycles." if pct<60 else f"Plan replacement within {rul} cycles." if pct<85 else "STOP. Replace tool before next cycle."
        st.markdown(f"""
        <div style="background:#071209;border:1px solid rgba(201,168,76,0.2);border-left:3px solid {bc};
             border-radius:4px;padding:14px 20px;margin-top:14px">
        <div style="font-size:9px;color:#3d6645;text-transform:uppercase;letter-spacing:.15em;
             font-family:monospace;margin-bottom:5px">◆ RECOMMENDATION</div>
        <div style="font-size:13px;color:#f5f0e8;font-weight:500;font-family:'Playfair Display',serif">{rec}</div>
        <div style="font-size:10px;color:#2a4a2e;margin-top:6px;font-family:monospace">
        {'Powered by real trained ML model · Gradient Boosting + XGBoost' if models_loaded else 'Physics-based fallback active'}</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="padding:70px 20px;text-align:center">
        <div style="font-size:48px;margin-bottom:18px;opacity:.12">⚙️</div>
        <div style="font-size:16px;color:#2a4a2e;font-weight:600;margin-bottom:8px;font-family:'Playfair Display',serif">Configure inputs — Click Predict</div>
        <div style="font-size:11px;color:#1a3a1e;line-height:1.8;font-family:monospace">
        Set cutting parameters and sensor readings on the left.<br>
        The roulette gauge and all charts appear here.
        </div></div>
        <div class="sec">Model Performance · BTP Results</div>
        <div class="ic">
        <div class="ir"><span class="ik">wear_prediction_r2</span><span class="iv">0.9802</span></div>
        <div class="ir"><span class="ik">rul_prediction_r2</span><span class="iv">0.9494</span></div>
        <div class="ir"><span class="ik">wear_rmse</span><span class="iv">13.24 µm</span></div>
        <div class="ir"><span class="ik">rul_rmse</span><span class="iv">0.895 cycles</span></div>
        <div class="ir"><span class="ik">validation_method</span><span class="iv">GroupKFold K=10</span></div>
        <div class="ir"><span class="ik">best_wear_model</span><span class="iv">Gradient Boosting</span></div>
        <div class="ir"><span class="ik">best_rul_model</span><span class="iv">XGBoost</span></div>
        <div class="ir"><span class="ik">failure_threshold</span><span class="iv">300 µm</span></div>
        </div>""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
