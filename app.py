import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import matplotlib.patheffects as pe
import joblib, os

st.set_page_config(page_title="ToolSense",page_icon="⚙️",layout="wide",initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

*{font-family:'DM Sans',sans-serif!important}
.stApp{background:#0c0c0f}
.block-container{padding:0!important;max-width:100%!important}

.hero{background:#0c0c0f;border-bottom:1px solid #1e1e28;padding:40px 64px 32px}
.h-eyebrow{font-size:10px;font-weight:500;letter-spacing:.2em;text-transform:uppercase;
           color:#555;margin-bottom:10px;font-family:'DM Mono',monospace!important}
.h-title{font-size:44px;font-weight:600;color:#fff;margin:0 0 6px;letter-spacing:-1.5px;line-height:1}
.h-title em{font-style:normal;color:#d4a843}
.h-sub{font-size:13px;color:#444;margin:0 0 28px;font-family:'DM Mono',monospace!important;letter-spacing:.02em}
.h-pills{display:flex;gap:8px;flex-wrap:wrap}
.pill{background:#141418;border:1px solid #2a2a35;border-radius:4px;
      padding:8px 16px;font-size:11px;color:#666;font-family:'DM Mono',monospace!important}
.pill strong{color:#d4a843;font-weight:500}

.mp{padding:36px 64px}
.sec{font-size:10px;font-weight:500;letter-spacing:.2em;text-transform:uppercase;color:#444;
     margin-bottom:14px;display:flex;align-items:center;gap:12px;font-family:'DM Mono',monospace!important}
.sec::after{content:'';flex:1;height:1px;background:#1e1e28}
.pnl{background:#0f0f13;border:1px solid #1e1e28;border-radius:8px;padding:22px;margin-bottom:14px}

.stSlider>div>div>div>div{background:#d4a843!important}
div[data-testid="stSlider"] label{color:#555!important;font-size:11px!important;
  font-family:'DM Mono',monospace!important;letter-spacing:.05em!important}
.stNumberInput input{background:#0c0c0f!important;border:1px solid #1e1e28!important;
  color:#e8e8e8!important;border-radius:6px!important;
  font-family:'DM Mono',monospace!important;font-size:13px!important}
.stNumberInput input:focus{border-color:#d4a843!important;box-shadow:none!important}
div[data-testid="stNumberInput"] label{color:#555!important;font-size:11px!important;
  font-family:'DM Mono',monospace!important;letter-spacing:.05em!important}

.stButton>button{
  background:#fff!important;color:#0c0c0f!important;
  border:none!important;border-radius:6px!important;
  font-size:12px!important;font-weight:600!important;
  letter-spacing:.15em!important;padding:15px 0!important;
  text-transform:uppercase!important;width:100%!important;
  font-family:'DM Sans',sans-serif!important;
  transition:all .15s!important;
}
.stButton>button:hover{background:#d4a843!important;color:#0c0c0f!important}

.sc{border-radius:8px;padding:20px 24px;margin:18px 0;
    display:flex;align-items:center;gap:16px}
.s-h{background:#0f1a12;border:1px solid #1a3d22}
.s-w{background:#191308;border:1px solid #3d2e00}
.s-c{background:#180c0c;border:1px solid #3d0f0f}
.sl{font-size:20px;font-weight:600;line-height:1;margin-bottom:5px;letter-spacing:-.3px}
.sd{font-size:12px;color:#555;font-family:'DM Mono',monospace!important}

.mg{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:14px 0}
.mc{background:#0f0f13;border:1px solid #1e1e28;border-radius:8px;padding:18px 20px}
.mv{font-size:30px;font-weight:600;line-height:1;margin-bottom:5px;
    letter-spacing:-1px;font-family:'DM Mono',monospace!important}
.ml{font-size:10px;color:#444;text-transform:uppercase;letter-spacing:.15em;font-weight:500}
.ms{font-size:11px;color:#2a2a35;margin-top:3px;font-family:'DM Mono',monospace!important}

.bd{background:#0f0f13;border:1px solid #1e1e28;border-radius:8px;overflow:hidden;margin:12px 0}
.br{display:flex;justify-content:space-between;align-items:center;
    padding:11px 18px;border-bottom:1px solid #141418}
.br:last-child{border-bottom:none}
.bk{font-size:11px;color:#444;font-family:'DM Mono',monospace!important}
.bv{font-size:12px;color:#e8e8e8;font-weight:500;font-family:'DM Mono',monospace!important}
.bg_{font-size:9px;font-weight:600;padding:2px 8px;border-radius:3px;
     margin-left:8px;font-family:'DM Mono',monospace!important;letter-spacing:.08em}
.bh{background:#3d0f0f;color:#f87171}
.bm{background:#3d2a00;color:#fbbf24}
.bl{background:#0f2a14;color:#4ade80}

.ic{background:#0f0f13;border:1px solid #1e1e28;border-radius:8px;overflow:hidden}
.ir{display:flex;justify-content:space-between;padding:10px 18px;
    border-bottom:1px solid #141418;font-size:12px}
.ir:last-child{border-bottom:none}
.ik{color:#444;font-family:'DM Mono',monospace!important}
.iv{color:#d4a843;font-family:'DM Mono',monospace!important;font-weight:500}
</style>
""", unsafe_allow_html=True)

THRESHOLD = 300.0

@st.cache_resource
def load_models():
    for w,r in [("best_wear_model.pkl","best_rul_model.pkl"),
                ("best_wear_model (1).pkl","best_rul_model (1).pkl")]:
        if os.path.exists(w) and os.path.exists(r):
            return joblib.load(w),joblib.load(r),True
    return None,None,False

wear_model,rul_model,models_loaded = load_models()

def predict(speed,feed,doc,cycle,vib_x,vib_y,vib_z,spindle,roughness):
    vm=(vib_x**2+vib_y**2+vib_z**2)**0.5
    mrr=speed*feed*doc
    wr=0.5*(speed/1200)+0.3*(feed/0.18)+0.2*(doc/1.0)
    sf=0.4*(spindle/70)+0.3*(vm/2.0)+0.3*(roughness/1.5)
    if models_loaded:
        X=np.array([[speed,feed,doc,vib_x,vib_y,vib_z,spindle,roughness,
                     vm,mrr,vm/(spindle+1e-6),np.log1p(cycle)]])
        wear=float(np.clip(wear_model.predict(X)[0],0,THRESHOLD-0.5))
        rul=max(0,int(rul_model.predict(X)[0]))
    else:
        wear=max(0.0,min(14*wr*sf*(cycle**1.28),THRESHOLD-0.5))
        rul=max(0,int((THRESHOLD-wear)*(cycle/max(wear,0.1))*0.85))
    return wear,rul,(wear/THRESHOLD)*100,vm,mrr,wr,sf

def bdg(val,ref):
    r=val/ref
    return ("HIGH","bh") if r>0.85 else ("MED","bm") if r>0.55 else ("LOW","bl")

def color(pct):
    return "#22c55e" if pct<60 else "#f59e0b" if pct<85 else "#ef4444"

def draw_roulette(pct, wear, rul):
    bc = color(pct)
    fig, ax = plt.subplots(figsize=(5.6,4.6), facecolor="#0c0c0f")
    ax.set_facecolor("#0c0c0f")
    ax.set_xlim(-1.55,1.55); ax.set_ylim(-1.2,1.6)
    ax.set_aspect('equal'); ax.axis('off')

    # — outer bezel
    ax.add_patch(plt.Circle((0,0),1.44,fill=False,edgecolor="#2a2a35",linewidth=10,alpha=1,zorder=1))
    ax.add_patch(plt.Circle((0,0),1.44,fill=False,edgecolor="#d4a843",linewidth=1.2,alpha=0.6,zorder=2))
    ax.add_patch(plt.Circle((0,0),1.30,fill=False,edgecolor="#d4a843",linewidth=0.6,alpha=0.2,zorder=2))

    # — roulette segments
    n=36
    for i in range(n):
        t1=180-(i*180/n); t2=180-((i+1)*180/n)
        # alternating red/black like real roulette, with zone coloring
        if i < int(n*0.60):
            base = "#8b1a1a" if i%2==0 else "#1a1a1a"
        elif i < int(n*0.85):
            base = "#7a5c00" if i%2==0 else "#1a1a1a"
        else:
            base = "#6b1a1a" if i%2==0 else "#1a1a1a"
        ax.add_patch(Wedge((0,0),1.27,t2,t1,width=0.15,
                           facecolor=base,edgecolor="#0c0c0f",linewidth=1.5,alpha=0.95))

    # — segment numbers (every 4th)
    for i in range(0,n,4):
        angle=np.radians(180-(i+0.5)*180/n)
        tx,ty=1.20*np.cos(angle),1.20*np.sin(angle)
        ax.text(tx,ty,str(i+1),ha='center',va='center',
                fontsize=5.5,color="#888",fontfamily='monospace',alpha=0.7)

    # — gold dots between segments
    for i in range(n):
        angle=np.radians(180-i*180/n)
        dx,dy=1.28*np.cos(angle),1.28*np.sin(angle)
        ax.add_patch(plt.Circle((dx,dy),0.016,color="#d4a843",zorder=6,alpha=0.8))

    # — inner track
    ax.add_patch(plt.Circle((0,0),1.10,fill=False,edgecolor="#2a2a35",linewidth=1,alpha=0.8))

    # — gauge face
    ax.add_patch(Wedge((0,0),1.06,0,180,width=0.44,
                       facecolor="#0f0f13",edgecolor="#1e1e28",linewidth=1))

    # — zone arcs on gauge face (subtle)
    for z0,z1,zcol in [(0,0.60,"#22c55e"),(0.60,0.85,"#f59e0b"),(0.85,1.0,"#ef4444")]:
        ax.add_patch(Wedge((0,0),1.05,180-z1*180,180-z0*180,
                           width=0.42,facecolor=zcol,edgecolor="none",alpha=0.06))

    # — progress arc
    if pct>0:
        ax.add_patch(Wedge((0,0),0.98,180-pct*1.8,180,
                           width=0.28,facecolor=bc,edgecolor="none",alpha=0.18))
        # bright edge
        ax.add_patch(Wedge((0,0),0.98,180-pct*1.8,180-pct*1.8+2,
                           width=0.28,facecolor=bc,edgecolor="none",alpha=0.7))

    # — tick marks
    for pm,lbl,col in [(0,"0","#333"),(60,"60","#22c55e"),(85,"85","#f59e0b"),(100,"100","#ef4444")]:
        ang=np.radians(180-pm*1.8)
        ax.plot([0.64*np.cos(ang),0.73*np.cos(ang)],
                [0.64*np.sin(ang),0.73*np.sin(ang)],color=col,lw=1.8,alpha=0.9,zorder=7)
        xl,yl=0.80*np.cos(ang),0.80*np.sin(ang)
        ax.text(xl,yl,lbl,ha='center',va='center',fontsize=7,
                color=col,fontfamily='monospace',fontweight='bold')

    # — needle
    na=np.radians(180-pct*1.8)
    nx,ny=0.58*np.cos(na),0.58*np.sin(na)
    # shadow
    ax.plot([0,nx],[0,ny],color="#000",lw=4,zorder=8,
            solid_capstyle='round',alpha=0.5)
    # needle body
    ax.plot([0,nx],[0,ny],color="#e8e8e8",lw=2.2,zorder=9,
            solid_capstyle='round',
            path_effects=[pe.withStroke(linewidth=3.5,foreground="#0c0c0f")])
    # needle tip
    ax.add_patch(plt.Circle((nx,ny),0.022,color=bc,zorder=11))
    # center hub
    ax.add_patch(plt.Circle((0,0),0.058,color="#1e1e28",zorder=10))
    ax.add_patch(plt.Circle((0,0),0.038,color="#d4a843",zorder=11))
    ax.add_patch(plt.Circle((0,0),0.018,color="#0c0c0f",zorder=12))

    # — center readout
    ax.text(0,0.26,f"{pct:.1f}%",ha='center',va='center',
            fontsize=21,fontweight='600',color=bc,fontfamily='monospace',
            path_effects=[pe.withStroke(linewidth=4,foreground="#0f0f13")])
    ax.text(0,0.10,"LIFE CONSUMED",ha='center',va='center',
            fontsize=6.5,color="#444",fontfamily='monospace',fontweight='500',
            letter_spacing=3)

    # — divider line
    ax.plot([-0.60,0.60],[-0.18,-0.18],color="#1e1e28",lw=1)
    ax.plot([0,0],[-0.18,-0.72],color="#1e1e28",lw=0.8)

    # — bottom stats
    ax.text(-0.55,-0.34,f"{wear:.1f}",ha='center',va='center',
            fontsize=16,fontweight='600',color="#d4a843",fontfamily='monospace')
    ax.text(-0.55,-0.47,"µm wear",ha='center',va='center',
            fontsize=7,color="#444",fontfamily='monospace')

    rc="#22c55e" if rul>10 else "#f59e0b" if rul>3 else "#ef4444"
    ax.text(0.55,-0.34,f"{rul}",ha='center',va='center',
            fontsize=16,fontweight='600',color=rc,fontfamily='monospace')
    ax.text(0.55,-0.47,"cycles left",ha='center',va='center',
            fontsize=7,color="#444",fontfamily='monospace')

    plt.tight_layout(pad=0.3)
    return fig

def draw_wear_chart(cycle,wear,rul,wr,sf):
    fut=list(range(1,cycle+rul+8))
    proj=[min(14*wr*sf*(c**1.28),THRESHOLD) for c in fut]
    bc=color((wear/THRESHOLD)*100)

    fig,ax=plt.subplots(figsize=(5,3.8),facecolor="#0f0f13")
    ax.set_facecolor("#0f0f13")
    ax.axhspan(0,180,alpha=0.04,color="#22c55e")
    ax.axhspan(180,255,alpha=0.04,color="#f59e0b")
    ax.axhspan(255,320,alpha=0.04,color="#ef4444")
    ax.fill_between(fut,proj,alpha=0.08,color=bc)
    ax.plot(fut,proj,color=bc,lw=2,zorder=3)
    ax.axhline(THRESHOLD,color="#ef4444",ls="--",lw=1,alpha=0.5)
    ax.axhline(THRESHOLD*0.85,color="#f59e0b",ls=":",lw=1,alpha=0.4)
    ax.axvline(cycle,color="#d4a843",ls="--",lw=0.8,alpha=0.4)
    ax.scatter([cycle],[wear],color=bc,s=65,zorder=6,edgecolors="#0f0f13",lw=2)
    ax.text(cycle+0.3,wear+13,f"{wear:.0f}µm",color=bc,fontsize=8,
            fontfamily="monospace",fontweight='500')
    if rul>0:
        ax.scatter([cycle+rul],[THRESHOLD],color="#ef4444",s=45,zorder=6,marker="x",lw=2)
        ax.text(cycle+rul+0.2,THRESHOLD-22,"FAIL",color="#ef4444",
                fontsize=7,fontfamily="monospace",alpha=0.7)

    ax.set_xlabel("Cycle",color="#444",fontsize=9,labelpad=6)
    ax.set_ylabel("Wear (µm)",color="#444",fontsize=9,labelpad=6)
    ax.tick_params(colors="#333",labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor("#1e1e28")
    ax.set_title("Wear Trajectory",color="#555",fontsize=9,
                 fontfamily='monospace',pad=8,fontweight='500')
    ax.set_ylim(0,THRESHOLD+35)
    fig.patch.set_edgecolor("#1e1e28")
    plt.tight_layout(pad=1.2)
    return fig

def draw_influence(speed,feed,doc,vm,spindle,roughness):
    fac={"Speed":0.5*(speed/1200),"Feed":0.3*(feed/0.18),"DoC":0.2*(doc/1.0),
         "Vibration":0.3*(vm/2),"Spindle":0.4*(spindle/70),"Roughness":0.3*(roughness/1.5)}
    vals=list(fac.values()); total=sum(vals)
    pcts=[v/total*100 for v in vals]
    cols=["#d4a843","#6366f1","#22c55e","#f97316","#ec4899","#38bdf8"]

    fig,ax=plt.subplots(figsize=(10.5,2.8),facecolor="#0f0f13")
    ax.set_facecolor("#0f0f13")
    bars=ax.barh(list(fac.keys()),vals,color=cols,alpha=0.85,height=0.48,edgecolor="none")
    for bar,v,p,col in zip(bars,vals,pcts,cols):
        ax.text(v+0.004,bar.get_y()+bar.get_height()/2,
                f"{p:.0f}%",va="center",color=col,fontsize=9,
                fontfamily="monospace",fontweight='500')
    ax.set_xlabel("Influence",color="#444",fontsize=9)
    ax.tick_params(colors="#555",labelsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor("#1e1e28")
    ax.set_title("Parameter Influence on Wear",color="#555",fontsize=9,
                 fontfamily='monospace',pad=8,fontweight='500')
    ax.set_xlim(0,max(vals)*1.45)
    plt.tight_layout(pad=1.2)
    return fig

# ── HERO
st.markdown(f"""
<div class="hero">
<div class="h-eyebrow">BTP · Micro-Turning · Predictive Maintenance · {'Gradient Boosting + XGBoost' if models_loaded else 'Fallback'}</div>
<div class="h-title">Tool<em>Sense</em></div>
<div class="h-sub">real-time tool wear & remaining useful life predictor</div>
<div class="h-pills">
  <div class="pill">Wear R² <strong>0.98</strong></div>
  <div class="pill">RUL R² <strong>0.95</strong></div>
  <div class="pill">Validation <strong>GroupKFold K=10</strong></div>
  <div class="pill">Threshold <strong>300µm</strong></div>
  <div class="pill">Models <strong>5 trained</strong></div>
  <div class="pill">Status <strong>{'✓ ML Active' if models_loaded else '⚠ Fallback'}</strong></div>
</div></div>
""", unsafe_allow_html=True)

st.markdown('<div class="mp">', unsafe_allow_html=True)
c1,_,c2=st.columns([4,0.25,6])

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
    st.markdown("<div style='height:6px'></div>",unsafe_allow_html=True)
    clicked=st.button("PREDICT TOOL CONDITION",use_container_width=True)

with c2:
    if clicked:
        wear,rul,pct,vm,mrr,wr,sf=predict(speed,feed,doc,cycle,vib_x,vib_y,vib_z,spindle,roughness)
        bc=color(pct)
        rc="#22c55e" if rul>10 else "#f59e0b" if rul>3 else "#ef4444"
        mc="mc-g" if pct<60 else "mc-y" if pct<85 else "mc-r"

        if pct<60:   sc,icon,lbl="s-h","●","HEALTHY"; desc=f"Tool within safe operating range — {rul} cycles remaining."
        elif pct<85: sc,icon,lbl="s-w","▲","WARNING"; desc=f"Elevated wear detected — replace within {rul} cycles."
        else:        sc,icon,lbl="s-c","■","CRITICAL — REPLACE NOW"; desc=f"Tool near failure. {rul} cycles remaining. Stop operation."

        st.markdown(f"""
        <div class="sc {sc}">
          <div style="font-size:22px;color:{bc};line-height:1">{icon}</div>
          <div>
            <div class="sl" style="color:{bc}">{lbl}</div>
            <div class="sd">{desc}</div>
          </div>
        </div>
        <div class="mg">
          <div class="mc">
            <div class="mv" style="color:#d4a843">{wear:.1f}</div>
            <div class="ml">Flank Wear</div>
            <div class="ms">µm · threshold 300</div>
          </div>
          <div class="mc">
            <div class="mv" style="color:{rc}">{rul}</div>
            <div class="ml">Remaining Cycles</div>
            <div class="ms">until failure</div>
          </div>
          <div class="mc">
            <div class="mv" style="color:{bc}">{pct:.1f}%</div>
            <div class="ml">Life Consumed</div>
            <div class="ms">{100-pct:.1f}% remaining</div>
          </div>
        </div>""", unsafe_allow_html=True)

        g1,g2=st.columns(2)
        with g1: st.pyplot(draw_roulette(pct,wear,rul),use_container_width=True)
        with g2: st.pyplot(draw_wear_chart(cycle,wear,rul,wr,sf),use_container_width=True)

        st.markdown('<div class="sec" style="margin-top:20px">Parameter Influence</div>',unsafe_allow_html=True)
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

        rec=("Continue operation normally. Next inspection after 10 cycles." if pct<60
             else f"Schedule tool replacement within the next {rul} cycles."
             if pct<85 else "Stop operation immediately. Replace tool before next cycle.")

        st.markdown(f"""
        <div style="background:#0f0f13;border:1px solid #1e1e28;border-left:3px solid {bc};
             border-radius:6px;padding:16px 20px;margin-top:14px">
          <div style="font-size:9px;color:#333;text-transform:uppercase;letter-spacing:.2em;
               font-family:'DM Mono',monospace;margin-bottom:6px">Recommendation</div>
          <div style="font-size:14px;color:#e8e8e8;font-weight:500;letter-spacing:-.2px">{rec}</div>
          <div style="font-size:10px;color:#222;margin-top:8px;font-family:'DM Mono',monospace">
          {'Gradient Boosting (wear) · XGBoost (RUL) · GroupKFold K=10' if models_loaded else 'Physics fallback active'}</div>
        </div>""", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="padding:90px 20px;text-align:center">
          <div style="font-size:11px;color:#222;letter-spacing:.2em;text-transform:uppercase;
               font-family:'DM Mono',monospace;margin-bottom:12px">Configure → Predict</div>
          <div style="font-size:28px;font-weight:600;color:#1e1e28;letter-spacing:-1px">
          Set parameters on the left.</div>
        </div>
        <div class="sec">Model Performance</div>
        <div class="ic">
          <div class="ir"><span class="ik">wear_r2</span><span class="iv">0.9802</span></div>
          <div class="ir"><span class="ik">rul_r2</span><span class="iv">0.9494</span></div>
          <div class="ir"><span class="ik">wear_rmse</span><span class="iv">13.24 µm</span></div>
          <div class="ir"><span class="ik">rul_rmse</span><span class="iv">0.895 cycles</span></div>
          <div class="ir"><span class="ik">validation</span><span class="iv">GroupKFold K=10</span></div>
          <div class="ir"><span class="ik">best_wear_model</span><span class="iv">Gradient Boosting</span></div>
          <div class="ir"><span class="ik">best_rul_model</span><span class="iv">XGBoost</span></div>
          <div class="ir"><span class="ik">failure_threshold</span><span class="iv">300 µm</span></div>
        </div>""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
