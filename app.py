import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import matplotlib.patheffects as pe
import joblib, os

st.set_page_config(page_title="ToolSense",page_icon="⚙️",layout="wide",initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@300;400;500;600&family=Barlow+Condensed:wght@500;600;700&family=DM+Mono:wght@400;500&display=swap');

*{font-family:'Barlow',sans-serif!important;box-sizing:border-box;margin:0;padding:0}
.stApp{background:#09090b}
.block-container{
  padding:0 !important;
  max-width:100% !important;
}
section[data-testid="stSidebar"]{display:none}

/* ── HERO */
.hero{
  padding:36px 48px 28px;
  border-bottom:1px solid #27272a;
}
.h-title{
  font-size:46px;font-weight:700;color:#fafafa;
  line-height:1;letter-spacing:-2px;
  font-family:'Barlow Condensed',sans-serif!important;
  text-transform:uppercase;margin-bottom:6px;
}
.h-title em{font-style:normal;color:#d4a843}
.h-sub{
  font-size:12px;color:#52525b;
  font-family:'DM Mono',monospace!important;
  letter-spacing:.03em;margin-bottom:20px;
}
.h-pills{display:flex;gap:6px;flex-wrap:wrap}
.pill{
  border:1px solid #27272a;border-radius:3px;
  padding:6px 12px;font-size:10px;color:#52525b;
  font-family:'DM Mono',monospace!important;
  letter-spacing:.04em;
}
.pill strong{color:#d4a843;font-weight:500}
.pill.on{border-color:#d4a843;color:#d4a843}

/* ── BODY */
.body{padding:28px 48px}

.sec{
  font-size:9px;font-weight:600;letter-spacing:.22em;
  text-transform:uppercase;color:#3f3f46;
  margin-bottom:10px;display:flex;align-items:center;gap:10px;
  font-family:'DM Mono',monospace!important;
}
.sec::after{content:'';flex:1;height:1px;background:#27272a}

.pnl{
  background:#0f0f12;border:1px solid #27272a;
  border-radius:6px;padding:20px 20px 14px;margin-bottom:12px;
}

/* slider overrides */
div[data-testid="stSlider"]{padding-left:0!important;padding-right:0!important}
div[data-testid="stSlider"] label{
  color:#52525b!important;font-size:10px!important;
  font-family:'DM Mono',monospace!important;
  letter-spacing:.08em!important;text-transform:uppercase!important;
}
.stSlider>div>div>div>div{background:#d4a843!important}

/* number input overrides */
div[data-testid="stNumberInput"]{padding:0!important}
div[data-testid="stNumberInput"] label{
  color:#52525b!important;font-size:10px!important;
  font-family:'DM Mono',monospace!important;
  letter-spacing:.08em!important;text-transform:uppercase!important;
}
.stNumberInput input{
  background:#09090b!important;border:1px solid #27272a!important;
  color:#e4e4e7!important;border-radius:4px!important;
  font-family:'DM Mono',monospace!important;font-size:13px!important;
}
.stNumberInput input:focus{border-color:#d4a843!important;box-shadow:none!important}

/* button */
.stButton>button{
  background:#fafafa!important;color:#09090b!important;
  border:none!important;border-radius:4px!important;
  font-size:11px!important;font-weight:700!important;
  letter-spacing:.2em!important;padding:14px 0!important;
  text-transform:uppercase!important;
  font-family:'Barlow Condensed',sans-serif!important;
}
.stButton>button:hover{background:#d4a843!important}

/* status */
.sc{border-radius:4px;padding:16px 20px;margin:12px 0;display:flex;align-items:center;gap:14px}
.s-h{background:#0c170e;border:1px solid #1a3d22}
.s-w{background:#161005;border:1px solid #3d2c00}
.s-c{background:#160808;border:1px solid #3d1010}
.sl{font-size:20px;font-weight:700;line-height:1;margin-bottom:4px;
    letter-spacing:-.2px;text-transform:uppercase;
    font-family:'Barlow Condensed',sans-serif!important}
.sd{font-size:11px;color:#52525b;font-family:'DM Mono',monospace!important}

/* metrics */
.mg{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:10px 0}
.mc{background:#0f0f12;border:1px solid #27272a;border-radius:4px;padding:15px 16px}
.mv{font-size:30px;font-weight:700;line-height:1;margin-bottom:4px;
    letter-spacing:-1px;font-family:'Barlow Condensed',sans-serif!important}
.ml{font-size:9px;color:#3f3f46;text-transform:uppercase;letter-spacing:.18em;font-weight:600}
.ms{font-size:10px;color:#27272a;margin-top:2px;font-family:'DM Mono',monospace!important}

/* table */
.bd{background:#0f0f12;border:1px solid #27272a;border-radius:4px;overflow:hidden;margin:10px 0}
.br{display:flex;justify-content:space-between;align-items:center;
    padding:9px 16px;border-bottom:1px solid #18181b}
.br:last-child{border-bottom:none}
.bk{font-size:10px;color:#52525b;font-family:'DM Mono',monospace!important}
.bv{font-size:11px;color:#e4e4e7;font-weight:500;font-family:'DM Mono',monospace!important}
.bg_{font-size:9px;font-weight:600;padding:2px 6px;border-radius:2px;
     margin-left:6px;letter-spacing:.08em;font-family:'DM Mono',monospace!important}
.bh{background:#3d0f0f;color:#fca5a5}
.bm{background:#3d2800;color:#fcd34d}
.bl{background:#0c2c14;color:#86efac}

/* info */
.ic{background:#0f0f12;border:1px solid #27272a;border-radius:4px;overflow:hidden}
.ir{display:flex;justify-content:space-between;padding:9px 16px;
    border-bottom:1px solid #18181b;font-size:11px}
.ir:last-child{border-bottom:none}
.ik{color:#52525b;font-family:'DM Mono',monospace!important}
.iv{color:#d4a843;font-family:'DM Mono',monospace!important;font-weight:500}

/* hide streamlit chrome */
#MainMenu,footer,header{visibility:hidden}
.stDeployButton{display:none}
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

wear_model,rul_model,ml = load_models()

def predict(speed,feed,doc,cycle,vx,vy,vz,spindle,rough):
    vm=(vx**2+vy**2+vz**2)**0.5
    mrr=speed*feed*doc
    wr=0.5*(speed/1200)+0.3*(feed/0.18)+0.2*(doc/1.0)
    sf=0.4*(spindle/70)+0.3*(vm/2.0)+0.3*(rough/1.5)
    if ml:
        X=np.array([[speed,feed,doc,vx,vy,vz,spindle,rough,vm,mrr,vm/(spindle+1e-6),np.log1p(cycle)]])
        wear=float(np.clip(wear_model.predict(X)[0],0,THRESHOLD-0.5))
        rul=max(0,int(rul_model.predict(X)[0]))
    else:
        wear=max(0.0,min(14*wr*sf*(cycle**1.28),THRESHOLD-0.5))
        rul=max(0,int((THRESHOLD-wear)*(cycle/max(wear,0.1))*0.85))
    return wear,rul,(wear/THRESHOLD)*100,vm,mrr,wr,sf

def gc(pct): return "#22c55e" if pct<60 else "#f59e0b" if pct<85 else "#ef4444"
def bdg(v,r):
    x=v/r
    return ("HIGH","bh") if x>0.85 else ("MED","bm") if x>0.55 else ("LOW","bl")

def roulette(pct,wear,rul):
    bc=gc(pct)
    fig,ax=plt.subplots(figsize=(5.4,4.4),facecolor="#09090b")
    ax.set_facecolor("#09090b")
    ax.set_xlim(-1.55,1.55);ax.set_ylim(-1.2,1.6)
    ax.set_aspect('equal');ax.axis('off')

    ax.add_patch(plt.Circle((0,0),1.45,fill=False,edgecolor="#27272a",linewidth=12,zorder=1))
    ax.add_patch(plt.Circle((0,0),1.45,fill=False,edgecolor="#d4a843",linewidth=1.4,alpha=0.65,zorder=2))
    ax.add_patch(plt.Circle((0,0),1.31,fill=False,edgecolor="#d4a843",linewidth=0.5,alpha=0.18,zorder=2))

    n=36
    for i in range(n):
        t1=180-(i*180/n);t2=180-((i+1)*180/n)
        if i<int(n*0.60):   c="#7a1515" if i%2==0 else "#1c1c20"
        elif i<int(n*0.85): c="#6b4e00" if i%2==0 else "#1c1c20"
        else:                c="#5c1212" if i%2==0 else "#1c1c20"
        ax.add_patch(Wedge((0,0),1.28,t2,t1,width=0.15,facecolor=c,edgecolor="#09090b",linewidth=1.8))

    for i in range(0,n,6):
        ang=np.radians(180-(i+0.5)*180/n)
        ax.text(1.20*np.cos(ang),1.20*np.sin(ang),str(i+1),
                ha='center',va='center',fontsize=5.5,color="#555",fontfamily='monospace')
    for i in range(n):
        ang=np.radians(180-i*180/n)
        ax.add_patch(plt.Circle((1.29*np.cos(ang),1.29*np.sin(ang)),0.017,color="#d4a843",zorder=6,alpha=0.8))

    ax.add_patch(plt.Circle((0,0),1.11,fill=False,edgecolor="#27272a",linewidth=1))
    ax.add_patch(Wedge((0,0),1.07,0,180,width=0.46,facecolor="#0f0f12",edgecolor="#27272a",linewidth=1))

    for z0,z1,zc in [(0,.60,"#22c55e"),(.60,.85,"#f59e0b"),(.85,1.,"#ef4444")]:
        ax.add_patch(Wedge((0,0),1.06,180-z1*180,180-z0*180,width=0.44,facecolor=zc,edgecolor="none",alpha=0.05))

    if pct>0:
        ax.add_patch(Wedge((0,0),0.99,180-pct*1.8,180,width=0.30,facecolor=bc,edgecolor="none",alpha=0.16))
        ax.add_patch(Wedge((0,0),0.99,180-pct*1.8,180-pct*1.8+1.8,width=0.30,facecolor=bc,edgecolor="none",alpha=0.8))

    for pm,lbl,tc in [(0,"0","#3f3f46"),(60,"60%","#22c55e"),(85,"85%","#f59e0b"),(100,"100%","#ef4444")]:
        ang=np.radians(180-pm*1.8)
        ax.plot([0.64*np.cos(ang),0.74*np.cos(ang)],[0.64*np.sin(ang),0.74*np.sin(ang)],color=tc,lw=1.8,alpha=0.9,zorder=7)
        ax.text(0.81*np.cos(ang),0.81*np.sin(ang),lbl,ha='center',va='center',fontsize=7,color=tc,fontfamily='monospace',fontweight='bold')

    na=np.radians(180-pct*1.8);nx,ny=0.58*np.cos(na),0.58*np.sin(na)
    ax.plot([0,nx],[0,ny],color="#000",lw=5,zorder=8,solid_capstyle='round',alpha=0.5)
    ax.plot([0,nx],[0,ny],color="#e4e4e7",lw=2,zorder=9,solid_capstyle='round',
            path_effects=[pe.withStroke(linewidth=3,foreground="#09090b")])
    ax.add_patch(plt.Circle((nx,ny),0.020,color=bc,zorder=11))
    ax.add_patch(plt.Circle((0,0),0.058,color="#27272a",zorder=10))
    ax.add_patch(plt.Circle((0,0),0.038,color="#d4a843",zorder=11))
    ax.add_patch(plt.Circle((0,0),0.016,color="#09090b",zorder=12))

    ax.text(0,0.26,f"{pct:.1f}%",ha='center',va='center',fontsize=21,fontweight='700',
            color=bc,fontfamily='monospace',path_effects=[pe.withStroke(linewidth=4,foreground="#0f0f12")])
    ax.text(0,0.10,"LIFE CONSUMED",ha='center',va='center',fontsize=6.5,color="#3f3f46",fontfamily='monospace')
    ax.plot([-0.60,0.60],[-0.19,-0.19],color="#27272a",lw=0.8)
    ax.plot([0,0],[-0.19,-0.72],color="#27272a",lw=0.8)

    ax.text(-0.52,-0.34,f"{wear:.1f}",ha='center',va='center',fontsize=15,fontweight='700',color="#d4a843",fontfamily='monospace')
    ax.text(-0.52,-0.48,"µm wear",ha='center',va='center',fontsize=7,color="#3f3f46",fontfamily='monospace')
    rc="#22c55e" if rul>10 else "#f59e0b" if rul>3 else "#ef4444"
    ax.text(0.52,-0.34,f"{rul}",ha='center',va='center',fontsize=15,fontweight='700',color=rc,fontfamily='monospace')
    ax.text(0.52,-0.48,"cycles left",ha='center',va='center',fontsize=7,color="#3f3f46",fontfamily='monospace')

    plt.tight_layout(pad=0.3)
    return fig

def wear_chart(cycle,wear,rul,wr,sf):
    fut=list(range(1,cycle+rul+8))
    proj=[min(14*wr*sf*(c**1.28),THRESHOLD) for c in fut]
    bc=gc((wear/THRESHOLD)*100)
    fig,ax=plt.subplots(figsize=(5,3.8),facecolor="#0f0f12")
    ax.set_facecolor("#0f0f12")
    ax.axhspan(0,180,alpha=0.04,color="#22c55e")
    ax.axhspan(180,255,alpha=0.04,color="#f59e0b")
    ax.axhspan(255,320,alpha=0.04,color="#ef4444")
    ax.fill_between(fut,proj,alpha=0.08,color=bc)
    ax.plot(fut,proj,color=bc,lw=2.2,zorder=3)
    ax.axhline(THRESHOLD,color="#ef4444",ls="--",lw=1,alpha=0.5)
    ax.axhline(THRESHOLD*0.85,color="#f59e0b",ls=":",lw=1,alpha=0.4)
    ax.axvline(cycle,color="#d4a843",ls="--",lw=0.8,alpha=0.35)
    ax.scatter([cycle],[wear],color=bc,s=65,zorder=6,edgecolors="#0f0f12",lw=2)
    ax.text(cycle+0.3,wear+13,f"{wear:.0f}µm",color=bc,fontsize=8.5,fontfamily="monospace")
    if rul>0:
        ax.scatter([cycle+rul],[THRESHOLD],color="#ef4444",s=45,zorder=6,marker="x",lw=2.5)
        ax.text(cycle+rul+0.2,THRESHOLD-24,"FAIL",color="#ef4444",fontsize=7.5,fontfamily="monospace",alpha=0.8)
    ax.set_xlabel("Cycle",color="#52525b",fontsize=9)
    ax.set_ylabel("Wear (µm)",color="#52525b",fontsize=9)
    ax.tick_params(colors="#3f3f46",labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor("#27272a")
    ax.set_title("Wear Trajectory",color="#52525b",fontsize=9,fontfamily='monospace',pad=8)
    ax.set_ylim(0,THRESHOLD+35)
    plt.tight_layout(pad=1.2)
    return fig

def influence(speed,feed,doc,vm,spindle,rough):
    fac={"Speed":0.5*(speed/1200),"Feed":0.3*(feed/0.18),"DoC":0.2*(doc/1.0),
         "Vibration":0.3*(vm/2),"Spindle":0.4*(spindle/70),"Roughness":0.3*(rough/1.5)}
    vals=list(fac.values()); total=sum(vals); pcts=[v/total*100 for v in vals]
    cols=["#d4a843","#6366f1","#22c55e","#f97316","#ec4899","#38bdf8"]
    fig,ax=plt.subplots(figsize=(10,2.6),facecolor="#0f0f12")
    ax.set_facecolor("#0f0f12")
    bars=ax.barh(list(fac.keys()),vals,color=cols,alpha=0.85,height=0.46,edgecolor="none")
    for bar,v,p,c in zip(bars,vals,pcts,cols):
        ax.text(v+0.004,bar.get_y()+bar.get_height()/2,f"{p:.0f}%",
                va="center",color=c,fontsize=9,fontfamily="monospace")
    ax.set_xlabel("Influence",color="#52525b",fontsize=9)
    ax.tick_params(colors="#52525b",labelsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor("#27272a")
    ax.set_title("Parameter Influence on Wear",color="#52525b",fontsize=9,fontfamily='monospace',pad=8)
    ax.set_xlim(0,max(vals)*1.5)
    plt.tight_layout(pad=1.2)
    return fig

# ── HERO
st.markdown(f"""
<div class="hero">
  <div class="h-title">Tool<em>Sense</em></div>
  <div class="h-sub">tool wear monitoring & remaining useful life prediction</div>
  <div class="h-pills">
    <div class="pill">Wear R² <strong>0.9802</strong></div>
    <div class="pill">RUL R² <strong>0.9494</strong></div>
    <div class="pill">CV <strong>GroupKFold K=10</strong></div>
    <div class="pill">RMSE <strong>13.24 µm</strong></div>
    <div class="pill">Failure threshold <strong>300 µm</strong></div>
    <div class="pill {'on' if ml else ''}">{'✓ ML Active' if ml else '⚠ Fallback'}</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="body">', unsafe_allow_html=True)
c1,_,c2=st.columns([4,0.2,6])

with c1:
    st.markdown('<div class="sec">Cutting Parameters</div>',unsafe_allow_html=True)
    st.markdown('<div class="pnl">',unsafe_allow_html=True)
    speed=st.slider("Spindle speed — rpm",500,2000,1000,step=50)
    feed=st.slider("Feed rate — mm/rev",0.05,0.30,0.13,step=0.01)
    doc=st.slider("Depth of cut — mm",0.1,2.0,0.8,step=0.1)
    cycle=st.slider("Cycle number",1,100,8)
    st.markdown('</div>',unsafe_allow_html=True)
    st.markdown('<div class="sec">Sensor Readings</div>',unsafe_allow_html=True)
    st.markdown('<div class="pnl">',unsafe_allow_html=True)
    s1,s2=st.columns(2)
    with s1:
        vx=st.number_input("Vibration X — mm/s²",value=1.35,step=0.05,format="%.2f")
        vz=st.number_input("Vibration Z — mm/s²",value=0.75,step=0.05,format="%.2f")
        rough=st.number_input("Surface roughness — µm",value=1.4,step=0.1,format="%.1f")
    with s2:
        vy=st.number_input("Vibration Y — mm/s²",value=1.20,step=0.05,format="%.2f")
        spindle=st.slider("Spindle load — %",10,100,61)
    st.markdown('</div>',unsafe_allow_html=True)
    st.markdown("<div style='height:6px'></div>",unsafe_allow_html=True)
    clicked=st.button("Run Prediction",use_container_width=True)

with c2:
    if clicked:
        wear,rul,pct,vm,mrr,wr,sf=predict(speed,feed,doc,cycle,vx,vy,vz,spindle,rough)
        bc=gc(pct)
        rc="#22c55e" if rul>10 else "#f59e0b" if rul>3 else "#ef4444"

        if pct<60:   sc,icon,lbl="s-h","●","Healthy"; desc=f"Within safe operating limits — {rul} cycles remaining before replacement is required."
        elif pct<85: sc,icon,lbl="s-w","▲","Warning — Plan Replacement"; desc=f"Elevated wear detected. Replace within {rul} cycles to prevent failure."
        else:        sc,icon,lbl="s-c","■","Critical — Replace Now"; desc=f"Tool near failure. Only {rul} cycles remaining. Stop operation immediately."

        st.markdown(f"""
        <div class="sc {sc}">
          <div style="font-size:18px;color:{bc}">{icon}</div>
          <div>
            <div class="sl" style="color:{bc}">{lbl}</div>
            <div class="sd">{desc}</div>
          </div>
        </div>
        <div class="mg">
          <div class="mc">
            <div class="mv" style="color:#d4a843">{wear:.1f}</div>
            <div class="ml">Flank Wear — µm</div>
            <div class="ms">failure at 300 µm</div>
          </div>
          <div class="mc">
            <div class="mv" style="color:{rc}">{rul}</div>
            <div class="ml">Remaining Useful Life</div>
            <div class="ms">cycles until failure</div>
          </div>
          <div class="mc">
            <div class="mv" style="color:{bc}">{pct:.1f}%</div>
            <div class="ml">Life Consumed</div>
            <div class="ms">{100-pct:.1f}% remaining</div>
          </div>
        </div>""", unsafe_allow_html=True)

        g1,g2=st.columns(2)
        with g1: st.pyplot(roulette(pct,wear,rul),use_container_width=True)
        with g2: st.pyplot(wear_chart(cycle,wear,rul,wr,sf),use_container_width=True)

        st.markdown('<div class="sec" style="margin-top:16px">Parameter Influence</div>',unsafe_allow_html=True)
        st.pyplot(influence(speed,feed,doc,vm,spindle,rough),use_container_width=True)

        rows=[("speed_rpm",f"{speed} rpm",*bdg(speed,1200)),
              ("feed_mm_rev",f"{feed:.2f} mm/rev",*bdg(feed,0.18)),
              ("depth_of_cut_mm",f"{doc:.1f} mm",*bdg(doc,1.0)),
              ("vibration_magnitude",f"{vm:.3f} mm/s²",*bdg(vm,2.0)),
              ("spindle_load",f"{spindle}%",*bdg(spindle,70)),
              ("surface_roughness",f"{roughness:.1f} µm",*bdg(rough,1.5)),
              ("mrr",f"{mrr:.1f} mm³/min",*bdg(mrr,150))]
        html="".join([f'<div class="br"><span class="bk">{r[0]}</span><div><span class="bv">{r[1]}</span><span class="bg_ {r[3]}">{r[2]}</span></div></div>' for r in rows])
        st.markdown(f'<div class="sec" style="margin-top:14px">Feature Breakdown</div><div class="bd">{html}</div>',unsafe_allow_html=True)

        rec=("Continue operation. Schedule inspection in 10 cycles." if pct<60
             else f"Plan tool replacement within {rul} cycles."
             if pct<85 else "Halt operation. Replace tool before next cycle.")
        st.markdown(f"""
        <div style="background:#0f0f12;border:1px solid #27272a;border-left:3px solid {bc};
             border-radius:4px;padding:14px 18px;margin-top:12px">
          <div style="font-size:9px;color:#3f3f46;text-transform:uppercase;letter-spacing:.2em;
               font-family:'DM Mono',monospace;margin-bottom:5px">Recommendation</div>
          <div style="font-size:14px;color:#e4e4e7;font-weight:500">{rec}</div>
          <div style="font-size:10px;color:#27272a;margin-top:6px;font-family:'DM Mono',monospace">
          {'Gradient Boosting (wear) · XGBoost (RUL) · holdout R² 0.97' if ml else 'Physics fallback — upload PKL files to enable ML'}</div>
        </div>""", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="padding:100px 20px;text-align:center">
          <div style="font-size:32px;font-weight:700;color:#18181b;letter-spacing:-1px;
               font-family:'Barlow Condensed',sans-serif">Set parameters.<br>Run prediction.</div>
        </div>
        <div class="sec">Model Performance</div>
        <div class="ic">
          <div class="ir"><span class="ik">wear_r2</span><span class="iv">0.9802</span></div>
          <div class="ir"><span class="ik">rul_r2</span><span class="iv">0.9494</span></div>
          <div class="ir"><span class="ik">wear_rmse</span><span class="iv">13.24 µm</span></div>
          <div class="ir"><span class="ik">rul_rmse</span><span class="iv">0.895 cycles</span></div>
          <div class="ir"><span class="ik">holdout_r2</span><span class="iv">0.9749</span></div>
          <div class="ir"><span class="ik">validation</span><span class="iv">GroupKFold K=10</span></div>
          <div class="ir"><span class="ik">wear_model</span><span class="iv">Gradient Boosting</span></div>
          <div class="ir"><span class="ik">rul_model</span><span class="iv">XGBoost</span></div>
          <div class="ir"><span class="ik">failure_threshold</span><span class="iv">300 µm</span></div>
        </div>""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
