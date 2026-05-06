import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Arc
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
import joblib, os

st.set_page_config(page_title="ToolSense",page_icon="⚙️",layout="wide",initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@300;400;500;600&family=Barlow+Condensed:wght@500;600;700&family=DM+Mono:wght@400;500&display=swap');

*{font-family:'Barlow',sans-serif!important;box-sizing:border-box}
.stApp{background:#f4f4f5}
.block-container{padding:0!important;max-width:100%!important}
#MainMenu,footer,header,.stDeployButton{visibility:hidden;display:none}

/* HERO */
.hero{
  background:#ffffff;
  border-bottom:1px solid #e4e4e7;
  padding:36px 52px 28px;
}
.h-title{
  font-size:48px;font-weight:700;color:#09090b;
  line-height:1;letter-spacing:-2.5px;
  font-family:'Barlow Condensed',sans-serif!important;
  text-transform:uppercase;margin-bottom:5px;
}
.h-title em{font-style:normal;color:#b45309}
.h-sub{
  font-size:12px;color:#a1a1aa;
  font-family:'DM Mono',monospace!important;
  letter-spacing:.04em;margin-bottom:20px;
}
.h-pills{display:flex;gap:6px;flex-wrap:wrap}
.pill{
  background:#f4f4f5;border:1px solid #e4e4e7;border-radius:3px;
  padding:6px 12px;font-size:10px;color:#71717a;
  font-family:'DM Mono',monospace!important;letter-spacing:.04em;
}
.pill strong{color:#b45309;font-weight:500}
.pill.on{border-color:#b45309;color:#b45309;background:#fffbeb}

/* BODY */
.body{padding:28px 52px}

.sec{
  font-size:9px;font-weight:600;letter-spacing:.22em;text-transform:uppercase;
  color:#a1a1aa;margin-bottom:10px;display:flex;align-items:center;gap:10px;
  font-family:'DM Mono',monospace!important;
}
.sec::after{content:'';flex:1;height:1px;background:#e4e4e7}

.pnl{
  background:#ffffff;border:1px solid #e4e4e7;
  border-radius:8px;padding:20px;margin-bottom:12px;
}

/* sliders */
div[data-testid="stSlider"]{padding:0!important}
div[data-testid="stSlider"] label{
  color:#71717a!important;font-size:10px!important;
  font-family:'DM Mono',monospace!important;
  letter-spacing:.08em!important;text-transform:uppercase!important;
}
.stSlider>div>div>div>div{background:#b45309!important}

/* number inputs */
div[data-testid="stNumberInput"]{padding:0!important}
div[data-testid="stNumberInput"] label{
  color:#71717a!important;font-size:10px!important;
  font-family:'DM Mono',monospace!important;
  letter-spacing:.08em!important;text-transform:uppercase!important;
}
.stNumberInput input{
  background:#f9f9f9!important;border:1px solid #e4e4e7!important;
  color:#09090b!important;border-radius:4px!important;
  font-family:'DM Mono',monospace!important;font-size:13px!important;
}
.stNumberInput input:focus{border-color:#b45309!important;box-shadow:none!important;outline:none!important}

/* button */
.stButton>button{
  background:#09090b!important;color:#fafafa!important;
  border:none!important;border-radius:4px!important;
  font-size:11px!important;font-weight:700!important;
  letter-spacing:.2em!important;padding:15px 0!important;
  text-transform:uppercase!important;
  font-family:'Barlow Condensed',sans-serif!important;
}
.stButton>button:hover{background:#b45309!important}

/* status */
.sc{border-radius:6px;padding:18px 22px;margin:12px 0;display:flex;align-items:center;gap:14px}
.s-h{background:#f0fdf4;border:1px solid #bbf7d0}
.s-w{background:#fffbeb;border:1px solid #fde68a}
.s-c{background:#fff1f2;border:1px solid #fecdd3}
.sl{font-size:20px;font-weight:700;line-height:1;margin-bottom:4px;
    letter-spacing:-.3px;text-transform:uppercase;
    font-family:'Barlow Condensed',sans-serif!important}
.sd{font-size:11px;color:#71717a;font-family:'DM Mono',monospace!important}

/* metrics */
.mg{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:10px 0}
.mc{background:#ffffff;border:1px solid #e4e4e7;border-radius:6px;padding:16px 18px;
    box-shadow:0 1px 3px rgba(0,0,0,0.04)}
.mv{font-size:32px;font-weight:700;line-height:1;margin-bottom:4px;
    letter-spacing:-1.5px;font-family:'Barlow Condensed',sans-serif!important}
.ml{font-size:9px;color:#a1a1aa;text-transform:uppercase;letter-spacing:.2em;font-weight:600}
.ms{font-size:10px;color:#d4d4d8;margin-top:2px;font-family:'DM Mono',monospace!important}

/* table */
.bd{background:#ffffff;border:1px solid #e4e4e7;border-radius:6px;overflow:hidden;margin:10px 0;
    box-shadow:0 1px 3px rgba(0,0,0,0.04)}
.br{display:flex;justify-content:space-between;align-items:center;
    padding:10px 16px;border-bottom:1px solid #f4f4f5}
.br:nth-child(even){background:#fafafa}
.br:last-child{border-bottom:none}
.bk{font-size:10px;color:#a1a1aa;font-family:'DM Mono',monospace!important}
.bv{font-size:11px;color:#18181b;font-weight:500;font-family:'DM Mono',monospace!important}
.bg_{font-size:9px;font-weight:600;padding:2px 7px;border-radius:2px;
     margin-left:8px;letter-spacing:.08em;font-family:'DM Mono',monospace!important}
.bh{background:#fee2e2;color:#dc2626}
.bm{background:#fef3c7;color:#d97706}
.bl{background:#dcfce7;color:#16a34a}

/* info */
.ic{background:#ffffff;border:1px solid #e4e4e7;border-radius:6px;overflow:hidden;
    box-shadow:0 1px 3px rgba(0,0,0,0.04)}
.ir{display:flex;justify-content:space-between;padding:10px 16px;
    border-bottom:1px solid #f4f4f5;font-size:11px}
.ir:nth-child(even){background:#fafafa}
.ir:last-child{border-bottom:none}
.ik{color:#a1a1aa;font-family:'DM Mono',monospace!important}
.iv{color:#b45309;font-family:'DM Mono',monospace!important;font-weight:500}
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

wear_model,rul_model,ml=load_models()

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

def gc(pct): return "#16a34a" if pct<60 else "#d97706" if pct<85 else "#dc2626"
def bdg(v,r):
    x=v/r
    return ("HIGH","bh") if x>0.85 else ("MED","bm") if x>0.55 else ("LOW","bl")

def draw_roulette(pct,wear,rul):
    bc=gc(pct)
    fig,ax=plt.subplots(figsize=(5.6,5.0),facecolor="#ffffff")
    ax.set_facecolor("#ffffff")
    ax.set_xlim(-1.6,1.6);ax.set_ylim(-1.3,1.7)
    ax.set_aspect('equal');ax.axis('off')

    # ── outer shadow ring (fake drop shadow)
    shadow=plt.Circle((0.015,-0.015),1.48,color="#e4e4e7",zorder=0)
    ax.add_patch(shadow)

    # ── outer bezel - rich dark
    ax.add_patch(plt.Circle((0,0),1.46,color="#18181b",zorder=1))
    ax.add_patch(plt.Circle((0,0),1.44,fill=False,edgecolor="#d4a843",linewidth=2,alpha=0.9,zorder=3))
    ax.add_patch(plt.Circle((0,0),1.38,fill=False,edgecolor="#d4a843",linewidth=0.5,alpha=0.3,zorder=3))

    # ── roulette segments - proper casino alternating
    n=36
    # casino red numbers: 1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36
    red_nums={1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36}
    for i in range(n):
        t1=180-(i*180/n); t2=180-((i+1)*180/n)
        num=i+1
        # zone-tinted casino colors
        if i<int(n*0.60):
            c="#9b1c1c" if num in red_nums else "#1c1c1e"
        elif i<int(n*0.85):
            c="#78350f" if num in red_nums else "#1c1c1e"
        else:
            c="#7f1d1d" if num in red_nums else "#1c1c1e"
        ax.add_patch(Wedge((0,0),1.35,t2,t1,width=0.18,
                           facecolor=c,edgecolor="#18181b",linewidth=1.5,zorder=2))

    # ── gold separator lines between segments
    for i in range(n):
        ang=np.radians(180-i*180/n)
        x1,y1=1.17*np.cos(ang),1.17*np.sin(ang)
        x2,y2=1.35*np.cos(ang),1.35*np.sin(ang)
        ax.plot([x1,x2],[y1,y2],color="#d4a843",lw=0.6,alpha=0.5,zorder=4)

    # ── segment numbers
    for i in range(n):
        ang=np.radians(180-(i+0.5)*180/n)
        ax.text(1.26*np.cos(ang),1.26*np.sin(ang),str(i+1),
                ha='center',va='center',fontsize=5,color="#f5f5f5",
                fontfamily='monospace',fontweight='bold',zorder=5)

    # ── gold diamond markers at key positions
    for i in range(0,n,3):
        ang=np.radians(180-i*180/n)
        ax.plot(1.165*np.cos(ang),1.165*np.sin(ang),'D',
                color="#d4a843",markersize=3,zorder=6,alpha=0.9)

    # ── ball track ring
    ax.add_patch(plt.Circle((0,0),1.15,fill=False,edgecolor="#3f3f46",linewidth=2,zorder=4))
    ax.add_patch(plt.Circle((0,0),1.13,fill=False,edgecolor="#52525b",linewidth=0.5,zorder=4))

    # ── gauge face - clean white
    ax.add_patch(Wedge((0,0),1.10,0,180,width=0.50,
                       facecolor="#fafafa",edgecolor="#e4e4e7",linewidth=1,zorder=5))

    # ── zone color bands on gauge (very subtle)
    for z0,z1,zc,za in [(0,.60,"#16a34a",0.08),(.60,.85,"#d97706",0.08),(.85,1.,"#dc2626",0.10)]:
        ax.add_patch(Wedge((0,0),1.09,180-z1*180,180-z0*180,
                           width=0.48,facecolor=zc,edgecolor="none",alpha=za,zorder=6))

    # ── gauge arc track
    ax.add_patch(plt.Circle((0,0),0.88,fill=False,edgecolor="#e4e4e7",linewidth=6,zorder=6,alpha=0.8))

    # ── progress arc (colored)
    if pct>0:
        n_arc=max(2,int(pct*1.8))
        thetas=np.linspace(np.radians(180),np.radians(180-pct*1.8),n_arc+1)
        xs=0.88*np.cos(thetas); ys=0.88*np.sin(thetas)
        from matplotlib.lines import Line2D
        ax.plot(xs,ys,color=bc,linewidth=6,solid_capstyle='round',zorder=7,alpha=0.9)

    # ── tick marks
    for pm,lbl,tc in [(0,"0","#d4d4d8"),(60,"60%","#16a34a"),(85,"85%","#d97706"),(100,"100%","#dc2626")]:
        ang=np.radians(180-pm*1.8)
        ax.plot([0.75*np.cos(ang),0.84*np.cos(ang)],
                [0.75*np.sin(ang),0.84*np.sin(ang)],
                color=tc,lw=2,alpha=1,zorder=8)
        ax.text(0.70*np.cos(ang),0.70*np.sin(ang),lbl,
                ha='center',va='center',fontsize=7,color=tc,
                fontfamily='monospace',fontweight='bold',zorder=8)

    # ── needle - elegant thin
    na=np.radians(180-pct*1.8)
    nx,ny=0.74*np.cos(na),0.74*np.sin(na)
    # needle tail
    ax.plot([0,-0.12*np.cos(na)],[0,-0.12*np.sin(na)],
            color="#71717a",lw=3,zorder=9,solid_capstyle='round')
    # needle body
    ax.plot([0,nx],[0,ny],color="#18181b",lw=2,zorder=10,
            solid_capstyle='round',
            path_effects=[pe.withStroke(linewidth=3.5,foreground="#fafafa")])
    # needle tip dot
    ax.add_patch(plt.Circle((nx,ny),0.020,color=bc,zorder=12,
                             path_effects=[pe.withStroke(linewidth=2,foreground="#ffffff")]))

    # ── center hub
    ax.add_patch(plt.Circle((0,0),0.065,color="#18181b",zorder=11))
    ax.add_patch(plt.Circle((0,0),0.042,color="#d4a843",zorder=12))
    ax.add_patch(plt.Circle((0,0),0.020,color="#18181b",zorder=13))

    # ── center pct readout
    ax.text(0,0.28,f"{pct:.1f}%",ha='center',va='center',
            fontsize=22,fontweight='700',color=bc,fontfamily='monospace',zorder=14,
            path_effects=[pe.withStroke(linewidth=5,foreground="#fafafa")])
    ax.text(0,0.13,"LIFE CONSUMED",ha='center',va='center',
            fontsize=6,color="#a1a1aa",fontfamily='monospace',
            fontweight='600',zorder=14,letter_spacing=2)

    # ── horizontal + vertical dividers
    ax.plot([-0.62,0.62],[-0.20,-0.20],color="#e4e4e7",lw=1,zorder=14)
    ax.plot([0,0],[-0.20,-0.80],color="#e4e4e7",lw=0.8,zorder=14)

    # ── bottom stat blocks
    # left - wear
    ax.add_patch(plt.Rectangle((-0.62,-0.82),0.58,0.56,
                               facecolor="#f9f9f9",edgecolor="#e4e4e7",linewidth=0.8,
                               zorder=13,transform=ax.transData))
    ax.text(-0.33,-0.38,f"{wear:.1f}",ha='center',va='center',
            fontsize=17,fontweight='700',color="#b45309",fontfamily='monospace',zorder=14)
    ax.text(-0.33,-0.54,"µm",ha='center',va='center',
            fontsize=8,color="#a1a1aa",fontfamily='monospace',zorder=14)
    ax.text(-0.33,-0.67,"FLANK WEAR",ha='center',va='center',
            fontsize=6,color="#d4d4d8",fontfamily='monospace',fontweight='600',zorder=14)

    # right - rul
    rc="#16a34a" if rul>10 else "#d97706" if rul>3 else "#dc2626"
    ax.add_patch(plt.Rectangle((0.04,-0.82),0.58,0.56,
                               facecolor="#f9f9f9",edgecolor="#e4e4e7",linewidth=0.8,
                               zorder=13,transform=ax.transData))
    ax.text(0.33,-0.38,f"{rul}",ha='center',va='center',
            fontsize=17,fontweight='700',color=rc,fontfamily='monospace',zorder=14)
    ax.text(0.33,-0.54,"cycles",ha='center',va='center',
            fontsize=8,color="#a1a1aa",fontfamily='monospace',zorder=14)
    ax.text(0.33,-0.67,"RUL",ha='center',va='center',
            fontsize=6,color="#d4d4d8",fontfamily='monospace',fontweight='600',zorder=14)

    plt.tight_layout(pad=0.2)
    return fig

def draw_wear_chart(cycle,wear,rul,wr,sf):
    fut=list(range(1,cycle+rul+8))
    proj=[min(14*wr*sf*(c**1.28),THRESHOLD) for c in fut]
    bc=gc((wear/THRESHOLD)*100)
    fig,ax=plt.subplots(figsize=(5.2,4.0),facecolor="#ffffff")
    ax.set_facecolor("#ffffff")
    ax.axhspan(0,180,alpha=0.04,color="#16a34a")
    ax.axhspan(180,255,alpha=0.04,color="#d97706")
    ax.axhspan(255,320,alpha=0.04,color="#dc2626")

    # subtle grid
    ax.yaxis.grid(True,color="#f4f4f5",linewidth=1,zorder=0)
    ax.set_axisbelow(True)

    ax.fill_between(fut,proj,alpha=0.07,color=bc)
    ax.plot(fut,proj,color=bc,lw=2.5,zorder=3)

    ax.axhline(THRESHOLD,color="#dc2626",ls="--",lw=1.2,alpha=0.5,label="Failure 300µm")
    ax.axhline(THRESHOLD*0.85,color="#d97706",ls=":",lw=1,alpha=0.4,label="Warning 255µm")
    ax.axhline(180,color="#16a34a",ls=":",lw=0.8,alpha=0.35,label="Caution 180µm")

    ax.axvline(cycle,color="#b45309",ls="--",lw=0.8,alpha=0.4)
    ax.scatter([cycle],[wear],color=bc,s=75,zorder=6,
               edgecolors="#ffffff",lw=2.5)
    ax.text(cycle+0.3,wear+14,f"{wear:.0f}µm",color=bc,fontsize=8.5,
            fontfamily="monospace",fontweight='600')

    if rul>0:
        ax.scatter([cycle+rul],[THRESHOLD],color="#dc2626",s=55,
                   zorder=6,marker="x",lw=2.5)
        ax.text(cycle+rul+0.2,THRESHOLD-24,"FAIL",color="#dc2626",
                fontsize=7.5,fontfamily="monospace",fontweight='600',alpha=0.8)

    ax.set_xlabel("Cycle",color="#71717a",fontsize=9,labelpad=6)
    ax.set_ylabel("Flank Wear (µm)",color="#71717a",fontsize=9,labelpad=6)
    ax.tick_params(colors="#a1a1aa",labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor("#e4e4e7")
    legend=ax.legend(fontsize=7,facecolor="#ffffff",labelcolor="#71717a",
                     edgecolor="#e4e4e7",loc="upper left",framealpha=1)
    ax.set_title("Wear Trajectory",color="#71717a",fontsize=9,
                 fontfamily='monospace',pad=8,fontweight='500')
    ax.set_ylim(0,THRESHOLD+40)
    plt.tight_layout(pad=1.2)
    return fig

def draw_influence(speed,feed,doc,vm,spindle,rough):
    fac={"Speed":0.5*(speed/1200),"Feed":0.3*(feed/0.18),"DoC":0.2*(doc/1.0),
         "Vibration":0.3*(vm/2),"Spindle":0.4*(spindle/70),"Roughness":0.3*(rough/1.5)}
    vals=list(fac.values()); total=sum(vals); pcts=[v/total*100 for v in vals]
    cols=["#b45309","#6366f1","#16a34a","#f97316","#ec4899","#0ea5e9"]
    fig,ax=plt.subplots(figsize=(10,2.8),facecolor="#ffffff")
    ax.set_facecolor("#ffffff")
    ax.xaxis.grid(True,color="#f4f4f5",linewidth=1,zorder=0)
    ax.set_axisbelow(True)
    bars=ax.barh(list(fac.keys()),vals,color=cols,alpha=0.85,height=0.50,edgecolor="none",zorder=3)
    for bar,v,p,c in zip(bars,vals,pcts,cols):
        ax.text(v+0.005,bar.get_y()+bar.get_height()/2,f"{p:.0f}%",
                va="center",color=c,fontsize=9.5,fontfamily="monospace",fontweight='600')
    ax.set_xlabel("Influence Score",color="#71717a",fontsize=9)
    ax.tick_params(colors="#a1a1aa",labelsize=9.5)
    for sp in ax.spines.values(): sp.set_edgecolor("#e4e4e7")
    ax.set_title("Parameter Influence on Wear",color="#71717a",fontsize=9,
                 fontfamily='monospace',pad=8,fontweight='500')
    ax.set_xlim(0,max(vals)*1.5)
    plt.tight_layout(pad=1.2)
    return fig

# ── HERO
st.markdown(f"""
<div class="hero">
  <div class="h-title">Tool<em>Sense</em></div>
  <div class="h-sub">tool wear monitoring · remaining useful life prediction</div>
  <div class="h-pills">
    <div class="pill">Wear R² <strong>0.9802</strong></div>
    <div class="pill">RUL R² <strong>0.9494</strong></div>
    <div class="pill">CV <strong>GroupKFold K=10</strong></div>
    <div class="pill">Holdout R² <strong>0.9749</strong></div>
    <div class="pill">Threshold <strong>300 µm</strong></div>
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
        rc="#16a34a" if rul>10 else "#d97706" if rul>3 else "#dc2626"

        if pct<60:   sc,icon,lbl="s-h","●","Healthy"
        elif pct<85: sc,icon,lbl="s-w","▲","Warning — Plan Replacement"
        else:        sc,icon,lbl="s-c","■","Critical — Replace Now"

        desc=(f"Within safe operating limits — {rul} cycles remaining before replacement." if pct<60
              else f"Elevated wear detected. Replace within {rul} cycles to prevent failure."
              if pct<85 else f"Tool near failure. Only {rul} cycles remaining. Stop immediately.")

        st.markdown(f"""
        <div class="sc {sc}">
          <div style="font-size:20px;color:{bc};line-height:1">{icon}</div>
          <div>
            <div class="sl" style="color:{bc}">{lbl}</div>
            <div class="sd">{desc}</div>
          </div>
        </div>
        <div class="mg">
          <div class="mc">
            <div class="mv" style="color:#b45309">{wear:.1f}</div>
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
        with g1: st.pyplot(draw_roulette(pct,wear,rul),use_container_width=True)
        with g2: st.pyplot(draw_wear_chart(cycle,wear,rul,wr,sf),use_container_width=True)

        st.markdown('<div class="sec" style="margin-top:16px">Parameter Influence</div>',unsafe_allow_html=True)
        st.pyplot(draw_influence(speed,feed,doc,vm,spindle,rough),use_container_width=True)

        rows=[("speed_rpm",f"{speed} rpm",*bdg(speed,1200)),
              ("feed_mm_rev",f"{feed:.2f} mm/rev",*bdg(feed,0.18)),
              ("depth_of_cut_mm",f"{doc:.1f} mm",*bdg(doc,1.0)),
              ("vibration_magnitude",f"{vm:.3f} mm/s²",*bdg(vm,2.0)),
              ("spindle_load",f"{spindle}%",*bdg(spindle,70)),
              ("surface_roughness",f"{rough:.1f} µm",*bdg(rough,1.5)),
              ("mrr",f"{mrr:.1f} mm³/min",*bdg(mrr,150))]
        html="".join([f'<div class="br"><span class="bk">{r[0]}</span><div><span class="bv">{r[1]}</span><span class="bg_ {r[3]}">{r[2]}</span></div></div>' for r in rows])
        st.markdown(f'<div class="sec" style="margin-top:14px">Feature Breakdown</div><div class="bd">{html}</div>',unsafe_allow_html=True)

        rec=("Continue operation normally. Schedule inspection in 10 cycles." if pct<60
             else f"Plan tool replacement within the next {rul} cycles."
             if pct<85 else "Halt operation. Replace tool before proceeding.")
        st.markdown(f"""
        <div style="background:#ffffff;border:1px solid #e4e4e7;border-left:4px solid {bc};
             border-radius:6px;padding:16px 20px;margin-top:12px;
             box-shadow:0 1px 3px rgba(0,0,0,0.04)">
          <div style="font-size:9px;color:#a1a1aa;text-transform:uppercase;letter-spacing:.2em;
               font-family:'DM Mono',monospace;margin-bottom:6px">Recommendation</div>
          <div style="font-size:15px;color:#18181b;font-weight:500;letter-spacing:-.2px">{rec}</div>
          <div style="font-size:10px;color:#d4d4d8;margin-top:6px;font-family:'DM Mono',monospace">
          {'Gradient Boosting (wear) · XGBoost (RUL) · holdout R² 0.9749' if ml else 'Physics fallback — upload PKL files to enable ML'}</div>
        </div>""", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="padding:100px 20px;text-align:center">
          <div style="font-size:36px;font-weight:700;color:#e4e4e7;letter-spacing:-1.5px;
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
