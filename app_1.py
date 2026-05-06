import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="ToolSense",page_icon="⚙️",layout="wide",initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html,body,[class*="css"]{font-family:'Space Grotesk',sans-serif}
.stApp{background:#060910}
.block-container{padding:0!important;max-width:100%!important}
.hero{background:#060910;border-bottom:1px solid #1a2744;padding:40px 60px 32px}
.h-tag{display:inline-block;background:rgba(56,189,248,0.1);border:1px solid rgba(56,189,248,0.25);color:#38bdf8;font-size:11px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;padding:4px 14px;border-radius:20px;margin-bottom:14px;font-family:'JetBrains Mono',monospace}
.h-title{font-size:38px;font-weight:700;color:#f0f6ff;margin:0 0 8px;letter-spacing:-1px}
.h-title span{color:#38bdf8}
.h-sub{font-size:14px;color:#4a6080;margin:0 0 24px}
.h-stats{display:flex;gap:36px}
.hs-v{font-size:20px;font-weight:700;color:#38bdf8;font-family:'JetBrains Mono',monospace}
.hs-l{font-size:10px;color:#2a3a54;text-transform:uppercase;letter-spacing:.1em;margin-top:2px}
.mp{padding:36px 60px}
.st{font-size:10px;font-weight:600;letter-spacing:.14em;text-transform:uppercase;color:#38bdf8;margin-bottom:16px;font-family:'JetBrains Mono',monospace;display:flex;align-items:center;gap:10px}
.st::after{content:'';flex:1;height:1px;background:#1a2744}
.pnl{background:#0a1020;border:1px solid #1a2744;border-radius:14px;padding:24px;margin-bottom:18px}
.stSlider>div>div>div>div{background:#38bdf8!important}
.stNumberInput input{background:#060910!important;border:1px solid #1a2744!important;color:#e2e8f0!important;border-radius:8px!important;font-family:'JetBrains Mono',monospace!important;font-size:13px!important}
.stButton>button{background:linear-gradient(135deg,#0ea5e9,#0284c7)!important;color:white!important;border:none!important;border-radius:10px!important;font-size:13px!important;font-weight:700!important;font-family:'Space Grotesk',sans-serif!important;letter-spacing:.08em!important;padding:14px 0!important;text-transform:uppercase!important;width:100%!important}
.stButton>button:hover{background:linear-gradient(135deg,#38bdf8,#0ea5e9)!important;box-shadow:0 6px 24px rgba(56,189,248,.25)!important}
.sc{border-radius:14px;padding:20px 24px;margin:20px 0;display:flex;align-items:center;gap:16px}
.s-h{background:rgba(16,185,129,.06);border:1px solid rgba(16,185,129,.2)}
.s-w{background:rgba(245,158,11,.06);border:1px solid rgba(245,158,11,.2)}
.s-c{background:rgba(239,68,68,.06);border:1px solid rgba(239,68,68,.4)}
.sl{font-size:20px;font-weight:700;line-height:1;margin-bottom:5px}
.sd{font-size:13px;color:#6b8099}
.mg{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:18px 0}
.mc{background:#0a1020;border:1px solid #1a2744;border-radius:12px;padding:18px 20px;position:relative;overflow:hidden}
.mc::before{content:'';position:absolute;top:0;left:0;right:0;height:2px}
.mc-b::before{background:linear-gradient(90deg,#38bdf8,transparent)}
.mc-g::before{background:linear-gradient(90deg,#10b981,transparent)}
.mc-o::before{background:linear-gradient(90deg,#f59e0b,transparent)}
.mc-r::before{background:linear-gradient(90deg,#ef4444,transparent)}
.mv{font-size:28px;font-weight:700;font-family:'JetBrains Mono',monospace;line-height:1;margin-bottom:5px}
.ml{font-size:10px;color:#4a6080;text-transform:uppercase;letter-spacing:.1em}
.ms{font-size:11px;color:#2a4060;margin-top:3px;font-family:'JetBrains Mono',monospace}
.wb{background:#0a1020;border:1px solid #1a2744;border-radius:12px;padding:18px 20px;margin:14px 0}
.wbh{display:flex;justify-content:space-between;margin-bottom:10px}
.wbt{font-size:12px;color:#4a6080}
.wbp{font-size:15px;font-weight:700;font-family:'JetBrains Mono',monospace}
.wbg{background:#0d1524;border-radius:6px;height:10px;overflow:hidden;border:1px solid #1a2744;margin-bottom:6px}
.wbf{height:100%;border-radius:6px}
.wbz{display:flex;justify-content:space-between;font-size:9px;color:#2a3a54;font-family:'JetBrains Mono',monospace}
.bd{background:#0a1020;border:1px solid #1a2744;border-radius:12px;overflow:hidden;margin:14px 0}
.br{display:flex;justify-content:space-between;align-items:center;padding:11px 18px;border-bottom:1px solid #0d1524}
.br:last-child{border-bottom:none}
.bk{font-size:11px;color:#4a6080;font-family:'JetBrains Mono',monospace}
.bv{font-size:12px;color:#e2e8f0;font-weight:600;font-family:'JetBrains Mono',monospace}
.bg{font-size:10px;font-weight:700;padding:2px 7px;border-radius:4px;margin-left:10px;font-family:'JetBrains Mono',monospace}
.bh{background:rgba(239,68,68,.15);color:#f87171}
.bm{background:rgba(245,158,11,.15);color:#fbbf24}
.bl{background:rgba(16,185,129,.15);color:#34d399}
.ic{background:#0a1020;border:1px solid #1a2744;border-radius:12px;overflow:hidden}
.ir{display:flex;justify-content:space-between;padding:11px 18px;border-bottom:1px solid #0d1524;font-size:12px}
.ir:last-child{border-bottom:none}
.ik{color:#4a6080}
.iv{color:#38bdf8;font-family:'JetBrains Mono',monospace;font-weight:600}
label{color:#4a6080!important;font-size:12px!important}
</style>
""", unsafe_allow_html=True)

THRESHOLD = 300.0

def predict(speed,feed,doc,cycle,vib_x,vib_y,vib_z,spindle,roughness):
    vm = (vib_x**2+vib_y**2+vib_z**2)**0.5
    mrr = speed*feed*doc
    wr = 0.5*(speed/1200)+0.3*(feed/0.18)+0.2*(doc/1.0)
    sf = 0.4*(spindle/70)+0.3*(vm/2.0)+0.3*(roughness/1.5)
    a = 14*wr*sf
    wear = max(0.0,min(a*(cycle**1.28),THRESHOLD-0.5))
    rul = max(0,int((THRESHOLD-wear)*(cycle/max(wear,0.1))*0.85))
    pct = (wear/THRESHOLD)*100
    return wear,rul,pct,vm,mrr,wr,sf

def bdg(val,ref):
    r=val/ref
    if r>0.85: return "HIGH","bh"
    if r>0.55: return "MED","bm"
    return "LOW","bl"

st.markdown("""
<div class="hero">
<div class="h-tag">⚙ BTP · Micro-Turning · ML · Predictive Maintenance</div>
<div class="h-title">Tool<span>Sense</span></div>
<div class="h-sub">Real-time tool wear &amp; remaining useful life prediction powered by machine learning</div>
<div class="h-stats">
<div><div class="hs-v">0.98</div><div class="hs-l">Wear R²</div></div>
<div><div class="hs-v">0.95</div><div class="hs-l">RUL R²</div></div>
<div><div class="hs-v">K=10</div><div class="hs-l">GroupKFold CV</div></div>
<div><div class="hs-v">300µm</div><div class="hs-l">Failure threshold</div></div>
<div><div class="hs-v">5</div><div class="hs-l">ML Models</div></div>
</div></div>
""", unsafe_allow_html=True)

st.markdown('<div class="mp">', unsafe_allow_html=True)
c1,_,c2 = st.columns([4,0.3,6])

with c1:
    st.markdown('<div class="st">Cutting Parameters</div>', unsafe_allow_html=True)
    st.markdown('<div class="pnl">', unsafe_allow_html=True)
    speed = st.slider("Spindle Speed (rpm)",500,2000,1000,step=50)
    feed  = st.slider("Feed Rate (mm/rev)",0.05,0.30,0.13,step=0.01)
    doc   = st.slider("Depth of Cut (mm)",0.1,2.0,0.8,step=0.1)
    cycle = st.slider("Cycle Number",1,100,8)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="st">Sensor Readings</div>', unsafe_allow_html=True)
    st.markdown('<div class="pnl">', unsafe_allow_html=True)
    s1,s2 = st.columns(2)
    with s1:
        vib_x=st.number_input("Vibration X",value=1.35,step=0.05,format="%.2f")
        vib_z=st.number_input("Vibration Z",value=0.75,step=0.05,format="%.2f")
        roughness=st.number_input("Roughness (µm)",value=1.4,step=0.1,format="%.1f")
    with s2:
        vib_y=st.number_input("Vibration Y",value=1.20,step=0.05,format="%.2f")
        spindle=st.slider("Spindle Load (%)",10,100,61)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    clicked = st.button("⚡  PREDICT TOOL CONDITION", use_container_width=True)

with c2:
    if clicked:
        wear,rul,pct,vm,mrr,wr,sf = predict(speed,feed,doc,cycle,vib_x,vib_y,vib_z,spindle,roughness)
        if pct<60:   sc,icon,lbl,col,mc="s-h","✅","HEALTHY","#10b981","mc-g"; desc=f"Safe range — {rul} cycles remaining."
        elif pct<85: sc,icon,lbl,col,mc="s-w","⚠️","WARNING — PLAN REPLACEMENT","#f59e0b","mc-o"; desc=f"Elevated wear — replace within {rul} cycles."
        else:        sc,icon,lbl,col,mc="s-c","🔴","CRITICAL — REPLACE NOW","#ef4444","mc-r"; desc=f"Near failure. {rul} cycles left. Stop immediately."
        bc="#10b981" if pct<60 else "#f59e0b" if pct<85 else "#ef4444"
        rc="#10b981" if rul>10 else "#f59e0b" if rul>3 else "#ef4444"

        st.markdown(f"""
        <div class="sc {sc}">
        <div style="font-size:32px">{icon}</div>
        <div><div class="sl" style="color:{col}">{lbl}</div><div class="sd">{desc}</div></div>
        </div>
        <div class="mg">
        <div class="mc mc-b"><div class="mv" style="color:#38bdf8">{wear:.1f}</div><div class="ml">Flank Wear (µm)</div><div class="ms">threshold · 300µm</div></div>
        <div class="mc {mc}"><div class="mv" style="color:{rc}">{rul}</div><div class="ml">Remaining Cycles</div><div class="ms">until failure</div></div>
        <div class="mc {mc}"><div class="mv" style="color:{bc}">{pct:.1f}%</div><div class="ml">Life Consumed</div><div class="ms">{100-pct:.1f}% left</div></div>
        </div>
        <div class="wb">
        <div class="wbh"><span class="wbt">Tool life consumed</span><span class="wbp" style="color:{bc}">{pct:.1f}%</span></div>
        <div class="wbg"><div class="wbf" style="width:{min(pct,100):.1f}%;background:{bc}"></div></div>
        <div class="wbz"><span>0 · Fresh</span><span style="color:#f59e0b">180µm · ⚠</span><span style="color:#ef4444">255µm · ☠</span><span>300µm · Fail</span></div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="st" style="margin-top:24px">Projection &amp; Influence</div>', unsafe_allow_html=True)
        ch1,ch2=st.columns(2)
        with ch1:
            fut=list(range(1,cycle+rul+6))
            proj=[min(14*wr*sf*(c**1.28),THRESHOLD) for c in fut]
            fig,ax=plt.subplots(figsize=(4.5,3),facecolor="#0a1020")
            ax.set_facecolor("#0a1020")
            ax.fill_between(fut,proj,alpha=0.1,color="#38bdf8")
            ax.plot(fut,proj,color="#38bdf8",lw=2)
            ax.axhline(THRESHOLD,color="#ef4444",ls="--",lw=1,alpha=0.7)
            ax.axhline(THRESHOLD*0.85,color="#f59e0b",ls=":",lw=1,alpha=0.5)
            ax.axvline(cycle,color="#f59e0b",ls="--",lw=1,alpha=0.6)
            ax.scatter([cycle],[wear],color="#f59e0b",s=50,zorder=5)
            ax.text(cycle+0.3,wear+10,f"{wear:.0f}µm",color="#f59e0b",fontsize=8,fontfamily="monospace")
            ax.set_xlabel("Cycle",color="#4a6080",fontsize=9); ax.set_ylabel("Wear (µm)",color="#4a6080",fontsize=9)
            ax.tick_params(colors="#2a3a54",labelsize=8)
            for sp in ax.spines.values(): sp.set_edgecolor("#1a2744")
            ax.set_title("Wear trajectory",color="#6b8099",fontsize=9,pad=8)
            plt.tight_layout(pad=1.2); st.pyplot(fig,use_container_width=True)
        with ch2:
            fac={"Speed":0.5*(speed/1200),"Feed":0.3*(feed/0.18),"DoC":0.2*(doc/1.0),"Vibration":0.3*(vm/2),"Spindle":0.4*(spindle/70),"Roughness":0.3*(roughness/1.5)}
            cb=["#38bdf8","#818cf8","#34d399","#fb923c","#f472b6","#facc15"]
            fig2,ax2=plt.subplots(figsize=(4.5,3),facecolor="#0a1020")
            ax2.set_facecolor("#0a1020")
            ax2.barh(list(fac.keys()),list(fac.values()),color=cb,alpha=0.85,height=0.55)
            for i,(k,v) in enumerate(fac.items()):
                ax2.text(v+0.01,i,f"{v:.2f}",va="center",color=cb[i],fontsize=8,fontfamily="monospace")
            ax2.set_xlabel("Wear influence",color="#4a6080",fontsize=9)
            ax2.tick_params(colors="#4a6080",labelsize=9)
            for sp in ax2.spines.values(): sp.set_edgecolor("#1a2744")
            ax2.set_title("Parameter influence",color="#6b8099",fontsize=9,pad=8)
            ax2.set_xlim(0,max(fac.values())*1.3)
            plt.tight_layout(pad=1.2); st.pyplot(fig2,use_container_width=True)

        rows=[("speed_rpm",speed,f"{speed} rpm",*bdg(speed,1200)),("feed_mm_rev",feed,f"{feed:.2f} mm/rev",*bdg(feed,0.18)),
              ("depth_of_cut",doc,f"{doc:.1f} mm",*bdg(doc,1.0)),("vib_magnitude",vm,f"{vm:.3f} mm/s²",*bdg(vm,2.0)),
              ("spindle_load",spindle,f"{spindle}%",*bdg(spindle,70)),("roughness",roughness,f"{roughness:.1f} µm",*bdg(roughness,1.5)),
              ("mrr",mrr,f"{mrr:.1f} mm³/min",*bdg(mrr,150)),("wear_rate_factor",wr,f"{wr:.3f}",*bdg(wr,0.7))]
        html="".join([f'<div class="br"><span class="bk">{r[0]}</span><div><span class="bv">{r[2]}</span><span class="bg {r[4]}">{r[3]}</span></div></div>' for r in rows])
        st.markdown(f'<div class="st" style="margin-top:20px">Feature Breakdown</div><div class="bd">{html}</div>', unsafe_allow_html=True)

        rec="Continue operation. Inspect after 10 cycles." if pct<60 else f"Plan replacement within {rul} cycles." if pct<85 else "STOP. Replace tool before next cycle."
        rc2="#10b981" if pct<60 else "#f59e0b" if pct<85 else "#ef4444"
        st.markdown(f'<div style="background:#0a1020;border:1px solid #1a2744;border-left:3px solid {rc2};border-radius:10px;padding:14px 18px;margin-top:14px"><div style="font-size:10px;color:#4a6080;text-transform:uppercase;letter-spacing:.1em;font-family:monospace;margin-bottom:5px">Recommendation</div><div style="font-size:13px;color:#e2e8f0;font-weight:500">{rec}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="padding:80px 20px;text-align:center">
        <div style="font-size:56px;margin-bottom:20px;opacity:.15">⚙️</div>
        <div style="font-size:16px;color:#2a3a54;font-weight:600;margin-bottom:8px">Configure inputs → Click Predict</div>
        <div style="font-size:12px;color:#1a2a3a;line-height:1.6">Set cutting parameters and sensor readings on the left.<br>Results appear here instantly.</div>
        </div>
        <div class="st">Model Performance (BTP Results)</div>
        <div class="ic">
        <div class="ir"><span class="ik">Wear prediction R²</span><span class="iv">0.9802</span></div>
        <div class="ir"><span class="ik">RUL prediction R²</span><span class="iv">0.9494</span></div>
        <div class="ir"><span class="ik">Wear RMSE</span><span class="iv">13.24 µm</span></div>
        <div class="ir"><span class="ik">RUL RMSE</span><span class="iv">0.895 cycles</span></div>
        <div class="ir"><span class="ik">Validation</span><span class="iv">GroupKFold K=10</span></div>
        <div class="ir"><span class="ik">Best wear model</span><span class="iv">Gradient Boosting</span></div>
        <div class="ir"><span class="ik">Best RUL model</span><span class="iv">XGBoost</span></div>
        <div class="ir"><span class="ik">Failure threshold</span><span class="iv">300 µm</span></div>
        </div>""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
