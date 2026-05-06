import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import joblib
import os

st.set_page_config(page_title="ToolSense",page_icon="⚙️",layout="wide",initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html,body,[class*="css"]{font-family:'Space Grotesk',sans-serif}
.stApp{background:#07080a}
.block-container{padding:0!important;max-width:100%!important}
.hero{background:linear-gradient(180deg,#0d1117 0%,#07080a 100%);border-bottom:1px solid #1c2333;padding:36px 56px 28px;position:relative;overflow:hidden}
.hero::before{content:'';position:absolute;top:-120px;right:-80px;width:480px;height:480px;border-radius:50%;background:radial-gradient(circle,rgba(22,163,74,0.07) 0%,transparent 65%);pointer-events:none}
.h-tag{display:inline-block;background:rgba(22,163,74,0.12);border:1px solid rgba(22,163,74,0.3);color:#22c55e;font-size:10px;font-weight:600;letter-spacing:.14em;text-transform:uppercase;padding:4px 14px;border-radius:20px;margin-bottom:12px;font-family:'JetBrains Mono',monospace}
.h-title{font-size:36px;font-weight:700;color:#f0f6ff;margin:0 0 6px;letter-spacing:-1px}
.h-title span{color:#22c55e}
.h-sub{font-size:13px;color:#3d4f6b;margin:0 0 22px}
.h-stats{display:flex;gap:0;border:1px solid #1c2333;border-radius:12px;overflow:hidden;width:fit-content}
.hs{padding:12px 22px;border-right:1px solid #1c2333;background:#0d1117}
.hs:last-child{border-right:none}
.hs-v{font-size:18px;font-weight:700;color:#22c55e;font-family:'JetBrains Mono',monospace;line-height:1}
.hs-l{font-size:9px;color:#2a3a54;text-transform:uppercase;letter-spacing:.1em;margin-top:3px}
.mp{padding:32px 56px}
.sec{font-size:10px;font-weight:600;letter-spacing:.14em;text-transform:uppercase;color:#22c55e;margin-bottom:14px;font-family:'JetBrains Mono',monospace;display:flex;align-items:center;gap:10px}
.sec::after{content:'';flex:1;height:1px;background:#1c2333}
.pnl{background:#0d1117;border:1px solid #1c2333;border-radius:14px;padding:22px;margin-bottom:16px}
.stSlider>div>div>div>div{background:#22c55e!important}
.stNumberInput input{background:#07080a!important;border:1px solid #1c2333!important;color:#e2e8f0!important;border-radius:8px!important;font-family:'JetBrains Mono',monospace!important;font-size:13px!important}
.stButton>button{background:linear-gradient(135deg,#16a34a,#15803d)!important;color:white!important;border:none!important;border-radius:10px!important;font-size:13px!important;font-weight:700!important;font-family:'Space Grotesk',sans-serif!important;letter-spacing:.1em!important;padding:14px 0!important;text-transform:uppercase!important;width:100%!important;box-shadow:0 0 24px rgba(22,163,74,0.2)!important}
.stButton>button:hover{box-shadow:0 0 36px rgba(22,163,74,0.4)!important}
.sc{border-radius:14px;padding:18px 22px;margin:18px 0;display:flex;align-items:center;gap:14px}
.s-h{background:rgba(22,163,74,0.06);border:1px solid rgba(22,163,74,0.2)}
.s-w{background:rgba(234,179,8,0.06);border:1px solid rgba(234,179,8,0.2)}
.s-c{background:rgba(239,68,68,0.06);border:1px solid rgba(239,68,68,0.35)}
.sl{font-size:19px;font-weight:700;line-height:1;margin-bottom:4px}
.sd{font-size:12px;color:#5a7090}
.mg{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:16px 0}
.mc{background:#0d1117;border:1px solid #1c2333;border-radius:12px;padding:16px 18px;position:relative;overflow:hidden}
.mc::before{content:'';position:absolute;top:0;left:0;right:0;height:2px}
.mc-g::before{background:linear-gradient(90deg,#22c55e,transparent)}
.mc-y::before{background:linear-gradient(90deg,#eab308,transparent)}
.mc-r::before{background:linear-gradient(90deg,#ef4444,transparent)}
.mc-b::before{background:linear-gradient(90deg,#3b82f6,transparent)}
.mv{font-size:26px;font-weight:700;font-family:'JetBrains Mono',monospace;line-height:1;margin-bottom:4px}
.ml{font-size:10px;color:#3d4f6b;text-transform:uppercase;letter-spacing:.1em}
.ms{font-size:11px;color:#1e3040;margin-top:2px;font-family:'JetBrains Mono',monospace}
.bd{background:#0d1117;border:1px solid #1c2333;border-radius:12px;overflow:hidden;margin:12px 0}
.br{display:flex;justify-content:space-between;align-items:center;padding:10px 16px;border-bottom:1px solid #0d1117}
.br:nth-child(odd){background:#0a0e14}
.br:last-child{border-bottom:none}
.bk{font-size:11px;color:#3d4f6b;font-family:'JetBrains Mono',monospace}
.bv{font-size:12px;color:#e2e8f0;font-weight:600;font-family:'JetBrains Mono',monospace}
.bg_{font-size:10px;font-weight:700;padding:2px 7px;border-radius:4px;margin-left:8px;font-family:'JetBrains Mono',monospace}
.bh{background:rgba(239,68,68,0.15);color:#f87171}
.bm{background:rgba(234,179,8,0.15);color:#fbbf24}
.bl{background:rgba(22,163,74,0.15);color:#4ade80}
.ic{background:#0d1117;border:1px solid #1c2333;border-radius:12px;overflow:hidden}
.ir{display:flex;justify-content:space-between;padding:10px 16px;border-bottom:1px solid #0a0e14;font-size:12px}
.ir:nth-child(odd){background:#0a0e14}
.ir:last-child{border-bottom:none}
.ik{color:#3d4f6b}
.iv{color:#22c55e;font-family:'JetBrains Mono',monospace;font-weight:600}
label{color:#3d4f6b!important;font-size:12px!important}
</style>
""", unsafe_allow_html=True)

THRESHOLD = 300.0

# ── Load real ML models
@st.cache_resource
def load_models():
    wear_path = "best_wear_model (1).pkl"
    rul_path  = "best_rul_model (1).pkl"
    if os.path.exists(wear_path) and os.path.exists(rul_path):
        wear_model = joblib.load(wear_path)
        rul_model  = joblib.load(rul_path)
        return wear_model, rul_model, True
    return None, None, False

wear_model, rul_model, models_loaded = load_models()

def build_features(speed, feed, doc, cycle, vib_x, vib_y, vib_z, spindle, roughness):
    vm  = (vib_x**2 + vib_y**2 + vib_z**2)**0.5
    mrr = speed * feed * doc
    vib_load_ratio = vm / (spindle + 1e-6)
    log_cycle = np.log1p(cycle)
    return np.array([[speed, feed, doc, vib_x, vib_y, vib_z,
                      spindle, roughness, vm, mrr, vib_load_ratio, log_cycle]])

def predict(speed, feed, doc, cycle, vib_x, vib_y, vib_z, spindle, roughness):
    vm  = (vib_x**2 + vib_y**2 + vib_z**2)**0.5
    mrr = speed * feed * doc
    wr  = 0.5*(speed/1200) + 0.3*(feed/0.18) + 0.2*(doc/1.0)
    sf  = 0.4*(spindle/70) + 0.3*(vm/2.0) + 0.3*(roughness/1.5)

    if models_loaded:
        X = build_features(speed, feed, doc, cycle, vib_x, vib_y, vib_z, spindle, roughness)
        wear = float(np.clip(wear_model.predict(X)[0], 0, THRESHOLD - 0.5))
        rul  = max(0, int(rul_model.predict(X)[0]))
    else:
        a    = 14 * wr * sf
        wear = max(0.0, min(a * (cycle**1.28), THRESHOLD - 0.5))
        rul  = max(0, int((THRESHOLD - wear) * (cycle / max(wear, 0.1)) * 0.85))

    pct = (wear / THRESHOLD) * 100
    return wear, rul, pct, vm, mrr, wr, sf

def bdg(val, ref):
    r = val / ref
    if r > 0.85: return "HIGH", "bh"
    if r > 0.55: return "MED",  "bm"
    return "LOW", "bl"

def draw_roulette_gauge(pct, wear, rul):
    fig, ax = plt.subplots(figsize=(5.5, 4.4), facecolor="#07080a")
    ax.set_facecolor("#07080a")
    ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.0, 1.55)
    ax.set_aspect('equal'); ax.axis('off')

    n = 36
    seg_colors = []
    for i in range(n):
        if i < n * 0.6:   seg_colors.append("#16a34a")
        elif i < n * 0.85: seg_colors.append("#ca8a04")
        else:              seg_colors.append("#dc2626")

    for i, col in enumerate(seg_colors):
        t1 = 180 - (i * 180 / n)
        t2 = 180 - ((i+1) * 180 / n)
        ax.add_patch(Wedge((0,0), 1.28, t2, t1, width=0.14,
                           facecolor=col, edgecolor="#07080a", linewidth=2, alpha=0.9))

    for i in range(n):
        angle = np.radians(180 - (i + 0.5) * 180 / n)
        rx, ry = 1.21 * np.cos(angle), 1.21 * np.sin(angle)
        ax.add_patch(plt.Circle((rx, ry), 0.027,
                                color="#ffffff" if i%2==0 else "#07080a", zorder=5))

    for r, a in [(1.08, 0.12), (0.97, 0.08)]:
        ax.add_patch(plt.Circle((0,0), r, fill=False,
                                edgecolor="#22c55e", linewidth=0.5, alpha=a))

    ax.add_patch(Wedge((0,0), 0.88, 0, 180, width=0.38,
                       facecolor="#0d1117", edgecolor="#1c2333", linewidth=1))

    bc = "#22c55e" if pct < 60 else "#eab308" if pct < 85 else "#ef4444"
    fill_angle = 180 * (pct / 100)
    if fill_angle > 0:
        ax.add_patch(Wedge((0,0), 0.87, 180-fill_angle, 180, width=0.36,
                           facecolor=bc, edgecolor="none", alpha=0.25))

    for pm, lbl, col in [(0,"0","#2a3a54"),(60,"60%","#22c55e"),(85,"85%","#eab308"),(100,"100%","#ef4444")]:
        angle = np.radians(180 - pm * 1.8)
        x1,y1 = 0.90*np.cos(angle), 0.90*np.sin(angle)
        x2,y2 = 0.99*np.cos(angle), 0.99*np.sin(angle)
        ax.plot([x1,x2],[y1,y2], color=col, lw=1.5, alpha=0.7)
        xl,yl = 1.04*np.cos(angle), 1.04*np.sin(angle)
        ax.text(xl, yl, lbl, ha='center', va='center',
                fontsize=6.5, color=col, fontfamily='monospace', alpha=0.85)

    needle_angle = np.radians(180 - pct * 1.8)
    nx, ny = 0.70 * np.cos(needle_angle), 0.70 * np.sin(needle_angle)
    ax.annotate('', xy=(nx, ny), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=bc, lw=2.8, mutation_scale=15))
    ax.add_patch(plt.Circle((0,0), 0.065, color="#1c2333", zorder=10))
    ax.add_patch(plt.Circle((0,0), 0.038, color=bc, zorder=11))

    ax.text(0, 0.24, f"{pct:.1f}%", ha='center', va='center',
            fontsize=20, fontweight='bold', color=bc, fontfamily='monospace')
    ax.text(0, 0.07, "LIFE CONSUMED", ha='center', va='center',
            fontsize=6.5, color="#3d4f6b", fontfamily='monospace', fontweight='600')
    ax.plot([-0.88, 0.88], [-0.20, -0.20], color="#1c2333", lw=0.8)
    ax.plot([0, 0], [-0.20, -0.62], color="#1c2333", lw=0.8)

    ax.text(-0.55, -0.36, f"{wear:.1f}µm", ha='center', va='center',
            fontsize=13, fontweight='700', color="#3b82f6", fontfamily='monospace')
    ax.text(-0.55, -0.53, "FLANK WEAR", ha='center', va='center',
            fontsize=6.5, color="#2a3a54", fontfamily='monospace')

    rul_col = "#22c55e" if rul > 10 else "#eab308" if rul > 3 else "#ef4444"
    ax.text(0.55, -0.36, f"{rul}", ha='center', va='center',
            fontsize=13, fontweight='700', color=rul_col, fontfamily='monospace')
    ax.text(0.55, -0.53, "RUL CYCLES", ha='center', va='center',
            fontsize=6.5, color="#2a3a54", fontfamily='monospace')

    ax.set_title("Tool Health Monitor", color="#2a3a54",
                 fontsize=8.5, fontfamily='monospace', pad=4)
    plt.tight_layout(pad=0.5)
    return fig

def draw_wear_chart(cycle, wear, rul, wr, sf):
    fut  = list(range(1, cycle + rul + 6))
    proj = [min(14*wr*sf*(c**1.28), THRESHOLD) for c in fut]
    bc   = "#22c55e" if (wear/THRESHOLD)*100 < 60 else "#eab308" if (wear/THRESHOLD)*100 < 85 else "#ef4444"

    fig, ax = plt.subplots(figsize=(5, 3.4), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.axhspan(0, 180, alpha=0.04, color="#22c55e")
    ax.axhspan(180, 255, alpha=0.04, color="#eab308")
    ax.axhspan(255, 320, alpha=0.04, color="#ef4444")
    ax.fill_between(fut, proj, alpha=0.08, color=bc)
    ax.plot(fut, proj, color=bc, lw=2, zorder=3)
    ax.axhline(THRESHOLD, color="#ef4444", ls="--", lw=1, alpha=0.6)
    ax.axhline(THRESHOLD*0.85, color="#eab308", ls=":", lw=1, alpha=0.5)
    ax.axhline(180, color="#22c55e", ls=":", lw=0.8, alpha=0.4)
    ax.axvline(cycle, color="#3d4f6b", ls="--", lw=0.8, alpha=0.5)
    ax.scatter([cycle], [wear], color=bc, s=60, zorder=6, edgecolors="#07080a", lw=1.5)
    ax.text(cycle+0.3, wear+12, f"{wear:.0f}µm", color=bc, fontsize=8, fontfamily="monospace")
    if rul > 0:
        fail_cycle = cycle + rul
        ax.scatter([fail_cycle], [THRESHOLD], color="#ef4444", s=40, zorder=6, marker="x", lw=1.5)
        ax.text(fail_cycle+0.3, THRESHOLD-18, "FAIL", color="#ef4444", fontsize=7, fontfamily="monospace", alpha=0.7)
    ax.set_xlabel("Cycle", color="#3d4f6b", fontsize=9)
    ax.set_ylabel("Wear (µm)", color="#3d4f6b", fontsize=9)
    ax.tick_params(colors="#2a3a54", labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor("#1c2333")
    ax.set_title("Wear trajectory", color="#3d4f6b", fontsize=9, pad=6)
    ax.set_ylim(0, THRESHOLD + 30)
    plt.tight_layout(pad=1.2)
    return fig

def draw_influence_chart(speed, feed, doc, vm, spindle, roughness):
    fac = {"Speed": 0.5*(speed/1200), "Feed": 0.3*(feed/0.18),
           "DoC": 0.2*(doc/1.0), "Vibration": 0.3*(vm/2),
           "Spindle": 0.4*(spindle/70), "Roughness": 0.3*(roughness/1.5)}
    vals   = list(fac.values())
    total  = sum(vals)
    pcts   = [v/total*100 for v in vals]
    colors = ["#22c55e","#3b82f6","#a855f7","#f97316","#ec4899","#eab308"]

    fig, ax = plt.subplots(figsize=(5, 3.4), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    bars = ax.barh(list(fac.keys()), vals, color=colors, alpha=0.8, height=0.5, edgecolor="none")
    for bar, v, p, col in zip(bars, vals, pcts, colors):
        ax.text(v+0.005, bar.get_y()+bar.get_height()/2,
                f"{v:.2f}  ({p:.0f}%)", va="center", color=col,
                fontsize=8, fontfamily="monospace")
    ax.set_xlabel("Influence", color="#3d4f6b", fontsize=9)
    ax.tick_params(colors="#3d4f6b", labelsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor("#1c2333")
    ax.set_title("Parameter influence on wear", color="#3d4f6b", fontsize=9, pad=6)
    ax.set_xlim(0, max(vals)*1.5)
    plt.tight_layout(pad=1.2)
    return fig

# ── HERO
st.markdown(f"""
<div class="hero">
<div class="h-tag">⚙ BTP · Micro-Turning · {'Real ML Model' if models_loaded else 'Physics Model'} · Predictive Maintenance</div>
<div class="h-title">Tool<span>Sense</span></div>
<div class="h-sub">Real-time tool wear &amp; remaining useful life prediction · {'Gradient Boosting + XGBoost' if models_loaded else 'Formula-based fallback'}</div>
<div class="h-stats">
<div class="hs"><div class="hs-v">0.98</div><div class="hs-l">Wear R²</div></div>
<div class="hs"><div class="hs-v">0.95</div><div class="hs-l">RUL R²</div></div>
<div class="hs"><div class="hs-v">K=10</div><div class="hs-l">GroupKFold CV</div></div>
<div class="hs"><div class="hs-v">300µm</div><div class="hs-l">Threshold</div></div>
<div class="hs"><div class="hs-v">{'✓ ML' if models_loaded else '⚠ FB'}</div><div class="hs-l">{'Model loaded' if models_loaded else 'Fallback'}</div></div>
</div></div>
""", unsafe_allow_html=True)

if not models_loaded:
    st.warning("⚠️ PKL models not found in repo root. Using physics fallback. Upload best_wear_model.pkl and best_rul_model.pkl to fix this.")

st.markdown('<div class="mp">', unsafe_allow_html=True)
c1, _, c2 = st.columns([4, 0.3, 6])

with c1:
    st.markdown('<div class="sec">Cutting Parameters</div>', unsafe_allow_html=True)
    st.markdown('<div class="pnl">', unsafe_allow_html=True)
    speed = st.slider("Spindle Speed (rpm)", 500, 2000, 1000, step=50)
    feed  = st.slider("Feed Rate (mm/rev)",  0.05, 0.30, 0.13, step=0.01)
    doc   = st.slider("Depth of Cut (mm)",   0.1, 2.0, 0.8, step=0.1)
    cycle = st.slider("Cycle Number",        1, 100, 8)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec">Sensor Readings</div>', unsafe_allow_html=True)
    st.markdown('<div class="pnl">', unsafe_allow_html=True)
    s1, s2 = st.columns(2)
    with s1:
        vib_x     = st.number_input("Vibration X (mm/s²)", value=1.35, step=0.05, format="%.2f")
        vib_z     = st.number_input("Vibration Z (mm/s²)", value=0.75, step=0.05, format="%.2f")
        roughness = st.number_input("Roughness (µm)",       value=1.4,  step=0.1,  format="%.1f")
    with s2:
        vib_y   = st.number_input("Vibration Y (mm/s²)", value=1.20, step=0.05, format="%.2f")
        spindle = st.slider("Spindle Load (%)", 10, 100, 61)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    clicked = st.button("⚡  PREDICT TOOL CONDITION", use_container_width=True)

with c2:
    if clicked:
        wear, rul, pct, vm, mrr, wr, sf = predict(
            speed, feed, doc, cycle, vib_x, vib_y, vib_z, spindle, roughness)

        bc = "#22c55e" if pct < 60 else "#eab308" if pct < 85 else "#ef4444"
        rc = "#22c55e" if rul > 10 else "#eab308" if rul > 3 else "#ef4444"
        mc = "mc-g"    if pct < 60 else "mc-y"    if pct < 85 else "mc-r"

        if pct < 60:
            sc, icon, lbl = "s-h", "✅", "HEALTHY"
            desc = f"Safe operating range — {rul} cycles remaining before replacement."
        elif pct < 85:
            sc, icon, lbl = "s-w", "⚠️", "WARNING — PLAN REPLACEMENT"
            desc = f"Elevated wear — schedule replacement within {rul} cycles."
        else:
            sc, icon, lbl = "s-c", "🔴", "CRITICAL — REPLACE NOW"
            desc = f"Near failure. Only {rul} cycles remaining. Stop immediately."

        st.markdown(f"""
        <div class="sc {sc}">
        <div style="font-size:30px">{icon}</div>
        <div><div class="sl" style="color:{bc}">{lbl}</div><div class="sd">{desc}</div></div>
        </div>
        <div class="mg">
        <div class="mc mc-b"><div class="mv" style="color:#3b82f6">{wear:.1f}</div><div class="ml">Flank Wear (µm)</div><div class="ms">threshold · 300µm</div></div>
        <div class="mc {mc}"><div class="mv" style="color:{rc}">{rul}</div><div class="ml">Remaining Cycles</div><div class="ms">until failure</div></div>
        <div class="mc {mc}"><div class="mv" style="color:{bc}">{pct:.1f}%</div><div class="ml">Life Consumed</div><div class="ms">{100-pct:.1f}% remaining</div></div>
        </div>""", unsafe_allow_html=True)

        g1, g2 = st.columns(2)
        with g1:
            st.pyplot(draw_roulette_gauge(pct, wear, rul), use_container_width=True)
        with g2:
            st.pyplot(draw_wear_chart(cycle, wear, rul, wr, sf), use_container_width=True)

        st.markdown('<div class="sec" style="margin-top:20px">Parameter Influence</div>', unsafe_allow_html=True)
        st.pyplot(draw_influence_chart(speed, feed, doc, vm, spindle, roughness), use_container_width=True)

        rows = [
            ("speed_rpm",        speed,     f"{speed} rpm",         *bdg(speed,    1200)),
            ("feed_mm_rev",      feed,      f"{feed:.2f} mm/rev",   *bdg(feed,     0.18)),
            ("depth_of_cut",     doc,       f"{doc:.1f} mm",        *bdg(doc,      1.0 )),
            ("vib_magnitude",    vm,        f"{vm:.3f} mm/s²",      *bdg(vm,       2.0 )),
            ("spindle_load",     spindle,   f"{spindle}%",          *bdg(spindle,  70  )),
            ("surface_roughness",roughness, f"{roughness:.1f} µm",  *bdg(roughness,1.5 )),
            ("mrr",              mrr,       f"{mrr:.1f} mm³/min",   *bdg(mrr,      150 )),
        ]
        html = "".join([
            f'<div class="br"><span class="bk">{r[0]}</span>'
            f'<div><span class="bv">{r[2]}</span>'
            f'<span class="bg_ {r[4]}">{r[3]}</span></div></div>'
            for r in rows])
        st.markdown(f'<div class="sec" style="margin-top:16px">Feature Breakdown</div>'
                    f'<div class="bd">{html}</div>', unsafe_allow_html=True)

        rec_txt = ("Continue operation. Inspect after 10 cycles." if pct < 60
                   else f"Plan replacement within {rul} cycles." if pct < 85
                   else "STOP. Replace tool before next cycle.")
        st.markdown(f"""
        <div style="background:#0d1117;border:1px solid #1c2333;border-left:3px solid {bc};
             border-radius:10px;padding:14px 18px;margin-top:14px">
        <div style="font-size:9px;color:#3d4f6b;text-transform:uppercase;letter-spacing:.12em;
             font-family:monospace;margin-bottom:5px">Recommendation</div>
        <div style="font-size:13px;color:#e2e8f0;font-weight:500">{rec_txt}</div>
        <div style="font-size:10px;color:#2a3a54;margin-top:6px;font-family:monospace">
        {'Powered by real trained ML model (Gradient Boosting + XGBoost)' if models_loaded else 'Physics-based fallback active'}</div>
        </div>""", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="padding:80px 20px;text-align:center">
        <div style="font-size:52px;margin-bottom:20px;opacity:.1">⚙️</div>
        <div style="font-size:15px;color:#2a3a54;font-weight:600;margin-bottom:8px">Configure inputs → Click Predict</div>
        <div style="font-size:12px;color:#1a2a3a;line-height:1.7">
        Set cutting parameters and sensor readings on the left.<br>The roulette gauge and all charts appear here.
        </div></div>
        <div class="sec">Model Performance (BTP Results)</div>
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
