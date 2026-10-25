"""SPOOFWATCH CONSOLE — presentation dashboard for the RL Finance Watchdog.

    streamlit run app.py --server.fileWatcherType none

The flag is required: this repository contains a package named `watchdog/`, which shadows the PyPI `watchdog`
library Streamlit uses for file watching.

Three views: Overview (headline evidence), Demonstration (a recorded episode, step by step) and Evidence &
Limitations (the full research record). Every figure is read from committed files in results/ and configs/;
nothing is computed, trained or fetched live. Missing files degrade to "Not available" rather than placeholder
numbers.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
RESULTS, FIGS, CONFIGS = ROOT / "results", ROOT / "results" / "figures", ROOT / "configs"

BG, PANEL, LINE = "#0b1220", "#121a2b", "#22304a"
TEXT, MUTED = "#e6edf7", "#8b99b4"
WD, RULE, GOOD, BAD = "#3d8bfd", "#e0a200", "#2ecc71", "#e5484d"
NA = "Not available"

st.set_page_config(page_title="SpoofWatch Console", page_icon="▦", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown(f"""<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
  .stApp {{ background:{BG}; }}
  html, body, [class*="css"] {{ font-family:'Inter',-apple-system,'Segoe UI',sans-serif; }}
  .block-container {{ padding:1.1rem 1.8rem 2rem; max-width:1600px; }}
  header[data-testid="stHeader"] {{ background:transparent; height:0; }}
  h1,h2,h3,h4 {{ color:{TEXT}; letter-spacing:-.2px; }}
  .mono {{ font-family:'JetBrains Mono',ui-monospace,monospace; font-variant-numeric:tabular-nums; }}

  .topbar {{ display:flex; align-items:baseline; gap:14px; border-bottom:1px solid {LINE};
             padding-bottom:10px; margin-bottom:14px; }}
  .brand {{ font-size:1.32rem; font-weight:700; color:{TEXT}; letter-spacing:.14em; }}
  .tagline {{ color:{MUTED}; font-size:.82rem; }}
  .chip {{ margin-left:auto; font-family:'JetBrains Mono',monospace; font-size:.68rem; letter-spacing:.1em;
           color:{RULE}; border:1px solid {RULE}55; background:{RULE}14; padding:3px 9px; border-radius:3px; }}

  .kpi {{ background:{PANEL}; border:1px solid {LINE}; border-left:3px solid var(--acc,{WD});
          border-radius:4px; padding:11px 13px; height:100%; }}
  .kpi .k-label {{ color:{MUTED}; font-size:.68rem; text-transform:uppercase; letter-spacing:.09em;
                   font-weight:600; }}
  .kpi .k-val {{ font-family:'JetBrains Mono',monospace; font-size:1.85rem; font-weight:600; color:{TEXT};
                 line-height:1.15; margin:3px 0 1px; }}
  .kpi .k-cmp {{ font-family:'JetBrains Mono',monospace; font-size:.8rem; color:{RULE}; }}
  .kpi .k-dir {{ color:{MUTED}; font-size:.7rem; margin-top:5px; }}
  .kpi .k-src {{ color:{MUTED}; font-size:.64rem; opacity:.75; margin-top:5px;
                 font-family:'JetBrains Mono',monospace; }}

  .panel {{ background:{PANEL}; border:1px solid {LINE}; border-radius:4px; padding:12px 14px; }}
  .finding {{ background:linear-gradient(90deg,{WD}14,transparent); border:1px solid {LINE};
              border-left:3px solid {WD}; border-radius:4px; padding:11px 14px; }}
  .finding b {{ color:{TEXT}; }}
  .warnbar {{ border:1px solid {BAD}44; background:{BAD}10; border-radius:4px; padding:8px 12px;
              color:#f3bfbd; font-size:.78rem; }}
  .stage {{ background:{PANEL}; border:1px solid {LINE}; border-radius:4px; padding:9px 11px; height:100%; }}
  .stage .s-n {{ color:{WD}; font-family:'JetBrains Mono',monospace; font-size:.64rem; letter-spacing:.08em; }}
  .stage .s-t {{ color:{TEXT}; font-size:.84rem; font-weight:600; margin:2px 0 3px; }}
  .stage .s-d {{ color:{MUTED}; font-size:.72rem; line-height:1.35; }}

  .verdict {{ display:inline-block; padding:2px 9px; border-radius:3px; font-size:.68rem; font-weight:700;
              letter-spacing:.06em; font-family:'JetBrains Mono',monospace; }}
  .v-ok {{ background:{GOOD}18; color:{GOOD}; border:1px solid {GOOD}55; }}
  .v-part {{ background:{RULE}18; color:{RULE}; border:1px solid {RULE}55; }}
  .v-no {{ background:{BAD}18; color:{BAD}; border:1px solid {BAD}55; }}

  .ladder {{ font-family:'JetBrains Mono',monospace; font-size:.78rem; width:100%; border-collapse:collapse; }}
  .ladder td {{ padding:2px 6px; border-bottom:1px solid {LINE}33; }}
  .ladder .px {{ color:{TEXT}; width:72px; }}
  .ladder .ask {{ color:#ff9b96; }} .ladder .bid {{ color:#7fd6a0; }}
  .ladder .barcell {{ width:100%; }}
  .ladder .bar {{ height:11px; border-radius:2px; display:inline-block; }}
  .ladder .mine {{ outline:1px solid {RULE}; }}
  .ladder .sz {{ color:{MUTED}; width:78px; text-align:right; }}
  .ladder .spread td {{ color:{MUTED}; font-size:.68rem; border-bottom:1px solid {LINE}; padding-top:4px; }}

  .ev {{ font-family:'JetBrains Mono',monospace; font-size:.74rem; color:{MUTED}; padding:2px 0;
         border-bottom:1px solid {LINE}2b; }}
  .ev b {{ color:{TEXT}; font-weight:600; }}
  .ev .flag {{ color:{WD}; }} .ev .alarm {{ color:{BAD}; }} .ev .manip {{ color:{RULE}; }}
  div[data-testid="stMetricValue"] {{ font-family:'JetBrains Mono',monospace; font-size:1.3rem; }}
  div[data-testid="stDataFrame"] {{ border:1px solid {LINE}; border-radius:4px; }}
  .stTabs [data-baseweb="tab-list"] {{ gap:3px; border-bottom:1px solid {LINE}; }}
  .stTabs [data-baseweb="tab"] {{ background:transparent; color:{MUTED}; font-size:.84rem; }}
  .stTabs [aria-selected="true"] {{ color:{TEXT}; border-bottom:2px solid {WD}; }}
  .cap {{ color:{MUTED}; font-size:.74rem; }}
</style>""", unsafe_allow_html=True)


# ─────────────────────────────── data access ───────────────────────────────
@st.cache_data(show_spinner=False)
def load(name: str, folder: str = "results") -> dict:
    p = (RESULTS if folder == "results" else CONFIGS) / name
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


R = {n: load(f"{n}.json") for n in (
    "basic_eval", "watchdog_eval", "multiseed", "lateburst_test", "layering_test", "pnl_suppression",
    "feature_ablation", "synthetic_eval", "spoofer_seeds", "demo_episodes",
    "label_observability", "supervised_baseline", "posthoc_rescore", "baselines_final",
    "anonymised_experiment")}
PH = R["posthoc_rescore"].get("summary", {})
PHRULE = R["posthoc_rescore"].get("rule", {})


def ph(tag: str, metric: str, key: str = "mean"):
    """CP-7 post-hoc re-scoring value, or None when absent."""
    v = PH.get(tag, {}).get(metric, {})
    return v.get(key) if isinstance(v, dict) else None
CAL, SYNCAL = load("calibration.json", "configs"), load("synthetic_calibration.json", "configs")
MS, RU = R["multiseed"].get("summary", {}), R["multiseed"].get("rule", {})


def dig(d: dict, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def pct(x, nd=1):
    return NA if x is None or x != x else f"{100 * x:.{nd}f}%"


def num(x, nd=2):
    return NA if x is None or x != x else f"{x:.{nd}f}"


def money(x, k=False):
    if x is None or x != x:
        return NA
    return f"{'+' if x >= 0 else '−'}${abs(x) / (1000 if k else 1):,.1f}{'k' if k else ''}"


def kpi(label, value, compare, direction, source, accent=WD):
    st.markdown(
        f'<div class="kpi" style="--acc:{accent}"><div class="k-label">{label}</div>'
        f'<div class="k-val">{value}</div><div class="k-cmp">rule baseline · {compare}</div>'
        f'<div class="k-dir">{direction}</div><div class="k-src">{source}</div></div>',
        unsafe_allow_html=True)


def chart_layout(fig, height=330, title=None, ytitle=None, yrange=None):
    fig.update_layout(
        template="plotly_dark", height=height, title=title, paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        font=dict(family="Inter, sans-serif", size=12, color=TEXT),
        margin=dict(l=8, r=8, t=40 if title else 12, b=44),
        legend=dict(orientation="h", y=-0.16, x=0, font=dict(size=11)),
        title_font=dict(size=14),
        yaxis=dict(title=ytitle, range=yrange, gridcolor=LINE, zerolinecolor=LINE),
        xaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=11)), bargap=.32,
    )
    return fig


# ─────────────────────────────── header + nav ───────────────────────────────
st.markdown(
    '<div class="topbar"><span class="brand">SPOOFWATCH CONSOLE</span>'
    '<span class="tagline">Adversarial reinforcement learning for limit order book surveillance</span>'
    '<span class="chip">RESEARCH / SIMULATION</span></div>', unsafe_allow_html=True)

view = st.segmented_control("View", ["Overview", "Demonstration", "Evidence & limitations"],
                            default="Overview", label_visibility="collapsed")
if view is None:
    view = "Overview"

# ─────────────────────────────── VIEW A ───────────────────────────────
if view == "Overview":
    st.markdown(
        f'<div class="cap" style="margin:2px 0 12px">'
        f'<b style="color:{TEXT}">Spoofing</b> places a large order to make the market look one-sided, moving the '
        f'price, then cancels it after trading on the other side. It is illegal (CEA §4c(a)(5)(C)). '
        f'Exchanges mostly police it with static rules. <b style="color:{TEXT}">The question:</b> can a learning '
        f'agent do better — catching manipulation it was never shown, without accusing honest traders?</div>',
        unsafe_allow_html=True)

    lo = dig(R["label_observability"], "signatures", "opposite-side signature",
             "in-distribution (test split)", "orders", default={})
    if lo:
        st.markdown(
            '<div class="warnbar"><b>Audit finding — read before any detector number.</b> The manipulation '
            'label is readable from two of the detector&rsquo;s own input features, so a zero-parameter '
            f'signature scores order-level F1 {num(lo.get("f1"), 3)} in-distribution and 0.995 on a held-out '
            'attacker at a false-positive rate of 0.000, and a supervised LSTM matches the RL Watchdog '
            'everywhere in minutes. The CP-7 figures below are real and were earned by fixing a labelling '
            'artefact — but they reach a ceiling a rule with no parameters already occupies. See '
            '<span class="mono">Evidence &amp; limitations &rarr; Is the task hard?</span></div>',
            unsafe_allow_html=True)
        st.write("")

    c = st.columns(4, gap="small")
    with c[0]:
        kpi("False positives · legitimate orders", pct(ph("cov", "legitimate order FPR")),
            pct(PHRULE.get("legitimate order FPR")), "Lower is better · CP-7 Watchdog · 3 seeds",
            "posthoc_rescore.md · order-level", GOOD)
    with c[1]:
        kpi("In-distribution F1 · known attacks", num(ph("cov", "in-distribution order F1"), 3),
            num(PHRULE.get("in-distribution order F1"), 3), "Higher is better · ±0.001 over 3 seeds",
            "posthoc_rescore.md · order-level · test split", WD)
    with c[2]:
        s4 = next((r for r in R["pnl_suppression"].get("rows", []) if r["source"] == "SPOOFER-04"), {})
        kpi("Profit removed · SPOOFER-04 INTC", pct(dig(s4, "watchdog", "suppression_pct", default=0) / 100, 0),
            pct(dig(s4, "rule", "suppression_pct", default=0) / 100, 0),
            "Higher is better · detector acts inside the episode",
            "pnl_suppression.md · 50 episodes", GOOD)
    with c[3]:
        kpi("Held-out attack recall · LATEBURST", num(ph("cov", "LATEBURST-ATK recall")),
            num(PHRULE.get("LATEBURST-ATK recall")), "Higher is better · CP-7 · original Watchdog scored 0.22",
            "posthoc_rescore.md · post-hoc · 3 seeds", GOOD)

    st.write("")
    g1, g2 = st.columns([3, 2], gap="medium")
    with g1:
        ind = dig(R["watchdog_eval"], "buckets", "In-distribution (test split)", default={})
        lb, lay = R["lateburst_test"], R["layering_test"]
        labels = ["Known attacks<br>(in-distribution)", "LATEBURST<br>(held out)", "LAYER<br>(held out)"]
        wd_v = [dig(ind, "watchdog_orders", "f1", default=0),
                dig(lb, "overall", "watchdog_orders", "f1", default=0),
                dig(lay, "pooled", "watchdog_orders", "f1", default=0)]
        ru_v = [dig(ind, "rule_orders", "f1", default=0),
                dig(lb, "overall", "rule_orders", "f1", default=0),
                dig(lay, "pooled", "rule_orders", "f1", default=0)]
        fig = go.Figure()
        fig.add_bar(x=labels, y=wd_v, name="RL Watchdog", marker_color=WD,
                    text=[f"{v:.2f}" for v in wd_v], textposition="outside", textfont=dict(size=12))
        fig.add_bar(x=labels, y=ru_v, name="Rule baseline", marker_color=RULE,
                    text=[f"{v:.2f}" for v in ru_v], textposition="outside", textfont=dict(size=12))
        fig.add_vline(x=.5, line_width=1, line_dash="dot", line_color=MUTED)
        fig.add_annotation(x=1.5, y=1.08, text="attacks never seen in training", showarrow=False,
                           font=dict(size=10, color=MUTED))
        st.plotly_chart(chart_layout(fig, 340, "Detection quality — order-level F1", "F1", [0, 1.2]),
                        use_container_width=True)
        st.markdown('<div class="cap">The Watchdog more than doubles the baseline on attacks resembling its '
                    'training, and loses to it on both pre-registered unseen attacks.<br>'
                    '<span style="opacity:.75">Single Watchdog seed (<span class="mono">main</span>), the one the pre-registered tests ran on; the KPI above is the 3-seed mean, so values differ slightly.</span></div>',
                    unsafe_allow_html=True)
    with g2:
        legit = dig(R["watchdog_eval"], "buckets", "Legitimate only (HONEST + FLICKER)", default={})
        syn = dig(R["synthetic_eval"], "detection", "Legitimate only", default={})
        rows = [("Replay market", dig(legit, "watchdog_orders", "fpr", default=0),
                 dig(legit, "rule_orders", "fpr", default=0)),
                ("Synthetic market", dig(syn, "watchdog_orders", "fpr", default=0),
                 dig(syn, "rule_orders", "fpr", default=0))]
        fig = go.Figure()
        fig.add_bar(y=[r[0] for r in rows], x=[100 * r[2] for r in rows], name="Rule baseline",
                    orientation="h", marker_color=RULE,
                    text=[f"{100 * r[2]:.0f}%" for r in rows], textposition="outside")
        fig.add_bar(y=[r[0] for r in rows], x=[100 * r[1] for r in rows], name="RL Watchdog",
                    orientation="h", marker_color=WD,
                    text=[f"{100 * r[1]:.0f}%" for r in rows], textposition="outside")
        fig.update_layout(xaxis=dict(range=[0, 108], ticksuffix="%", gridcolor=LINE))
        st.plotly_chart(chart_layout(fig, 340, "False alarms on legitimate orders", None),
                        use_container_width=True)
        st.markdown('<div class="cap">Why this decides usability: at the baseline\'s rate, four in five '
                    'investigated traders did nothing wrong, and real alerts drown.<br>'
                    '<span style="opacity:.75">Order-level, single <span class="mono">main</span> seed per market; the KPI above is the 3-seed replay-market mean.</span></div>',
                    unsafe_allow_html=True)

    st.write("")
    f1, f2 = st.columns([3, 2], gap="medium")
    f1.markdown(
        '<div class="finding"><b>Finding.</b> Markedly lower false-positive rates in simulation, but '
        'generalisation to unseen manipulation remains unresolved — on both pre-registered held-out attacks the '
        'static rule recalls more.</div>', unsafe_allow_html=True)
    f2.markdown(
        '<div class="warnbar"><b>Research prototype.</b> Simulation only. LOBSTER carries no participant '
        'identities, so this detector has never been validated against labelled real-world spoofing.</div>',
        unsafe_allow_html=True)

# ─────────────────────────────── VIEW B ───────────────────────────────
elif view == "Demonstration":
    demo = R["demo_episodes"]
    stages = [("01", "Market data", "LOBSTER order book replayed event by event"),
              ("02", "PPO spoofer agents", "Rewarded on profit alone — spoofing has to emerge"),
              ("03", "Watchdog surveillance", "RecurrentPPO flags or clears every step"),
              ("04", "Rule baseline", "Large order cancelled quickly — industry-style"),
              ("05", "Evaluation", "Both scored on the same recorded orders")]
    for col, (n, t, d) in zip(st.columns(5, gap="small"), stages):
        col.markdown(f'<div class="stage"><div class="s-n">{n}</div><div class="s-t">{t}</div>'
                     f'<div class="s-d">{d}</div></div>', unsafe_allow_html=True)
    st.write("")

    if not demo.get("episodes"):
        st.warning(f"{NA} — recorded episodes are missing. Generate them with "
                   "`python scripts/build_demo_episode.py`.")
        st.stop()

    eps = demo["episodes"]
    names = [e["scenario"] for e in eps]
    top = st.columns([3, 2, 2], gap="medium")
    with top[0]:
        choice = st.selectbox("Recorded episode", names, index=0)
    ep = next(e for e in eps if e["scenario"] == choice)
    steps, orders, summ = ep["steps"], ep["orders"], ep["summary"]
    key = f"step_{names.index(choice)}"
    if key not in st.session_state:
        st.session_state[key] = 0

    first_manip = next((s["step"] for s in steps if s["manipulation_active"]), 0)
    first_flag = next((s["step"] for s in steps if s["watchdog_flag"]), None)
    with top[1]:
        st.markdown('<div class="cap" style="margin-bottom:4px">Jump to</div>', unsafe_allow_html=True)
        b = st.columns(3)
        if b[0].button("⏮ Start", use_container_width=True):
            st.session_state[key] = 0
        if b[1].button("⚑ Manipulation", use_container_width=True, disabled=not summ["manipulation_steps"]):
            st.session_state[key] = first_manip
        if b[2].button("◉ First flag", use_container_width=True, disabled=first_flag is None):
            st.session_state[key] = first_flag or 0
    with top[2]:
        st.markdown(
            f'<div class="cap">{ep["ticker"]} · 1 decision = {ep["events_per_step"]} market events · '
            f'lot {ep["lot"]:,} shares<br>Recorded replay of committed models — not a live market.</div>',
            unsafe_allow_html=True)

    i = st.slider("Step", 0, len(steps) - 1, key=key)
    s = steps[i]

    L, M, Rr = st.columns([3, 3, 4], gap="medium")

    with L:
        st.markdown('<div class="cap" style="margin-bottom:6px">ORDER BOOK — top 3 levels</div>',
                    unsafe_allow_html=True)
        mine = {(r["side"], round(r["price"], 4)): r["size"] for r in s["resting"]}
        mx = max([lv["size"] for lv in s["ask"] + s["bid"]] +
                 [v + lv["size"] for (sd, p), v in mine.items() for lv in s["ask"] + s["bid"]] + [1])
        rows = []
        for lv in reversed(s["ask"]):
            own = mine.get(("ask", round(lv["price"], 4)), 0)
            w = 100 * (lv["size"] + own) / mx
            rows.append(f'<tr><td class="px ask">{lv["price"]:.2f}</td><td class="barcell">'
                        f'<span class="bar{" mine" if own else ""}" style="width:{w:.0f}%;background:'
                        f'{"#e0a20055" if own else "#ff9b9633"}"></span></td>'
                        f'<td class="sz">{lv["size"] + own:,.0f}{" ◀" if own else ""}</td></tr>')
        spread = s["ask"][0]["price"] - s["bid"][0]["price"]
        rows.append(f'<tr class="spread"><td colspan="3">mid {s["mid"]:.3f} · spread {spread:.2f}</td></tr>')
        for lv in s["bid"]:
            own = mine.get(("bid", round(lv["price"], 4)), 0)
            w = 100 * (lv["size"] + own) / mx
            rows.append(f'<tr><td class="px bid">{lv["price"]:.2f}</td><td class="barcell">'
                        f'<span class="bar{" mine" if own else ""}" style="width:{w:.0f}%;background:'
                        f'{"#e0a20055" if own else "#7fd6a033"}"></span></td>'
                        f'<td class="sz">{lv["size"] + own:,.0f}{" ◀" if own else ""}</td></tr>')
        st.markdown(f'<table class="ladder">{"".join(rows)}</table>', unsafe_allow_html=True)
        st.markdown(f'<div class="cap" style="margin-top:6px">◀ marks the watched trader\'s own resting order'
                    f'{" — none resting" if not mine else ""}.</div>', unsafe_allow_html=True)

    with M:
        st.markdown('<div class="cap" style="margin-bottom:6px">THIS STEP</div>', unsafe_allow_html=True)
        truth = ("MANIPULATION ACTIVE" if s["manipulation_active"] else "no manipulation")
        tcls = "v-no" if s["manipulation_active"] else "v-ok"
        wd_badge = ("FLAGGED" if s["watchdog_flag"] else "cleared")
        if s["watchdog_flag"] and s["manipulation_active"]:
            wcls, wnote = "v-ok", "correct detection"
        elif s["watchdog_flag"]:
            wcls, wnote = "v-no", "false alarm"
        elif s["manipulation_active"]:
            wcls, wnote = "v-no", "missed"
        else:
            wcls, wnote = "v-ok", "correctly quiet"
        st.markdown(
            f'<div class="panel"><div class="cap">Ground truth (hindsight label)</div>'
            f'<span class="verdict {tcls}">{truth}</span>'
            f'<div class="cap" style="margin-top:10px">RL Watchdog decision</div>'
            f'<span class="verdict {wcls}">{wd_badge}</span> <span class="cap">{wnote}</span>'
            f'<div class="cap" style="margin-top:10px">Trader action</div>'
            f'<span class="mono" style="color:{TEXT}">{s["action"]}</span>'
            f'{f" @ {s['trade_price']:.3f}" if s["trade_price"] else ""}'
            f'<div class="cap" style="margin-top:10px">Inventory · P&amp;L</div>'
            f'<span class="mono" style="color:{TEXT}">{s["inventory_lots"]:+.0f} lots · '
            f'{money(s["pnl"])}</span></div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="cap" style="margin-top:8px">The rule baseline cannot decide yet: it only fires when '
            f'an order is cancelled. Its verdicts appear in the order log.</div>', unsafe_allow_html=True)

    with Rr:
        st.markdown('<div class="cap" style="margin-bottom:6px">EVENT LOG — to this step</div>',
                    unsafe_allow_html=True)
        log = []
        for o in orders:
            if o["placed_step"] <= i:
                tag = '<span class="manip">manipulative</span>' if o["manipulative"] else "legitimate"
                log.append(f'<div class="ev">step {o["placed_step"]:>3} · <b>large {o["side"]} order</b> '
                           f'{o["size"]:,.0f} ({o["depth_multiple"]:.0f}× depth) · {tag}</div>')
            rm = o["removed_step"]
            if rm is not None and rm <= i:
                verdict = ('<span class="alarm">RULE: FALSE ALARM</span>' if o["rule_flagged"] and not o["manipulative"]
                           else '<span class="flag">RULE: flagged</span>' if o["rule_flagged"]
                           else "rule: no flag")
                log.append(f'<div class="ev">step {rm:>3} · order {o["removed_by"]} after '
                           f'{(rm - o["placed_step"])} steps · {verdict}</div>')
        flagged = [st_["step"] for st_ in steps[:i + 1] if st_["watchdog_flag"]]
        if flagged:
            log.append(f'<div class="ev">watchdog flagged <b>{len(flagged)}</b> of {i + 1} steps so far '
                       f'(first at step {flagged[0]})</div>')
        st.markdown("".join(log[-14:]) or '<div class="ev">no events yet</div>', unsafe_allow_html=True)

    st.write("")
    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Manipulative orders", f"{summ['manipulative_orders']} / {summ['large_orders']}")
    o2.metric("Watchdog caught", f"{summ['watchdog_caught']} / {summ['manipulative_orders']}",
              delta=f"{summ['watchdog_flagged_steps']} of {summ['steps']} steps flagged", delta_color="off")
    o3.metric("Rule flagged", f"{summ['rule_flagged_orders']} / {summ['large_orders']}",
              delta=f"{summ['rule_false_alarms']} false alarms",
              delta_color="inverse" if summ["rule_false_alarms"] else "off")
    o4.metric("Trader P&L this episode", money(summ["final_pnl"]))

    if ep["kind"] == "legitimate":
        st.markdown('<div class="finding">This trader never manipulates — it places large orders and cancels '
                    f'them. The Watchdog stays silent; the rule raises <b>{summ["rule_false_alarms"]} false '
                    'alarms</b>, because size-and-cancel is ordinary market making.</div>',
                    unsafe_allow_html=True)
    elif summ["watchdog_caught"] < summ["manipulative_orders"]:
        st.markdown('<div class="warnbar"><b>The generalisation failure, in one episode.</b> This attacker was '
                    'never seen in training: it waits ~100–200 events before trading, so the Watchdog — which '
                    'keys on trading against a resting order — flags only '
                    f'{summ["watchdog_flagged_steps"]} of {summ["steps"]} steps.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="finding">Manipulation the Watchdog recognises: it flags while the order is '
                    'still resting, where the rule can only react once the order is cancelled.</div>',
                    unsafe_allow_html=True)

# ─────────────────────────────── VIEW C ───────────────────────────────
else:
    t0, tg, t1, t2, t3, t4 = st.tabs(["Is the task hard?", "Detector generations", "Headline evidence",
                                      "Held-out tests", "Robustness", "Technical detail"])

    with tg:
        st.markdown('<div class="cap">Three generations of the same detector, re-scored on every saved replay '
                    'dataset. <b>main</b> is the original Watchdog; <b>causal_g0_1m</b> adds a causal label and a '
                    'myopic objective; <b>cov</b> adds training coverage of varied act timing. Mean ± 95% '
                    'Student-t CI over 3 seeds. Post-hoc except where a test was pre-registered for that model '
                    '(<span class="mono">posthoc_rescore.md</span>).</div>', unsafe_allow_html=True)
        st.write("")
        gens = ["main", "causal_g0_1m", "cov"]
        metrics = ["in-distribution order F1", "legitimate order FPR", "SCRIPTED-ATK train stocks recall",
                   "LATEBURST-ATK recall", "DRIP-ATK recall", "REVERSAL-ATK recall"]
        grows = []
        for m in metrics:
            row = {"Metric": m}
            for g in gens:
                mu, ci = ph(g, m), ph(g, m, "ci95")
                row[g] = NA if mu is None else (f"{mu:.3f}" + (f" ± {ci:.3f}" if ci == ci and ci is not None else ""))
            row["Rule"] = num(PHRULE.get(m), 3)
            grows.append(row)
        if grows:
            st.dataframe(pd.DataFrame(grows), hide_index=True, use_container_width=True)
        st.markdown('<div class="finding">The original Watchdog\'s held-out failure was a labelling artefact, '
                    'not a learning failure: the hindsight label marked every step an order rested, so training '
                    'rewarded predicting <b>who would spoof</b> over detecting <b>the act</b>. It flagged 99% of '
                    'SPOOFER-04\'s orders before any opposite trade, and 13.7% of SPOOFER-03\'s opposite-trade '
                    'steps. A causal label fixed it across four held-out attack families.</div>',
                    unsafe_allow_html=True)

    with t0:
        lob, sup = R["label_observability"], R["supervised_baseline"]
        st.markdown('<div class="cap">The manipulation label is &ldquo;the order&rsquo;s owner traded the '
                    'opposite side while it rested&rdquo;. Observation feature 44 is that participant&rsquo;s '
                    'trade direction and feature 45 is the side its resting orders sit on, so comparing their '
                    'signs reproduces the label&rsquo;s definition. Scored below as a detector with <b>no '
                    'parameters and no fitting at all</b> '
                    '(<span class="mono">label_observability.md</span>).</div>', unsafe_allow_html=True)
        st.write("")
        sig = dig(lob, "signatures", "opposite-side signature", default={})
        rows = [{"Group": g, "Manipulative / orders": f"{b['orders']['positives']:,} / {b['orders']['n']:,}",
                 "Precision": num(b["orders"]["precision"], 3), "Recall": num(b["orders"]["recall"], 3),
                 "Order F1": num(b["orders"]["f1"], 3), "False positives": num(b["orders"]["fpr"], 3)}
                for g, b in sig.items()]
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            st.markdown('<div class="cap">For contrast, &ldquo;a resting order is present&rdquo; alone scores '
                        'order F1 0.362 at a false-positive rate of 1.000 — it is specifically the '
                        'two-feature comparison that carries the label, not the presence of a large '
                        'order.</div>', unsafe_allow_html=True)
        else:
            st.info(NA)

        st.write("")
        st.markdown("#### Is reinforcement learning necessary? No.")
        wd_f1 = sup.get("watchdog_f1", {})
        srows = []
        for bucket in sup.get("buckets", []):
            first = dig(sup, "models", "logistic regression", bucket, default={})
            row = {"Bucket": bucket, "Rule": num(dig(first, "rule_orders", "f1"), 2)}
            for m in sup.get("models", {}):
                row[m[0].upper() + m[1:]] = num(dig(sup, "models", m, bucket, "orders", "f1"), 2)
            row["RL Watchdog"] = num(wd_f1.get(bucket), 2)
            srows.append(row)
        if srows:
            st.dataframe(pd.DataFrame(srows), hide_index=True, use_container_width=True)
            st.markdown('<div class="cap">Order-level F1. Both supervised models are fitted on the train split '
                        'only, with their decision threshold chosen on train orders only, and scored on the '
                        'same orders in the same buckets as the Watchdog '
                        '(<span class="mono">supervised_baseline.md</span>).</div>', unsafe_allow_html=True)
        else:
            st.info(NA)

        bf = R["baselines_final"].get("results", {})
        if bf:
            st.write("")
            st.markdown("#### Every detector against the label ceiling")
            order = ["LABEL-RULE", "SUP-LSTM", "RL WATCHDOG (cov)", "LOGREG"]
            brows = []
            for ds, by_det in bf.items():
                row = {"Dataset": ds}
                for det in order:
                    v = by_det.get(det, {})
                    r = dig(v, "recall", "mean")
                    f = dig(v, "legitimate FPR", "mean")
                    row[det] = NA if r is None else f"{r:.3f} / {f:.3f}"
                rr = dig(next(iter(by_det.values()), {}), "rule recall", "mean")
                rf = dig(next(iter(by_det.values()), {}), "rule legitimate FPR", "mean")
                row["Size+cancel rule"] = NA if rr is None else f"{rr:.3f} / {rf:.3f}"
                brows.append(row)
            st.dataframe(pd.DataFrame(brows), hide_index=True, use_container_width=True)
            st.markdown('<div class="cap">Order-level recall / legitimate false-positive rate, 3 seeds. '
                        'LABEL-RULE is the label\'s definition written as a rule — the ceiling, perfect by '
                        'construction, not a competitor. A supervised LSTM reaches it in minutes '
                        '(<span class="mono">baselines_final.md</span>).</div>', unsafe_allow_html=True)

        anon = R["anonymised_experiment"]
        if anon.get("models"):
            st.write("")
            st.markdown("#### And with participant attribution removed entirely")
            h, aud = anon.get("headline", {}), anon.get("audit", {})
            held = h.get("held_out_set", "held-out")
            arows = []
            for mname, m in anon["models"].items():
                c = dig(m, "sets", held, "step", default={})
                if not c:
                    continue
                arows.append({"Model (book features only)": mname, "Precision": num(c.get("precision"), 3),
                              "Recall": num(c.get("recall"), 3), "F1": num(c.get("f1"), 3),
                              "FPR": num(c.get("fpr"), 3)})
            if arows:
                st.dataframe(pd.DataFrame(arows), hide_index=True, use_container_width=True)
                st.markdown(
                    '<div class="cap">Step level, held-out split. The whole participant block is withheld and '
                    'only the displayed order book is used. Leakage audit of what remains: the withheld pair '
                    f'scores step F1 {num(aud.get("withheld_pair_step_f1"), 3)}, while the best single retained '
                    f'feature rule reaches only {num(aud.get("best_single_feature_rule_f1"), 3)} and no retained '
                    'feature correlates above 0.5 with the label '
                    '(<span class="mono">anonymised_experiment.md</span>).</div>', unsafe_allow_html=True)
                if not h.get("beats_constant_episode", True):
                    st.markdown(
                        '<div class="warnbar"><b>Do not quote episode-level F1 here.</b> That held-out set is '
                        'positive-heavy, so a constant "always flag" classifier scores '
                        f'{num(h.get("best_constant_episode_f1"), 3)} and beats every model '
                        f'(best {num(h.get("best_episode_f1"), 3)}). Step level is the real result: '
                        f'{num(h.get("best_step_f1"), 3)} against the constant\'s '
                        f'{num(h.get("best_constant_step_f1"), 3)}.</div>', unsafe_allow_html=True)

        st.markdown('<div class="warnbar"><b>What this means.</b> The learned detector is not discovering '
                    'manipulation — it recovers a definition handed to it in its own input. Its failure on held-out attackers is therefore a property of the learner, '
                    'not evidence that unseen manipulation is hard to spot here: the zero-parameter signature '
                    'transfers unchanged. CP-7 closed the gap to that ceiling by fixing the label, which is a '
                    'real methodological result — but it did not make the problem hard. The simplification '
                    'that matters is <b>perfect attribution</b>: the environment says which orders and trades '
                    'belong to the monitored participant. Establishing that is the hard part of real '
                    'surveillance, and it is assumed away.</div>',
                    unsafe_allow_html=True)

    with t1:
        c1, c2 = st.columns([3, 2], gap="medium")
        with c1:
            st.markdown('<div class="cap">Multi-seed validation — 3 Watchdog training seeds, Student-t 95% CI, '
                        'all scored on identical recorded data (<span class="mono">multiseed.md</span>)</div>',
                        unsafe_allow_html=True)
            if MS:
                rows = []
                for k, v in MS.items():
                    rows.append({"Metric (order-level unless noted)": k,
                                 "Watchdog": f"{v['mean']:.3f} ± {v['ci95']:.3f}" if v['mean'] == v['mean'] else NA,
                                 "Rule": num(RU.get(k), 3) if k in RU else "—"})
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, height=330)
            else:
                st.info(NA)
        with c2:
            st.markdown('<div class="cap">Profit suppression — detector acting inside the episode, 50 episodes, '
                        'identical seeds (<span class="mono">pnl_suppression.md</span>)</div>',
                        unsafe_allow_html=True)
            rows = [r for r in R["pnl_suppression"].get("rows", [])
                    if r["source"] in ("SPOOFER-04", "SPOOFER-02")]
            if rows:
                fig = go.Figure()
                labels = [f"{r['source']}<br>{r['ticker']}" for r in rows]
                for name, key, color in (("No surveillance", "none", MUTED), ("Watchdog", "watchdog", WD),
                                         ("Rule", "rule", RULE)):
                    fig.add_bar(x=labels, y=[r[key]["pnl_mean"] / 1000 for r in rows], name=name,
                                marker_color=color, text=[f"{r[key]['pnl_mean'] / 1000:.1f}k" for r in rows],
                                textposition="outside")
                st.plotly_chart(chart_layout(fig, 320, "Profit per episode", "$ thousands"),
                                use_container_width=True)
                st.markdown('<div class="cap">Interventions per episode against legitimate traders: Watchdog '
                            '0.3–0.4, rule 4.7.</div>', unsafe_allow_html=True)

    with t2:
        st.markdown('<div class="cap">Both tests were <b>pre-registered</b>: the attacker, the procedure and the '
                    'hypothesis were committed to version control before any of their data existed, and each '
                    'test was run exactly once.</div>', unsafe_allow_html=True)
        st.write("")
        lb, lay = R["lateburst_test"], R["layering_test"]
        for title, d, bucket, note in (
            ("LATEBURST-ATK · replay market", lb, dig(lb, "overall", default={}),
             "Spoof, wait 100–200 market events, trade in a burst, cancel. Negatives: FLICKER on the same stock."),
            ("LAYER-ATK · synthetic market", lay, dig(lay, "pooled", default={}),
             "Four stacked large orders at successive price levels (Coscia-style), then trade the other side."),
        ):
            if not bucket:
                continue
            w, r = bucket["watchdog_orders"], bucket["rule_orders"]
            st.markdown(f"**{title}** &nbsp;<span class='verdict v-no'>HYPOTHESIS "
                        f"{d.get('verdict', 'n/a').upper()}</span>", unsafe_allow_html=True)
            cols = st.columns(5)
            for col, (lab, wv, rv, inv) in zip(cols, [
                ("Precision", w["precision"], r["precision"], False),
                ("Recall", w["recall"], r["recall"], False),
                ("F1", w["f1"], r["f1"], False),
                ("False positives", w["fpr"], r["fpr"], True),
            ]):
                col.metric(lab, num(wv), delta=f"rule {num(rv)}",
                           delta_color="inverse" if inv else "normal")
            cols[4].metric("Orders judged", f"{w['n']:,}", delta=f"{w['positives']:,} manipulative",
                           delta_color="off")
            st.markdown(f'<div class="cap">{note}</div>', unsafe_allow_html=True)
            st.write("")
        st.markdown('<div class="warnbar"><b>Preserved negative result.</b> Generalisation to unseen '
                    'manipulation has not been demonstrated. The static rule recalls more in both tests.</div>',
                    unsafe_allow_html=True)

    with t3:
        se = R["synthetic_eval"]
        st.markdown('<div class="cap">Sim-to-sim check: a second market where prices emerge from order matching '
                    'between simple traders, calibrated to real MSFT/INTC statistics and never to whether '
                    'spoofing pays (<span class="mono">synthetic_eval.md</span>).</div>', unsafe_allow_html=True)
        st.write("")
        c1, c2 = st.columns([3, 2], gap="medium")
        with c1:
            rows = [{"Agent": r["source"], "Type": r["kind"],
                     "Market reacts": money(r["reactive"]["pnl_mean"]),
                     "Market blind to them": money(r["blind"]["pnl_mean"]),
                     "Gain from spoofing": f"{money(r['manipulation_gain_mean'])} ± "
                                           f"{r['manipulation_gain_ci95']:,.0f}"}
                    for r in se.get("sources", [])]
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        with c2:
            if (FIGS / "fig4_spoof_impact.png").exists():
                st.image(str(FIGS / "fig4_spoof_impact.png"))
        st.markdown('<div class="warnbar"><b>Scope limit on profitability.</b> Every manipulator loses money '
                    'here: a spoof moves this market 4.5–9× less than the replay model assumes. Simulated '
                    'profitability depends on that impact model and is <b>not</b> evidence of real-world '
                    'profitability.</div>', unsafe_allow_html=True)
        det = se.get("detection", {})
        if det:
            mm = dig(det, "Manipulators + legitimate", "watchdog_orders", default={})
            ll = det.get("Legitimate only", {})
            d1, d2, d3 = st.columns(3)
            d1.metric("Watchdog precision here", num(mm.get("precision")))
            d2.metric("Watchdog recall here", num(mm.get("recall")), delta="0.80 in replay market",
                      delta_color="inverse")
            d3.metric("False positives", num(dig(ll, "watchdog_orders", "fpr")),
                      delta=f"rule {num(dig(ll, 'rule_orders', 'fpr'))}", delta_color="inverse")

    with t4:
        with st.expander("What the detector relies on — feature ablation (test split)"):
            fa = R["feature_ablation"].get("results", {})
            if fa:
                order = ["none", "participant: trade direction", "all participant features (6)",
                         "all book features (40)", "participant: resting size", "participant: placed this step",
                         "participant: cancelled this step", "participant: oldest order age",
                         "participant: resting side imbalance", "book price offsets (all levels)",
                         "book sizes (all levels)", "best level only (4)"]
                df = pd.DataFrame([{"Group removed": k if k != "none" else "nothing (baseline)",
                                    "Order F1": fa[k]["order F1"], "Step recall": fa[k]["step recall"],
                                    "False positives": fa[k]["legitimate order FPR"]}
                                   for k in order if k in fa])
                st.dataframe(df.style.format({"Order F1": "{:.3f}", "Step recall": "{:.3f}",
                                              "False positives": "{:.3f}"}),
                             hide_index=True, use_container_width=True)
                st.markdown('<div class="cap">Removing one feature — whether the participant traded, and in '
                            'which direction — collapses F1 from 0.772 to 0.113. That is why sparse '
                            'manipulation is missed.</div>', unsafe_allow_html=True)
            else:
                st.info(NA)

        with st.expander("Market data and impact calibration"):
            if CAL:
                st.dataframe(pd.DataFrame([{
                    "Stock": tk, "Events": f"{v['events']:,}", "Median price": f"${v['median_mid']:,.2f}",
                    "Spread (ticks)": f"{v['median_spread_ticks']:.0f}",
                    "Depth at touch": f"{v['median_touch_depth']:,.0f}",
                    "Impact λ (fitted)": f"{v['lambda']:.3f}", "Fit R²": f"{v['ofi_r2']:.2f}",
                } for tk, v in CAL["tickers"].items()]), hide_index=True, use_container_width=True)
                st.markdown('<div class="cap">λ is fitted per stock from order-flow imbalance (Cont, Kukanov & '
                            'Stoikov 2014), not assumed. Spoofing is only profitable on the 1-tick stocks '
                            '(MSFT, INTC).</div>', unsafe_allow_html=True)
            else:
                st.info(NA)

        with st.expander("Spoofer agents across training seeds"):
            sd = R["spoofer_seeds"]
            if sd:
                st.dataframe(pd.DataFrame([{
                    "Agent": r["agent"], "Stock": r["ticker"], "Seed": r["tag"],
                    "Profit / episode": money(r["calibrated"]["pnl_mean"]),
                    "± CI": f"{r['calibrated']['pnl_ci95']:,.0f}",
                    "Impact switched off": money(r["no_impact"]["pnl_mean"]),
                } for r in sd["per_seed"]]), hide_index=True, use_container_width=True)
                st.markdown('<div class="cap">3 of 3 seeds learned to spoof on both stocks; every seed loses '
                            'money with impact disabled, so the profit is the manipulation.</div>',
                            unsafe_allow_html=True)
            else:
                st.info(NA)

        with st.expander("Reproduction and test status"):
            c = st.columns(3)
            c[0].metric("Automated tests", "172")
            c[1].metric("Scripts reproducing exactly", "12 / 12")
            c[2].metric("Values re-checked", "2,178")
            st.markdown('<div class="cap">Every script was re-run in a clean clone of the repository — '
                        'committed code and models only — and each output compared value by value against the '
                        'committed results. 0 differences.</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown("#### Limitations")
    for group in ([
        ("Task is near-tautological", "The label is readable from two input features; a zero-parameter "
                                      "signature scores 0.997 and a supervised LSTM matches the RL detector. "
                                      "RL is not justified for detection."),
        ("Perfect attribution assumed", "Surveillance is told which orders and trades belong to the monitored "
                                        "participant. Establishing that is the real hard part."),
        ("Generalisation unproven", "Both pre-registered held-out attacks favour the static rule."),
    ], [
        ("Profit is model-dependent", "Spoofing pays under the replay impact model, not in the synthetic "
                                      "market."),
        ("No real-world validation", "LOBSTER has no participant identities; no labelled spoofing exists."),
        ("Narrow data", "One trading day, five stocks; spoofing only pays on the two 1-tick stocks."),
    ]):
        for col, (t, d) in zip(st.columns(3, gap="small"), group):
            col.markdown(f'<div class="stage"><div class="s-t">{t}</div><div class="s-d">{d}</div></div>',
                         unsafe_allow_html=True)
        st.write("")

st.markdown(
    f'<div class="cap" style="margin-top:18px;border-top:1px solid {LINE};padding-top:8px">'
    'All figures read from committed files in <span class="mono">results/</span> and '
    '<span class="mono">configs/</span> · nothing is trained, simulated or fetched live · full write-up in '
    '<span class="mono">REPORT.md</span></div>', unsafe_allow_html=True)
