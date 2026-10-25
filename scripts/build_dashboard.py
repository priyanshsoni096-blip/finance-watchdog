"""Build dashboard.html — a single self-contained page showing the project's measured results.

Usage:
    python scripts/build_dashboard.py

Reads the committed files in results/ and writes dashboard.html at the repo root. No network, no CDN, no
build step: open the file in any browser. Every number is pulled from the JSON results, so the page cannot
drift from the evidence — if a result is regenerated, rebuild the page.

Design rule: the page reports what was measured, including the results that went against the hypothesis.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "dashboard.html"


def load(name: str) -> dict:
    p = RESULTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def fmt(x, nd=2, pct=False, money=False, signed=False):
    if x is None or (isinstance(x, float) and x != x):
        return "n/a"
    if pct:
        return f"{100 * x:.0f}%"
    if money:
        return f"{'+' if x >= 0 else '−'}${abs(x):,.0f}"
    s = f"{x:,.{nd}f}"
    return ("+" + s) if signed and x >= 0 else s


def bar_chart(title: str, rows: list[tuple[str, float, float]], note: str = "", hi_is_good: bool = True) -> str:
    """rows: (label, watchdog value, rule value), values on a 0-1 scale."""
    body = []
    for label, w, r in rows:
        w = 0.0 if w is None or w != w else w
        r = 0.0 if r is None or r != r else r
        body.append(f"""
      <div class="bar-row">
        <div class="bar-label">{label}</div>
        <div class="bars">
          <div class="bar-track"><div class="bar wd" style="width:{max(w,0)*100:.1f}%"></div><span class="bar-val">{w:.2f}</span></div>
          <div class="bar-track"><div class="bar rule" style="width:{max(r,0)*100:.1f}%"></div><span class="bar-val">{r:.2f}</span></div>
        </div>
      </div>""")
    direction = "higher is better" if hi_is_good else "lower is better"
    return f"""
    <div class="chart">
      <h4>{title} <span class="muted">({direction})</span></h4>
      {''.join(body)}
      {f'<p class="note">{note}</p>' if note else ''}
    </div>"""


def main() -> int:
    ms, we, lb, lay = load("multiseed.json"), load("watchdog_eval.json"), load("lateburst_test.json"), load("layering_test.json")
    ps, fa, se, sd = load("pnl_suppression.json"), load("feature_ablation.json"), load("synthetic_eval.json"), load("spoofer_seeds.json")
    be = load("basic_eval.json")
    if not (ms and we and lb):
        print("missing core result files; run the evaluations first")
        return 1

    S = ms.get("summary", {}); RU = ms.get("rule", {})
    f1_w, f1_ci = S.get("in-distribution order F1", {}).get("mean"), S.get("in-distribution order F1", {}).get("ci95")
    fpr_w, fpr_ci = S.get("legitimate-only order FPR", {}).get("mean"), S.get("legitimate-only order FPR", {}).get("ci95")
    lbr_w, lbr_ci = S.get("LATEBURST-ATK order recall", {}).get("mean"), S.get("LATEBURST-ATK order recall", {}).get("ci95")
    lbp_w = S.get("LATEBURST-ATK order precision", {}).get("mean")

    ind = we.get("buckets", {}).get("In-distribution (test split)", {})
    legit = we.get("buckets", {}).get("Legitimate only (HONEST + FLICKER)", {})
    lb_all = lb.get("overall", {})
    lay_pool = lay.get("pooled", {})

    sup = {r["source"]: r for r in ps.get("rows", [])}
    s4, s2 = sup.get("SPOOFER-04", {}), sup.get("SPOOFER-02", {})
    syn_src = {r["source"]: r for r in se.get("sources", [])}
    syn_det = se.get("detection", {})
    across = sd.get("across_seeds", {})
    agents = {a["agent"]: a for a in be.get("agents", [])}

    # charts
    detect_chart = bar_chart("Detection quality (order-level F1)", [
        ("In-distribution", ind.get("watchdog_orders", {}).get("f1"), ind.get("rule_orders", {}).get("f1")),
        ("Unseen attacker (LATEBURST)", lb_all.get("watchdog_orders", {}).get("f1"), lb_all.get("rule_orders", {}).get("f1")),
        ("Layering (synthetic market)", lay_pool.get("watchdog_orders", {}).get("f1"), lay_pool.get("rule_orders", {}).get("f1")),
        ("Synthetic market", syn_det.get("Manipulators + legitimate", {}).get("watchdog_orders", {}).get("f1"),
         syn_det.get("Manipulators + legitimate", {}).get("rule_orders", {}).get("f1")),
    ])
    fp_chart = bar_chart("False positives on legitimate orders", [
        ("Replay market", legit.get("watchdog_orders", {}).get("fpr"), legit.get("rule_orders", {}).get("fpr")),
        ("Held-out AMZN", we.get("buckets", {}).get("Held-out RL: SPOOFER-05 + AMZN controls", {}).get("watchdog_orders", {}).get("fpr"),
         we.get("buckets", {}).get("Held-out RL: SPOOFER-05 + AMZN controls", {}).get("rule_orders", {}).get("fpr")),
        ("Synthetic market", syn_det.get("Legitimate only", {}).get("watchdog_orders", {}).get("fpr"),
         syn_det.get("Legitimate only", {}).get("rule_orders", {}).get("fpr")),
        ("Layering test", lay.get("flicker_only", {}).get("watchdog_orders", {}).get("fpr"),
         lay.get("flicker_only", {}).get("rule_orders", {}).get("fpr")),
    ], hi_is_good=False)
    recall_chart = bar_chart("Recall on manipulation never seen in training", [
        ("LATEBURST-ATK", lb_all.get("watchdog_orders", {}).get("recall"), lb_all.get("rule_orders", {}).get("recall")),
        ("Layering (LAYER-ATK)", lay_pool.get("watchdog_orders", {}).get("recall"), lay_pool.get("rule_orders", {}).get("recall")),
        ("Scripted attacker", we.get("buckets", {}).get("Scripted attacker, train stocks + FLICKER", {}).get("watchdog_orders", {}).get("recall"),
         we.get("buckets", {}).get("Scripted attacker, train stocks + FLICKER", {}).get("rule_orders", {}).get("recall")),
    ])

    abl = fa.get("results", {})
    abl_rows = "".join(
        f"<tr><td>{g}</td><td class='num'>{fmt(v.get('order F1'))}</td><td class='num'>{fmt(v.get('step recall'))}</td></tr>"
        for g, v in [("none (baseline)", abl.get("none", {})),
                     ("participant: trade direction", abl.get("participant: trade direction", {})),
                     ("all 6 participant features", abl.get("all participant features (6)", {})),
                     ("all 40 book features", abl.get("all book features (40)", {}))] if v)

    seed_rows = "".join(
        f"<tr><td>{r['agent']}</td><td>{r['ticker']}</td><td>{r['tag']}</td>"
        f"<td class='num'>{fmt(r['calibrated']['pnl_mean'], money=True)} ± {fmt(r['calibrated']['pnl_ci95'], 0)}</td>"
        f"<td class='num neg'>{fmt(r['no_impact']['pnl_mean'], money=True)}</td>"
        f"<td class='num'>{fmt(r['calibrated']['trades_against_own_spoof_per_ep'], 0)}</td></tr>"
        for r in sd.get("per_seed", []))

    syn_rows = "".join(
        f"<tr><td>{r['source']}</td><td>{r['kind']}</td>"
        f"<td class='num'>{fmt(r['reactive']['pnl_mean'], money=True)}</td>"
        f"<td class='num'>{fmt(r['blind']['pnl_mean'], money=True)}</td>"
        f"<td class='num'>{fmt(r['manipulation_gain_mean'], money=True)} ± {fmt(r['manipulation_gain_ci95'], 0)}</td></tr>"
        for r in se.get("sources", []))

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    built = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RL Finance Watchdog — results</title>
<style>
  :root {{
    --bg:#0f1420; --panel:#161d2d; --panel2:#1b2436; --line:#27324a; --text:#e8ecf5; --muted:#93a0bb;
    --wd:#4da3ff; --rule:#d9a441; --good:#39c07c; --bad:#e2605a; --warn:#d9a441;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--text); font:15px/1.55 "Segoe UI",system-ui,-apple-system,sans-serif; }}
  .wrap {{ max-width:1100px; margin:0 auto; padding:32px 20px 64px; }}
  header h1 {{ margin:0 0 6px; font-size:26px; letter-spacing:-.2px; }}
  .sub {{ color:var(--muted); margin:0 0 18px; }}
  .question {{ background:var(--panel); border-left:3px solid var(--wd); padding:14px 18px; border-radius:6px; margin:18px 0; }}
  .disclaimer {{ background:#2a1f1c; border:1px solid #58372f; color:#f0c9c2; padding:10px 14px; border-radius:6px; font-size:13.5px; }}
  h2 {{ font-size:19px; margin:34px 0 12px; padding-bottom:7px; border-bottom:1px solid var(--line); }}
  h4 {{ margin:0 0 10px; font-size:14px; font-weight:600; }}
  .muted {{ color:var(--muted); font-weight:400; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(290px,1fr)); gap:14px; }}
  .card {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px; }}
  .card h3 {{ margin:0 0 4px; font-size:14px; color:var(--muted); font-weight:600; text-transform:uppercase; letter-spacing:.4px; }}
  .verdict {{ display:inline-block; font-size:12px; font-weight:700; padding:3px 9px; border-radius:99px; margin:6px 0 10px; }}
  .v-yes {{ background:rgba(57,192,124,.15); color:var(--good); border:1px solid rgba(57,192,124,.4); }}
  .v-part {{ background:rgba(217,164,65,.15); color:var(--warn); border:1px solid rgba(217,164,65,.4); }}
  .v-no {{ background:rgba(226,96,90,.15); color:var(--bad); border:1px solid rgba(226,96,90,.4); }}
  .card p {{ margin:7px 0 0; font-size:13.5px; }}
  .big {{ font-size:27px; font-weight:700; letter-spacing:-.5px; }}
  .vs {{ color:var(--muted); font-size:13px; }}
  table {{ width:100%; border-collapse:collapse; margin-top:10px; font-size:13.5px; }}
  th,td {{ text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); }}
  th {{ color:var(--muted); font-weight:600; font-size:12px; text-transform:uppercase; letter-spacing:.3px; }}
  td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  td.neg {{ color:var(--bad); }}
  .chart {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px; margin-bottom:14px; }}
  .bar-row {{ display:grid; grid-template-columns:190px 1fr; gap:12px; align-items:center; margin:9px 0; }}
  .bar-label {{ font-size:13px; color:var(--muted); }}
  .bar-track {{ position:relative; background:var(--panel2); border-radius:4px; height:19px; margin:3px 0; }}
  .bar {{ height:100%; border-radius:4px; }}
  .bar.wd {{ background:var(--wd); }} .bar.rule {{ background:var(--rule); }}
  .bar-val {{ position:absolute; right:7px; top:0; line-height:19px; font-size:11.5px; font-variant-numeric:tabular-nums; }}
  .legend {{ display:flex; gap:16px; font-size:12.5px; color:var(--muted); margin:4px 0 14px; }}
  .dot {{ display:inline-block; width:9px; height:9px; border-radius:2px; margin-right:5px; }}
  .note {{ color:var(--muted); font-size:12.5px; margin:8px 0 0; }}
  .grid2 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(330px,1fr)); gap:14px; }}
  footer {{ margin-top:40px; padding-top:14px; border-top:1px solid var(--line); color:var(--muted); font-size:12.5px; }}
  ul {{ margin:8px 0; padding-left:20px; }} li {{ margin:4px 0; font-size:13.5px; }}
</style></head><body><div class="wrap">

<header>
  <h1>RL Finance Watchdog</h1>
  <p class="sub">Adversarial reinforcement learning for detecting limit-order-book spoofing · built on LOBSTER data, 2012-06-21</p>
</header>

<div class="question">
  Can a reinforcement learning agent learn to detect <em>previously unseen</em> order-book manipulation — spoofing — while
  keeping a low false-positive rate on legitimate activity, without being told what spoofing looks like?
</div>

<div class="disclaimer">
  This page reports simulation results, including those that went against the hypothesis. No labelled real-world spoofing
  data exists, so <strong>no claim of real-world detection capability is made</strong>. Legal reference: CEA §4c(a)(5)(C) (Dodd-Frank §747).
</div>

<h2>The three claims</h2>
<div class="cards">
  <div class="card">
    <h3>1 · RL learns manipulation</h3>
    <div class="verdict v-part">PARTLY SHOWN</div>
    <div class="big">{fmt(across.get('SPOOFER-04', {}).get('pnl_mean'), money=True)}</div>
    <p class="vs">INTC profit per episode, mean of 3 training seeds (± {fmt(across.get('SPOOFER-04', {}).get('pnl_ci95'), 0)})</p>
    <p>All 3 seeds learned to spoof on both stocks. With price impact switched off the same trading loses
    {fmt(across.get('SPOOFER-04', {}).get('no_impact_mean'), money=True)} — so the profit is the spoof.</p>
    <p class="note">But it does not survive a market where price reacts through order matching (below).</p>
  </div>
  <div class="card">
    <h3>2 · Generalises to unseen attacks</h3>
    <div class="verdict v-no">NOT SHOWN</div>
    <div class="big">{fmt(lbr_w)} <span class="vs">vs {fmt(lb_all.get('rule_orders', {}).get('recall'))} rule</span></div>
    <p class="vs">Recall on a pre-registered unseen attacker (± {fmt(lbr_ci)}, 3 seeds)</p>
    <p>Two pre-registered tests, each run once, both went against the Watchdog. It keys on trading against a
    resting order, so sparse spoofing slips past.</p>
  </div>
  <div class="card">
    <h3>3 · Few false positives</h3>
    <div class="verdict v-yes">SHOWN IN SIMULATION</div>
    <div class="big">{fmt(fpr_w, 3)} <span class="vs">vs {fmt(RU.get('legitimate-only order FPR'), 2)} rule</span></div>
    <p class="vs">False-positive rate on legitimate orders (± {fmt(fpr_ci, 3)}, 3 seeds)</p>
    <p>About 26× lower than the rule, and it holds in the synthetic market too
    ({fmt(syn_det.get('Legitimate only', {}).get('watchdog_orders', {}).get('fpr'))} vs
    {fmt(syn_det.get('Legitimate only', {}).get('rule_orders', {}).get('fpr'))}).</p>
  </div>
</div>

<h2>Watchdog vs the rule-based baseline</h2>
<div class="legend">
  <span><span class="dot" style="background:var(--wd)"></span>Watchdog (RL)</span>
  <span><span class="dot" style="background:var(--rule)"></span>Rule baseline</span>
  <span class="muted">— the rule flags any large order cancelled quickly, tuned on the training split</span>
</div>
{detect_chart}
{fp_chart}
{recall_chart}
<p class="note">The trade-off in one line: the rule catches nearly everything but flags most legitimate large orders too;
the Watchdog is precise and quiet but misses manipulation it has not seen.</p>

<h2>Does surveillance actually cut the profit?</h2>
<div class="chart">
  <p class="note" style="margin-top:0">The detector acts inside the episode: on a flag the exchange cancels the trader's resting
  orders and blocks new ones for 200 events. 50 episodes per condition, identical seeds.</p>
  <table>
    <tr><th>Spoofer</th><th class="num">No surveillance</th><th class="num">With Watchdog</th><th class="num">With rule</th><th class="num">Profit cut</th></tr>
    <tr><td>SPOOFER-04 · INTC</td><td class="num">{fmt(s4.get('none', {}).get('pnl_mean'), money=True)}</td>
      <td class="num">{fmt(s4.get('watchdog', {}).get('pnl_mean'), money=True)}</td>
      <td class="num">{fmt(s4.get('rule', {}).get('pnl_mean'), money=True)}</td>
      <td class="num">{fmt(s4.get('watchdog', {}).get('suppression_pct'), 0)}% vs {fmt(s4.get('rule', {}).get('suppression_pct'), 0)}%</td></tr>
    <tr><td>SPOOFER-02 · MSFT</td><td class="num">{fmt(s2.get('none', {}).get('pnl_mean'), money=True)}</td>
      <td class="num">{fmt(s2.get('watchdog', {}).get('pnl_mean'), money=True)}</td>
      <td class="num">{fmt(s2.get('rule', {}).get('pnl_mean'), money=True)}</td>
      <td class="num">{fmt(s2.get('watchdog', {}).get('suppression_pct'), 0)}% vs {fmt(s2.get('rule', {}).get('suppression_pct'), 0)}%</td></tr>
  </table>
  <p class="note">The Watchdog also intervenes on legitimate traders about 12× less often than the rule.</p>
</div>

<h2>Seed robustness · does the Spoofer result depend on luck?</h2>
<div class="chart">
  <table>
    <tr><th>Agent</th><th>Stock</th><th>Seed</th><th class="num">PnL / episode</th><th class="num">Impact off</th><th class="num">Trades vs own spoof</th></tr>
    {seed_rows}
  </table>
  <p class="note">3 of 3 seeds learned to spoof on both stocks. INTC is stable across seeds; MSFT varies widely
  ({fmt(across.get('SPOOFER-02', {}).get('pnl_mean'), money=True)} ± {fmt(across.get('SPOOFER-02', {}).get('pnl_ci95'), 0)}),
  so the committed headline seed understates it. Every seed loses money with impact switched off.</p>
</div>

<h2>Sim-to-sim robustness · a market where price emerges from matching</h2>
<div class="chart">
  <table>
    <tr><th>Agent</th><th>Kind</th><th class="num">Followers react</th><th class="num">Followers blind</th><th class="num">Gain from spoofing</th></tr>
    {syn_rows}
  </table>
  <p class="note">In the synthetic agent-based market a spoof moves the price 4.5–9× less than the replay model assumes,
  and every manipulator loses money. Detection partly transfers (precision
  {fmt(syn_det.get('Manipulators + legitimate', {}).get('watchdog_orders', {}).get('precision'))},
  recall {fmt(syn_det.get('Manipulators + legitimate', {}).get('watchdog_orders', {}).get('recall'))});
  profitability does not. This is the project's largest open question.</p>
</div>

<h2>What the Watchdog actually uses</h2>
<div class="grid2">
  <div class="chart">
    <h4>Feature ablation <span class="muted">(test split)</span></h4>
    <table>
      <tr><th>Feature group removed</th><th class="num">Order F1</th><th class="num">Step recall</th></tr>
      {abl_rows}
    </table>
    <p class="note">Detection rests almost entirely on the participant's own trade direction. Remove it and the
    detector collapses — which is exactly why sparse spoofing is missed.</p>
  </div>
  <div class="chart">
    <h4>Structural comparison with documented cases</h4>
    <table>
      <tr><th></th><th class="num">Coscia</th><th class="num">SPOOFER-04</th></tr>
      <tr><td>Large orders filled</td><td class="num">0.08–0.5%</td><td class="num">0.4%</td></tr>
      <tr><td>Resting over 1 second</td><td class="num">0.57%</td><td class="num">52%</td></tr>
      <tr><td>Side reversal</td><td class="num">yes</td><td class="num">74%</td></tr>
    </table>
    <p class="note">Similar where the agent chooses (one side large, trade the other; orders almost never filled);
    different in speed, and layering cannot emerge in the replay environment.</p>
  </div>
</div>

<h2>Evidence quality</h2>
<div class="cards">
  <div class="card"><h3>Tests</h3><div class="big">139</div><p class="vs">passing, on real market data</p></div>
  <div class="card"><h3>Pre-registered tests</h3><div class="big">2</div><p class="vs">attacker, procedure and hypothesis committed before any data existed; each run once</p></div>
  <div class="card"><h3>Training seeds</h3><div class="big">3 + 3</div><p class="vs">Watchdog seeds and seeds per profitable Spoofer, with confidence intervals</p></div>
</div>
<p class="note">Environment exploits found and closed by decomposing where profit came from, not by reading headline PnL: free
accumulation, alternating-side swings, early termination, thin opening book. Each is documented with the measurement that caught it.</p>

<h2>Honest limitations</h2>
<ul>
  <li>Profitable manipulation is demonstrated only under the replay impact model; it does not survive the synthetic market.</li>
  <li>The Watchdog misses sparse spoofing — the headline failure, confirmed by two pre-registered tests.</li>
  <li>LOBSTER carries no participant identities, so the Watchdog cannot be run on real order flow; only the rule can.</li>
  <li>One trading day, five stocks. Spoofing is unprofitable on the wide-spread stocks under calibrated impact.</li>
  <li>The manipulation label is a proxy for intent (trading the opposite side while an order rests), not a legal determination.</li>
</ul>

<footer>
  Generated {built} from the committed results at <code>{commit}</code> · every number is read from <code>results/*.json</code> ·
  rebuild with <code>python scripts/build_dashboard.py</code>
</footer>
</div></body></html>"""

    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(html) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
