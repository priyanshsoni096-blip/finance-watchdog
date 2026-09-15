# Structural comparison with documented spoofing cases

This project demonstrates the feasibility of an RL-based surveillance approach in a realistic limit-order-book simulation, and evaluates the structural similarity of the emergently-learned manipulation strategy to documented real-world manipulation cases.

This is a comparison of *order structure*, not a validation of detection on real manipulation. No labelled real-world spoofing data exists to test detection against.

## Sources

Real-case figures are taken only from these documents and were checked against the text:

- **[CFTC Order, Panther Energy Trading LLC and Michael J. Coscia, Docket 13-26, July 22, 2013](https://www.cftc.gov/sites/default/files/idc/groups/public/@lrenforcementactions/documents/legalpleading/enfpantherorder072213.pdf)**
- **[United States v. Coscia, 7th Cir. 2017](https://caselaw.findlaw.com/court/us-7th-circuit/1869999.html)** (opinion; figures come from trial evidence it recounts)
- **[CFTC press release 7156-15 on Navinder Singh Sarao, April 2015](https://www.cftc.gov/PressRoom/PressReleases/7156-15)**

Simulated figures come from `python scripts/real_case_stats.py` (`results/real_case_stats.md`, 20 episodes per source, sampled actions) and `results/basic_eval.md`.

Two figures seen in a search-engine summary were **not** found in the primary documents and are not used: "over 400,000 large orders cancelled over 98% of the time" and small-order fill rates of "46.5% vs 7.6%".

## Comparison

The right-hand column says whether the simulated value is chosen by the agents (emergent) or set by the environment design (fixed).

| Dimension | Coscia | Sarao | SPOOFER-04 (INTC) | SPOOFER-02 (MSFT) | Emergent or fixed |
|---|---|---|---|---|---|
| Core pattern | Small order on one side, larger orders on the other, cancelled once the small order fills (CFTC order) | Exceptionally large sell orders layered into the visible book (CFTC press release) | Large order on one side, market orders on the other; 97.1% of large orders are manipulative | Same pattern; 70.5% of large orders are manipulative | Emergent |
| Reversal after a cycle | Algorithm "designed to promptly operate in reverse" (CFTC order) | — | Consecutive large orders switch side 73.8% of the time | 21.3% | Emergent |
| Large-order placement | Several larger orders at progressively better prices, e.g. bids at $85.26, $85.27, $85.28 against a 17-contract offer at $85.29 (CFTC order) | Four to six orders one price level apart, kept at least three or four levels from the best ask (CFTC press release) | One large order at the best price | One large order at the best price | Fixed: the environment allows one large order per side, at the touch |
| Large vs small order size | Small orders as small as five contracts; large orders fifty or more (opinion) | Orders described as exceptionally large | ~17–20× the genuine trade lot | ~20× | Fixed: spoof = 10× touch depth, lot = ½ touch depth |
| Large orders filled | 0.08% on CME and 0.5% on ICE (opinion) | Vast majority cancelled without resulting in trades (CFTC press release) | 0.4% filled by the market; 93.2% cancelled by owner; 6.4% resting at episode end | 1.4% filled; 95.8% cancelled; 2.8% at episode end | Emergent (the episode-end share is a simulation artefact) |
| Speed | Only 0.57% of large orders on the market for more than one second; a cycle took about two-thirds of a second (opinion) | Layering Algorithm run continuously for over two hours on May 6, 2010 (CFTC press release) | Median resting time of manipulative large orders 1.09 s; 52.2% rest over 1 s | 0.52 s; 43.8% over 1 s | Emergent, but shaped by design (see below) |
| Small / genuine orders filled | 35.61% of small orders filled (opinion) | — | Genuine orders are market orders and always fill | Same | Fixed; not comparable |
| Scale | About $1.4 million net profit, August 8 – October 18, 2011 (CFTC order) | Over $40 million profit since April 2010 (CFTC press release) | +$13,208 per 2,000-event episode | +$2,437 per episode | Not comparable (one simulated trading day, calibrated impact model) |

## What this shows

**Structurally similar, and emergent in the trained Spoofers:**
- Large visible orders on one side while trading on the other.
- The large order is almost never allowed to fill: 0.4–1.4% filled, against Coscia's 0.08–0.5%.
- SPOOFER-04 alternates sides much like Coscia's reversing algorithm (73.8% of consecutive large orders).

**Structurally different:**
- **Speed.** 43.8% (SPOOFER-02) and 52.2% (SPOOFER-04) of the simulated manipulative large orders rest longer than one second, against 0.57% of Coscia's large orders. That share is about 77–92× higher. This follows from the environment: a spoof's price impact builds up over about 50 market events (calibrated from real order-flow response), so waiting is rewarded. It should not be read as realism.
- **Layering.** Coscia's program priced several larger orders progressively, and Sarao layered four to six orders several levels away from the best price. The environment allows only one large order per side at the best price, so neither pattern can emerge. A layering attacker would need a multi-order environment.

**Not comparable:**
- The large-to-small size ratio (fixed by the environment).
- Small-order fill rates (simulated genuine orders are market orders).
- Profit scale.
