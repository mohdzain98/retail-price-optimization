# Methodology

How the pipeline turns weekly retail data into price recommendations, the assumptions behind each step, and where
each approach comes from. Sections are added as phases land. Implemented so far: data preparation (§1), price elasticity
(§2), baseline decomposition (§3) and price optimization (§4).

---

## 1. Data preparation

**Source.** dunnhumby *Breakfast at the Frat*: 156 weeks (Jan 2009 – Jan 2012) of store × product × week sales
for 55 products in four categories (bag snacks, cold cereal, frozen pizza, oral hygiene), with the shelf price, the
regular (base) price, and in-store promotion flags. See [data.md](data.md) for the schema.

**Panel.** Store rows are aggregated to **product × week**:

| Quantity | Rule | Why |
|---|---|---|
| Units, spend | sum over stores | total demand |
| Price | spend ÷ units | unit-weighted average price actually paid |
| Base price | unit-weighted mean, floored at price | a promotion lowers price below base, never above |
| Feature / display / TPR | share of selling stores running the tactic | intensity of promotion, 0–1 |
| `n_stores` | stores with sales | distribution (coverage) |

**Cleaning.** Rows with zero units or a missing/zero price are dropped (24 of 524,950). Missing base prices are
filled within the store-product series, then from the price.

**Eligibility.** A product gets its own elasticity estimate only if it has at least 52 weeks of data and at least 5%
price variation (max/min − 1). Ineligible products are kept and later receive an imputed elasticity. On this data
all 55 products are eligible.

**Features.**

| Feature | Definition | Purpose |
|---|---|---|
| `ln_units` | log units | log-log model: slopes are elasticities |
| `ln_base_price` | log regular (base) price | **regular-price** level |
| `ln_promo_price` | ln(price ÷ base price) ≤ 0 | **temporary discount**; `ln_price = ln_base_price + ln_promo_price` |
| `feature`, `display`, `feature_display`, `tpr` | share of selling stores running each tactic (mutually exclusive in the raw data) | merchandising lift beyond the price cut |
| `promo_flag` | discount ≥ 5%, or feature/display share ≥ 25% | promo weeks (RGMx-style) |
| `pl1..pl3` | `promo_flag` lagged 1–3 weeks | **pantry loading**: the post-promotion dip as shoppers use up stock |
| `ln_n_stores` | log of selling stores | separates coverage changes from price response |
| Fourier terms | sin/cos of day-of-year, order 3 | smooth yearly seasonality |
| Holiday flags | week contains a US public holiday; separate flags for Thanksgiving, Christmas and Super Bowl Sunday | demand spikes around events |
| `ln_trend` | LOESS of the **residual** of a per-product driver regression | slow drift not explained by the drivers |

**Residual trend (RGMx approach).** For each product, `ln_units` is first regressed on all the drivers above (price
terms, tactics, coverage, holidays, pantry loading, seasonality) using training weeks only. LOESS is then applied to
the residual over time. The trend therefore captures only drift the drivers cannot explain, such as distribution gains
or brand growth. It never absorbs price or promotion effects, which a LOESS on raw volume would partly do. The window
is 78 weeks (longer than a year), so seasonality stays with the Fourier terms, and the trend is held flat across the
holdout.

**Time split.** The last 13 weeks (from 2011-10-12) are a global holdout. Every training week precedes every test
week, so model accuracy is measured on genuinely unseen future data.

---

## 2. Price elasticity

### 2.1 Model: two elasticities per product

The price paid is split into a regular level and a temporary deviation, `ln P = ln P_base + ln(P / P_base)`, and
each part gets its own slope:

```
ln Q_it = α_i + β_i · (ln P_base,it − mean_i) + δ_i · ln(P_it / P_base,it) + γ · X_it + ε_it
```

- **`β_i`: regular-price elasticity.** The response to a lasting change in the shelf price. This is what the price
  optimizer needs. Mixing it with promotional responses would overstate how much volume a regular-price cut wins.
- **`δ_i`: promo-price elasticity.** The response to a temporary discount. It is usually larger, because shoppers
  react to deals and buy ahead. Phase 5 promo planning uses this.
- **`X_it` (controls):** feature, display, feature+display and TPR shares; pantry loading `pl1..pl3`; `ln_n_stores`;
  holidays; Fourier seasonality; the residual trend. With the TPR share in the model, `δ` measures the effect of
  discount *depth*, and the TPR share captures the lift from being on deal at all. These are the same promo drivers
  RGMx uses (ACV feature, display, F&D and TPR, plus pantry flags).
- **Identification check.** The regular price moves for all 55 products (median range 19%). Regular-price changes
  carry about a quarter of the price variance and correlate only −0.13 with discount depth, so the two slopes are
  separately identifiable.

The constant-elasticity form gives the demand curve `Q(P) = Q₀ · (P/P₀)^β` that the optimizer uses in Phase 4.

### 2.2 Four estimators

| Model | How the slopes are estimated | Strength | Weakness |
|---|---|---|---|
| **OLS per product** | separate regression per product | no shared assumptions | noisy where a price term barely moves |
| **Pooled OLS** | product intercepts, one slope per category and price term | very stable | every product in a category gets the same slopes |
| **MixedLM** | category fixed slopes + random intercept and slopes per product (REML) | partial pooling (as in RGMx) | approximate intervals |
| **Bayesian hierarchical** | both slopes follow `global → category → product` | partial pooling plus a full posterior | about 4 minutes to sample |

The Bayesian model:

- **Two non-centred hierarchies.** `β_i = μ_cat + σ_prod · z_i`, `μ_cat = μ_global + σ_cat · z_cat`, and the same
  structure for `δ`. A product with clear price variation keeps its own slope; a weakly identified one is pulled
  toward its category, by an amount the data decides.
- **Priors.** Regular: `μ_global ~ N(−1.5, 1)`. Promo: `μ_global ~ N(−2.5, 1.5)`, reflecting the stronger response to
  deals. Group scales are `HalfNormal(0.5)`. All are weak relative to 7,000+ observations.
- **Control effects per category.** Promotion, pantry, season and trend effects differ by category. Controls are
  demeaned within product and standardised.
- **Student-t likelihood.** Heavy tails keep launch and phase-out outliers from dragging the slopes.
- **Sampling.** NUTS, 4 chains × 1,500 draws, `target_accept` 0.95. Latest run: r-hat 1.008, minimum bulk ESS 774,
  0 divergences.

### 2.3 Model selection and imputation (RGMx post-processing)

Each model is scored **per product**, and each product uses the first model in the priority list
(`bayes_hierarchical → mixed_effects → ols_per_product`) that passes every quality gate:

| Gate | Threshold (config) |
|---|---|
| In-sample MAPE on units | ≤ 0.30 |
| In-sample R² on log units, within product | ≥ 0.45 |
| Regular elasticity | within [−5, −0.3] |

Products no model passes, and ineligible products, are imputed **hierarchically**. They take the median of valid
estimates (and interval bounds) at the most specific level that has one:

1. sub-category + manufacturer
2. sub-category
3. category
4. all products

The promo family is checked separately against [−8, −0.3] and imputed the same way. Every product carries a source
(`estimated` / `imputed`) and a reason for each family. RGMx uses the same pattern: quality gates, model priority,
then hierarchical imputation over brand and retailer levels.

### 2.4 Validation

- **Recovery tests.** On a synthetic panel with known regular and promo elasticities, every model recovers both, and
  the Bayesian 94% HDIs cover the truth for most products.
- **Holdout accuracy.** WAPE on units over the last 13 weeks.

### 2.5 Results

| Model | Holdout WAPE | Median regular β | Within bounds | Passes gates |
|---|---|---|---|---|
| OLS per product | 18.9% | −1.21 | 82% | 82% |
| Pooled OLS | 19.1% | −1.34 | 100% | 80% |
| MixedLM | 17.9% | −1.21 | 84% | 73% |
| **Bayesian hierarchical** | **16.8%** | −1.48 | 93% | 78% |

- **The Bayesian model is now also the most accurate on the holdout.** Separate promo effects per category and the
  richer promo controls are what changed.
- **Selection:** 43 products use the Bayesian estimate, 6 fall back to per-product OLS, and 6 are imputed (for
  example a private-label pretzel with a positive slope, and Honey Nut Toasted Oats with R² 0.23).

| Category | Median regular β | Median promo δ |
|---|---|---|
| Bag snacks | −1.42 | −1.26 |
| Cold cereal | −1.40 | −2.23 |
| Frozen pizza | −2.22 | −2.40 |
| Oral hygiene | −1.51 | −1.93 |

In cereal, pizza and oral care, temporary discounts move volume more than regular-price changes do, as expected. A
single price slope would have blended the two.

Figures: `02_elasticity_forest.png` (regular), `02_promo_elasticity_forest.png`, `02_model_comparison.png`.

### 2.6 Limitations

- **Price endogeneity.** Retailers set prices partly in response to expected demand. Instruments such as the same
  product's price in other stores would address this. Not done yet.
- **No cross-price effects.** Competitor price indices (RGMx's `xpi` / `opi`) and substitution between products are not
  modelled yet.
- **Aggregation.** The product-week price averages across stores with different prices, which smooths the price
  signal.
- **In-sample gates.** MAPE and R² are measured on training weeks, as in RGMx. Holdout WAPE is reported separately for
  each model.

---

## 3. Baseline and due-to decomposition

### 3.1 Definition

The **baseline** is the volume a product would sell in a week at its regular price, with no temporary discount, no
feature or display, and no post-promotion dip. Holidays, seasonality and the trend stay in it. Phase 4 optimizes the
regular price on top of this baseline.

### 3.2 Top-down formula (RGMx)

The baseline is built **top-down**: the promotional effects are removed from *actual* volume. This keeps it anchored to
what really sold, unlike a bottom-up model prediction.

```
baseline_t = exp( ln units_t − price_t − promo_t − pantry_t − error_t )
```

| Effect term | Definition |
|---|---|
| `price` | final promo elasticity δ × ln(price ÷ base price): only the temporary-discount deviation |
| `promo` | contribution of feature, display, feature+display and TPR from the product's model; kept only in promo weeks and only if positive |
| `pantry` | contribution of `pl1..pl3`; usually negative, so removing it **restores** the post-promotion dip |
| `error` | residual of the product's model, kept only in promo weeks with a positive promo effect and floored at 0 |

- **Contributions are model-agnostic.** Each one is computed as the model's prediction minus its prediction with those
  drivers set to zero. For a linear model this equals Σ coefficient × driver.
- **Which model:** each product uses its selected model from Phase 2. If none passed the gates, it uses the first model
  in priority. Ineligible products (no model) get the price effect only, using their imputed δ.

### 3.3 Bad-baseline rules (RGMx)

Week-to-week noise in consecutive promo weeks can make the raw baseline implausible. Each rule flags such weeks and
names its fix:

| Rule | Flags | Fix |
|---|---|---|
| 2 · zero baseline | baseline = 0 while units > 0 | actual volume |
| 3 · implausibly high | baseline > 1.5 × units, or baseline > units with no promo in the last 3 weeks | actual volume |
| 4 · promo dip | promo week, baseline below both neighbours | 7-week centred rolling mean |
| 5 · promo spike | promo week, no holiday, baseline above both neighbours | rolling mean |
| 6 · promo drop | promo week, baseline < 75% of a neighbour | rolling mean |

- **Precedence:** rolling-mean fixes win over actual-volume fixes, as in RGMx.
- **Edges:** RGMx fills missing neighbours with 0; here a series edge simply has no neighbour, so the neighbour rules
  cannot fire there.
- **Float tolerance:** comparisons use a small relative tolerance, so `exp(ln units)` never counts as "above units".

### 3.4 Due-to decomposition

Incremental units (`actual − baseline`) are split across the effect terms in proportion to their log effects, so the
lifts always sum exactly to the incremental units. Weeks with no effects but a corrected baseline carry the difference
as error lift.

### 3.5 Results

On the real data, 1,502 of 8,304 product-weeks (18%) received a rolling-mean fix, all from the promo-week rules (dip
862, spike 544, drop 306). The zero and implausibly-high rules did not fire.

| Category | Baseline | Price lift | Promo lift | Pantry | Error |
|---|---|---|---|---|---|
| Bag snacks | 85.0% | 4.8% | 7.2% | +1.6% | 1.4% |
| Cold cereal | 72.9% | 14.2% | 10.2% | −0.3% | 3.0% |
| Frozen pizza | 63.8% | 21.7% | 12.6% | −0.4% | 2.3% |
| Oral hygiene | 78.0% | 19.0% | 2.8% | −1.3% | 1.6% |
| **All** | **75.0%** | 13.0% | 9.4% | +0.1% | 2.5% |

- **Frozen pizza** is the most promotion-dependent category: 36% of its volume is incremental.
- **Bag snacks** sell mostly at baseline.
- **Oral care** responds to price cuts, not merchandising.
- **Cereal promotions** roughly triple weekly volume (median promo-week lift +197%).

Figures: `03_due_to_by_category.png`, `03_baseline_weekly.png`.

### 3.6 Limitations

- **Promo-week error.** Attributing the positive promo-week residual to promotion (as RGMx does) can overstate
  promotional lift when an unrelated shock happens in a promo week.
- **Promo flag threshold.** A 5% discount marks 33% of product-weeks as promo weeks. A stricter threshold would change
  which weeks the rules can touch.

---

## 4. Price optimization (NLopt)

### 4.1 Why not the linear formula

The RGMx notebook and the source article optimise:

```
ΔVolume = Baseline × Elasticity × (P − P̄)/P̄,   maximise (Baseline + ΔVolume) × P
```

This leaves out several things an optimal price depends on:

- **Wrong demand curve.** It is a first-order (linear) approximation of the log-log model the elasticity came from. At
  β = −2 and ±20% it predicts −40% / +40% volume, where the true curve gives −31% / +56%.
- **Revenue, not profit.** Without costs, inelastic products always run to the upper price bound.
- **Mismatched reference.** P̄ is the mean of all prices (promo weeks included), while the baseline is defined at the
  regular price.
- **No portfolio view.** No volume, price-index or ladder constraints.
- **No uncertainty.** A single elasticity value is used.

The notebook also had implementation bugs: a log baseline used as volume, a volume constraint with the wrong sign for
NLopt, and a ×1000 price scaling.

### 4.2 Formulation

Decision variables are relative prices `r_i = P_i / P₀_i`, all close to 1, which keeps the problem well-conditioned.

| Piece | Definition |
|---|---|
| Current price `P₀` | median base price over the last 13 weeks |
| Base volume `B` | mean weekly baseline over the same weeks (Phase 3) |
| Elasticity `β` | final **regular-price** elasticity (Phase 2) |
| Unit cost `c` | `(1 − margin_category) · P₀`, using the assumed margins in config (the data has no cost) |
| Demand | `Q_i = B_i · r_i^β_i`, the constant-elasticity curve implied by the log-log model |
| Objective | maximise `Σ (P_i − c_eff,i) Q_i`, where the **effective cost** sets the goal: profit uses `c`, revenue uses `0`, and a profit/revenue mix uses `w·c`. Both **profit and revenue** are solved in every run and stored side by side |

Constraints, all written as `fc(r) ≤ 0` with analytic Jacobians and tested for sign:

| Constraint | Form |
|---|---|
| Bounds | `0.8 ≤ r_i ≤ 1.2` |
| Volume floor | `0.95 − Σ Q(r) / Σ Q(1) ≤ 0` |
| Average price change | `Σ w_i r_i − 1.05 ≤ 0`, with `w` = current revenue shares |
| Price ladder | `1.1 · (unit price_PL / unit price_brand) · r_PL − r_brand ≤ 0` for private-label/brand pairs in the same sub-category and size unit; only ladders that already hold are enforced (90 pairs) |

**Correctness check:** for one product with no portfolio constraints, the optimum is the closed form
`P* = c·β/(1+β)` (for β < −1). A unit test checks that NLopt recovers it.

### 4.3 Pricing-direction rule (RGMx) and the profit version

RGMx states the classic **revenue** rule: if E < −1, lowering price raises revenue; if E > −1, raising price raises
revenue. Each product's own optimum (no portfolio constraints) is

```
P* = c_eff · β / (1 + β)   for β < −1, else the upper bound
```

- **Revenue** (`c_eff = 0`): P* is the lower bound for elastic products and the upper bound for inelastic ones. This is
  exactly the RGMx rule.
- **Profit** (`c_eff = c`): **raise if |β| < 1 / margin, cut if |β| > 1 / margin.** With 28–40% margins the cut-off is
  |β| ≈ 2.5–3.6. A mildly elastic product (β = −1.5) should be *raised*, because the volume a cut wins is worth less than
  the margin it gives up.

Every recommendation carries `own_optimal_price`, `expected_direction`, `actual_direction` and `rule_status`. A
`constraint_override` status means a portfolio constraint moved the product against its own optimum. On the real data:

| Goal | Follows the rule | Constraint override |
|---|---|---|
| Revenue | 49 / 54 | 5 |
| Profit | 31 / 54 | 23 (the +5% average cap reallocates moves) |

Inelastic products are never cut under either goal.

### 4.4 Solvers

| Algorithm | Type | Weekly profit | Evaluations | Time |
|---|---|---|---|---|
| **LD_SLSQP** (primary) | local, gradient (analytic) | **+13.93%** | 106 | 0.4 s |
| LN_COBYLA | local, derivative-free | +13.92% | 2,513 | 10.6 s |
| GN_ISRES | global, evolutionary | +11.14% | 20,000 (limit) | 2.6 s |

- **SLSQP and COBYLA agree.** The problem is smooth with a single optimum, so the gradient method is fastest by far.
- **ISRES** shows what a global search costs in 54 dimensions.
- **Scaling matters.** Before the objective and constraints were normalised to O(1), SLSQP stopped early ("roundoff
  limited") at +11.5%.
- **Time limit.** Each solve is capped at 20 s (`max_seconds`). Under the revenue goal, COBYLA ran into its 20,000
  evaluation limit (160 s); it now stops at the cap with the same optimum as SLSQP.

### 4.5 Price points, safeguard, robustness and the linear-formula check

- **Price points.** The SLSQP optimum is rounded to "9" endings (x.09 … x.99), then repaired. Any product whose
  rounding breaks a constraint is stepped back one price point toward its current price until every constraint holds.
  Rounding keeps +12.9% of the +13.9%.
- **Safeguard.** If the rounded and repaired prices would score below current prices on the goal being optimised, the
  current prices are kept and `safeguard_applied` is set. The portfolio total therefore never gets worse than today
  (by the model). It did not trigger on the real data.
- **Robustness.** The portfolio is re-solved for 200 posterior draws of the Bayesian elasticities. The 5th–95th
  percentile range of each product's optimal price has median width 8.6% of the current price, and up to 34% for
  products with uncertain elasticities. Wide ranges are candidates for a price test before rollout.
- **Linear formula, scored on the true curve.** Its prices give +13.2% profit at −2.3% volume. The recommended prices
  give +12.9% profit at only −0.2% volume. The linear formula over-predicts how much volume a price rise loses. That is
  only a 0.7-point profit gap with the guardrails on, but 2.1 points without them (27.8% vs 25.7%).

### 4.6 Results

Recommended (rounded) prices, baseline business, weekly change vs current:

| Goal | Volume | Revenue | Profit | Annual profit impact |
|---|---|---|---|---|
| **Profit** | −0.2% | +6.2% | **+12.9%** | ≈ +$305k on ≈ $2.4M baseline profit |
| Revenue | +23.4% | **+15.1%** | −18.9% | ≈ −$446k |

**Maximising revenue cuts prices on elastic products.** It buys volume and revenue but gives up a fifth of the profit.
That is why profit is the default goal. The revenue run is kept so the simulator can show the trade-off.

Details below are for the **profit** goal.

- **Direction:** 25 products go up, 27 go down and 2 stay flat.
- **Median moves by category:**
  - bag snacks +7.1%;
  - cold cereal +2.8%;
  - frozen pizza −5.1%;
  - oral hygiene −6.5%, since elastic premium mouthwashes go to the −20% bound.
- **Binding constraint:** the +5% average cap is the one that binds. The optimizer therefore *cuts* prices on elastic,
  high-revenue products (for example Newman's Own pizza, below its own optimum) to fund larger increases on inelastic
  ones (cereal, pretzels).
- **Without the guardrails** (bounds only), profit would rise 27.8% from a 15.6% average price increase. The
  constraints hold back about half of that on purpose.

Figures (one set per goal): `04_scenarios_{profit,revenue}.png`, `04_price_changes_{profit,revenue}.png`,
`04_objective_curve_{profit,revenue}.png`.

### 4.7 Limitations

- **Assumed margins.** Profit gains scale with the margin assumption (28–40% by category).
- **Regular-price business only.** Promotions keep their current depth and calendar. Phase 5 plans promotions.
- **No cross-price effects or competitor reaction.** Each product's demand depends only on its own price.
- **Local elasticity.** The elasticity is estimated within the observed price range. Moves near ±20% extrapolate.

---

## Sources of the approach

- **Overall pipeline (elasticity → baseline → NLopt optimization)** follows the pricing workflow described in
  *Mastering the Art of Pricing Optimization*
  ([Towards Data Science](https://medium.com/data-science/mastering-the-art-of-pricing-optimization-a-data-science-solution-eb8befb79425)).
- **RGM econometrics practice (RGMx)**, from the author's earlier industry work, contributed:
  - the log-volume mixed model with promo-tactic, distribution and pantry-loading drivers;
  - the trend fitted on model residuals;
  - per-pair quality gates (MAPE, R², elasticity range) with model priority;
  - hierarchical imputation;
  - the top-down baseline with its effect terms and bad-baseline rules (Phase 3).
- **Model specification and estimators** are standard econometrics, chosen and implemented for this project:
  - the log-log constant-elasticity demand model, here with a regular/promo price split;
  - fixed and random effects;
  - hierarchical Bayesian partial pooling, as in Gelman & Hill, *Data Analysis Using Regression and
    Multilevel/Hierarchical Models* (2007);
  - LOESS smoothing (Cleveland, 1979).
- **Libraries:** statsmodels (OLS, MixedLM, LOWESS), PyMC (NUTS sampling) and NLopt (SLSQP, COBYLA, ISRES).
