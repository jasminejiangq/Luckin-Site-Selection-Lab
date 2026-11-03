# Luckin Manhattan Site-Selection — Milestone 4

Data retrieved: 2026-11-03T06:11:18.344388+00:00 (UTC).

## Executive summary

This report solves the competitive location model on the downloaded Manhattan public-data instance. All financial and demand outputs are **scenario-based model estimates**, not observed sales, validated store forecasts, or guarantees of profitability. Read the limitations section before using any selected address as a real-estate recommendation.

Model instance: 310 census tracts, 2345 candidate storefronts, 22 provisional Luckin establishments, and 1332 coffee/tea competitors.
The instance has 58,995 tract-candidate choice pairs, 61,650 MILP variables, and 186,109 constraints.

## Scenario results

| Scenario | New sites | 10-year incremental NPV | Solver status | Bound | Relative gap | Runtime (s) | Min incumbent retention |
|---|---:|---:|---|---:|---:|---:|---:|
| Low | 10 | $70,313,420 | Optimization terminated successfully. (HiGHS Status 7: Optimal) | $70,324,476 | 0.0157% | 238.4 | 99.9% |
| Base | 10 | $143,250,000 | Optimization terminated successfully. (HiGHS Status 7: Optimal) | $143,270,570 | 0.0144% | 233.6 | 99.9% |
| High | 10 | $230,773,896 | Optimization terminated successfully. (HiGHS Status 7: Optimal) | $230,807,287 | 0.0145% | 277.6 | 99.9% |

Gap interpretation: the table reports the independently recomputed relative gap `(best bound - feasible NPV)/abs(feasible NPV)`. HiGHS terminates on its linear objective, which omits the constant baseline-NPV term; therefore its internal relative gap can differ slightly from the true reported-NPV gap.

- **Low:** HiGHS terminated with status Optimal at its internal relative gap 0.0100%; the independently recomputed gap on reported net NPV is 0.0157%. The bound is an upper bound; this is solver-tolerance optimality, not exact equality of incumbent and bound.
- **Base:** HiGHS terminated with status Optimal at its internal relative gap 0.0097%; the independently recomputed gap on reported net NPV is 0.0144%. The bound is an upper bound; this is solver-tolerance optimality, not exact equality of incumbent and bound.
- **High:** HiGHS terminated with status Optimal at its internal relative gap 0.0100%; the independently recomputed gap on reported net NPV is 0.0145%. The bound is an upper bound; this is solver-tolerance optimality, not exact equality of incumbent and bound.

### Selected sites and estimated demand (base scenario)

The order estimates below are model-implied choice shares, not sales observations. Rent in the cost column is the uniform fallback assumption because listing-specific asking rents are unavailable.

| Candidate ID | Address | ZIP | Estimated new-store orders/day | Annual rent used | 10-year cost PV |
|---|---|---:|---:|---:|---:|
| VAC01962 | 1858 LEXIGTON AVE ST 2 | 10029 | 3,180.3 | $129,600 | $3,179,779 |
| VAC02096 | 2253 ADAM C POWELL BLVD | 10027 | 3,360.5 | $129,600 | $3,179,779 |
| VAC02117 | 2310 ADAM C POWELL BLVD | 10030 | 3,529.2 | $129,600 | $3,179,779 |
| VAC02150 | 2328 JOE DIMAGGIO HIGHWAY | 10027 | 3,293.8 | $129,600 | $3,179,779 |
| VAC02158 | 201 WEST 144 STREET | 10030 | 3,322.2 | $129,600 | $3,179,779 |
| VAC02178 | 741 ST NICHOLAS AVENUE | 10031 | 3,178.3 | $129,600 | $3,179,779 |
| VAC02222 | 3862 Broadway | 10032 | 3,399.4 | $129,600 | $3,179,779 |
| VAC02284 | 4316 Broadway | 10033 | 3,448.7 | $129,600 | $3,179,779 |
| VAC02291 | 623 WEST 185 STREET | 10033 | 3,336.8 | $129,600 | $3,179,779 |
| VAC02332 | 565 WEST 207 STREET | 10034 | 3,450.2 | $129,600 | $3,179,779 |

Base-case gross new-store demand is 33,499.4 orders/day, reconciled as 30,975.2 from competitor attraction + 2,522.6 from the outside option + 1.6 cannibalized from incumbent Luckin locations. Reconciliation residual: 7.28e-12 orders/day.

Greedy feasible benchmark: $143,146,065 NPV using 10 openings. The MILP solution should be no worse than this feasible benchmark.

### Cross-scenario regret ($)

| Plan | base | high | low |
|---|---:|---:|---:|
| base | $0 | $0 | $0 |
| high | $0 | $0 | $0 |
| low | $0 | $0 | $0 |

## Mathematical model

### Sets and parameters

- `I`: Manhattan census tracts; `J`: reported vacant storefront candidate locations; `L`: provisional incumbent Luckin locations; `C`: coffee/tea competitor locations.
- `J_i`: candidate sites within the 1,600 m choice cutoff of tract `i`; distance uses a rotated Manhattan-grid approximation.
- `h_i`: equal-weight average of the quintile scores for ACS population, LODES workplace jobs, and MTA average daily subway ridership. The tier scores are 0.50, 0.75, 1.00, 1.25, 1.50.
- `gamma_i`: baseline Luckin attraction share; `U_i = u_0 + Σ_{l∈L} V_il + Σ_{c∈C} V_ic`; `V_ik = exp(-β d_ik)`, `β=ln(2)/400 m`, `u_0=0.35`.
- Opportunity scale: `lambda = |L| vbar / Σ_i h_i gamma_i^ref`, `w_i=lambda h_i`. This calibrates the assumed baseline to `|L| × vbar`; it is not fitted to observed Luckin transaction data.
- Annual contribution margin `m=$2.33/order`; horizon `T=10` years; discount rate `r=10%`; opening limit `K=10`; capital budget `$5,000,000`; minimum incumbent retention `85%`; new-store spacing `152.4 m`.

### Choice and objective

For a selected plan `x`, let `S_i = Σ_{j∈J_i} V_ij x_j`. The exact multinomial-logit shares are `P_i0 = U_i/(U_i+S_i)` and `P_ij = V_ij x_j/(U_i+S_i)`. The maximized objective is

`Z = 365 m [Σ_{t=1}^T (1+r)^(-t)] Σ_i w_i (1-gamma_i)(1-P_i0) - Σ_j Phi_j x_j`.

`Phi_j = C_j + [Σ_t (1+r)^(-t)] (R_j + F_L + F_O)`, where annual rent is `R_j = 12 × 1,000 sq ft × 1.20 × rent_psf_month`.

### MILP constraints

The linearized formulation uses `P_i0` and `P_ij` variables:

1. Choice normalization: `P_i0 + Σ_j P_ij = 1`.
2. Share upper links: `P_ij ≤ (V_ij/U_i)P_i0` and `P_ij ≤ [V_ij/(U_i+V_ij)]x_j`.
3. Lower link: `P_ij ≥ (V_ij/U_i)(P_i0 - 1 + x_j)`. This is needed for exact shares when retention constraints are present.
4. Opening cap and capital budget: `Σ_j x_j ≤ 10`, `Σ_j C_j x_j ≤ $5,000,000`.
5. Eligibility, candidate-to-candidate spacing, and incumbent exclusion constraints.
6. For every incumbent `l`, `Σ_i w_i(V_il/U_i)P_i0 ≥ 0.85 Σ_i w_i(V_il/U_i)`.
7. Optional valid tangent inequalities tighten the LP relaxation without excluding integer plans.
8. `x_j ∈ {0,1}`, `0≤P_i0≤1`, `0≤P_ij≤1`.

Tiny-instance regression tests enumerate all 256 opening plans, compare exact logit shares and the MILP solution against brute force, and test the necessity of the lower link.

## Data provenance and cleaning

| Model input | Official source / vintage |
|---|---|
| Tract points | [Census 2024 Gazetteer](https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2024_Gazetteer/2024_Gaz_tracts_national.zip) |
| Population | [ACS 2024 5-year, B01003](https://www2.census.gov/programs-surveys/acs/summary_file/2024/table-based-SF/data/5YRData/acsdt5y2024-b01003.dat) |
| Workplace jobs | [LEHD LODES 8, NY WAC JT01, 2023](https://lehd.ces.census.gov/data/lodes/LODES8/ny/wac/ny_wac_S000_JT01_2023.csv.gz) |
| Subway ridership | [MTA Subway Hourly Ridership (`5wq4-mkjj`)](https://data.ny.gov/resource/5wq4-mkjj.json), August 2026 |
| Competitors/incumbents | [NYC DOHMH inspections (`43nn-pn8j`)](https://data.cityofnewyork.us/resource/43nn-pn8j.json) |
| Vacant storefront candidates | [NYC DOF storefront registrations (`92iy-9c3n`)](https://data.cityofnewyork.us/resource/92iy-9c3n.json), reporting year 2025 |

The pipeline retained 310 census tracts, 121 station complexes, 1332 competitor records, 22 Luckin inspection records, and 2345 unique candidate coordinates. Input and output SHA-256 hashes are recorded in the source/build manifests.

## Limitations and interpretation

1. **Not a sales forecast.** Quintile-weighted population, jobs, and transit are demand-potential proxies. The opportunity scale is calibrated to an assumed per-store order volume (`vbar=350/600/900`), not observed purchase data. The resulting NPV is conditional on these assumptions.
2. **Provisional store network.** DOHMH inspection records establish inspected establishments, not current operating status. The 22 Luckin records and competitor set need a current, independently verified store list before investment decisions.
3. **No measured site rents or fit-out.** Candidate rent quotes are absent; all use the configured `$9/sq ft/month` fallback and the reusable-fitout flag is assumed zero. Costs are therefore uniform across candidate sites and may materially alter the selected locations.
4. **Location accuracy.** DOF coordinates are storefront/building-level geocodes, not verified doors. Demand-point locations are tract representative points. Grid distances are approximations, not pedestrian routes, and can affect both choice and selected sites.
5. **Proxy interpretation.** LODES jobs and MTA ridership measure workplace employment and subway entries, respectively; neither is a direct count of coffee customers. Population, jobs, and transit are equally weighted after within-feature ranking, not empirically estimated demand coefficients.

## Deliverables

- `outputs/real/summary.json`: solver status, bound, gap, objectives, parameters, and checks.
- `outputs/real/plan_<scenario>.csv`: selected site IDs, addresses, coordinates, costs, and modeled orders.
- `outputs/real/cross_scenario.csv`: cross-scenario plan performance/regret, when all scenarios were solved.
- `outputs/real/figures/site_selection_maps.png`, `scenario_dashboard.png`, and `demand_potential.png`.
- `data/raw/source_manifest.json` and `data/processed/processed_manifest.json`: provenance and file hashes.
