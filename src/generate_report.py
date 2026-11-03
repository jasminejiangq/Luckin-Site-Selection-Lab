"""Generate the Milestone 4 report from audited data and solver outputs."""

import json
from pathlib import Path

import pandas as pd

from config import OUT, RAW, PROC, BASE
from params import summary as instance_summary
import params


def _money(value):
    return f"${value:,.0f}"


def _report_status(result):
    status = str(result["status"])
    gap = result.get("gap")
    solver_gap = result.get("solver_reported_gap")
    if "optimal" in status.lower():
        return (
            f"HiGHS terminated with status Optimal at its internal relative gap "
            f"{float(solver_gap):.4%}; the independently recomputed gap on reported net NPV "
            f"is {float(gap):.4%}. The bound is an upper bound; this is solver-tolerance "
            "optimality, not exact equality of incumbent and bound."
        )
    return "Best feasible incumbent only; global optimality has NOT been proven."


def main():
    out_dir = OUT / "real"
    result_file = out_dir / "summary.json"
    if not result_file.exists():
        raise FileNotFoundError(f"Run the real-data scenarios first: {result_file}")
    result = json.loads(result_file.read_text(encoding="utf-8"))
    manifest = json.loads((RAW / "source_manifest.json").read_text(encoding="utf-8"))
    ins = params.build(PROC, BASE)
    stats = instance_summary(ins)
    scenarios = [name for name in ("low", "base", "high") if name in result]
    if not scenarios:
        raise ValueError("No low/base/high scenarios found in the solver summary")

    lines = [
        "# Luckin Manhattan Site-Selection — Milestone 4",
        "",
        f"Data retrieved: {manifest['retrieved_at_utc']} (UTC).",
        "",
        "## Executive summary",
        "",
        "This report solves the competitive location model on the downloaded Manhattan public-data instance. "
        "All financial and demand outputs are **scenario-based model estimates**, not observed sales, "
        "validated store forecasts, or guarantees of profitability. Read the limitations section before "
        "using any selected address as a real-estate recommendation.",
        "",
        f"Model instance: {stats['|I|']} census tracts, {stats['|J|']} candidate storefronts, "
        f"{stats['|L|']} provisional Luckin establishments, and {stats['|C|']} coffee/tea competitors.",
        f"The instance has {stats['M = sum|J_i|']:,} tract-candidate choice pairs, "
        f"{result[scenarios[0]]['n_vars']:,} MILP variables, and "
        f"{result[scenarios[0]]['n_cons']:,} constraints.",
        "",
        "## Scenario results",
        "",
        "| Scenario | New sites | 10-year incremental NPV | Solver status | Bound | Relative gap | Runtime (s) | Min incumbent retention |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for name in scenarios:
        r = result[name]
        lines.append(
            f"| {name.title()} | {r['n_open']} | {_money(r['Z'])} | {r['status']} | "
            f"{_money(r['bound'])} | {float(r['gap']):.4%} | {float(r['runtime_s']):.1f} | "
            f"{float(r['min_retention']):.1%} |"
        )
    lines.extend(
        [
            "",
            "Gap interpretation: the table reports the independently recomputed relative gap "
            "`(best bound - feasible NPV)/abs(feasible NPV)`. HiGHS terminates on its linear "
            "objective, which omits the constant baseline-NPV term; therefore its internal "
            "relative gap can differ slightly from the true reported-NPV gap.",
            "",
        ]
    )
    for name in scenarios:
        lines.append(f"- **{name.title()}:** {_report_status(result[name])}")
    lines.append("")
    lines.extend(
        [
            "### Selected sites and estimated demand (base scenario)",
            "",
            "The order estimates below are model-implied choice shares, not sales observations. "
            "Rent in the cost column is the uniform fallback assumption because listing-specific asking rents are unavailable.",
            "",
            "| Candidate ID | Address | ZIP | Estimated new-store orders/day | Annual rent used | 10-year cost PV |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    if "base" in result:
        base_plan = pd.read_csv(out_dir / "plan_base.csv")
        for _, row in base_plan.iterrows():
            address = str(row.get("property_street_address_or", "Address unavailable"))
            zipcode = str(row.get("zip_code", ""))
            lines.append(
                f"| {row['cand_id']} | {address} | {zipcode} | {row['orders_per_day']:,.1f} | "
                f"{_money(row['annual_rent'])} | {_money(row['Phi'])} |"
            )
        base = result["base"]
        lines.extend(
            [
                "",
                f"Base-case gross new-store demand is {base['gross_new_orders']:,.1f} orders/day, "
                f"reconciled as {base['from_competitors']:,.1f} from competitor attraction + "
                f"{base['from_no_purchase']:,.1f} from the outside option + "
                f"{base['lost_incumbent_orders']:,.1f} cannibalized from incumbent Luckin locations. "
                f"Reconciliation residual: {base.get('order_reconciliation_residual', float('nan')):.3g} orders/day.",
                "",
                f"Greedy feasible benchmark: {_money(base['greedy_Z'])} NPV using {base['greedy_n']} openings. "
                "The MILP solution should be no worse than this feasible benchmark.",
            ]
        )

    if (out_dir / "cross_scenario.csv").exists():
        cross = pd.read_csv(out_dir / "cross_scenario.csv")
        pivot = cross.pivot(index="plan", columns="scenario", values="regret")
        columns = list(pivot.columns)
        lines.extend(
            [
                "",
                "### Cross-scenario regret ($)",
                "",
                "| Plan | " + " | ".join(columns) + " |",
                "|---|" + "|".join("---:" for _ in columns) + "|",
            ]
        )
        for plan_name, values in pivot.iterrows():
            lines.append(
                f"| {plan_name} | "
                + " | ".join(_money(values[column]) for column in columns)
                + " |"
            )
        lines.append("")

    lines.extend(
        [
            "## Mathematical model",
            "",
            "### Sets and parameters",
            "",
            "- `I`: Manhattan census tracts; `J`: reported vacant storefront candidate locations; "
            "`L`: provisional incumbent Luckin locations; `C`: coffee/tea competitor locations.",
            "- `J_i`: candidate sites within the 1,600 m choice cutoff of tract `i`; "
            "distance uses a rotated Manhattan-grid approximation.",
            "- `h_i`: equal-weight average of the quintile scores for ACS population, LODES workplace jobs, "
            "and MTA average daily subway ridership. The tier scores are 0.50, 0.75, 1.00, 1.25, 1.50.",
            "- `gamma_i`: baseline Luckin attraction share; `U_i = u_0 + Σ_{l∈L} V_il + Σ_{c∈C} V_ic`; "
            "`V_ik = exp(-β d_ik)`, `β=ln(2)/400 m`, `u_0=0.35`.",
            "- Opportunity scale: `lambda = |L| vbar / Σ_i h_i gamma_i^ref`, `w_i=lambda h_i`. "
            "This calibrates the assumed baseline to `|L| × vbar`; it is not fitted to observed Luckin transaction data.",
            "- Annual contribution margin `m=$2.33/order`; horizon `T=10` years; discount rate `r=10%`; "
            "opening limit `K=10`; capital budget `$5,000,000`; minimum incumbent retention `85%`; "
            "new-store spacing `152.4 m`.",
            "",
            "### Choice and objective",
            "",
            "For a selected plan `x`, let `S_i = Σ_{j∈J_i} V_ij x_j`. "
            "The exact multinomial-logit shares are `P_i0 = U_i/(U_i+S_i)` and "
            "`P_ij = V_ij x_j/(U_i+S_i)`. The maximized objective is",
            "",
            "`Z = 365 m [Σ_{t=1}^T (1+r)^(-t)] Σ_i w_i (1-gamma_i)(1-P_i0) - Σ_j Phi_j x_j`.",
            "",
            "`Phi_j = C_j + [Σ_t (1+r)^(-t)] (R_j + F_L + F_O)`, where annual rent is "
            "`R_j = 12 × 1,000 sq ft × 1.20 × rent_psf_month`.",
            "",
            "### MILP constraints",
            "",
            "The linearized formulation uses `P_i0` and `P_ij` variables:",
            "",
            "1. Choice normalization: `P_i0 + Σ_j P_ij = 1`.",
            "2. Share upper links: `P_ij ≤ (V_ij/U_i)P_i0` and "
            "`P_ij ≤ [V_ij/(U_i+V_ij)]x_j`.",
            "3. Lower link: `P_ij ≥ (V_ij/U_i)(P_i0 - 1 + x_j)`. "
            "This is needed for exact shares when retention constraints are present.",
            "4. Opening cap and capital budget: `Σ_j x_j ≤ 10`, `Σ_j C_j x_j ≤ $5,000,000`.",
            "5. Eligibility, candidate-to-candidate spacing, and incumbent exclusion constraints.",
            "6. For every incumbent `l`, `Σ_i w_i(V_il/U_i)P_i0 ≥ 0.85 Σ_i w_i(V_il/U_i)`.",
            "7. Optional valid tangent inequalities tighten the LP relaxation without excluding integer plans.",
            "8. `x_j ∈ {0,1}`, `0≤P_i0≤1`, `0≤P_ij≤1`.",
            "",
            "Tiny-instance regression tests enumerate all 256 opening plans, compare exact logit shares and "
            "the MILP solution against brute force, and test the necessity of the lower link.",
            "",
            "## Data provenance and cleaning",
            "",
            "| Model input | Official source / vintage |",
            "|---|---|",
            "| Tract points | [Census 2024 Gazetteer](https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2024_Gazetteer/2024_Gaz_tracts_national.zip) |",
            "| Population | [ACS 2024 5-year, B01003](https://www2.census.gov/programs-surveys/acs/summary_file/2024/table-based-SF/data/5YRData/acsdt5y2024-b01003.dat) |",
            "| Workplace jobs | [LEHD LODES 8, NY WAC JT01, 2023](https://lehd.ces.census.gov/data/lodes/LODES8/ny/wac/ny_wac_S000_JT01_2023.csv.gz) |",
            "| Subway ridership | [MTA Subway Hourly Ridership (`5wq4-mkjj`)](https://data.ny.gov/resource/5wq4-mkjj.json), August 2026 |",
            "| Competitors/incumbents | [NYC DOHMH inspections (`43nn-pn8j`)](https://data.cityofnewyork.us/resource/43nn-pn8j.json) |",
            "| Vacant storefront candidates | [NYC DOF storefront registrations (`92iy-9c3n`)](https://data.cityofnewyork.us/resource/92iy-9c3n.json), reporting year 2025 |",
            "",
            f"The pipeline retained {manifest['cleaned_row_counts']['tract_points']} census tracts, "
            f"{manifest['cleaned_row_counts']['subway_station_complexes']} station complexes, "
            f"{manifest['cleaned_row_counts']['competitor_establishments']} competitor records, "
            f"{manifest['cleaned_row_counts']['luckin_establishments']} Luckin inspection records, and "
            f"{manifest['cleaned_row_counts']['vacant_storefront_candidates']} unique candidate coordinates. "
            "Input and output SHA-256 hashes are recorded in the source/build manifests.",
            "",
            "## Limitations and interpretation",
            "",
            "1. **Not a sales forecast.** Quintile-weighted population, jobs, and transit are demand-potential proxies. "
            "The opportunity scale is calibrated to an assumed per-store order volume (`vbar=350/600/900`), "
            "not observed purchase data. The resulting NPV is conditional on these assumptions.",
            "2. **Provisional store network.** DOHMH inspection records establish inspected establishments, not "
            "current operating status. The 22 Luckin records and competitor set need a current, independently "
            "verified store list before investment decisions.",
            "3. **No measured site rents or fit-out.** Candidate rent quotes are absent; all use the configured "
            "`$9/sq ft/month` fallback and the reusable-fitout flag is assumed zero. Costs are therefore uniform "
            "across candidate sites and may materially alter the selected locations.",
            "4. **Location accuracy.** DOF coordinates are storefront/building-level geocodes, not verified doors. "
            "Demand-point locations are tract representative points. Grid distances are approximations, not "
            "pedestrian routes, and can affect both choice and selected sites.",
            "5. **Proxy interpretation.** LODES jobs and MTA ridership measure workplace employment and subway "
            "entries, respectively; neither is a direct count of coffee customers. Population, jobs, and transit "
            "are equally weighted after within-feature ranking, not empirically estimated demand coefficients.",
            "",
            "## Deliverables",
            "",
            "- `outputs/real/summary.json`: solver status, bound, gap, objectives, parameters, and checks.",
            "- `outputs/real/plan_<scenario>.csv`: selected site IDs, addresses, coordinates, costs, and modeled orders.",
            "- `outputs/real/cross_scenario.csv`: cross-scenario plan performance/regret, when all scenarios were solved.",
            "- `outputs/real/figures/site_selection_maps.png`, `scenario_dashboard.png`, and `demand_potential.png`.",
            "- `data/raw/source_manifest.json` and `data/processed/processed_manifest.json`: provenance and file hashes.",
            "",
        ]
    )
    report_path = out_dir / "MILESTONE4_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
