"""
run.py — STEP 4: solve the MILP for each demand scenario, verify, and save.
  python src/run.py --data synthetic --solver highs            # test pipeline, no licence
  python src/run.py --data real --solver gurobi --scenario all # Milestone 4 run
Outputs in outputs/<data>/ : summary.json, plan_<s>.csv, cross_scenario.csv
"""
import argparse, json, sys, time
import numpy as np, pandas as pd
from config import ROOT, OUT, PROC, scenario, DEMAND_SCENARIOS
import params
from evaluate import evaluate, greedy

def solve(ins, solver, start=None, verbose=False):
    if solver == "gurobi":
        from model_gurobi import solve_gurobi
        return solve_gurobi(ins, start=start, verbose=verbose)
    from model_highs import solve_highs
    return solve_highs(ins, verbose=verbose)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="synthetic", choices=["synthetic", "real"])
    ap.add_argument("--solver", default="gurobi", choices=["gurobi", "highs"])
    ap.add_argument("--scenario", default="all", choices=list(DEMAND_SCENARIOS) + ["all"])
    ap.add_argument("--time-limit", type=float, default=600)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    data_dir = ROOT / "data" / "synthetic" if a.data == "synthetic" else PROC
    out = OUT / a.data; out.mkdir(parents=True, exist_ok=True)
    names = list(DEMAND_SCENARIOS) if a.scenario == "all" else [a.scenario]
    report, plans, insts = {}, {}, {}
    for s in names:
        ins = params.build(data_dir, scenario(s, time_limit_s=a.time_limit)); insts[s] = ins
        print(f"\n=== scenario {s}: vbar={ins.p.vbar}  ({a.data} data, {a.solver})")
        for k, v in params.summary(ins).items(): print(f"   {k}: {v}")
        t = time.time(); xg = greedy(ins); eg = evaluate(ins, xg)
        print(f"   greedy: {eg['n_open']} stores, Z=${eg['Z']:,.0f} ({time.time()-t:.1f}s)")
        res = solve(ins, a.solver, start=xg, verbose=a.verbose)
        solver_gap = res.get("gap", np.nan)
        ev = evaluate(ins, res["x"])
        # --- sanity checks required for M4 ---
        assert ev["feasible"], f"plan violates {ev['checks']}"
        rel = abs(ev["Z"] - res["obj"]) / max(1.0, abs(ev["Z"]))
        assert rel < 1e-4, f"MILP objective {res['obj']:,.0f} != closed-form {ev['Z']:,.0f}"
        if ev["Z"] < eg["Z"] - 1e-6 * max(1, abs(eg["Z"])):
            # only possible when the solver stopped early (time limit) without using the greedy start
            print(f"   WARNING: solver incumbent (${ev['Z']:,.0f}) below greedy (${eg['Z']:,.0f}); "
                  f"status {res['status']} -> reporting greedy plan, bound still from solver")
            res = dict(res, x=xg, obj=eg["Z"]); ev = eg
        # gap on the TRUE objective Z (HiGHS reports it on its internal, constant-shifted objective)
        res["gap"] = (res["bound"] - ev["Z"]) / max(1.0, abs(ev["Z"])) if np.isfinite(res["bound"]) else np.nan
        print(f"   MILP [{res['status']}]: {ev['n_open']} stores, Z=${ev['Z']:,.0f}, bound=${res['bound']:,.0f}, "
              f"gap={res['gap']:.2e}, {res['runtime']:.1f}s, {res['n_vars']} vars, {res['n_cons']} cons")
        print(f"   gross new orders/day {ev['gross_new_orders']:,.0f} = from competitors {ev['from_competitors']:,.0f}"
              f" + from no-purchase {ev['from_no_purchase']:,.0f} + cannibalised Luckin {ev['lost_incumbent_orders']:,.0f}")
        print(f"   min incumbent retention {ev['min_retention']:.3f} (floor {1-ins.p.eps:.2f})")
        plans[s] = res["x"]
        sel = ins.cands.iloc[ev["opened"]].copy()
        sel["orders_per_day"] = ev["store_orders"]; sel["capex"] = ins.C[ev["opened"]]
        sel["Phi"] = ins.Phi[ev["opened"]]; sel["annual_rent"] = ins.R[ev["opened"]]
        sel.to_csv(out / f"plan_{s}.csv", index=False)
        gross_reconciliation = (
            ev["lost_incumbent_orders"] + ev["from_competitors"] + ev["from_no_purchase"]
        )
        report[s] = {"vbar": ins.p.vbar, "lambda": ins.lam, "solver": a.solver,
                     "status": str(res["status"]), "Z": ev["Z"],
                     "bound": res["bound"], "gap": res["gap"], "solver_reported_gap": solver_gap,
                     "runtime_s": res["runtime"], "n_vars": res["n_vars"], "n_cons": res["n_cons"],
                     "n_open": ev["n_open"], "opened": ins.cands.cand_id.iloc[ev["opened"]].tolist(),
                     "gross_new_orders": ev["gross_new_orders"], "incremental_orders": ev["incremental_orders"],
                     "lost_incumbent_orders": ev["lost_incumbent_orders"], "from_competitors": ev["from_competitors"],
                     "from_no_purchase": ev["from_no_purchase"], "min_retention": ev["min_retention"],
                     "order_reconciliation_residual": ev["gross_new_orders"] - gross_reconciliation,
                     "greedy_Z": eg["Z"], "greedy_n": eg["n_open"],
                     "parameters": {
                         "half_distance_m": ins.p.half_dist_m,
                         "choice_cutoff_m": ins.p.d_max_m,
                         "outside_option_attraction": ins.p.u0,
                         "annual_discount_rate": ins.p.r,
                         "horizon_years": ins.p.T,
                         "contribution_margin_per_order": ins.p.margin,
                         "max_new_stores": ins.p.K,
                         "capital_budget": ins.p.B,
                         "incumbent_retention_floor": 1 - ins.p.eps,
                         "spacing_m": ins.p.spacing_m,
                         "mip_gap_tolerance": ins.p.mip_gap,
                     },
                     "size": params.summary(ins)}
    if len(names) > 1:          # evaluate every scenario-optimal plan in every scenario -> regret
        rows = []
        for ps, x in plans.items():
            for s, ins in insts.items():
                e = evaluate(ins, x)
                rows.append({"plan": ps, "scenario": s, "NPV": e["Z"], "regret": report[s]["Z"] - e["Z"],
                             "feasible": e["feasible"], "n_open": e["n_open"]})
        cs = pd.DataFrame(rows); cs.to_csv(out / "cross_scenario.csv", index=False)
        print("\n", cs.pivot(index="plan", columns="scenario", values="regret").round(0))
    json.dump(report, open(out / "summary.json", "w"), indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"\nsaved -> {out}")

if __name__ == "__main__":
    main()
