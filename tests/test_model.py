"""
test_model.py — validation required by the M3 plan (Section 5):
  1. exactness: for EVERY opening plan of a tiny instance, the linear constraints
     (5a)-(5c) pin P to the logit shares (4)  (min and max of each P_i0 coincide)
  2. brute-force enumeration optimum == MILP optimum (HiGHS and Gurobi code)
  3. no openings -> Z = 0
  4. without (5c) the retention constraint lets the solver 'fake' shares
Run:  python tests/test_model.py
"""
import sys, itertools, pathlib, tempfile
import numpy as np, pandas as pd
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from dataclasses import replace
from config import BASE
from geo import grid_walk
import params
from evaluate import evaluate
from model_highs import build_matrices, solve_highs
from scipy.optimize import linprog

def tiny(seed=1, nT=8, nJ=8, nL=2, nC=6, scale=900):
    rng = np.random.default_rng(seed); d = pathlib.Path(tempfile.mkdtemp())
    pt = lambda n: (40.75 + rng.uniform(0, 0.03, n), -73.99 + rng.uniform(0, 0.03, n))
    la, lo = pt(nT)
    tr = pd.DataFrame(dict(tract_id=[f"T{k}" for k in range(nT)], lat=la, lon=lo,
                           residents=rng.integers(500, 6000, nT), jobs=rng.integers(0, 30000, nT),
                           transit=rng.integers(0, 20000, nT)))
    la, lo = pt(nJ); cd = pd.DataFrame(dict(cand_id=[f"J{k}" for k in range(nJ)], lat=la, lon=lo,
                                            rent_psf_month=np.nan, reusable_fitout=rng.integers(0, 2, nJ), eligible=1))
    la, lo = pt(nL); lk = pd.DataFrame(dict(store_id=[f"L{k}" for k in range(nL)], lat=la, lon=lo))
    la, lo = pt(nC); cp = pd.DataFrame(dict(store_id=[f"C{k}" for k in range(nC)], lat=la, lon=lo))
    st = pd.concat([cd.rename(columns={"cand_id": "store_id"})[["store_id", "lat", "lon"]].assign(kind="cand"),
                    lk.assign(kind="luckin"), cp.assign(kind="comp")], ignore_index=True)
    D = grid_walk(tr.lat.values[:, None], tr.lon.values[:, None], st.lat.values[None], st.lon.values[None])
    ii, kk = np.nonzero(D <= 2400)
    wd = pd.DataFrame(dict(tract_id=tr.tract_id.values[ii], kind=st.kind.values[kk], store_id=st.store_id.values[kk], dist_m=D[ii, kk]))
    for n, df in [("tracts", tr), ("candidates", cd), ("luckin", lk), ("competitors", cp), ("walk_dist", wd)]:
        df.to_csv(d / f"{n}.csv", index=False)
    return d

def test_exactness(ins):
    c, const, A, lo, hi, lb, ub, integ = build_matrices(ins)
    nJ, nI = ins.nJ, ins.nI
    # drop policy rows (keep only choice rows 0 .. nI+3M) so every plan is allowed
    nchoice = nI + 3 * ins.M; A, lo, hi = A[:nchoice], lo[:nchoice], hi[:nchoice]
    from scipy.optimize import LinearConstraint, milp, Bounds
    worst = 0.0
    for plan in itertools.product([0, 1], repeat=nJ):
        x = np.array(plan, float); l2, u2 = lb.copy(), ub.copy(); u2[:nJ] = 1
        l2[:nJ] = x; u2[:nJ] = x
        exact = ins.U / (ins.U + np.bincount(ins.pi, ins.pV * x[ins.pj], minlength=nI))
        for sgn in (+1, -1):            # minimise and maximise sum of P0
            obj = np.zeros(A.shape[1]); obj[nJ:nJ + nI] = sgn
            r = milp(obj, bounds=Bounds(l2, u2), constraints=LinearConstraint(A, lo, hi))
            worst = max(worst, np.abs(r.x[nJ:nJ + nI] - exact).max())
    # (5d) tangent cuts must hold at every integer plan (valid inequalities)
    from params import tangent_cuts
    for plan in itertools.product([0, 1], repeat=nJ):
        x = np.array(plan, float); S = np.bincount(ins.pi, ins.pV * x[ins.pj], minlength=nI)
        for S0, a_, b_ in tangent_cuts(ins):
            assert (S / (ins.U + S) <= a_ + b_ * S + 1e-12).all(), "tangent cut invalid"
    print(f"[1b] tangent cuts (5d) valid at all {2**nJ} plans")
    print(f"[1] exactness over all {2**nJ} plans: max |P0_LP - P0_logit| = {worst:.2e}")
    assert worst < 1e-6

def brute(ins):
    best = (-np.inf, None)
    for plan in itertools.product([0, 1], repeat=ins.nJ):
        e = evaluate(ins, plan)
        if e["feasible"] and e["Z"] > best[0]: best = (e["Z"], np.array(plan))
    return best

if __name__ == "__main__":
    d = tiny()
    for vbar in (900, 3000, 8000):
        p = replace(BASE, vbar=vbar, K=3, B=1.2e6, eps=0.10)
        ins = params.build(d, p)
        if vbar == 900:
            test_exactness(ins)
            e0 = evaluate(ins, np.zeros(ins.nJ)); print(f"[3] no openings -> Z = {e0['Z']:.6f}"); assert abs(e0["Z"]) < 1e-6
        Zb, xb = brute(ins)
        h = solve_highs(ins)
        sys.path.insert(0, str(HERE / "fake_gurobipy"))
        import model_gurobi
        g = model_gurobi.solve_gurobi(ins, verbose=False)
        print(f"[2] vbar={vbar}: brute-force Z={Zb:,.0f} open={np.nonzero(xb)[0].tolist()} | "
              f"HiGHS Z={h['obj']:,.0f} open={np.nonzero(h['x'])[0].tolist()} | "
              f"gurobipy-code Z={g['obj']:,.0f} open={np.nonzero(g['x'])[0].tolist()}")
        assert abs(Zb - h["obj"]) < 1e-3 * max(1, abs(Zb)) and abs(Zb - g["obj"]) < 1e-3 * max(1, abs(Zb))
        assert abs(evaluate(ins, h["x"])["Z"] - h["obj"]) < 1e-3 * max(1, abs(Zb))
    # [4] why (5c) matters: max sum of P0 without the lower link, with x fixed open
    ins = params.build(d, replace(BASE, vbar=2500))
    c, const, A, lo, hi, lb, ub, integ = build_matrices(ins)
    nI, M, nJ = ins.nI, ins.M, ins.nJ
    keep = np.r_[0:nI + 2 * M]                   # (5a)+(5b) only
    from scipy.optimize import milp, LinearConstraint, Bounds
    l2, u2 = lb.copy(), ub.copy(); l2[:nJ] = 1; u2[:nJ] = 1
    obj = np.zeros(A.shape[1]); obj[nJ:nJ + nI] = -1
    r = milp(obj, bounds=Bounds(l2, u2), constraints=LinearConstraint(A[keep], lo[keep], hi[keep]))
    exact = ins.U / (ins.U + np.bincount(ins.pi, ins.pV, minlength=nI))
    gap = r.x[nJ:nJ + nI] - exact; k = gap.argmax()
    print(f"[4] all candidates open, WITHOUT (5c): node {k} P0 can be inflated from {exact[k]:.3f} (logit) "
          f"to {r.x[nJ + k]:.3f} -> (5c) is needed because retention (12) rewards a high P0")
    assert gap.max() > 0.05
    print("ALL TESTS PASSED")
