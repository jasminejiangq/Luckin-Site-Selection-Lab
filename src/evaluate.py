"""
evaluate.py — closed-form evaluation of ANY opening plan x using the original
nonlinear logit shares, equation (4).  Used to (a) verify the MILP solution,
(b) evaluate the base plan under other scenarios (regret), (c) greedy heuristic.
"""
import numpy as np
from params import Instance

def shares(ins: Instance, x):
    x = np.asarray(x, float)
    S = np.bincount(ins.pi, ins.pV * x[ins.pj], minlength=ins.nI)    # sum of open candidates' attraction
    P0 = ins.U / (ins.U + S)                                         # (4) aggregate pre-existing share
    Pij = ins.pV * x[ins.pj] / (ins.U + S)[ins.pi]                   # (4) per pair
    return P0, Pij

def evaluate(ins: Instance, x, tol=1e-6):
    p = ins.p; x = np.round(np.asarray(x, float)); P0, Pij = shares(ins, x)
    w, g = ins.w, ins.gamma
    bracket = np.bincount(ins.pi, Pij, minlength=ins.nI) + g * P0 - g          # objective (7) bracket
    assert np.allclose(bracket, (1 - g) * (1 - P0), atol=1e-9), "identity (1-g)(1-P0) failed"
    inc = w * bracket
    Z = ins.obj_coef * inc.sum() - (ins.Phi * x).sum()
    lost = (w * g * (1 - P0)).sum()
    Vcomp = ins.U - p.u0 - ins.Vbar
    from_comp = (w * Vcomp / ins.U * (1 - P0)).sum(); from_out = (w * p.u0 / ins.U * (1 - P0)).sum()
    store_orders = np.bincount(ins.pj, ins.w[ins.pi] * Pij, minlength=ins.nJ)
    assert np.isclose(
        store_orders.sum(), lost + from_comp + from_out, rtol=1e-9, atol=1e-7
    ), "new-store orders do not reconcile to incumbent cannibalization and external demand"
    # retention (12)
    a = ins.w[ins.lk_i] * ins.lk_V / ins.U[ins.lk_i]
    base_l = np.bincount(ins.lk_l, a, minlength=ins.nL)
    now_l = np.bincount(ins.lk_l, a * P0[ins.lk_i], minlength=ins.nL)
    ret = np.divide(now_l, base_l, out=np.ones(ins.nL), where=base_l > 0)
    opened = np.nonzero(x > 0.5)[0]
    S = set(map(int, opened))
    checks = {
        "K (8)": len(opened) <= p.K,
        "budget (9)": (ins.C * x).sum() <= p.B + tol,
        "eligibility (10)": all(ins.elig[j] == 1 for j in opened),
        "spacing new-new (11)": not any(a_ in S and b_ in S for a_, b_ in ins.E),
        "spacing new-incumbent (11)": not any(j in S for j in ins.N),
        "retention (12)": bool((ret >= 1 - p.eps - tol).all()),
    }
    return {"Z": Z, "opened": opened, "n_open": len(opened), "capex": (ins.C * x).sum(),
            "gross_new_orders": store_orders[opened].sum(), "store_orders": store_orders[opened],
            "lost_incumbent_orders": lost, "from_competitors": from_comp, "from_no_purchase": from_out,
            "incremental_orders": inc.sum(), "retention": ret, "min_retention": ret.min() if ins.nL else 1,
            "feasible": all(checks.values()), "checks": checks}

def greedy(ins: Instance, verbose=False):
    """Add the store with the largest positive marginal NPV that keeps the plan feasible."""
    x = np.zeros(ins.nJ); blocked = set(np.nonzero(ins.elig == 0)[0]) | set(map(int, ins.N))
    nbr = {}
    for a_, b_ in ins.E: nbr.setdefault(a_, []).append(b_); nbr.setdefault(b_, []).append(a_)
    while x.sum() < ins.p.K:
        S = np.bincount(ins.pi, ins.pV * x[ins.pj], minlength=ins.nI)
        U = ins.U[ins.pi]; Si = S[ins.pi]
        dP0 = U / (U + Si) - U / (U + Si + ins.pV)
        gain = ins.obj_coef * np.bincount(ins.pj, ins.w[ins.pi] * (1 - ins.gamma[ins.pi]) * dP0,
                                          minlength=ins.nJ) - ins.Phi
        gain[list(blocked | set(np.nonzero(x)[0]))] = -np.inf
        added = False
        for j in np.argsort(-gain):
            if gain[j] <= 0: break
            y = x.copy(); y[j] = 1
            if evaluate(ins, y)["feasible"]:
                x = y; added = True; blocked |= set(nbr.get(j, []))
                if verbose: print(f"  + {ins.cands.cand_id.iat[j]} gain ${gain[j]:,.0f}")
                break
        if not added: break
    return x
