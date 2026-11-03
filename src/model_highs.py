"""
model_highs.py — the SAME MILP as model_gurobi.py, built as sparse matrices
and solved with HiGHS through scipy.optimize.milp (no licence needed).
Used as an independent cross-check of the Gurobi implementation.
Variable order: [x_j (nJ) | P0_i (nI) | P_ij (M)]
"""
import time
import numpy as np, scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds
from params import Instance, tangent_cuts

def build_matrices(ins: Instance):
    p = ins.p; nJ, nI, M = ins.nJ, ins.nI, ins.M
    ix = lambda j: j; i0 = lambda i: nJ + i; ip = lambda k: nJ + nI + k
    n = nJ + nI + M; k = np.arange(M)
    i, j, V = ins.pi, ins.pj, ins.pV; U = ins.U[i]; r = V / U
    rows, cols, vals, lo, hi = [], [], [], [], []
    def block(R, Cc, Vv, l, h):
        off = sum(len(x) for x in lo)
        rows.append(np.asarray(R) + off); cols.append(np.asarray(Cc)); vals.append(np.asarray(Vv, float))
        lo.append(np.asarray(l, float)); hi.append(np.asarray(h, float))
    # (5a)  P0_i + sum_j P_ij = 1
    block(np.r_[np.arange(nI), i], np.r_[i0(np.arange(nI)), ip(k)], np.ones(nI + M), np.ones(nI), np.ones(nI))
    # (5b-1) P_ij - (V/U) P0_i <= 0
    block(np.r_[k, k], np.r_[ip(k), i0(i)], np.r_[np.ones(M), -r], np.full(M, -np.inf), np.zeros(M))
    # (5b-2) P_ij - V/(U+V) x_j <= 0
    block(np.r_[k, k], np.r_[ip(k), ix(j)], np.r_[np.ones(M), -V / (U + V)], np.full(M, -np.inf), np.zeros(M))
    # (5c)  P_ij - (V/U) P0_i - (V/U) x_j >= -(V/U)
    block(np.r_[k, k, k], np.r_[ip(k), i0(i), ix(j)], np.r_[np.ones(M), -r, -r], -r, np.full(M, np.inf))
    # (8) store count, (9) budget
    block(np.zeros(nJ), np.arange(nJ), np.ones(nJ), [-np.inf], [p.K])
    block(np.zeros(nJ), np.arange(nJ), ins.C, [-np.inf], [p.B])
    # (11) pairwise spacing
    if len(ins.E):
        e = np.arange(len(ins.E))
        block(np.r_[e, e], np.r_[ins.E[:, 0], ins.E[:, 1]], np.ones(2 * len(e)), np.full(len(e), -np.inf), np.ones(len(e)))
    # (12) retention: sum_i a_il P0_i >= (1-eps) sum_i a_il
    a = ins.w[ins.lk_i] * ins.lk_V / ins.U[ins.lk_i]
    rhs = (1 - p.eps) * np.bincount(ins.lk_l, a, minlength=ins.nL)
    block(ins.lk_l, i0(ins.lk_i), a, rhs, np.full(ins.nL, np.inf))
    # (5d) tangent cuts:  -P0_i - b_i sum_j V_ij x_j <= a_i - 1
    for S0, a_, b_ in tangent_cuts(ins):
        block(np.r_[np.arange(nI), i], np.r_[i0(np.arange(nI)), ix(j)], np.r_[-np.ones(nI), -b_[i] * V],
              np.full(nI, -np.inf), a_ - 1)
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(sum(len(x) for x in lo), n))
    # bounds: (10) eligibility and N fixed to 0; (13) domains
    ub = np.ones(n); ub[:nJ] = ins.elig; ub[ins.N] = 0
    has_c = np.bincount(i, minlength=nI) > 0
    lb = np.zeros(n); lb[nJ:nJ + nI][~has_c] = 1          # nodes without candidates: P0 = 1
    # objective (7):  coef * sum_i w_i (sum_j P_ij + gamma_i P0_i) - sum_j Phi_j x_j  - const
    c = np.zeros(n); c[:nJ] = -ins.Phi
    c[nJ:nJ + nI] = ins.obj_coef * ins.w * ins.gamma
    c[nJ + nI:] = ins.obj_coef * ins.w[i]
    const = -ins.obj_coef * (ins.w * ins.gamma).sum()
    integ = np.zeros(n); integ[:nJ] = 1
    return c, const, A, np.concatenate(lo), np.concatenate(hi), lb, ub, integ

def solve_highs(ins: Instance, time_limit=None, verbose=False):
    c, const, A, lo, hi, lb, ub, integ = build_matrices(ins)
    t = time.time()
    res = milp(-c, integrality=integ, bounds=Bounds(lb, ub), constraints=LinearConstraint(A, lo, hi),
               options={"time_limit": time_limit or ins.p.time_limit_s, "mip_rel_gap": ins.p.mip_gap, "disp": verbose})
    if res.x is None:
        raise RuntimeError(f"HiGHS returned no solution: {res.message}")
    bound = -getattr(res, "mip_dual_bound", np.nan) + const if getattr(res, "mip_dual_bound", None) is not None else np.nan
    return {"x": np.round(res.x[:ins.nJ]), "obj": -res.fun + const, "bound": bound,
            "gap": getattr(res, "mip_gap", np.nan), "status": res.message, "runtime": time.time() - t,
            "n_vars": A.shape[1], "n_cons": A.shape[0], "P0": res.x[ins.nJ:ins.nJ + ins.nI]}
