"""
params.py — STEP 3: compute every model coefficient from the frozen tables.
Maps 1-to-1 onto draft equations (1)-(3), (6) and the sets E, N of (11).
"""
from dataclasses import dataclass
import numpy as np, pandas as pd
from config import Params, BASE
from geo import haversine

@dataclass
class Instance:
    p: Params
    tracts: pd.DataFrame; cands: pd.DataFrame; luckin: pd.DataFrame; comps: pd.DataFrame
    h: np.ndarray            # (1) demand index per node, dimensionless
    tiers: np.ndarray        # z: tier (1..5) per node x feature
    cuts: dict               # quantile cut-offs per feature
    lam: float               # (2) scale lambda_s
    w: np.ndarray            # (2) purchase opportunities / day per node
    U: np.ndarray            # (3) pre-existing attraction per node
    Vbar: np.ndarray         # (3) Luckin attraction per node
    gamma: np.ndarray        # (3) baseline Luckin share per node
    pi: np.ndarray; pj: np.ndarray; pV: np.ndarray   # demand-candidate pairs within D_max (M of them)
    lk_i: np.ndarray; lk_l: np.ndarray; lk_V: np.ndarray  # node-incumbent pairs within D_max
    C: np.ndarray            # opening cash per candidate, $
    R: np.ndarray            # annual rent per candidate, $/yr
    Phi: np.ndarray          # (6) PV of opening + fixed operating cost, $
    elig: np.ndarray         # e_j
    E: np.ndarray            # (11) candidate pairs closer than spacing
    N: np.ndarray            # (11) candidates too close to an existing Luckin

    @property
    def nI(self): return len(self.tracts)
    @property
    def nJ(self): return len(self.cands)
    @property
    def nL(self): return len(self.luckin)
    @property
    def M(self): return len(self.pi)
    @property
    def obj_coef(self):   # $ per (daily order sustained for T years)
        return self.p.days * self.p.margin * self.p.aT

def demand_index(tr, p):
    """Equation (1): within-feature quintile tiers -> scores b_q -> equal-weight average h_i."""
    b = np.array(p.tier_scores); tiers = np.zeros((len(tr), len(p.features)), int); cuts = {}
    for g, f in enumerate(p.features):
        v = tr[f].to_numpy(float)
        c = np.percentile(v, p.tier_cuts_pct); cuts[f] = c
        tiers[:, g] = np.searchsorted(c, v, side="left") + 1     # ties at a cut-off -> LOWER tier
    h = b[tiers - 1].mean(axis=1)
    return h, tiers, cuts

def choice_terms(nI, wd, tid, lk_id, cd_id, p):
    """Equation (3) for given choice parameters. Returns U, Vbar, comp sum, pair arrays."""
    wd = wd[wd.dist_m <= p.d_max_m]
    V = np.exp(-p.beta * wd.dist_m.to_numpy())
    i = wd.tract_id.map(tid).to_numpy()
    kind = wd.kind.to_numpy()
    is_l, is_c, is_j = kind == "luckin", kind == "comp", kind == "cand"
    Vbar = np.bincount(i[is_l], V[is_l], minlength=nI)
    Vcomp = np.bincount(i[is_c], V[is_c], minlength=nI)
    U = p.u0 + Vbar + Vcomp
    lk = (i[is_l], wd.store_id[is_l].map(lk_id).to_numpy(), V[is_l])
    cj = (i[is_j], wd.store_id[is_j].map(cd_id).to_numpy(), V[is_j])
    return U, Vbar, lk, cj

def build(data_dir, p: Params = BASE) -> Instance:
    rd = lambda n: pd.read_csv(f"{data_dir}/{n}.csv", dtype={"tract_id": str, "store_id": str, "cand_id": str})
    tr, cd, lk, cp, wd = rd("tracts"), rd("candidates"), rd("luckin"), rd("competitors"), rd("walk_dist")
    tid = {t: k for k, t in enumerate(tr.tract_id)}
    lk_id = {s: k for k, s in enumerate(lk.store_id)}; cd_id = {s: k for k, s in enumerate(cd.cand_id)}
    wd = wd[wd.tract_id.isin(tid)]

    h, tiers, cuts = demand_index(tr, p)
    # (2) lambda uses gamma^ref under the BASE choice parameters -> w frozen in choice tests
    Uref, Vbref, _, _ = choice_terms(len(tr), wd, tid, lk_id, cd_id, BASE)
    gref = Vbref / Uref
    denom = (h * gref).sum()
    assert denom > 0, "sum_i h_i gamma_i^ref must be positive (no incumbent within D_max of any node?)"
    lam = len(lk) * p.vbar / denom
    w = lam * h

    U, Vbar, (li, ll, lV), (ci, cj, cV) = choice_terms(len(tr), wd, tid, lk_id, cd_id, p)
    gamma = Vbar / U

    # (6) costs
    rho = cd.rent_psf_month.fillna(p.rent_benchmark_psf_month).to_numpy(float)
    R = 12 * p.area_sf * p.rent_uplift * rho
    C = np.where(cd.reusable_fitout.to_numpy() == 1, p.capex_reuse, p.capex) * p.capex_mult
    Phi = C + p.aT * (R + p.labor + p.other_fixed)

    # (11) spacing sets from GEODESIC distance (not walking distance)
    la, lo = cd.lat.to_numpy(), cd.lon.to_numpy()
    D = haversine(la[:, None], lo[:, None], la[None], lo[None])
    a, b = np.nonzero(np.triu(D <= p.spacing_m, k=1)); E = np.stack([a, b], 1)
    DL = haversine(la[:, None], lo[:, None], lk.lat.to_numpy()[None], lk.lon.to_numpy()[None])
    N = np.nonzero(DL.min(1) <= p.spacing_m)[0] if len(lk) else np.array([], int)

    return Instance(p, tr, cd, lk, cp, h, tiers, cuts, lam, w, U, Vbar, gamma,
                    ci, cj, cV, li, ll, lV, C, R, Phi, cd.eligible.to_numpy(int), E, N)

def summary(ins: Instance) -> dict:
    I1 = len(np.unique(ins.pi))
    return {"|I|": ins.nI, "|J|": ins.nJ, "|L|": ins.nL, "|C|": len(ins.comps),
            "M = sum|J_i|": ins.M, "nodes with J_i nonempty": I1,
            "share vars (M+|I|)": ins.M + ins.nI,
            "choice constraints (3M+|I|)": 3 * ins.M + ins.nI,
            "|E| spacing pairs": len(ins.E), "|N| blocked by incumbents": len(ins.N),
            "lambda": round(ins.lam, 2), "total opportunities/day": round(ins.w.sum()),
            "baseline Luckin orders/day": round((ins.w * ins.gamma).sum(), 1),
            "Phi range $": (round(ins.Phi.min()), round(ins.Phi.max())),
            "break-even incremental orders/day (median Phi)": round(np.median(ins.Phi) / ins.obj_coef, 1)}

def tangent_cuts(ins: Instance):
    """(5d) valid inequalities. With S_i = sum_j V_ij x_j, the exact share 1 - P_i0 = S_i/(U_i+S_i) is
    CONCAVE in S_i, so every tangent line lies above it:
        1 - P_i0 <= a_i + b_i * sum_j V_ij x_j,   b_i = U_i/(U_i+S0)^2,  a_i = S0/(U_i+S0) - b_i*S0.
    Every integer plan still satisfies them (no solution is cut off); they only remove fractional
    LP points that 'open half a store' in many places. Returns list of (S0, a[nI], b[nI])."""
    out = []
    for S0 in ins.p.tangent_S0:
        b = ins.U / (ins.U + S0) ** 2
        out.append((S0, S0 / (ins.U + S0) - b * S0, b))
    return out
