"""
config.py — every number the model uses, in one place.
All values follow the Milestone 3 draft (Group 36). Each one is an
ASSUMPTION (A), DATA (D) or DERIVED ESTIMATE (E) — see the comment tag.
Change values here only; never hard-code them in other files.
"""
from dataclasses import dataclass, field, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # repo root, no hard-coded user paths
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"

@dataclass(frozen=True)
class Params:
    # ---- demand index (Section 1) ----
    tier_scores: tuple = (0.50, 0.75, 1.00, 1.25, 1.50)   # b_q, A
    tier_cuts_pct: tuple = (20, 40, 60, 80)               # quantile cut-offs, A
    features: tuple = ("residents", "jobs", "transit")    # G
    vbar: float = 600.0          # orders / existing store / day, A (scenario)
    # ---- customer choice (Section 2) ----
    half_dist_m: float = 400.0   # attraction halves every 400 m -> beta = ln2/400, A
    d_max_m: float = 1600.0      # choice-set cut-off D_max, A
    u0: float = 0.35             # no-purchase attraction, A
    # ---- finance (Section 3) ----
    margin: float = 2.33         # m, $/order contribution, A
    area_sf: float = 1000.0      # A_j, A
    rent_uplift: float = 1.20    # kappa, A
    rent_benchmark_psf_month: float = 9.0   # fallback when site rent is unavailable, A
    staff: int = 2; hours: float = 14; wage: float = 19.50; burden: float = 1.237  # F^L, A
    other_fixed: float = 60_000.0          # F^O $/yr, A
    capex: float = 500_000.0               # C_j standard, A
    capex_reuse: float = 395_000.0         # C_j with reusable food-service fit-out, A
    capex_mult: float = 1.0                # for +-25% sensitivity
    r: float = 0.10; T: int = 10; days: int = 365   # A
    # ---- policy constraints (Section 4) ----
    K: int = 10                  # max openings, A
    B: float = 5_000_000.0       # capital budget $, A
    spacing_m: float = 152.4     # 500 ft, A
    eps: float = 0.15            # max share of baseline orders an incumbent may lose, A
    # ---- solver ----
    tangent_S0: tuple = (0.3, 0.7, 1.5, 3.0)   # (5d) valid tangent cuts; () switches them off
    time_limit_s: float = 600.0
    mip_gap: float = 1e-4

    @property
    def beta(self):
        import math
        return math.log(2) / self.half_dist_m          # 1/m

    @property
    def aT(self):
        return sum((1 + self.r) ** -t for t in range(1, self.T + 1))

    @property
    def labor(self):
        return self.staff * self.hours * self.wage * self.burden * self.days

BASE = Params()
DEMAND_SCENARIOS = {"low": 350.0, "base": 600.0, "high": 900.0}

def scenario(name: str, **overrides) -> Params:
    """Params for a demand scenario plus optional overrides, e.g. scenario('base', u0=1.0)."""
    return replace(BASE, vbar=DEMAND_SCENARIOS[name], **overrides)
