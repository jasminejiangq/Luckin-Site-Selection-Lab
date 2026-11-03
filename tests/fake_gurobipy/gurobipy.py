"""
TEST-ONLY stand-in for gurobipy: implements the small API subset used in
src/model_gurobi.py and solves with HiGHS (scipy). Lets us execute the exact
Gurobi model code without a licence. Real runs must use real gurobipy.
"""
import time
import numpy as np, scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds

class GRB:
    BINARY, CONTINUOUS, INTEGER = "B", "C", "I"
    MAXIMIZE, MINIMIZE = -1, 1
    OPTIMAL, TIME_LIMIT = 2, 9

class LinExpr:
    __hash__ = None
    def __init__(self, terms=None, const=0.0):
        self.t = dict(terms or {}); self.c = float(const)
    @staticmethod
    def of(o):
        if isinstance(o, LinExpr): return o
        if isinstance(o, Var): return LinExpr({o.idx: 1.0})
        return LinExpr(const=float(o))
    def __add__(self, o):
        o = LinExpr.of(o); t = dict(self.t)
        for k, v in o.t.items(): t[k] = t.get(k, 0.0) + v
        return LinExpr(t, self.c + o.c)
    __radd__ = __add__
    def __neg__(self): return LinExpr({k: -v for k, v in self.t.items()}, -self.c)
    def __sub__(self, o): return self + (-LinExpr.of(o))
    def __rsub__(self, o): return LinExpr.of(o) + (-self)
    def __mul__(self, a):
        assert not isinstance(a, (LinExpr, Var)), "nonlinear"
        return LinExpr({k: v * a for k, v in self.t.items()}, self.c * a)
    __rmul__ = __mul__
    def __le__(self, o): return (self - o, "<")
    def __ge__(self, o): return (self - o, ">")
    def __eq__(self, o): return (self - o, "=")

class Var:
    __hash__ = object.__hash__
    def __init__(self, idx, lb, ub, vtype):
        self.idx, self.LB, self.UB, self.vtype, self.Start, self.X = idx, lb, ub, vtype, None, None
    def _e(self): return LinExpr({self.idx: 1.0})
    def __add__(self, o): return self._e() + o
    __radd__ = __add__
    def __sub__(self, o): return self._e() - o
    def __rsub__(self, o): return LinExpr.of(o) - self._e()
    def __mul__(self, a): return self._e() * a
    __rmul__ = __mul__
    def __neg__(self): return -self._e()
    def __le__(self, o): return self._e() <= o
    def __ge__(self, o): return self._e() >= o
    def __eq__(self, o): return self._e() == o

def quicksum(it):
    t, c = {}, 0.0
    for o in it:
        o = LinExpr.of(o); c += o.c
        for k, v in o.t.items(): t[k] = t.get(k, 0.0) + v
    return LinExpr(t, c)

class _P: pass

class Model:
    def __init__(self, name=""):
        self.vars, self.cons, self.Params = [], [], _P()
        self.Params.TimeLimit, self.Params.MIPGap, self.Params.OutputFlag = 1e9, 1e-4, 0
        self.obj, self.sense, self.SolCount = LinExpr(), 1, 0
    def addVar(self, lb=0.0, ub=float("inf"), vtype="C", name=""):
        v = Var(len(self.vars), lb, ub, vtype); self.vars.append(v); return v
    def addVars(self, n, lb=0.0, ub=float("inf"), vtype="C", name=""):
        if vtype == GRB.BINARY: lb, ub = 0.0, 1.0
        return {k: self.addVar(lb, ub, vtype) for k in range(n)}
    def addConstr(self, tc, name=""):
        self.cons.append(tc)
    def setObjective(self, e, sense): self.obj, self.sense = LinExpr.of(e), sense
    def update(self): pass
    @property
    def NumVars(self): return len(self.vars)
    @property
    def NumConstrs(self): return len(self.cons)
    def optimize(self):
        n = len(self.vars); R, C, V, lo, hi = [], [], [], [], []
        for r, (e, s) in enumerate(self.cons):
            for k, v in e.t.items(): R.append(r); C.append(k); V.append(v)
            lo.append(-e.c if s in "=>" else -np.inf); hi.append(-e.c if s in "=<" else np.inf)
        A = sp.csr_matrix((V, (R, C)), shape=(len(self.cons), n))
        c = np.zeros(n)
        for k, v in self.obj.t.items(): c[k] = v
        c *= self.sense
        lb = np.array([v.LB for v in self.vars]); ub = np.array([v.UB for v in self.vars])
        integ = np.array([v.vtype != "C" for v in self.vars], float)
        t = time.time()
        res = milp(c, integrality=integ, bounds=Bounds(lb, ub), constraints=LinearConstraint(A, lo, hi),
                   options={"time_limit": self.Params.TimeLimit, "mip_rel_gap": self.Params.MIPGap})
        self.Runtime = time.time() - t
        if res.x is None: self.SolCount = 0; self.Status = 3; return
        self.SolCount = 1; self.Status = GRB.OPTIMAL if res.status == 0 else GRB.TIME_LIMIT
        for v, val in zip(self.vars, res.x): v.X = val
        self.ObjVal = self.sense * res.fun + self.obj.c
        db = getattr(res, "mip_dual_bound", None)
        self.ObjBound = self.sense * db + self.obj.c if db is not None else np.nan
        self.MIPGap = getattr(res, "mip_gap", 0.0)
