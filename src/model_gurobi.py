"""
model_gurobi.py — the Milestone-4 MILP in gurobipy, written so each block
matches one numbered equation in the report.

    max  Z_s = 365 m a_T sum_i w_i ( sum_j P_ij + gamma_i P_i0 - gamma_i ) - sum_j Phi_j x_j      (7)
    s.t. P_i0 + sum_{j in J_i} P_ij = 1                                   for all i               (5a)
         P_ij <= (V_ij/U_i) P_i0                                          for all (i,j)           (5b)
         P_ij <= V_ij/(U_i+V_ij) x_j                                      for all (i,j)           (5b)
         P_ij >= (V_ij/U_i) (P_i0 - 1 + x_j)                              for all (i,j)           (5c)
         sum_j x_j <= K                                                                           (8)
         sum_j C_j x_j <= B                                                                       (9)
         x_j <= e_j                                                       for all j               (10)
         x_j + x_j' <= 1 for (j,j') in E;  x_j = 0 for j in N                                     (11)
         sum_i w_i (V_il/U_i) P_i0 >= (1-eps) sum_i w_i (V_il/U_i)        for all l               (12)
         1 - P_i0 <= a_i + b_i sum_j V_ij x_j   (valid tangent cuts, optional)  for all i     (5d)
         x binary, 0 <= P <= 1                                                                    (13)
"""
from collections import defaultdict
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from params import Instance, tangent_cuts

def build_model(ins: Instance, start=None, verbose=True):
    p = ins.p
    I, J = range(ins.nI), range(ins.nJ)
    m = gp.Model("luckin_manhattan_expansion")
    m.Params.OutputFlag = 1 if verbose else 0
    m.Params.TimeLimit = p.time_limit_s
    m.Params.MIPGap = p.mip_gap

    # ---------- decision variables (13) ----------
    x = m.addVars(ins.nJ, vtype=GRB.BINARY, name="x")                  # 1 if candidate j opens
    P0 = m.addVars(ins.nI, lb=0.0, ub=1.0, name="P0")                  # share choosing ANY pre-existing option
    P = m.addVars(ins.M, lb=0.0, ub=1.0, name="P")                     # P[k] = P_{i(k) j(k)}, only pairs within D_max

    pairs_of_node = defaultdict(list)
    for k in range(ins.M):
        pairs_of_node[int(ins.pi[k])].append(k)

    # ---------- choice constraints (5a)-(5c) ----------
    for i in I:
        m.addConstr(P0[i] + gp.quicksum(P[k] for k in pairs_of_node[i]) == 1, name=f"alloc[{i}]")   # (5a)
    for k in range(ins.M):
        i, j, V = int(ins.pi[k]), int(ins.pj[k]), float(ins.pV[k]); U = float(ins.U[i])
        m.addConstr(P[k] <= (V / U) * P0[i], name=f"ratio_ub[{k}]")                               # (5b)
        m.addConstr(P[k] <= V / (U + V) * x[j], name=f"open_ub[{k}]")                             # (5b)
        m.addConstr(P[k] >= (V / U) * (P0[i] - 1 + x[j]), name=f"ratio_lb[{k}]")                  # (5c)

    # ---------- (5d) valid tangent cuts: tighten the LP relaxation, cut off no integer plan ----------
    for S0, a_, b_ in tangent_cuts(ins):
        for i in I:
            m.addConstr(1 - P0[i] <= float(a_[i]) + float(b_[i]) * gp.quicksum(float(ins.pV[k]) * x[int(ins.pj[k])]
                                                                             for k in pairs_of_node[i]),
                        name=f"tangent[{i},{S0}]")

    # ---------- policy constraints ----------
    m.addConstr(gp.quicksum(x[j] for j in J) <= p.K, name="store_count")                          # (8)
    m.addConstr(gp.quicksum(float(ins.C[j]) * x[j] for j in J) <= p.B, name="capital_budget")     # (9)
    for j in J:
        if ins.elig[j] == 0:
            x[j].UB = 0                                                                            # (10)
    for j in ins.N:
        x[int(j)].UB = 0                                                                           # (11) near incumbent
    for a, b in ins.E:
        m.addConstr(x[int(a)] + x[int(b)] <= 1, name=f"spacing[{a},{b}]")                         # (11) new-new
    coef_il = defaultdict(list)
    for i, l, V in zip(ins.lk_i, ins.lk_l, ins.lk_V):
        coef_il[int(l)].append((int(i), float(ins.w[i] * V / ins.U[i])))
    for l, terms in coef_il.items():
        m.addConstr(gp.quicksum(a * P0[i] for i, a in terms) >= (1 - p.eps) * sum(a for _, a in terms),
                    name=f"retention[{l}]")                                                        # (12)

    # ---------- objective (7) ----------
    coef = ins.obj_coef                                   # 365 * m * a_T   ($ per daily order)
    revenue = gp.quicksum(float(ins.w[i]) * (gp.quicksum(P[k] for k in pairs_of_node[i])
                                             + float(ins.gamma[i]) * P0[i] - float(ins.gamma[i])) for i in I)
    cost = gp.quicksum(float(ins.Phi[j]) * x[j] for j in J)
    m.setObjective(coef * revenue - cost, GRB.MAXIMIZE)

    if start is not None:                                 # e.g. greedy plan as MIP start
        for j in J:
            x[j].Start = float(start[j])
    m.update()
    return m, x, P0, P

def solve_gurobi(ins: Instance, start=None, verbose=True):
    m, x, P0, P = build_model(ins, start, verbose)
    m.optimize()
    if m.SolCount == 0:
        raise RuntimeError(f"Gurobi found no solution (status {m.Status})")
    return {"x": np.array([round(x[j].X) for j in range(ins.nJ)]), "obj": m.ObjVal, "bound": m.ObjBound,
            "gap": m.MIPGap, "status": {GRB.OPTIMAL: "OPTIMAL", GRB.TIME_LIMIT: "TIME_LIMIT"}.get(m.Status, m.Status),
            "runtime": m.Runtime, "n_vars": m.NumVars, "n_cons": m.NumConstrs,
            "P0": np.array([P0[i].X for i in range(ins.nI)])}
