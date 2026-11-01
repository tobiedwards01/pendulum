"""Swing-up planner for the double pendulum on a trolley.

Uses direct collocation, a trajectory optimisation method:
  - Split the swing-up into 50 short time slices. The unknowns are the state (trolley position
    and speed, both angles and both angular speeds) and the trolley acceleration at every slice.
  - Ask scipy's optimiser to find values that start hanging still, end upright and still, obey the
    physics between every pair of slices, and use as little trolley acceleration as possible.
  - Solving the whole path at once like this is much less likely to get stuck than tweaking the
    trolley moves one at a time.

The autopilot then follows the plan with time-varying LQR feedback, which corrects small errors
along the way.

Planning takes 5-20 seconds, so the autopilot runs this file as a separate process:
    python planner.py "<json params>" <output.npz>
"""
import json
import math
import sys

import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.linalg import solve_discrete_are
from scipy.optimize import minimize

U_MAX = 20.0       # planned trolley acceleration limit (m/s^2), leaves headroom for feedback
X_MAX = 2.5        # how far the trolley may travel from home during the swing-up (m)
N_SLICES = 50
TRACK_DT = 0.01    # time step of the finished plan (s)
GOAL = np.array([0.0, 0.0, math.pi, 0.0, math.pi, 0.0])   # trolley home, both arms up, still


def derivs(S, U, p):
    """Vectorised rate of change of the state. S has shape (n, 6), U has shape (n,).
    State = [trolley x, trolley v, theta1, omega1, theta2, omega2]. Both masses are 1.
    This is the same physics as Sim.accels in main.py."""
    g, L1, L2, b1, b2 = p
    _, v, t1, w1, t2, w2 = S.T
    d = t1 - t2
    s, c = np.sin(d), np.cos(d)
    M11 = 2 * L1 * L1
    M12 = L1 * L2 * c
    M22 = L2 * L2
    F1 = -L1 * L2 * w2 * w2 * s - 2 * L1 * (g * np.sin(t1) + U * np.cos(t1))
    F2 = L1 * L2 * w1 * w1 * s - L2 * (g * np.sin(t2) + U * np.cos(t2))
    det = M11 * M22 - M12 * M12
    a1 = (F1 * M22 - M12 * F2) / det - b1 * w1
    a2 = (M11 * F2 - M12 * F1) / det - b2 * w2
    return np.stack([v, U, w1, a1, w2, a2], axis=1)


def rk4_step(S, U, dt, p):
    """One RK4 step for a batch of states, with the acceleration U held constant."""
    k1 = derivs(S, U, p)
    k2 = derivs(S + k1 * (dt / 2), U, p)
    k3 = derivs(S + k2 * (dt / 2), U, p)
    k4 = derivs(S + k3 * dt, U, p)
    return S + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def jacobians(fn, X, U, eps=1e-6):
    """Finite-difference derivatives of fn(X, U) with respect to the state and the control,
    for every row of X at once. Returns arrays of shape (n, 6, 6) and (n, 6)."""
    n = len(U)
    S = np.repeat(X[:, None, :], 14, axis=1)
    V = np.repeat(U[:, None], 14, axis=1)
    for i in range(6):
        S[:, 2 * i, i] += eps
        S[:, 2 * i + 1, i] -= eps
    V[:, 12] += eps
    V[:, 13] -= eps
    out = fn(S.reshape(-1, 6), V.reshape(-1)).reshape(n, 14, 6)
    A = np.stack([(out[:, 2 * i] - out[:, 2 * i + 1]) / (2 * eps) for i in range(6)], axis=2)
    B = (out[:, 12] - out[:, 13]) / (2 * eps)
    return A, B


def collocate(p, T, N=N_SLICES):
    """Trapezoidal direct collocation from hanging still to upright. Returns (X, U, success)."""
    h = T / N
    nx = (N + 1) * 6
    x0 = np.zeros(6)

    def split(w):
        return w[:nx].reshape(N + 1, 6), w[nx:]

    def cost(w):
        return h * np.sum(w[nx:] ** 2)

    def cost_grad(w):
        g = np.zeros_like(w)
        g[nx:] = 2 * h * w[nx:]
        return g

    def defects(w):
        """Zero when consecutive slices obey the physics (trapezoid rule)."""
        X, U = split(w)
        F = derivs(X, U, p)
        return (X[1:] - X[:-1] - h / 2 * (F[1:] + F[:-1])).ravel()

    def defects_jac(w):
        X, U = split(w)
        Fx, Fu = jacobians(lambda S, V: derivs(S, V, p), X, U)
        J = np.zeros((N * 6, len(w)))
        I = np.eye(6)
        for k in range(N):
            r = slice(6 * k, 6 * k + 6)
            J[r, 6 * k:6 * k + 6] = -I - h / 2 * Fx[k]
            J[r, 6 * (k + 1):6 * (k + 2)] = I - h / 2 * Fx[k + 1]
            J[r, nx + k] = -h / 2 * Fu[k]
            J[r, nx + k + 1] = -h / 2 * Fu[k + 1]
        return J

    # starting guess: both arms turning steadily from down to up, trolley still
    frac = np.linspace(0, 1, N + 1)[:, None]
    X_guess = x0 + (GOAL - x0) * frac
    X_guess[:, 3] = X_guess[:, 5] = math.pi / T
    w0 = np.concatenate([X_guess.ravel(), np.zeros(N + 1)])

    bounds = [(None, None)] * nx + [(-U_MAX, U_MAX)] * (N + 1)
    for k in range(N + 1):
        bounds[6 * k] = (-X_MAX, X_MAX)
    for i in range(6):
        bounds[i] = (x0[i], x0[i])
        bounds[6 * N + i] = (GOAL[i], GOAL[i])

    r = minimize(cost, w0, jac=cost_grad, method="SLSQP", bounds=bounds,
                 constraints=[{"type": "eq", "fun": defects, "jac": defects_jac}],
                 options={"maxiter": 300, "ftol": 1e-6})
    X, U = split(r.x)
    ok = r.success and np.max(np.abs(defects(r.x))) < 1e-3
    return X, U, ok


def make_plan(X, U, T, p):
    """Turn the 50-slice solution into a fine-grained path plus LQR gains for following it."""
    N = len(U) - 1
    t_nodes = np.linspace(0, T, N + 1)
    path = CubicHermiteSpline(t_nodes, X, derivs(X, U, p), axis=0)
    t = np.arange(0, T + 1e-9, TRACK_DT)
    xs = path(t)
    us = np.interp(t[:-1] + TRACK_DT / 2, t_nodes, U)

    step = lambda S, V: rk4_step(S, V, TRACK_DT, p)
    A, B = jacobians(step, xs[:-1], us)
    A_up, B_up = jacobians(step, GOAL[None], np.array([0.0]))
    Q = np.diag([10.0, 1.0, 200.0, 10.0, 200.0, 10.0])
    R = 0.02
    P = solve_discrete_are(A_up[0], B_up[0][:, None], Q, np.array([[R]]))
    K = np.zeros((len(us), 6))
    for k in range(len(us) - 1, -1, -1):   # time-varying LQR, worked backwards from upright
        PB = P @ B[k]
        denom = R + B[k] @ PB
        K[k] = (PB @ A[k]) / denom
        P = Q + A[k].T @ P @ A[k] - np.outer(A[k].T @ PB, PB @ A[k]) / denom
        P = 0.5 * (P + P.T)
    return {"xs": xs, "us": us, "K": K, "dt": TRACK_DT}


def plan_swing_up(p, durations=(3.5, 4.5, 2.5)):
    """Try a few swing-up lengths; return the first plan that works, or None."""
    for T in durations:
        X, U, ok = collocate(p, T)
        if ok:
            return make_plan(X, U, T, p)
    return None


if __name__ == "__main__":
    params = json.loads(sys.argv[1])
    plan = plan_swing_up(tuple(params))
    if plan is None:
        np.savez(sys.argv[2], ok=False)
    else:
        np.savez(sys.argv[2], ok=True, **plan)
