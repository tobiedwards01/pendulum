"""Autopilot for the pendulum on a trolley.

Everything here is normal physics and control theory (no machine learning):

1. MEASURE  - wiggle the trolley for a few seconds and record the swing.
              The model (with the known lengths and gravity) predicts each arm's angular
              acceleration *without* damping. Whatever is left over must be the damping,
              which is  -b * omega,  so a least-squares fit gives b for each arm.

2. SWING UP
   single:  pump energy in by moving the trolley in time with the swing, until the pendulum
            has just enough energy to reach the top.
   double:  a) SETTLE: take energy out until it hangs still (so it works from anywhere).
            b) PLAN: planner.py works out a whole swing-up path with trajectory optimisation.
               It runs in the background and is remembered, so it only happens once per setup.
            c) FOLLOW the plan, using time-varying LQR feedback to correct small errors.

3. BALANCE  - LQR (linear quadratic regulator). Linearise the model around upright using
              the measured damping, then let scipy find the best feedback gains K.
              The trolley acceleration is then simply  u = -K . (state error).
"""
import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np
from scipy.linalg import expm, solve_continuous_are

MAX_A = 25.0                   # autopilot trolley acceleration limit (m/s^2)
MEASURE_TIME = 3.0             # seconds spent measuring the damping
WIGGLE_A = 3.0                 # trolley wiggle while measuring (m/s^2)
WIGGLE_W = 2 * math.pi / 1.5   # wiggle frequency (rad/s)
SWING_A = 20.0                 # acceleration limit while swinging up (m/s^2)
SWING_K = 20.0                 # how hard to pump energy in
ENERGY_MARGIN = 1.05           # aim for slightly more energy than needed to reach the top
CATCH_ANGLE = 0.5              # switch to balancing within this many radians of upright...
CATCH_W = 6.0                  # ...and when spinning slower than this (rad/s)
SETTLE_K = 1.5                 # how hard to take energy out when settling the double pendulum
SETTLE_SWITCH = 0.2            # below this much swing energy, hand over to the hanging-down LQR
POS_WEIGHT = 1.0               # how much the balancer cares about trolley position/speed
PREVIEW_DT = 0.01              # time step for planning a balanced move to a target (s)
PREVIEW_POS = 50.0             # how closely a balanced move should follow the smooth path
TARGET_V = 1.2                 # top speed when moving to a new target position (m/s)
TARGET_A = 1.0                 # how quickly it speeds up / slows down for a new target (m/s^2)

PLANNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "planner.py")


def wrap(a):
    """Wrap an angle to the range -pi..pi."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def clip(u, limit):
    return max(-limit, min(limit, u))


class Autopilot:
    def __init__(self):
        self.active = False
        self.mode = "off"
        self.b_est = None
        self.plans = {}          # finished swing-up plans, remembered for each setup
        self.failed = set()      # setups the planner couldn't solve
        self.proc = None         # background planner process
        self.proc_key = None
        self.proc_out = None
        self.t = 0.0

    def start(self, sim):
        self.active = True
        self.mode = "measure"
        self.t = 0.0
        self.samples = []      # (time, theta1, omega1, theta2, omega2, u)
        self.x0 = sim.x        # where the trolley should be right now (glides towards the target)
        self.v_ref = 0.0       # how fast that reference point is moving
        self.a_ref = 0.0       # ...and how fast it is speeding up
        self.target = sim.x    # where the user wants the trolley to end up
        self.move = None       # smooth path from the old position to the target
        self.preview = None    # planned balanced move (states and accelerations over time)
        self.b_est = None

    def set_target(self, x, sim=None):
        """Plan a smooth minimum-jerk path (a 5th-order polynomial) from where the reference is
        now, including its current speed and acceleration, to the target, ending still."""
        self.target = x
        x0, v0, a0 = self.x0, self.v_ref, self.a_ref
        D = abs(x - x0)
        # long enough that the peak speed and acceleration stay within the limits
        T = max(1.875 * D / TARGET_V, math.sqrt(5.77 * D / TARGET_A), 1.0)
        M = np.array([[T ** 3, T ** 4, T ** 5],
                      [3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
                      [6 * T, 12 * T ** 2, 20 * T ** 3]])
        rhs = np.array([x - x0 - v0 * T - a0 / 2 * T ** 2, -v0 - a0 * T, -a0])
        c3, c4, c5 = np.linalg.solve(M, rhs)
        self.move = {"c": (x0, v0, a0 / 2, c3, c4, c5), "T": T, "tau": 0.0}
        self.preview = self.plan_preview(sim) if sim is not None and self.mode == "balance" else None

    def plan_preview(self, sim):
        """Preview tracking for a move while balancing.

        A balanced pendulum can't just follow the trolley: to speed up, the trolley first has to
        duck backwards so the pendulum leans forwards, and the reverse to stop. So we look ahead
        along the whole smooth path, using the linear model of the upright pendulum, and solve
        a finite-horizon LQR tracking problem. That gives the best trolley moves *and* the lean
        to expect at every moment, which the balancer then follows.
        """
        A, B = self.A_up, self.B_up
        n = A.shape[0]
        dt = PREVIEW_DT
        # exact discrete-time version of the linear model (acceleration held for each step)
        M = expm(np.block([[A, B[:, None]], [np.zeros((1, n + 1))]]) * dt)
        Ad, Bd = M[:n, :n], M[:n, n]

        m = self.move
        N = int((m["T"] + 3.0) / dt)          # carry on 3 s past the end so it settles
        tau = np.minimum(np.arange(N + 1) * dt, m["T"])
        c0, c1, c2, c3, c4, c5 = m["c"]
        r = np.zeros((N + 1, n))
        r[:, 0] = c0 + tau * (c1 + tau * (c2 + tau * (c3 + tau * (c4 + tau * c5)))) - sim.x
        r[:, 1] = c1 + tau * (2 * c2 + tau * (3 * c3 + tau * (4 * c4 + tau * 5 * c5)))
        r[tau >= m["T"], 1] = 0.0
        Q = np.diag([PREVIEW_POS, 1, 200, 10, 200, 10] if n == 6 else [PREVIEW_POS, 1, 100, 10])
        R = 0.05

        # backward pass: value function x'Px + 2s'x, gains u = -K x + kff
        P = Q.copy()
        svec = -Q @ r[N]
        Ks = np.zeros((N, n))
        kff = np.zeros(N)
        for k in range(N - 1, -1, -1):
            PB = P @ Bd
            den = R + Bd @ PB
            K = (PB @ Ad) / den
            kf = -(Bd @ svec) / den
            Acl = Ad - np.outer(Bd, K)
            svec = Acl.T @ (PB * kf + svec) - R * kf * K - Q @ r[k]
            P = Q + Ad.T @ P @ Acl
            P = 0.5 * (P + P.T)
            Ks[k], kff[k] = K, kf

        # forward pass: the planned states and accelerations, starting from how it is now
        xs = np.zeros((N + 1, n))
        us = np.zeros(N)
        p1, p2 = wrap(sim.theta - math.pi), wrap(sim.theta2 - math.pi)
        xs[0] = [0.0, sim.v, p1, sim.omega] + ([p2, sim.omega2] if n == 6 else [])
        for k in range(N):
            us[k] = -Ks[k] @ xs[k] + kff[k]
            xs[k + 1] = Ad @ xs[k] + Bd * us[k]
        return {"xs": xs, "us": us, "dt": dt, "x_start": sim.x, "tau": 0.0}

    def move_reference(self, dt):
        """Step the reference point along the planned path towards the target."""
        if self.preview is not None:
            self.preview["tau"] += dt
            if self.preview["tau"] >= len(self.preview["us"]) * self.preview["dt"]:
                self.preview = None
        if self.move is None:
            return
        m = self.move
        m["tau"] = tau = min(m["tau"] + dt, m["T"])
        c0, c1, c2, c3, c4, c5 = m["c"]
        self.x0 = c0 + tau * (c1 + tau * (c2 + tau * (c3 + tau * (c4 + tau * c5))))
        self.v_ref = c1 + tau * (2 * c2 + tau * (3 * c3 + tau * (4 * c4 + tau * 5 * c5)))
        self.a_ref = 2 * c2 + tau * (6 * c3 + tau * (12 * c4 + tau * 20 * c5))
        if tau >= m["T"]:
            self.x0, self.v_ref, self.a_ref = self.target, 0.0, 0.0
            self.move = None

    def at_target(self):
        return self.move is None

    def stop(self):
        self.active = False
        self.mode = "off"
        self.kill_planner()

    def override(self, sim):
        """The user is pushing the trolley by hand."""
        if self.mode == "measure":
            self.start(sim)          # the push spoils the measurement, so start again
        elif self.mode == "follow":
            self.mode = "settle"     # knocked off the plan: settle down and try again

    # ---------- main entry point: called every physics step ----------
    def control(self, sim, dt, g, L1, L2):
        """Return the trolley acceleration to use for this physics step."""
        self.t += dt
        if self.mode == "measure":
            return self.measure(sim, dt, g, L1, L2)
        if self.mode != "balance":
            self.preview = None      # the planned balanced move only makes sense while balancing
        if self.mode != "follow":    # the swing-up plan needs a fixed home, so don't move it then
            self.move_reference(dt)

        p1, p2 = wrap(sim.theta - math.pi), wrap(sim.theta2 - math.pi)   # angles from upright
        near_top = abs(p1) < 0.3 and abs(sim.omega) < 2.0
        if sim.double:
            near_top = near_top and abs(p2) < 0.3 and abs(sim.omega2) < 2.0

        if self.mode == "balance":
            fell = abs(p1) > 0.7 or (sim.double and abs(p2) > 0.7)
            if not fell:
                return self.lqr(sim, self.K_up, p1, p2)
            self.mode = "settle" if sim.double else "swing"

        if not sim.double:   # single pendulum: energy pumping, then catch it
            if abs(p1) < CATCH_ANGLE and abs(sim.omega) < CATCH_W:
                self.mode = "balance"
                return self.lqr(sim, self.K_up, p1, p2)
            return self.swing_up(sim, g, L1)

        # ----- double pendulum -----
        key = (round(g, 3), round(L1, 3), round(L2, 3), round(self.b_est[0], 3), round(self.b_est[1], 3))
        self.check_planner(key)

        if self.mode == "follow":
            if self.follow_t >= self.plan["T"]:
                self.mode = "balance" if near_top else "settle"
            else:
                u = self.follow(sim, dt)
                if u is not None:
                    return u
                self.mode = "settle"     # drifted too far from the plan

        # settle
        if near_top:
            self.mode = "balance"
            return self.lqr(sim, self.K_up, p1, p2)
        if key in self.plans and self.at_target() and self.is_settled(sim, g, L1, L2):
            self.begin_follow(sim, self.plans[key])
            return self.follow(sim, dt) or 0.0
        return self.settle(sim, g, L1, L2)

    # ---------- stage 1: measure damping ----------
    def measure(self, sim, dt, g, L1, L2):
        u = clip(WIGGLE_A * math.cos(WIGGLE_W * self.t) - 2.0 * (sim.x - self.x0), MAX_A)
        self.samples.append((self.t, sim.theta, sim.omega, sim.theta2, sim.omega2, u))
        if self.t >= MEASURE_TIME:
            self.b_est = self.fit_damping(sim, g, L1, L2)
            self.K_up = self.design_lqr(sim, g, L1, L2, upright=True)
            if sim.double:
                self.K_down = self.design_lqr(sim, g, L1, L2, upright=False)
            self.mode = "settle" if sim.double else "swing"
        return u

    def fit_damping(self, sim, g, L1, L2):
        S = self.samples
        num, den = [0.0, 0.0], [0.0, 0.0]
        for k in range(1, len(S) - 1):
            t_a, _, w1_a, _, w2_a, u_a = S[k - 1]
            _, th1, w1, th2, w2, u = S[k]
            t_b, _, w1_b, _, w2_b, _ = S[k + 1]
            h = t_b - t_a
            measured = ((w1_b - w1_a) / h, (w2_b - w2_a) / h)
            predicted = sim.accels(th1, w1, th2, w2, (u_a + u) / 2, g, L1, L2, 0.0, 0.0)
            for i, w in enumerate((w1, w2)):
                num[i] += (measured[i] - predicted[i]) * w   # leftover = -b * omega
                den[i] += w * w
        return tuple(max(0.0, -num[i] / den[i]) if den[i] > 1e-6 else 0.0 for i in range(2))

    # ---------- single pendulum swing-up ----------
    def swing_up(self, sim, g, L):
        th, w = sim.theta, sim.omega
        energy = 0.5 * L * L * w * w - g * L * math.cos(th)   # per unit mass
        target = g * L * ENERGY_MARGIN                          # just over "standing still at the top"
        u = SWING_K * (energy - target) * w * math.cos(th)
        if abs(w) < 0.05 and abs(wrap(th)) < 0.05:
            u = SWING_A                                         # hanging still: give it a kick
        u += -1.0 * (sim.x - self.x0) - 1.0 * (sim.v - self.v_ref)
        return clip(u, SWING_A)

    # ---------- double pendulum: settle, plan, follow ----------
    @staticmethod
    def energy(sim, g, L1, L2):
        """Swing energy (per unit mass) above hanging still, and horizontal momentum of the bobs
        relative to the trolley. Accelerating the trolley at u changes the energy at -u * momentum."""
        v1x, v1y = L1 * math.cos(sim.theta) * sim.omega, L1 * math.sin(sim.theta) * sim.omega
        v2x = v1x + L2 * math.cos(sim.theta2) * sim.omega2
        v2y = v1y + L2 * math.sin(sim.theta2) * sim.omega2
        kinetic = 0.5 * (v1x * v1x + v1y * v1y + v2x * v2x + v2y * v2y)
        potential = g * (2 * L1 * (1 - math.cos(sim.theta)) + L2 * (1 - math.cos(sim.theta2)))
        return kinetic + potential, v1x + v2x

    def is_settled(self, sim, g, L1, L2):
        e, _ = self.energy(sim, g, L1, L2)
        return e < 0.002 * g * (2 * L1 + L2) and abs(sim.x - self.x0) < 0.1 and abs(sim.v) < 0.1

    def settle(self, sim, g, L1, L2):
        e, momentum = self.energy(sim, g, L1, L2)
        if e > SETTLE_SWITCH * g * (2 * L1 + L2):
            # lots of swing: move the trolley *with* the bobs to soak up energy
            u = SETTLE_K * momentum - 1.0 * (sim.x - self.x0) - 1.5 * (sim.v - self.v_ref)
            return clip(u, SWING_A)
        # nearly still: LQR around hanging down finishes the job and parks the trolley
        return self.lqr(sim, self.K_down, wrap(sim.theta), wrap(sim.theta2), upright=False)

    def check_planner(self, key):
        """Start the background planner for this setup if needed, and collect its result."""
        if key in self.plans or key in self.failed:
            return
        if self.proc is not None and self.proc_key != key:
            self.kill_planner()                  # settings changed: plan for the new ones
        if self.proc is None:
            fd, self.proc_out = tempfile.mkstemp(suffix=".npz", prefix="pendulum_plan_")
            os.close(fd)
            self.proc = subprocess.Popen([sys.executable, PLANNER, json.dumps(list(key)), self.proc_out],
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.proc_key = key
            self.proc_started = self.t
            self.next_poll = self.t
            return
        if self.t < self.next_poll:
            return
        self.next_poll = self.t + 0.25
        if self.proc.poll() is None:
            return                               # still working
        try:
            with np.load(self.proc_out) as data:
                if bool(data["ok"]):
                    us = data["us"]
                    self.plans[key] = {"xs": data["xs"], "us": us, "K": data["K"],
                                       "dt": float(data["dt"]), "T": len(us) * float(data["dt"])}
                else:
                    self.failed.add(key)
        except (OSError, KeyError, ValueError):
            self.failed.add(key)
        self.kill_planner()

    def kill_planner(self):
        if self.proc is not None:
            if self.proc.poll() is None:
                self.proc.kill()
                self.proc.wait()
            try:
                os.remove(self.proc_out)
            except OSError:
                pass
        self.proc = None
        self.proc_key = None

    def begin_follow(self, sim, plan):
        self.mode = "follow"
        self.plan = plan
        self.follow_t = 0.0
        # the plan starts at angle 0; the real angles may be whole turns away from that
        self.turns = (round((sim.theta - wrap(sim.theta)) / (2 * math.pi)),
                      round((sim.theta2 - wrap(sim.theta2)) / (2 * math.pi)))

    def follow(self, sim, dt):
        """Follow the planned path. Returns None if it has drifted too far to recover."""
        plan = self.plan
        xs, us, K, pdt = plan["xs"], plan["us"], plan["K"], plan["dt"]
        k = min(int(self.follow_t / pdt), len(us) - 1)
        f = min(1.0, self.follow_t / pdt - k)
        ref = xs[k] * (1 - f) + xs[k + 1] * f
        err = np.array([sim.x - self.x0 - ref[0], sim.v - ref[1],
                        sim.theta - 2 * math.pi * self.turns[0] - ref[2], sim.omega - ref[3],
                        sim.theta2 - 2 * math.pi * self.turns[1] - ref[4], sim.omega2 - ref[5]])
        if abs(err[2]) > 0.6 or abs(err[4]) > 0.6:
            return None
        self.follow_t += dt
        return clip(us[k] - float(K[k] @ err), MAX_A)

    # ---------- LQR (balancing upright, or settling hanging down) ----------
    def design_lqr(self, sim, g, L1, L2, upright):
        b1, b2 = self.b_est
        double = sim.double

        def f(s, u):
            """Rate of change of the state, using the *measured* damping."""
            if double:
                _, v, t1, w1, t2, w2 = s
                a1, a2 = sim.accels(t1, w1, t2, w2, u, g, L1, L2, b1, b2)
                return np.array([v, u, w1, a1, w2, a2])
            _, v, t1, w1 = s
            a1, _ = sim.accels(t1, w1, 0.0, 0.0, u, g, L1, L2, b1, b2)
            return np.array([v, u, w1, a1])

        n = 6 if double else 4
        s0 = np.zeros(n)
        if upright:
            s0[2] = math.pi
            if double:
                s0[4] = math.pi
        eps = 1e-6
        A = np.column_stack([(f(s0 + eps * e, 0.0) - f(s0 - eps * e, 0.0)) / (2 * eps) for e in np.eye(n)])
        B = ((f(s0, eps) - f(s0, -eps)) / (2 * eps)).reshape(n, 1)
        if upright:
            Q = np.diag([POS_WEIGHT, POS_WEIGHT, 200, 10, 200, 10] if double else [POS_WEIGHT, POS_WEIGHT, 100, 10])
            R = np.array([[0.05]])
        else:
            Q = np.diag([4, 4, 20, 2, 20, 2])
            R = np.array([[0.2]])
        if upright:
            self.A_up, self.B_up = A, B.ravel()   # kept for planning balanced moves
        P = solve_continuous_are(A, B, Q, R)
        return np.linalg.solve(R, B.T @ P).ravel()

    def lqr(self, sim, K, p1, p2, upright=True):
        state = [sim.x, sim.v, p1, sim.omega] + ([p2, sim.omega2] if sim.double else [])
        pv = self.preview if upright else None
        if pv is not None:
            # follow the planned balanced move: planned state (including lean) + planned acceleration
            k = min(int(pv["tau"] / pv["dt"]), len(pv["us"]) - 1)
            ref = pv["xs"][k].copy()
            ref[0] += pv["x_start"]
            u_ff = pv["us"][k]
        else:
            ref = [self.x0, self.v_ref, 0.0, 0.0] + ([0.0, 0.0] if sim.double else [])
            u_ff = 0.0
        err = np.subtract(state, ref)
        return clip(u_ff - float(np.dot(K, err)), MAX_A)

    # ---------- text for the on-screen status card ----------
    def status(self, sim, b1, b2):
        names = {
            "measure": f"Measuring damping... {max(0.0, MEASURE_TIME - self.t):.1f} s",
            "swing": "Swinging up",
            "balance": "Balancing",
            "settle": "Settling (letting it hang still)",
            "follow": "Swinging up (following plan)",
        }
        lines = [("Autopilot", names.get(self.mode, "Off"))]
        if self.mode != "measure":
            moving = "" if self.at_target() else "  (moving)"
            lines.append(("Target position", f"{self.target:+.1f} m{moving}"))
        if self.mode == "settle" and self.proc is not None:
            lines.append(("Planning swing-up", f"{self.t - self.proc_started:.0f} s"))
        elif self.mode == "settle" and self.failed and self.proc is None and sim.double:
            lines.append(("Planner", "no swing-up found for this setup"))
        if self.mode == "follow":
            lines.append(("Plan progress", f"{self.follow_t:.1f} / {self.plan['T']:.1f} s"))
        if self.b_est:
            lines.append(("Measured damping 1", f"{self.b_est[0]:.3f} /s  (true {b1:.3f})"))
            if sim.double:
                lines.append(("Measured damping 2", f"{self.b_est[1]:.3f} /s  (true {b2:.3f})"))
        return lines
