import math
from collections import deque

import pygame
import pygame.gfxdraw

from controller import Autopilot

# ---------- window & layout ----------
W, H = 1100, 740
PANEL_H = 190
PANEL_Y = H - PANEL_H
SCALE = 150              # pixels per metre (zooms out for long pendulums)
MAX_SCALE = 150
REACH_PX = 220           # the whole pendulum fits in this many pixels above or below the pivot
RAIL_Y = 270             # screen y of the overhead rail (middle, so it fits upright too)
PIVOT_Y = RAIL_Y + 40    # where the pendulum hangs from

# ---------- colours ----------
BG_TOP = (26, 30, 44)
BG_BOTTOM = (12, 14, 20)
PANEL_BG = (30, 34, 46)
PANEL_EDGE = (52, 58, 78)
TEXT = (225, 230, 240)
MUTED = (140, 148, 168)
ACCENT = (90, 170, 255)
RAIL = (120, 128, 148)
TROLLEY = (70, 130, 220)
TROLLEY_DARK = (40, 75, 140)
WHEEL = (35, 38, 50)
ROD = (210, 215, 225)
BOB = (240, 90, 80)
BOB_LIGHT = (255, 160, 150)
BOB2 = (255, 185, 70)
BTN = (48, 54, 72)
BTN_HOVER = (62, 70, 94)
BTN_ACTIVE = (60, 110, 190)
TARGET = (110, 220, 140)

# ---------- trolley settings ----------
CART_ACCEL = 4.0      # m/s^2 while a move key/button is held
CART_FRICTION = 2.0   # m/s^2 slow-down when nothing is held
CART_MAX_V = 6.0      # m/s
AUTO_MAX_V = 10.0     # m/s speed limit when the autopilot is driving
SUBSTEPS = 10         # physics steps per frame (keeps it stable)
WHEEL_R = 10          # px

pygame.init()
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("Pendulum on a trolley")
clock = pygame.time.Clock()
font = pygame.font.SysFont("segoeui", 16)
font_small = pygame.font.SysFont("segoeui", 14)
font_big = pygame.font.SysFont("segoeui", 22, bold=True)

# pre-drawn background gradient
background = pygame.Surface((W, PANEL_Y))
for y in range(PANEL_Y):
    t = y / PANEL_Y
    c = [int(BG_TOP[i] + (BG_BOTTOM[i] - BG_TOP[i]) * t) for i in range(3)]
    pygame.draw.line(background, c, (0, y), (W, y))


# ---------- drawing helpers ----------
def lerp_colour(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def aa_circle(surf, colour, centre, r):
    x, y = int(centre[0]), int(centre[1])
    pygame.gfxdraw.filled_circle(surf, x, y, r, colour)
    pygame.gfxdraw.aacircle(surf, x, y, r, colour)


def thick_line(surf, colour, p1, p2, width):
    (x1, y1), (x2, y2) = p1, p2
    length = math.hypot(x2 - x1, y2 - y1) or 1
    nx = -(y2 - y1) / length * width / 2
    ny = (x2 - x1) / length * width / 2
    pts = [(round(x1 + nx), round(y1 + ny)), (round(x2 + nx), round(y2 + ny)),
           (round(x2 - nx), round(y2 - ny)), (round(x1 - nx), round(y1 - ny))]
    pygame.gfxdraw.filled_polygon(surf, pts, colour)
    pygame.gfxdraw.aapolygon(surf, pts, colour)


# ---------- UI widgets ----------
class Button:
    def __init__(self, rect, label, on_click=None, arrow=0):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.on_click = on_click   # None = a "hold" button
        self.arrow = arrow         # -1 left arrow, +1 right arrow, 0 none
        self.active = False

    def click(self, pos):
        if self.on_click and self.rect.collidepoint(pos):
            self.on_click()
            return True
        return False

    def draw(self, surf, mouse):
        hover = self.rect.collidepoint(mouse)
        colour = BTN_ACTIVE if self.active else (BTN_HOVER if hover else BTN)
        pygame.draw.rect(surf, colour, self.rect, border_radius=8)
        pygame.draw.rect(surf, PANEL_EDGE, self.rect, 1, border_radius=8)
        txt = font.render(self.label, True, TEXT)
        surf.blit(txt, txt.get_rect(center=self.rect.center))
        if self.arrow:
            cy = self.rect.centery
            ax = self.rect.left + 18 if self.arrow < 0 else self.rect.right - 18
            d = 6 * self.arrow
            pts = [(ax + d, cy), (ax - d, cy - 7), (ax - d, cy + 7)]
            pygame.gfxdraw.filled_polygon(surf, pts, TEXT)
            pygame.gfxdraw.aapolygon(surf, pts, TEXT)


class Slider:
    def __init__(self, x, y, w, label, vmin, vmax, value, fmt):
        self.x, self.y = x, y
        self.track = pygame.Rect(x + 95, y - 3, w, 6)
        self.label, self.vmin, self.vmax = label, vmin, vmax
        self.value, self.fmt = value, fmt
        self.dragging = False
        self.enabled = True

    def knob_x(self):
        frac = (self.value - self.vmin) / (self.vmax - self.vmin)
        return self.track.x + frac * self.track.w

    def set_from_mouse(self, mx):
        frac = min(1.0, max(0.0, (mx - self.track.x) / self.track.w))
        self.value = self.vmin + frac * (self.vmax - self.vmin)

    def handle(self, event):
        if not self.enabled:
            self.dragging = False
            return False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.track.inflate(20, 24).collidepoint(event.pos):
                self.dragging = True
                self.set_from_mouse(event.pos[0])
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self.set_from_mouse(event.pos[0])
        return False

    def draw(self, surf):
        accent = ACCENT if self.enabled else BTN_HOVER
        text = TEXT if self.enabled else MUTED
        lbl = font.render(self.label, True, MUTED)
        surf.blit(lbl, lbl.get_rect(midleft=(self.x, self.y)))
        pygame.draw.rect(surf, BTN, self.track, border_radius=3)
        kx = self.knob_x()
        filled = pygame.Rect(self.track.x, self.track.y, kx - self.track.x, self.track.h)
        pygame.draw.rect(surf, accent, filled, border_radius=3)
        aa_circle(surf, accent, (kx, self.y), 9)
        aa_circle(surf, text, (kx, self.y), 6)
        val = font.render(self.fmt.format(self.value), True, text)
        surf.blit(val, val.get_rect(midleft=(self.track.right + 16, self.y)))


# ---------- simulation ----------
class Sim:
    def __init__(self):
        self.double = False
        self.reset()

    def reset(self):
        self.x = 0.0                    # trolley position (m)
        self.v = 0.0                    # trolley velocity (m/s)
        self.reset_pendulum()

    def reset_pendulum(self):
        # angles are measured from straight down; omegas are angular velocities (rad/s)
        if self.double:
            self.theta, self.theta2 = math.radians(90), math.radians(120)
        else:
            self.theta, self.theta2 = math.radians(30), math.radians(30)
        self.omega = self.omega2 = 0.0
        self.trail1 = deque(maxlen=400)
        self.trail2 = deque(maxlen=400)

    def accels(self, t1, w1, t2, w2, a, g, L1, L2, b1, b2):
        """Angular accelerations of both arms, with the pivot accelerating at a.
        b1 and b2 are the damping on each arm."""
        if not self.double:
            return -(g * math.sin(t1) + a * math.cos(t1)) / L1 - b1 * w1, 0.0
        m1 = m2 = 1.0
        d = t1 - t2
        # solve the 2x2 system  [M11 M12; M12 M22] [alpha1; alpha2] = [F1; F2]
        M11 = (m1 + m2) * L1 * L1
        M12 = m2 * L1 * L2 * math.cos(d)
        M22 = m2 * L2 * L2
        F1 = -m2 * L1 * L2 * w2 * w2 * math.sin(d) - (m1 + m2) * L1 * (g * math.sin(t1) + a * math.cos(t1))
        F2 = m2 * L1 * L2 * w1 * w1 * math.sin(d) - m2 * L2 * (g * math.sin(t2) + a * math.cos(t2))
        det = M11 * M22 - M12 * M12
        alpha1 = (F1 * M22 - M12 * F2) / det
        alpha2 = (M11 * F2 - M12 * F1) / det
        return alpha1 - b1 * w1, alpha2 - b2 * w2

    def step(self, dt, push, g, L1, L2, b1, b2, holding_bob, auto_a=None):
        # trolley: the autopilot sets the acceleration directly; otherwise
        # accelerate while pushed, and let friction slow it down
        vmax = CART_MAX_V
        if auto_a is not None:
            a, vmax = auto_a, AUTO_MAX_V
        elif push:
            a = push * CART_ACCEL
        elif abs(self.v) <= CART_FRICTION * dt:
            a = -self.v / dt
        else:
            a = -math.copysign(CART_FRICTION, self.v)
        if abs(self.v) >= vmax and a * self.v > 0:
            a = 0.0
        self.v += a * dt
        self.x += self.v * dt

        if holding_bob:
            return

        # RK4 integration (accurate enough for the chaotic double pendulum)
        def f(s):
            a1, a2 = self.accels(s[0], s[1], s[2], s[3], a, g, L1, L2, b1, b2)
            return (s[1], a1, s[3], a2)

        def nudge(s, k, h):
            return tuple(s[i] + k[i] * h for i in range(4))

        s = (self.theta, self.omega, self.theta2, self.omega2)
        k1 = f(s)
        k2 = f(nudge(s, k1, dt / 2))
        k3 = f(nudge(s, k2, dt / 2))
        k4 = f(nudge(s, k3, dt))
        self.theta, self.omega, self.theta2, self.omega2 = (
            s[i] + dt / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(4))


sim = Sim()
autopilot = Autopilot()
paused = False
show_trail = True
cam_x = 0.0
dragging_bob = 0   # 0 = none, 1 = first bob, 2 = second bob


def toggle_pause():
    global paused
    paused = not paused
    pause_btn.label = "Play" if paused else "Pause"


def stop_autopilot():
    autopilot.stop()
    auto_btn.active = False


def reset():
    stop_autopilot()
    sim.reset()


def toggle_trail():
    global show_trail
    show_trail = not show_trail
    trail_btn.active = show_trail
    sim.trail1.clear()
    sim.trail2.clear()


def toggle_double():
    stop_autopilot()
    sim.double = not sim.double
    double_btn.active = sim.double
    length2_sl.enabled = sim.double
    damping2_sl.enabled = sim.double
    sim.reset_pendulum()


def toggle_autopilot():
    if autopilot.active:
        stop_autopilot()
    else:
        autopilot.start(sim)
        auto_btn.active = True


def set_upright():
    # stand it (almost) straight up, so you can watch the autopilot catch it
    sim.theta = sim.theta2 = math.pi + 0.05
    sim.omega = sim.omega2 = 0.0


pause_btn = Button((20, PANEL_Y + 20, 80, 42), "Pause", toggle_pause)
reset_btn = Button((105, PANEL_Y + 20, 80, 42), "Reset", reset)
trail_btn = Button((190, PANEL_Y + 20, 75, 42), "Trail", toggle_trail)
trail_btn.active = True
double_btn = Button((270, PANEL_Y + 20, 80, 42), "Double", toggle_double)
left_btn = Button((20, PANEL_Y + 72, 160, 42), "Move left", arrow=-1)
right_btn = Button((190, PANEL_Y + 72, 160, 42), "Move right", arrow=1)
auto_btn = Button((20, PANEL_Y + 124, 160, 42), "Autopilot", toggle_autopilot)
upright_btn = Button((190, PANEL_Y + 124, 160, 42), "Stand upright", set_upright)
buttons = [pause_btn, reset_btn, trail_btn, double_btn, left_btn, right_btn, auto_btn, upright_btn]

damping_sl = Slider(390, PANEL_Y + 28, 260, "Damping 1", 0.0, 1.5, 0.1, "{:.2f} /s")
damping2_sl = Slider(390, PANEL_Y + 61, 260, "Damping 2", 0.0, 1.5, 0.1, "{:.2f} /s")
length_sl = Slider(390, PANEL_Y + 94, 260, "Length 1", 0.3, 2.0, 1.0, "{:.2f} m")
length2_sl = Slider(390, PANEL_Y + 127, 260, "Length 2", 0.3, 2.0, 1.0, "{:.2f} m")
gravity_sl = Slider(390, PANEL_Y + 160, 260, "Gravity", 1.0, 25.0, 9.81, "{:.2f} m/s²")
damping2_sl.enabled = length2_sl.enabled = False
sliders = [damping_sl, damping2_sl, length_sl, length2_sl, gravity_sl]


def to_screen_x(xw):
    return W / 2 + (xw - cam_x) * SCALE


def bob_world(which):
    """World position (m) of bob 1 or 2: x along the rail, y downwards from the pivot."""
    x = sim.x + length_sl.value * math.sin(sim.theta)
    y = length_sl.value * math.cos(sim.theta)
    if which == 2:
        x += length2_sl.value * math.sin(sim.theta2)
        y += length2_sl.value * math.cos(sim.theta2)
    return x, y


def bob_screen(which=1):
    x, y = bob_world(which)
    return to_screen_x(x), PIVOT_Y + y * SCALE


def bob2_screen():
    return bob_screen(2)


def draw_rail():
    pygame.draw.rect(screen, RAIL, (0, RAIL_Y - 3, W, 6))
    # a tick every 0.5 m, labelled every 1 m
    left = cam_x - (W / 2) / SCALE
    right = cam_x + (W / 2) / SCALE
    for k in range(math.floor(left * 2) - 1, math.ceil(right * 2) + 2):
        xw = k / 2
        sx = to_screen_x(xw)
        major = k % 2 == 0
        colour = ACCENT if k == 0 else (TEXT if major else MUTED)
        h = 14 if major else 7
        pygame.draw.line(screen, colour, (sx, RAIL_Y - 3), (sx, RAIL_Y - 3 - h), 2 if major else 1)
        if major:
            lbl = font_small.render(f"{int(xw)} m", True, colour)
            screen.blit(lbl, lbl.get_rect(midbottom=(sx, RAIL_Y - 20)))


def draw_trolley():
    cx = to_screen_x(sim.x)
    spin = sim.x * SCALE / WHEEL_R   # wheels roll with the trolley
    for wx in (cx - 35, cx + 35):
        wy = RAIL_Y - 3 - WHEEL_R
        thick_line(screen, TROLLEY_DARK, (wx, wy), (wx, RAIL_Y + 12), 6)
        aa_circle(screen, WHEEL, (wx, wy), WHEEL_R)
        pygame.gfxdraw.aacircle(screen, int(wx), int(wy), WHEEL_R, RAIL)
        for i in range(3):
            ang = spin + i * 2 * math.pi / 3
            pygame.draw.aaline(screen, RAIL, (wx, wy),
                               (wx + math.cos(ang) * (WHEEL_R - 2), wy + math.sin(ang) * (WHEEL_R - 2)))
    body = pygame.Rect(0, 0, 110, 30)
    body.midtop = (cx, RAIL_Y + 8)
    pygame.draw.rect(screen, TROLLEY, body, border_radius=7)
    pygame.draw.rect(screen, TROLLEY_DARK, body, 2, border_radius=7)


def draw_bob(centre, colour, r):
    x, y = centre
    aa_circle(screen, lerp_colour(colour, (0, 0, 0), 0.35), (x, y), r + 2)
    aa_circle(screen, colour, (x, y), r)
    aa_circle(screen, lerp_colour(colour, (255, 255, 255), 0.45), (x - r * 0.3, y - r * 0.3), max(3, r // 3))


def draw_pendulum():
    px = to_screen_x(sim.x)
    b1 = bob_screen(1)
    thick_line(screen, ROD, (px, PIVOT_Y), b1, 3)
    if sim.double:
        b2 = bob_screen(2)
        thick_line(screen, ROD, b1, b2, 3)
    aa_circle(screen, TROLLEY_DARK, (px, PIVOT_Y), 6)
    aa_circle(screen, ROD, (px, PIVOT_Y), 3)
    if sim.double:
        draw_bob(b1, BOB, 14)
        draw_bob(b2, BOB2, 14)
    else:
        draw_bob(b1, BOB, 17)


def draw_trail(trail, colour):
    pts = [(to_screen_x(x), PIVOT_Y + y * SCALE) for x, y in trail]
    n = len(pts)
    for i in range(1, n):
        c = lerp_colour(BG_BOTTOM, colour, 0.8 * i / n)
        pygame.draw.aaline(screen, c, pts[i - 1], pts[i])


def draw_card(rows, pos, width, anchor="bottomleft"):
    card = pygame.Surface((width, 24 * len(rows) + 20), pygame.SRCALPHA)
    pygame.draw.rect(card, (255, 255, 255, 14), card.get_rect(), border_radius=10)
    for i, (name, val) in enumerate(rows):
        y = 12 + i * 24
        card.blit(font_small.render(name, True, MUTED), (14, y))
        v = font.render(val, True, TEXT)
        card.blit(v, v.get_rect(topright=(width - 14, y - 2)))
    screen.blit(card, card.get_rect(**{anchor: pos}))


def draw_readout():
    L, g = length_sl.value, gravity_sl.value

    def wrap(t):
        return (math.degrees(t) + 180) % 360 - 180

    if sim.double:
        rows = [
            ("Angle 1", f"{wrap(sim.theta):+.1f}°"),
            ("Angle 2", f"{wrap(sim.theta2):+.1f}°"),
            ("Angular speed 1", f"{sim.omega:+.2f} rad/s"),
            ("Angular speed 2", f"{sim.omega2:+.2f} rad/s"),
            ("Trolley speed", f"{sim.v:+.2f} m/s"),
            ("Trolley position", f"{sim.x:+.2f} m"),
        ]
    else:
        rows = [
            ("Angle", f"{wrap(sim.theta):+.1f}°"),
            ("Angular speed", f"{sim.omega:+.2f} rad/s"),
            ("Trolley speed", f"{sim.v:+.2f} m/s"),
            ("Trolley position", f"{sim.x:+.2f} m"),
            ("Period (small swings)", f"{2 * math.pi * math.sqrt(L / g):.2f} s"),
        ]
    draw_card(rows, (20, PANEL_Y - 16), 270)
    if autopilot.active:
        draw_card(autopilot.status(sim, damping_sl.value, damping2_sl.value),
                  (W - 20, PANEL_Y - 16), 380, anchor="bottomright")


def draw_panel(mouse):
    pygame.draw.rect(screen, PANEL_BG, (0, PANEL_Y, W, PANEL_H))
    pygame.draw.line(screen, PANEL_EDGE, (0, PANEL_Y), (W, PANEL_Y))
    for b in buttons:
        b.draw(screen, mouse)
    for s in sliders:
        s.draw(screen)
    help_lines = ["Space  pause / play", "R  reset", "← / →  or  A / D  move trolley",
                  "Drag a bob to set its angle", "Autopilot swings up and balances",
                  "Move keys override it (knock it!)", "Autopilot on: click rail = target"]
    for i, line in enumerate(help_lines):
        screen.blit(font_small.render(line, True, MUTED), (860, PANEL_Y + 12 + i * 24))


def draw_target():
    """A marker on the rail where the autopilot is taking the trolley."""
    if not autopilot.active or autopilot.mode == "measure":
        return
    tx = to_screen_x(autopilot.target)
    y = RAIL_Y - 3
    pts = [(tx, y - 2), (tx - 9, y - 18), (tx + 9, y - 18)]
    pygame.gfxdraw.filled_polygon(screen, pts, TARGET)
    pygame.gfxdraw.aapolygon(screen, pts, TARGET)
    pygame.draw.line(screen, TARGET, (tx, RAIL_Y + 3), (tx, RAIL_Y + 14), 2)
    lbl = font_small.render(f"target {autopilot.target:+.1f} m", True, TARGET)
    screen.blit(lbl, lbl.get_rect(midbottom=(tx, y - 38)))


# ---------- main loop ----------
running = True
while running:
    dt = min(clock.tick(60) / 1000.0, 1 / 30)
    mouse = pygame.mouse.get_pos()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                toggle_pause()
            elif event.key == pygame.K_r:
                reset()
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if any(b.click(event.pos) for b in buttons):
                continue
            if any(s.handle(event) for s in sliders):
                continue
            if sim.double and math.dist(event.pos, bob_screen(2)) < 24:
                dragging_bob = 2
            elif math.dist(event.pos, bob_screen(1)) < 24:
                dragging_bob = 1
            elif autopilot.active and abs(event.pos[1] - RAIL_Y) < 45:
                # click on the rail: send the trolley there (snapped to 0.1 m)
                xw = cam_x + (event.pos[0] - W / 2) / SCALE
                autopilot.set_target(round(xw, 1), sim)
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            dragging_bob = 0
            for s in sliders:
                s.handle(event)
        elif event.type == pygame.MOUSEMOTION:
            for s in sliders:
                s.handle(event)

    # dragging the bob sets the angle directly
    if dragging_bob == 1:
        sim.theta = math.atan2(mouse[0] - to_screen_x(sim.x), mouse[1] - PIVOT_Y)
    elif dragging_bob == 2:
        bx, by = bob_screen(1)
        sim.theta2 = math.atan2(mouse[0] - bx, mouse[1] - by)
    if dragging_bob:
        sim.omega = sim.omega2 = 0.0
        if autopilot.mode == "measure":
            autopilot.start(sim)   # a drag spoils the measurement, so start it again

    # trolley controls: keyboard or hold the on-screen buttons.
    # These override the autopilot while held, so you can knock the pendulum.
    keys = pygame.key.get_pressed()
    mouse_down = pygame.mouse.get_pressed()[0]
    left_btn.active = keys[pygame.K_LEFT] or keys[pygame.K_a] or (mouse_down and left_btn.rect.collidepoint(mouse))
    right_btn.active = keys[pygame.K_RIGHT] or keys[pygame.K_d] or (mouse_down and right_btn.rect.collidepoint(mouse))
    push = int(right_btn.active) - int(left_btn.active)
    if push and autopilot.active and not paused:
        autopilot.override(sim)

    if not paused:
        g, L1, L2 = gravity_sl.value, length_sl.value, length2_sl.value
        h = dt / SUBSTEPS
        for _ in range(SUBSTEPS):
            auto_a = None
            if autopilot.active and not dragging_bob and not push:
                auto_a = autopilot.control(sim, h, g, L1, L2)
            sim.step(h, push, g, L1, L2, damping_sl.value, damping2_sl.value, dragging_bob, auto_a)
        if show_trail:
            sim.trail1.append(bob_world(1))
            if sim.double:
                sim.trail2.append(bob_world(2))

    # zoom so the pendulum fits whether it hangs down or stands up
    reach = length_sl.value + (length2_sl.value if sim.double else 0.0)
    SCALE = min(MAX_SCALE, REACH_PX / reach)

    # camera smoothly follows the trolley, so the metre marks scroll past
    cam_x += (sim.x - cam_x) * min(1.0, 2.5 * dt)

    # draw
    screen.blit(background, (0, 0))
    screen.blit(font_big.render("Pendulum on a trolley", True, TEXT), (20, 16))
    draw_rail()
    draw_target()
    if show_trail:
        draw_trail(sim.trail1, BOB)
        if sim.double:
            draw_trail(sim.trail2, BOB2)
    draw_trolley()
    draw_pendulum()
    draw_readout()
    if paused:
        p = font_big.render("PAUSED", True, ACCENT)
        screen.blit(p, p.get_rect(midtop=(W / 2, 16)))
    draw_panel(mouse)
    pygame.display.flip()

autopilot.stop()   # also stops a background planner if one is running
pygame.quit()
