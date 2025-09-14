import math
import pygame
from dataclasses import dataclass
from collections import deque

# =========================
# Configurazione generale
# =========================
WIDTH, HEIGHT = 1200, 800
FPS = 120
DT = 1.0 / FPS
PIXELS_PER_METER = 15.0
DRAW_SCALE_VECTORS = False
SHOW_HUD = True

# Scie ruote (anteriore rossa, posteriore blu)
TRAIL_DURATION = 6.0
TRAIL_MIN_SPACING_M = 0.06
COLORS = {
    "front_left":  (220, 60, 40),
    "front_right": (220, 60, 40),
    "rear_left":   (20, 80, 220),
    "rear_right":  (20, 80, 220)
}

# =========================
# Parametri veicolo
# =========================
@dataclass
@dataclass
class VehicleParams:
    # massa & inerzia
    m: float = 1500.0
    Iz: float = 1600.0          # kg m^2 (berlina compatta)

    # geometria
    a: float = 1.30             # CG -> asse ant. (m)
    b: float = 1.40             # CG -> asse post. (m)  => L = 2.70 m
    track: float = 1.55         # carreggiata (m)
    h: float = 0.50             # altezza CG (m)

    # pneumatici (rigidezze per ASSE)
    Cf: float = 85000.0         # N/rad (front)
    Cr: float = 95000.0         # N/rad (rear)

    # aderenza & resistenze
    mu: float = 1.15            # asfalto asciutto buono (stradale sportivo)
    c_rr: float = 0.012         # resistenza al rotolamento
    CdA: float = 0.67           # coeff. aero*area (m^2)
    rho_air: float = 1.225
    g: float = 9.81

    # sterzo (angolo ruota “bicycle”)
    delta_max: float = math.radians(38.0)   # ~ realistico; niente 55° da kart ;)

    # “motore/freni” semplificati (limite di a_x)
    ax_max: float = 4.5          # m/s^2  (0–100 ≈ 7.5 s con drag/roll)
    ax_min: float = -8.0         # m/s^2  (frenata decisa su stradale buono)


# =========================
# Stato e input
# =========================
@dataclass
class State:
    x: float = -30.0
    y: float = 0.0
    psi: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    r: float = 0.0
    delta: float = 0.0

@dataclass
class Inputs:
    ax_cmd: float = 0.0

# =========================
# Utility
# =========================
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def sign(x):
    return 1.0 if x >= 0 else -1.0

# =========================
# Modello dinamico
# =========================
class BicycleModel:
    def __init__(self, p: VehicleParams):
        self.p = p
        self.kappa_f = 0.2

    def tire_forces(self, vx, vy, r, delta, Fx_f, Fx_r, Fzf, Fzr):
        eps = 1e-3
        vx_eff = vx if abs(vx) > eps else (eps * sign(vx) if vx != 0 else eps)
        alpha_f = math.atan2(vy + self.p.a * r, vx_eff) - delta
        alpha_r = math.atan2(vy - self.p.b * r, vx_eff)
        Fy0_f = -self.p.Cf * alpha_f
        Fy0_r = -self.p.Cr * alpha_r
        Fy_f = self._friction_limit(Fy0_f, Fx_f, Fzf)
        Fy_r = self._friction_limit(Fy0_r, Fx_r, Fzr)
        return Fy_f, Fy_r, alpha_f, alpha_r

    def _friction_limit(self, Fy0, Fx, Fz):
        muFz = self.p.mu * max(Fz, 1e-6)
        scale = 1.0 / math.sqrt(1.0 + (Fx / muFz) ** 2)
        Fy = Fy0 * scale
        Fy = clamp(Fy, -muFz * math.sqrt(max(0.0, 1.0 - (Fx / muFz) ** 2)),
                        muFz * math.sqrt(max(0.0, 1.0 - (Fx / muFz) ** 2)))
        return Fy

    def normal_loads(self, ax, ay):
        L = self.p.a + self.p.b
        Fzf0 = self.p.m * self.p.g * (self.p.b / L)
        Fzr0 = self.p.m * self.p.g * (self.p.a / L)
        dFz = self.p.m * ax * self.p.h / L
        return Fzf0 - dFz, Fzr0 + dFz

    def aero_and_roll(self, vx, vy):
        v = math.hypot(vx, vy)
        Fdrag = 0.5 * self.p.rho_air * self.p.CdA * v * v * sign(vx)
        Froll = self.p.c_rr * self.p.m * self.p.g * sign(vx)
        return Fdrag + Froll

    def dynamics(self, s: State, u: Inputs):
        p = self.p
        delta = s.delta
        delta_dot = 0.0
        ax_cmd = clamp(u.ax_cmd, p.ax_min, p.ax_max)
        ay_proxy = s.r * s.vx
        Fzf, Fzr = self.normal_loads(ax_cmd, ay_proxy)
        Fx_total = p.m * ax_cmd
        Fx_f = self.kappa_f * Fx_total
        Fx_r = (1.0 - self.kappa_f) * Fx_total
        Fres_long = self.aero_and_roll(s.vx, s.vy)
        Fy_f, Fy_r, alpha_f, alpha_r = self.tire_forces(s.vx, s.vy, s.r, delta, Fx_f, Fx_r, Fzf, Fzr)
        vxdot = (Fx_f + Fx_r - Fres_long) / p.m + s.r * s.vy
        vydot = (Fy_f + Fy_r) / p.m - s.r * s.vx
        rdot = (p.a * Fy_f - p.b * Fy_r) / p.Iz
        xdot = s.vx * math.cos(s.psi) - s.vy * math.sin(s.psi)
        ydot = s.vx * math.sin(s.psi) + s.vy * math.cos(s.psi)
        psidot = s.r
        beta = math.atan2(s.vy, max(abs(s.vx), 1e-6)) * (1 if s.vx >= 0 else -1)
        deriv = (xdot, ydot, psidot, vxdot, vydot, rdot, delta_dot)
        extras = {
            "Fy_f": Fy_f, "Fy_r": Fy_r,
            "Fx_f": Fx_f, "Fx_r": Fx_r,
            "Fzf": Fzf, "Fzr": Fzr,
            "alpha_f": alpha_f, "alpha_r": alpha_r,
            "beta": beta, "Fres_long": Fres_long
        }
        return deriv, extras, s.delta

# =========================
# RK4
# =========================
def rk4_step(model, s, u, dt):
    s1 = State(**s.__dict__)
    (k1, _, _) = model.dynamics(s1, u)
    s2 = State(**s.__dict__); _apply_k(s2, k1, dt/2)
    (k2, _, _) = model.dynamics(s2, u)
    s3 = State(**s.__dict__); _apply_k(s3, k2, dt/2)
    (k3, _, _) = model.dynamics(s3, u)
    s4 = State(**s.__dict__); _apply_k(s4, k3, dt)
    (k4, extras, _) = model.dynamics(s4, u)
    for attr, k_vals in zip(s.__dict__.keys(), zip(*[k1, k2, k3, k4])):
        setattr(s, attr, getattr(s, attr) + dt/6.0 * (k_vals[0] + 2*k_vals[1] + 2*k_vals[2] + k_vals[3]))
    s.psi = (s.psi + math.pi) % (2*math.pi) - math.pi
    return s, extras

def _apply_k(s, k, scale):
    for attr, val in zip(s.__dict__.keys(), k):
        setattr(s, attr, getattr(s, attr) + scale * val)

# =========================
# Rendering
# =========================
def world_to_screen(x_m, y_m):
    return int(WIDTH/2 + x_m * PIXELS_PER_METER), int(HEIGHT/2 - y_m * PIXELS_PER_METER)

def rotate_image(image, angle_deg):
    rot = pygame.transform.rotate(image, angle_deg)
    return rot, rot.get_rect()

def scale_image_to_length(image, length_px):
    w, h = image.get_width(), image.get_height()
    scale = length_px / max(w, h)
    return pygame.transform.smoothscale(image, (int(w*scale), int(h*scale)))

def draw_vector(surface, origin_xy, vec_m, scale=1.0, width=3, color=(20,20,20)):
    ox, oy = origin_xy
    vx, vy = vec_m
    end = (ox + vx * PIXELS_PER_METER * scale, oy - vy * PIXELS_PER_METER * scale)
    pygame.draw.line(surface, color, (ox, oy), end, width)
    ang = math.atan2(-vy, vx)
    arrow_size = 6
    left = (end[0] - arrow_size * math.cos(ang - math.pi/6),
            end[1] + arrow_size * math.sin(ang - math.pi/6))
    right = (end[0] - arrow_size * math.cos(ang + math.pi/6),
             end[1] + arrow_size * math.sin(ang + math.pi/6))
    pygame.draw.polygon(surface, color, [end, left, right])

# =========================
# Main
# =========================
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    params = VehicleParams()
    try:
        raw_img = pygame.image.load("car.png").convert_alpha()
    except:
        raise SystemExit("car.png mancante")
    car_len_px = (params.a + params.b) * PIXELS_PER_METER
    scaled_img = scale_image_to_length(raw_img, car_len_px)
    base_img = pygame.transform.rotate(scaled_img, -90)
    model = BicycleModel(params)
    state = State()
    inputs = Inputs()
    sim_time = 0.0
    trails = {name: deque() for name in COLORS}
    last_pos = {name: None for name in COLORS}
    font = pygame.font.SysFont("consolas", 18)
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT: running = False
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE] or keys[pygame.K_q]: running = False
        if keys[pygame.K_r]:
            state = State(); inputs = Inputs(); sim_time = 0
            for t in trails.values(): t.clear()
            for k in last_pos: last_pos[k] = None
        # sterzo fisso
        if keys[pygame.K_LEFT]:  state.delta = params.delta_max
        elif keys[pygame.K_RIGHT]: state.delta = -params.delta_max
        else: state.delta = 0.0
        # accel/freno: pedale solo se premuto
        if keys[pygame.K_UP]:
            inputs.ax_cmd = params.ax_max
        elif keys[pygame.K_DOWN]:
            inputs.ax_cmd = params.ax_min * 0.6
        elif keys[pygame.K_SPACE]:
            inputs.ax_cmd = params.ax_min
        else:
            inputs.ax_cmd = 0.0
        # sim
        state, extras = rk4_step(model, state, inputs, DT)
        sim_time += DT
        # posizioni ruote
        cospsi, sinpsi = math.cos(state.psi), math.sin(state.psi)
        half_track = params.track / 2 * 0.6
        wheel_offsets = {
            "front_left":  ( params.a,  half_track),
            "front_right": ( params.a, -half_track),
            "rear_left":   (-params.b,  half_track),
            "rear_right":  (-params.b, -half_track)
        }
        for name, (dx, dy) in wheel_offsets.items():
            wx = state.x + dx*cospsi - dy*sinpsi
            wy = state.y + dx*sinpsi + dy*cospsi
            if last_pos[name] is None or math.hypot(wx - last_pos[name][0], wy - last_pos[name][1]) >= TRAIL_MIN_SPACING_M:
                trails[name].append((wx, wy, sim_time))
                last_pos[name] = (wx, wy)
            while trails[name] and sim_time - trails[name][0][2] > TRAIL_DURATION:
                trails[name].popleft()
        # draw
        screen.fill((240, 246, 250))
        trail_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        for name, pts in trails.items():
            if len(pts) < 2: continue
            col = COLORS[name]
            plist = list(pts)
            for i in range(1, len(plist)):
                age = sim_time - plist[i][2]
                alpha = int(255 * (1.0 - age / TRAIL_DURATION))
                if alpha <= 0: continue
                c = (*col, alpha)
                pygame.draw.line(trail_surf, c, world_to_screen(plist[i-1][0], plist[i-1][1]),
                                                world_to_screen(plist[i][0], plist[i][1]), 2)
        screen.blit(trail_surf, (0, 0))
        car_angle_deg = math.degrees(state.psi)
        rot, rect = rotate_image(base_img, car_angle_deg)
        rect.center = world_to_screen(state.x, state.y)
        screen.blit(rot, rect.topleft)
        # frecce velocità/accel
        if DRAW_SCALE_VECTORS:
            sx, sy = world_to_screen(state.x, state.y)
            vxw = state.vx * math.cos(state.psi) - state.vy * math.sin(state.psi)
            vyw = state.vx * math.sin(state.psi) + state.vy * math.cos(state.psi)
            draw_vector(screen, (sx, sy), (vxw, vyw), scale=0.5, width=3, color=(0,0,0))
            ax_body = (extras["Fx_f"] + extras["Fx_r"] - extras["Fres_long"]) / params.m + state.r * state.vy
            ay_body = (extras["Fy_f"] + extras["Fy_r"]) / params.m - state.r * state.vx
            axw = ax_body * math.cos(state.psi) - ay_body * math.sin(state.psi)
            ayw = ax_body * math.sin(state.psi) + ay_body * math.cos(state.psi)
            draw_vector(screen, (sx, sy), (axw, ayw), scale=0.2, width=3, color=(150,0,0))
        # HUD
        if SHOW_HUD:
            v_mag = math.hypot(state.vx, state.vy)
            v_kmh = v_mag * 3.6
            lines = [
                f"Velocità: {v_kmh:6.1f} km/h ({v_mag:5.2f} m/s)",
                f"Sterzo: {math.degrees(state.delta):5.1f}°",
                "↑ accel  ↓ frena  ← tutto SX  → tutto DX  SPACE frena forte  R reset"
            ]
            y = 10
            for t in lines:
                screen.blit(font.render(t, True, (10,10,10)), (10, y)); y += 20
        pygame.display.flip()
        clock.tick(FPS)
    pygame.quit()

if __name__ == "__main__":
    main()
