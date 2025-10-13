# envs/parking2d.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .generators import gen_situation, ParkingSituation
from .geometry import (
    collide_car, make_grid, raycast_grid, CAR_LEN, CAR_WID,
    rect_corners, rect_inside_bounds
)
from .replay import ReplayLogger


class Parking2DEnv(gym.Env):
    """
    Variante 'global-only' per l'osservazione:
      obs = [cos dth, sin dth, dx_n, dy_n, tanh(v), steer_norm, rays_norm...]

    Fix inclusi:
      - Successo NON può avvenire nello stesso step di collisione o OOB.
      - inside_slot 'tight': considera le dimensioni dell'auto rispetto a quelle dello slot.
      - Ricompensa terminale di successo applicata solo se success=True dopo le precedenze.
      - Guardrail su NaN nello stato (terminazione sicura).
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        layout: str | None = None,
        max_steps: int = 500,
        rays: int = 16,
        log_replays: bool = False,
        replays_dir: str = "replays",
        save_every: int = 50,
        frame_stride: int = 4,
        success_angle_deg: float = 10.0,
        success_speed_max: float = 0.3,
        stall_speed: float = 0.05,
        stall_time_s: float = 3.0,
        ray_max_range: float = 15.0,
        global_norm_scale: float = 20.0,  # scala per normalizzare dx, dy globali
    ):
        super().__init__()
        self.layout = layout
        self.max_steps = int(max_steps)
        self.rays = int(rays)
        self.dt = 0.1
        self.L = 2.64

        # limiti cinematica
        self.v_max_f, self.v_max_b = 3.0, -1.5
        self.steer_max, self.steer_rate = 0.6, 1.5
        self.drag = 0.35

        # sensori
        self.ray_max_range = float(ray_max_range)

        # normalizzazione globale dx, dy
        self.global_norm_scale = float(global_norm_scale)

        # successo / termination
        self.success_angle = float(np.deg2rad(success_angle_deg))
        self.success_speed_max = float(success_speed_max)
        self.stall_speed = float(stall_speed)
        self.stall_steps = int(max(1, stall_time_s / self.dt))

        # osservazione: 6 core + rays
        # [cos dth, sin dth, dx_n, dy_n, tanh(v), steer_norm]
        self.obs_core_size = 6
        self.obs_size = self.obs_core_size + self.rays

        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.obs_size,), dtype=np.float32)

        self.logger = ReplayLogger(replays_dir) if log_replays else None
        self.save_every = int(max(1, save_every))
        self.frame_stride = int(max(1, frame_stride))

        # stati interni
        self._steer = 0.0
        self._prev_steer = 0.0
        self._ep_idx = 0
        self._save_this_ep = False
        self._prev_phi = 0.0
        self.prev_d = 0.0
        self.dwell = 0
        self._stall_count = 0

    # --------- Gym API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sit: ParkingSituation = gen_situation(layout=self.layout, seed=seed)
        self.state = np.array([*self.sit.car_start_pose, 0.0], dtype=float)  # x,y,th,v
        self.t = 0
        self.dwell = 0
        self._stall_count = 0
        self._prev_steer = 0.0
        self._steer = 0.0
        self._build_grid()

        self.prev_d = np.linalg.norm(self.state[:2] - self.sit.spot_pose[:2])
        self._prev_phi = self._compute_potential()

        # decide se salvare questo episodio
        self._ep_idx += 1
        self._save_this_ep = (self.logger is not None) and (self._ep_idx % self.save_every == 0)
        if self._save_this_ep:
            self.logger.reset()
            self.logger.push(self.state[:3])

        return self._obs(), {}

    def step(self, action):
        # guardrail: evita NaN in input
        if not np.all(np.isfinite(action)):
            action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)

        steer_cmd = float(np.clip(action[0], -1, 1))
        acc_cmd   = float(np.clip(action[1], -1, 1))

        x, y, th, v = self.state
        x, y, th, v, self._steer = self._bicycle_step(x, y, th, v, self._steer, acc_cmd, steer_cmd)
        self.state = np.array([x, y, th, v], dtype=float)
        self.t += 1

        # guardrail: NaN nello stato → termina
        if not np.all(np.isfinite(self.state)):
            infos = {"success": False, "collided": False, "stalled": False, "nan_state": True}
            return self._obs(), -10.0, True, False, infos

        # dwell con cap
        if self._inside_slot_tight() and abs(self.state[3]) < 0.15:
            self.dwell = min(self.dwell + 1, int(3.0 / self.dt))  # max ~3s
        else:
            self.dwell = 0

        # stall detection
        if (abs(self.state[3]) < self.stall_speed) and (not self._inside_slot_tight()):
            self._stall_count += 1
        else:
            self._stall_count = 0

        # EVENTI
        collided = collide_car(x, y, th, self.sit.obstacles)
        oob      = self._outside_bounds()

        # successo "raw"
        success_raw = self._success_tight()

        # PRECEDENZA: collisione / OOB vincono SEMPRE sul successo
        success = bool(success_raw and (not collided) and (not oob))

        timeout  = (self.t >= self.max_steps)
        stalled  = (self._stall_count >= self.stall_steps)

        done = collided or success or timeout or oob or stalled
        rew  = self._reward(collided, success, stalled) + (-0.5 if oob else 0.0)

        if self._save_this_ep:
            if (self.t % self.frame_stride) == 0:
                self.logger.push(self.state[:3], rew) # NOTE: modificato per aggiungere rew
            if done:
                if (self.t % self.frame_stride) != 0:
                    self.logger.push(self.state[:3], rew) # NOTE: modificato per aggiungere rew
                fname = f"{self.sit.layout}_seed{self.sit.seed}_t{self.t}_succ{int(success)}.json"
                self.logger.dump(self.sit, fname)

        infos = {"success": success, "collided": bool(collided), "stalled": bool(stalled), "oob": bool(oob)}
        return self._obs(), float(rew), bool(done), False, infos

    # --------- Dynamics / Sensors ----------
    def _bicycle_step(self, x, y, th, v, steer, a_cmd, turn_cmd):
        dt = self.dt
        L  = self.L

        self._prev_steer = steer
        steer += np.clip(turn_cmd, -1, 1) * self.steer_rate * dt
        steer  = float(np.clip(steer, -self.steer_max, self.steer_max))

        a = float(a_cmd) * 1.2  # più mansueta per manovre
        v = v + (a - self.drag * v) * dt
        v = float(np.clip(v, self.v_max_b, self.v_max_f))

        th = (th + (v / L) * np.tan(steer) * dt + np.pi) % (2 * np.pi) - np.pi
        x  = x + v * np.cos(th) * dt
        y  = y + v * np.sin(th) * dt
        return x, y, th, v, steer

    def _build_grid(self):
        P = np.vstack([rect_corners(*r) for r in np.vstack([self.sit.bounds, self.sit.obstacles])])
        x0, x1 = P[:, 0].min(), P[:, 0].max()
        y0, y1 = P[:, 1].min(), P[:, 1].max()
        pad = 8.0
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        w, h = (x1 - x0) + 2 * pad, (y1 - y0) + 2 * pad
        self.world_rect_big = (cx, cy, w, h, 0.0)
        self.grid, self.world_meta = make_grid(self.world_rect_big, self.sit.obstacles)

    def _outside_bounds(self):
        car = (self.state[0], self.state[1], CAR_LEN, CAR_WID, self.state[2])
        return not rect_inside_bounds(car, self.world_rect_big)

    # --------- Geometria / Utility ----------
    def _relative_pose(self):
        """(lx, ly, d, dth) nel frame del posto (non esposte all'agente)."""
        x, y, th, _ = self.state
        sx, sy, sth = self.sit.spot_pose
        c, s = np.cos(-sth), np.sin(-sth)
        lx = (x - sx) * c - (y - sy) * s
        ly = (x - sx) * s + (y - sy) * c
        d = np.hypot(lx, ly)
        dth = (sth - th + np.pi) % (2 * np.pi) - np.pi
        return lx, ly, d, dth

    def _global_delta(self):
        """Delta GLOBALE verso il centro del posto (dx, dy) in metri."""
        gx, gy, _ = self.sit.spot_pose
        dx = gx - self.state[0]
        dy = gy - self.state[1]
        return dx, dy

    def _slot_dims(self):
        """
        Stima le dimensioni dello slot (L,W) dai veicoli adiacenti; fallback a (5.0, 2.4).
        """
        if self.sit.obstacles is not None and len(self.sit.obstacles) > 0:
            # ostacoli: [x, y, len, wid, theta]
            L = float(self.sit.obstacles[0][2])
            W = float(self.sit.obstacles[0][3])
            return L, W
        return 5.0, 2.4  # fallback

    # --------- Osservazioni ----------
    def _obs(self):
        # orientamento relativo
        _, _, _, dth = self._relative_pose()
        cos_dth, sin_dth = np.cos(dth), np.sin(dth)

        # delta GLOBALE (dx, dy) → normalizza in [-1,1]
        dx, dy = self._global_delta()
        s = self.global_norm_scale
        #dx_n = np.clip(dx / s, -1.0, 1.0)
        #dy_n = np.clip(dy / s, -1.0, 1.0)
        dx_n = dx / s
        dy_n = dy / s

        # raggi → normalizza a [-1,1]
        th = self.state[2]
        angles = th + np.linspace(-0.9 * np.pi, 0.9 * np.pi, self.rays)
        rays = raycast_grid(self.grid, self.state[:2], angles, self.world_meta, max_range=self.ray_max_range)
        rays = np.asarray(rays, dtype=np.float32)
        rays_n01 = np.clip(rays / self.ray_max_range, 0.0, 1.0)
        rays_nm1p1 = rays_n01 * 2.0 - 1.0

        # stato dinamico
        v_n = np.tanh(self.state[3])
        steer_norm = self._steer / self.steer_max

        core = np.array([cos_dth, sin_dth, dx_n, dy_n, v_n, steer_norm], dtype=np.float32)
        obs = np.concatenate([core, rays_nm1p1.astype(np.float32)], axis=0)
        return obs

    # --------- Slot / Success / Reward ----------
    def _inside_slot_tight(self, safety_tol=(0.05, 0.05)):
        """
        TRUE se l'INTERO rettangolo dell'auto può stare nello slot:
        |lx| <= (L_slot/2 - L_car/2 - tol_x) e |ly| <= (W_slot/2 - W_car/2 - tol_y)
        """
        lx, ly, _, _ = self._relative_pose()
        Ls, Ws = self._slot_dims()
        tolx, toly = safety_tol

        halfL_allow = max(0.0, (Ls / 2.0) - (CAR_LEN / 2.0) - tolx)
        halfW_allow = max(0.0, (Ws / 2.0) - (CAR_WID / 2.0) - toly)

        return (abs(lx) <= halfL_allow) and (abs(ly) <= halfW_allow)

    def _success_tight(self):
        """
        Successo se: dentro slot 'tight', allineato e praticamente fermo.
        """
        x, y, th, v = self.state
        sx, sy, sth = self.sit.spot_pose
        dth = (sth - th + np.pi) % (2 * np.pi) - np.pi
        return self._inside_slot_tight() and (abs(dth) < self.success_angle) and (abs(v) < self.success_speed_max)

    def _compute_potential(self):
        """Potenziale φ = distanza + yaw error + offset laterale (tutti normalizzati)."""
        lx, ly, d, dth = self._relative_pose()
        d_term = (d / 15.0)
        yaw_term = (abs(dth) / np.pi)
        ly_term = (abs(ly) / 3.0)
        return 1.0 * d_term + 0.5 * yaw_term + 0.7 * ly_term

    def _reward(self, collided, success, stalled):
        # progresso potential-based
        phi = self._compute_potential()
        dphi = self._prev_phi - phi
        self._prev_phi = phi

        r = 0.2 * dphi
        r += -0.002  # piccolo costo temporale

        # termini di controllo (interni)
        lx, ly, d, dth = self._relative_pose()
        th = self.state[2]
        sth = self.sit.spot_pose[2]
        v = self.state[3]

        v_long = v * np.cos(th - sth)
        v_ref = np.clip(0.5 * d, 0.0, 2.0) * np.sign(-lx)

        if not self._inside_slot_tight():
            r += 0.3 * (1 - np.clip(d / 15.0, 0.0, 1.0))
            r += -0.05 * abs(dth)
            r += -0.05 * abs(v_long - v_ref)
            # lieve termine globale: più vicino al posto è meglio anche da lontano
            dx, dy = self._global_delta()
            r += -0.001 * np.hypot(dx, dy)
        else:
            r += 0.6 * (1 - np.clip(d / 15.0, 0.0, 1.0))
            r += -0.3 * abs(dth)
            r += -0.4 * abs(v)
            r += 0.02 * self.dwell  # max ~0.6

        # smoothness sterzo
        r += -0.01 * abs(self._steer)
        r += -0.02 * abs(self._steer - self._prev_steer)

        # terminali (success già ha precedenza applicata)
        if collided:
            r -= 5.0
        if success:
            r += 20.0
        if stalled:
            r -= 1.0

        # compat
        sx, sy, _ = self.sit.spot_pose
        self.prev_d = np.linalg.norm([self.state[0] - sx, self.state[1] - sy])

        return float(r)
