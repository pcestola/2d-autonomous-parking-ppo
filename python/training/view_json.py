import argparse, json, os, glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation
from lib.geometry import rect_corners, CAR_LEN, CAR_WID


# ---------- util ----------
def draw_rect(ax, cx, cy, L, W, th, edge=None, face=None, alpha=1.0, lw=1.5, z=0.0):
    pts = rect_corners(cx, cy, L, W, th)
    poly = Polygon(
        pts, closed=True,
        edgecolor=edge, facecolor=face, alpha=alpha,
        linewidth=lw, zorder=z
    )
    ax.add_patch(poly)
    return poly


def autolimits(ax, data, margin=2.0):
    xs, ys = [], []
    for r in data.get("bounds", []) + data.get("obstacles", []):
        pts = rect_corners(*r)
        xs.extend(pts[:, 0])
        ys.extend(pts[:, 1])
    for p in data.get("poses", []):
        xs.append(p[0])
        ys.append(p[1])
    if xs and ys:
        ax.set_aspect("equal", "box")
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(min(ys) - margin, max(ys) + margin)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def find_nth_latest(replays_dir="replays", index: int = 0):
    files = glob.glob(os.path.join(replays_dir, "*.json"))
    if not files:
        raise FileNotFoundError("Nessun JSON trovato in 'replays/'.")
    files.sort(key=os.path.getmtime, reverse=True)
    index = max(0, min(index, len(files) - 1))
    return files[index]


# ---------- main ----------
def main(path: str | None, interval_ms: int, tail_len: int, speed: float, index: int):
    if path is None:
        path = find_nth_latest(index=index)
        print(f"[i] Nessun file specificato: apro il file n°{index} (tra i più recenti):\n    {path}")
    data = load_json(path)

    poses = np.array(data.get("poses", []), dtype=float)
    if poses.size == 0:
        raise ValueError("Il JSON non contiene 'poses' da animare.")

    # --- rewards: carica, allinea, cumulativa ---
    rewards = np.array(data.get("rewards", []), dtype=float)
    if rewards.size != poses.shape[0]:
        print("[!] Lunghezza 'rewards' diversa da 'poses', verrà adattata (padding con 0).")
        if rewards.size > poses.shape[0]:
            rewards = rewards[:poses.shape[0]]
        else:
            rewards = np.pad(rewards, (0, poses.shape[0] - rewards.size), constant_values=0.0)
    cum_rewards = np.cumsum(rewards) if rewards.size else np.zeros(poses.shape[0], dtype=float)

    fig, ax = plt.subplots()
    plt.subplots_adjust(top=0.85)  # spazio per il titolo

    # statici: bounds/obstacles/spot (z molto bassi)
    for b in data.get("bounds", []):
        draw_rect(ax, *b, edge=(0.2, 0.2, 0.2), face=None, alpha=0.6, lw=2.0, z=0.05)
    for o in data.get("obstacles", []):
        draw_rect(ax, *o, edge=None, face=(0.45, 0.45, 0.45), alpha=0.9, z=0.10)
    s = data.get("spot_pose", None)
    if s:
        draw_rect(ax, s[0], s[1], 5.0, 2.4, s[2],
                  edge=(0, 0.6, 0), face=(0, 1, 0), alpha=0.25, lw=2.0, z=0.08)

    # posizioni fisse (partenza/arrivo) - SOTTO l'auto e la traiettoria
    draw_rect(ax, poses[0, 0], poses[0, 1], CAR_LEN, CAR_WID, poses[0, 2],
              edge=(0.0, 0.2, 0.9), face=None, lw=2.0, z=0.20)  # start
    draw_rect(ax, poses[-1, 0], poses[-1, 1], CAR_LEN, CAR_WID, poses[-1, 2],
              edge=(0.9, 0.0, 0.0), face=None, lw=2.0, z=0.20)  # end

    # oggetti dinamici: AUTO (medio) e TRAIETTORIA (sopra)
    # auto in movimento (colore diverso) - livello intermedio
    car_poly = draw_rect(ax, poses[0, 0], poses[0, 1], CAR_LEN, CAR_WID, poses[0, 2],
                         edge=(0.9, 0.45, 0.0), face=(1.0, 0.7, 0.2), alpha=0.95, lw=2.0, z=0.60)
    # traiettoria - livello più alto
    (traj_line,) = ax.plot([], [], '-r', linewidth=2.2, zorder=0.90)

    autolimits(ax, data)
    ax.grid(True, alpha=0.25, zorder=0.01)

    # stato animazione
    total = len(poses)
    step = max(1, int(speed))

    def update(frame_idx):
        i = min(frame_idx * step, total - 1)
        j0 = max(0, i - tail_len) if tail_len > 0 else 0

        # aggiorna traiettoria (sopra)
        traj_line.set_data(poses[j0:i + 1, 0], poses[j0:i + 1, 1])

        # aggiorna macchina (mezzo)
        pts = rect_corners(poses[i, 0], poses[i, 1], CAR_LEN, CAR_WID, poses[i, 2])
        car_poly.set_xy(pts)

        # titolo con reward
        ax.set_title(f"Reward: {rewards[i]:+.3f}    Totale: {cum_rewards[i]:.3f}",
                     fontsize=13, color="darkred", pad=15)

        return traj_line, car_poly

    frames = int(np.ceil(total / step))
    anim = FuncAnimation(
        fig, update, frames=frames, interval=interval_ms,
        blit=False, repeat=False
    )

    #anim.save('ok.gif', fps=30)
    plt.show()


# ---------- entry ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Animazione di un replay JSON (matplotlib).")
    parser.add_argument("--file", type=str, default=None,
                        help="Percorso del JSON (es: replays\\episodio.json). Se omesso, apre quello più recente.")
    parser.add_argument("--index", type=int, default=0,
                        help="Indice del file da aprire (0 = più recente, 1 = penultimo, ecc.)")
    parser.add_argument("--interval", type=int, default=33,
                        help="Intervallo tra frame in ms (default 33 ≈ 30 fps).")
    parser.add_argument("--tail", type=int, default=120,
                        help="Lunghezza coda traiettoria (frame). 0 = intera traccia.")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Velocità: >1 salta frame (riproduzione più rapida).")
    args = parser.parse_args()
    main(args.file, args.interval, args.tail, args.speed, args.index)
