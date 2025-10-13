import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from .generators import gen_situation

CAR_LEN, CAR_WID = 4.5, 2.0  # dimensioni indicative auto

def rect_corners(cx, cy, L, W, th):
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]])
    pts = np.array([[ L/2,  W/2],
                    [ L/2, -W/2],
                    [-L/2, -W/2],
                    [-L/2,  W/2]])
    return (pts @ R.T) + np.array([cx, cy])

def draw(ax, rect, **kw):
    cx, cy, L, W, th = rect
    pts = rect_corners(cx, cy, L, W, th)
    ax.add_patch(Polygon(pts, closed=True, **kw))

def main(layout=None, seed=None):
    s = gen_situation(layout=layout, seed=seed)

    fig, ax = plt.subplots()
    ax.set_aspect('equal', 'box')
    ax.grid(True, alpha=.3)
    ax.set_title(f"Layout={s.layout}  Seed={s.seed}")

    # strada e parcheggi
    for b in s.bounds:
        draw(ax, b, fill=False, linewidth=2.0)
    for o in s.obstacles:
        draw(ax, o, facecolor=(.5,.5,.5), edgecolor=None, alpha=.9)
    sp = s.spot_pose
    draw(ax, (sp[0], sp[1], 5.0, 2.4, sp[2]),
         facecolor=(0,1,0,.25), edgecolor=(0,.6,0), lw=2.0)

    # auto iniziale (blu)
    x, y, th = s.car_start_pose
    draw(ax, (x, y, CAR_LEN, CAR_WID, th),
         edgecolor=(0,0,1), facecolor=(.35,.6,1,.9), lw=2.0)

    # limiti grafici
    all_pts = np.vstack([rect_corners(*r) for r in np.vstack([s.bounds, s.obstacles])])
    m = 3
    ax.set_xlim(all_pts[:,0].min()-m, all_pts[:,0].max()+m)
    ax.set_ylim(all_pts[:,1].min()-m, all_pts[:,1].max()+m)

    plt.show()

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--layout", choices=["spina","colonna","esse"], default=None)
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args()
    main(a.layout, a.seed)
