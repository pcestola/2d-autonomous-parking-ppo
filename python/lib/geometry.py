import numpy as np

CAR_LEN = 4.6
CAR_WID = 2.0

def rotmat(th: float) -> np.ndarray:
    c, s = np.cos(th), np.sin(th)
    return np.array([[c,-s],[s,c]], dtype=float)

def rect_inside_bounds(car_rect, bounds_rect) -> bool:
    # car_rect: (cx,cy,L,W,th) — bounds_rect: (cx,cy,L,W,th≈0)
    from .geometry import rect_corners
    bx, by, BL, BW, bth = bounds_rect
    assert abs(bth) < 1e-3, "i bounds sono assunti axis-aligned"
    x0, x1 = bx - BL/2, bx + BL/2
    y0, y1 = by - BW/2, by + BW/2
    pts = rect_corners(*car_rect)  # 4x2
    return (pts[:,0].min() >= x0) and (pts[:,0].max() <= x1) and \
           (pts[:,1].min() >= y0) and (pts[:,1].max() <= y1)

def rect_corners(cx, cy, l, w, th) -> np.ndarray:
    # rettangolo centrato, lungo l asse x
    hw, hl = w/2.0, l/2.0
    pts = np.array([
        [ hl,  hw],
        [ hl, -hw],
        [-hl, -hw],
        [-hl,  hw],
    ], dtype=float)
    R = rotmat(th)
    return (pts @ R.T) + np.array([cx, cy])

def _proj_on_axis(pts: np.ndarray, axis: np.ndarray) -> tuple[float,float]:
    s = pts @ axis
    return s.min(), s.max()

def obb_overlap(a, b) -> bool:
    # a,b: (cx,cy,l,w,th)
    ax = rect_corners(*a)
    bx = rect_corners(*b)
    axes = []
    for poly in (ax, bx):
        e = np.diff(np.vstack([poly, poly[0]]), axis=0)
        # normali agli edge
        n = np.stack([ -e[:,1], e[:,0] ], axis=1)
        n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
        axes.extend([n[0], n[1]])  # bastano 2 assi per rettangoli
    for axis in axes:
        a0,a1 = _proj_on_axis(ax, axis)
        b0,b1 = _proj_on_axis(bx, axis)
        if a1 < b0 or b1 < a0:
            return False
    return True

def car_obb(x, y, th) -> tuple[float,float,float,float,float]:
    return (x, y, CAR_LEN, CAR_WID, th)

def collide_car(x,y,th, rects: np.ndarray) -> bool:
    c = car_obb(x,y,th)
    for r in rects:
        if obb_overlap(c, r):
            return True
    return False

# Occupancy grid raster + raycasting (semplice e veloce)
def make_grid(bounds_rect, obstacles: np.ndarray, res: float = 0.1):
    # bounds_rect: (cx,cy,L,W,th) assumiamo th≈0 nella nostra generazione
    cx, cy, L, W, _ = bounds_rect
    x0, x1 = cx - L/2, cx + L/2
    y0, y1 = cy - W/2, cy + W/2
    nx = int(np.ceil((x1-x0)/res))
    ny = int(np.ceil((y1-y0)/res))
    grid = np.zeros((ny, nx), dtype=np.uint8)
    def world_to_idx(px, py):
        ix = np.clip(((px - x0)/res).astype(int), 0, nx-1)
        iy = np.clip(((py - y0)/res).astype(int), 0, ny-1)
        return ix, iy
    # rasterizza rettangoli come poligoni pieni via bounding-box + mask
    for r in obstacles:
        pts = rect_corners(*r)  # 4x2
        minx, miny = pts[:,0].min(), pts[:,1].min()
        maxx, maxy = pts[:,0].max(), pts[:,1].max()
        xs = np.arange(minx, maxx+res, res)
        ys = np.arange(miny, maxy+res, res)
        if xs.size==0 or ys.size==0: 
            continue
        X,Y = np.meshgrid(xs, ys)
        P = np.stack([X.ravel(), Y.ravel()], axis=1)
        # test punto-in-poligono (winding) usando proiezioni su assi dei lati
        # trucco: consideriamo il rettangolo come OBB → mappiamo punti nello spazio locale
        cx, cy, l, w, th = r
        R = rotmat(th)
        Pin = (P - np.array([cx,cy])) @ R
        mask = (np.abs(Pin[:,0]) <= l/2) & (np.abs(Pin[:,1]) <= w/2)
        ix, iy = world_to_idx(P[mask,0], P[mask,1])
        grid[iy, ix] = 1
    return grid, (x0, y0, res)

def raycast_grid(grid, origin_xy, angles, world_meta, max_range: float = 15.0):
    x0, y0, res = world_meta
    ny, nx = grid.shape
    ox, oy = origin_xy
    dists = np.empty_like(angles, dtype=float)
    max_steps = int(max_range/res)
    for i, ang in enumerate(angles):
        x, y = ox, oy
        dx, dy = np.cos(ang)*res, np.sin(ang)*res
        dist = 0.0
        hit = False
        for _ in range(max_steps):
            x += dx; y += dy; dist += res
            ix = int((x - x0)/res); iy = int((y - y0)/res)
            if ix < 0 or ix >= nx or iy < 0 or iy >= ny:
                break
            if grid[iy, ix] != 0:
                hit = True
                break
        dists[i] = min(dist, max_range) / max_range
    return dists  # normalizzate [0,1]
