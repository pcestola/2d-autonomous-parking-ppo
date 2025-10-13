# generators.py
from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np

Layout = Literal["colonna", "spina", "esse"]

@dataclass
class ParkingSituation:
    seed: int
    layout: Layout
    spot_pose: np.ndarray       # [x, y, theta]
    car_start_pose: np.ndarray  # [x, y, theta]
    obstacles: np.ndarray       # shape (K, 5): [x, y, len, wid, theta]
    bounds: np.ndarray          # shape (1, 5): [cx, cy, w, h, theta]
    friction: float

# ------------------------------ Utils ---------------------------------

def _rng_from_seed(seed: Optional[int]) -> np.random.Generator:
    if seed is None:
        return np.random.default_rng(np.random.randint(1 << 31))
    return np.random.default_rng(seed)

def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation a*(1-t) + b*t with t in [0,1]."""
    return a * (1.0 - t) + b * t

def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))

def _normalize_layout_name(layout: Optional[str]) -> Layout:
    """
    Accetta alias comuni e li normalizza:
      - 'colonna'  → perpendicolari a 90° rispetto alla strada (theta ≈ -pi/2 sotto la strada)
      - 'spina'    → parcheggi inclinati (30°–60°) sotto la strada
      - 'esse'     → in linea (paralleli alla strada, theta ≈ 0)
    """
    if layout is None:
        return np.random.choice(["colonna", "spina", "esse"], p=[0.4, 0.4, 0.2])  # default
    layout = layout.lower()
    aliases = {
        "colonna": ["colonna", "perpendicolare", "90", "90°"],
        "spina":   ["spina", "spina_di_pesce", "pettine", "angolato", "angled"],
        "esse":    ["esse", "s", "linea", "in_linea", "parallelo", "parallel"]
    }
    for canonical, names in aliases.items():
        if layout in names:
            return canonical  # type: ignore
    raise ValueError(f"Layout sconosciuto: {layout}")

# ------------------------------ Generators ---------------------------------

def gen_colonna(seed: Optional[int] = None, difficulty: float = 0.0) -> ParkingSituation:
    """
    Parcheggi perpendicolari (a colonna) sotto la strada.
    difficulty: 0.0 = facile (più spazio), 1.0 = difficile (spazio realistico/strettino)
    """
    d = _clip01(difficulty)
    rng = _rng_from_seed(seed)

    # Spazio strada/bounds
    lane_w = _lerp(4.2, 3.2, d)                       # corsia più larga da facile→difficile
    road_L = float(rng.uniform(_lerp(60.0, 44.0, d), _lerp(72.0, 60.0, d)))
    bounds = np.array([[0.0, 0.0, road_L, lane_w, 0.0]])

    # Dimensioni posto e margini
    spot_len, spot_wid = 5.0, 2.4
    curb_off = _lerp(0.40, 0.25, d)                   # margine dal cordolo
    theta = -np.pi / 2                                # 90° sotto la strada

    # Spaziatura tra auto in parcheggio
    gap_min = _lerp(1.2, 0.6, d)                      # più gap quando facile
    gap_max = _lerp(1.6, 0.9, d)
    step = spot_wid + rng.uniform(gap_min, gap_max)

    # Numero e posizione posti
    n = 5
    x0 = rng.uniform(-_lerp(22.0, 18.0, d), -_lerp(10.0, 8.0, d))
    xs = x0 + np.arange(n) * step + rng.normal(0.0, _lerp(0.02, 0.05, d), size=n)
    y_slots = -(lane_w/2 + curb_off + spot_len/2 + rng.uniform(-_lerp(0.02, 0.05, d), _lerp(0.02, 0.05, d)))

    empty_idx = int(rng.integers(0, n))
    spot_pose = np.array([xs[empty_idx], y_slots, theta])

    # (facoltativo) lasciare un vicino vuoto per facilità
    leave_neighbor_empty = (rng.random() < _lerp(0.7, 0.2, d))
    empty_set = {empty_idx}
    if leave_neighbor_empty:
        if empty_idx == 0:
            empty_set.add(1)
        elif empty_idx == n - 1:
            empty_set.add(n - 2)
        else:
            empty_set.add(empty_idx + (1 if rng.random() < 0.5 else -1))

    obstacles = np.array(
        [[x, y_slots, spot_len, spot_wid, theta] for i, x in enumerate(xs) if i not in empty_set],
        dtype=float
    )

    # partenza: più vicina e centrata quando facile
    start_x = -road_L/2 + _lerp(6.0, 4.0, d)
    car_start_pose = np.array([start_x, 0.0, 0.0], dtype=float)

    return ParkingSituation(
        seed or int(rng.integers(1 << 31)),
        "colonna",
        spot_pose, car_start_pose, obstacles, bounds,
        float(rng.uniform(0.95, 1.00) if d < 0.5 else rng.uniform(0.90, 1.00))
    )

def gen_spina(seed: Optional[int] = None, difficulty: float = 0.0) -> ParkingSituation:
    """
    Parcheggi a spina (inclinati) sotto la strada.
    """
    d = _clip01(difficulty)
    rng = _rng_from_seed(seed)

    lane_w = _lerp(4.2, 3.2, d)
    road_L = float(rng.uniform(_lerp(60.0, 44.0, d), _lerp(72.0, 60.0, d)))
    bounds = np.array([[0.0, 0.0, road_L, lane_w, 0.0]])

    spot_len, spot_wid = 5.0, 2.4
    curb_off = _lerp(0.40, 0.25, d)
    alpha = rng.uniform(np.deg2rad(_lerp(25.0, 30.0, d)), np.deg2rad(_lerp(40.0, 60.0, d)))
    theta = -alpha

    # passo tra posti: proiezioni + gap
    gap_min = _lerp(1.2, 0.5, d)
    gap_max = _lerp(1.6, 0.9, d)
    step = abs(spot_len*np.cos(alpha)) + abs(spot_wid*np.sin(alpha)) + rng.uniform(gap_min, gap_max)

    n = 5
    x0 = rng.uniform(-_lerp(22.0, 18.0, d), -_lerp(10.0, 8.0, d))
    xs = x0 + np.arange(n) * step + rng.normal(0.0, _lerp(0.03, 0.08, d), size=n)
    y_slots = -(lane_w/2 + curb_off
                + 0.5*abs(spot_len*np.sin(alpha))
                + 0.5*abs(spot_wid*np.cos(alpha))
                + rng.uniform(-_lerp(0.02, 0.03, d), _lerp(0.02, 0.03, d)))

    empty_idx = int(rng.integers(0, n))
    spot_pose = np.array([xs[empty_idx], y_slots, theta])

    leave_neighbor_empty = (rng.random() < _lerp(0.6, 0.2, d))
    empty_set = {empty_idx}
    if leave_neighbor_empty:
        if empty_idx == 0:
            empty_set.add(1)
        elif empty_idx == n - 1:
            empty_set.add(n - 2)
        else:
            empty_set.add(empty_idx + (1 if rng.random() < 0.5 else -1))

    obstacles = np.array(
        [[x, y_slots, spot_len, spot_wid, theta] for i, x in enumerate(xs) if i not in empty_set],
        dtype=float
    )

    car_start_pose = np.array([-road_L/2 + _lerp(6.0, 4.0, d), 0.0, 0.0], dtype=float)

    return ParkingSituation(
        seed or int(rng.integers(1 << 31)),
        "spina",
        spot_pose, car_start_pose, obstacles, bounds,
        float(rng.uniform(0.95, 1.00) if d < 0.5 else rng.uniform(0.90, 1.00))
    )

def gen_esse(seed: Optional[int] = None, difficulty: float = 0.0) -> ParkingSituation:
    """
    Parcheggi in linea (a S) sotto la strada (theta ≈ 0).
    """
    d = _clip01(difficulty)
    rng = _rng_from_seed(seed)

    lane_w = _lerp(4.4, 3.2, d)  # un filo più larga perché l'innesto laterale è più difficile
    road_L = float(rng.uniform(_lerp(60.0, 44.0, d), _lerp(72.0, 60.0, d)))
    bounds = np.array([[0.0, 0.0, road_L, lane_w, 0.0]])

    spot_len, spot_wid = 5.0, 2.4
    curb_off = _lerp(0.40, 0.25, d)
    n = 5
    step = rng.uniform(_lerp(6.2, 5.6, d), _lerp(6.8, 6.0, d))  # più lungo quando facile
    x0 = rng.uniform(-_lerp(22.0, 18.0, d), -_lerp(10.0, 8.0, d))
    xs = x0 + np.arange(n) * step + rng.normal(0.0, _lerp(0.03, 0.10, d), size=n)
    y_slots = -(lane_w/2 + curb_off + spot_wid/2 + rng.uniform(-_lerp(0.02, 0.05, d), _lerp(0.02, 0.05, d)))

    empty_idx = int(rng.integers(0, n))
    spot_pose = np.array([xs[empty_idx], y_slots, 0.0])

    # in linea: per facilitare, spesso lasciamo un vicino vuoto
    leave_neighbor_empty = (rng.random() < _lerp(0.7, 0.3, d))
    empty_set = {empty_idx}
    if leave_neighbor_empty:
        if empty_idx == 0:
            empty_set.add(1)
        elif empty_idx == n - 1:
            empty_set.add(n - 2)
        else:
            empty_set.add(empty_idx + (1 if rng.random() < 0.5 else -1))

    obstacles = np.array(
        [[x, y_slots, spot_len, spot_wid, 0.0] for i, x in enumerate(xs) if i not in empty_set],
        dtype=float
    )

    car_start_pose = np.array([-road_L/2 + _lerp(7.0, 4.0, d), 0.0, 0.0], dtype=float)

    return ParkingSituation(
        seed or int(rng.integers(1 << 31)),
        "esse",
        spot_pose, car_start_pose, obstacles, bounds,
        float(rng.uniform(0.95, 1.00) if d < 0.5 else rng.uniform(0.90, 1.00))
    )

# ------------------------------ Public API ---------------------------------

def gen_situation(layout: Optional[Layout | str] = None, seed: Optional[int] = None,
                  difficulty: float = 0.0) -> ParkingSituation:
    """
    Genera una situazione con difficoltà controllata:
        difficulty in [0,1] : 0 = molto facile (più spazi, meno rumore), 1 = più stretto/difficile.
    layout accetta alias: 'colonna' (90°), 'spina' (angolo), 'esse' (in linea).
    """
    Lnorm: Layout = _normalize_layout_name(layout) if layout is not None else None  # type: ignore
    if Lnorm is None:
        Lnorm = np.random.choice(["colonna", "spina", "esse"], p=[0.4, 0.4, 0.2])  # default

    if Lnorm == "colonna":
        sit = gen_colonna(seed, difficulty=difficulty)
        # assert geometrico: theta ≈ -pi/2 (perpendicolare)
        if not (abs(abs(sit.spot_pose[2]) - (np.pi/2)) < np.deg2rad(5)):
            raise AssertionError("Layout 'colonna' atteso perpendicolare (~90°).")
        sit.layout = "colonna"
        return sit

    if Lnorm == "spina":
        sit = gen_spina(seed, difficulty=difficulty)
        # assert geometrico: theta inclinato (30°–60°)
        ang = abs(sit.spot_pose[2])
        if not (np.deg2rad(20) <= ang <= np.deg2rad(65)):
            raise AssertionError("Layout 'spina' atteso inclinato (20°–65°).")
        sit.layout = "spina"
        return sit

    if Lnorm == "esse":
        sit = gen_esse(seed, difficulty=difficulty)
        # assert geometrico: theta ≈ 0 (parallelo)
        if not (abs(sit.spot_pose[2]) < np.deg2rad(5)):
            raise AssertionError("Layout 'esse' atteso parallelo (~0°).")
        sit.layout = "esse"
        return sit

    raise ValueError(f"Layout sconosciuto: {layout}")
