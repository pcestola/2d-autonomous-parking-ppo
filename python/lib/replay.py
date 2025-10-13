# replay.py
import json, os
from typing import List
import numpy as np

class ReplayLogger:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.frames: list[list[float]] = []
        self.rewards: list[float] = []  # <--- aggiunta

    def reset(self):
        self.frames = []
        self.rewards = []  # <--- reset anche le reward

    def push(self, pose_xyth, reward: float | None = None):
        self.frames.append([float(p) for p in pose_xyth])
        if reward is not None:
            self.rewards.append(float(reward))
        else:
            # manteniamo lunghezze allineate
            self.rewards.append(0.0)

    def dump(self, situation, fname: str):
        data = {
            "seed": situation.seed,
            "layout": situation.layout,
            "spot_pose": situation.spot_pose.tolist(),
            "car_start_pose": situation.car_start_pose.tolist(),
            "obstacles": situation.obstacles.tolist(),
            "bounds": situation.bounds.tolist(),
            "poses": self.frames,
            "rewards": self.rewards,  # <--- aggiunta qui
        }
        path = os.path.join(self.out_dir, fname)
        with open(path, "w") as f:
            json.dump(data, f)
        return path
