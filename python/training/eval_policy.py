import argparse, os
from stable_baselines3 import PPO
from lib import Parking2DEnv

def main(model_path: str, episodes: int = 10, layout: str | None = None):
    env = Parking2DEnv(layout=layout, rays=16, max_steps=400, log_replays=True, replays_dir="replays")
    model = PPO.load(model_path)
    for ep in range(episodes):
        obs, _ = env.reset(seed=None)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, trunc, info = env.step(action)
    print("Replays salvati in ./replays")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--layout", choices=["spina","colonna","esse"], default=None)
    args = parser.parse_args()
    main(args.model, args.episodes, args.layout)
