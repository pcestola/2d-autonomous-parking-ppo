import argparse
from stable_baselines3 import PPO
from lib import Parking2DEnv


def evaluate(model_path: str, episodes: int = 10, layout: str | None = None):
    env = Parking2DEnv(layout=layout, rays=16, max_steps=500, log_replays=True, replays_dir="replays")
    model = PPO.load(model_path)

    stats = {"success": 0, "collision": 0, "timeout": 0, "oob": 0, "stalled": 0}
    steps_per_ep = []

    for _ in range(episodes):
        obs, _ = env.reset(seed=None)
        done = False
        ep_steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_steps += 1

        if info.get("success"):
            stats["success"] += 1
        if info.get("collided"):
            stats["collision"] += 1
        if info.get("oob"):
            stats["oob"] += 1
        if info.get("stalled"):
            stats["stalled"] += 1
        if (not info.get("success")) and (not info.get("collided")) and (not info.get("oob")) and (not info.get("stalled")):
            stats["timeout"] += 1

        steps_per_ep.append(ep_steps)

    print("=== Eval summary ===")
    for key, value in stats.items():
        print(f"{key}: {value}/{episodes}")
    print(f"avg_steps: {sum(steps_per_ep) / max(len(steps_per_ep), 1):.2f}")
    print("Replays salvati in ./replays")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--layout", choices=["spina", "colonna", "esse"], default=None)
    args = parser.parse_args()
    evaluate(args.model, args.episodes, args.layout)


