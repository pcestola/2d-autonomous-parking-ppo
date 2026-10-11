import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from lib import Parking2DEnv
from .callbacks import EveryNSteps

N_ENVS = 6
TOTAL_STEPS = 1_200_000
MODELS_DIR = "models"
VECNORM_PATH = os.path.join(MODELS_DIR, "vecnorm.pkl")


def make_env(layout=None, log_replays=False):
    def _thunk():
        env = Parking2DEnv(layout=layout, rays=16, max_steps=500, log_replays=log_replays, replays_dir="replays")
        return Monitor(env)
    return _thunk


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    venv = SubprocVecEnv([make_env(layout=None, log_replays=(i == 0)) for i in range(N_ENVS)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.995)

    model = PPO(
        "MlpPolicy",
        venv,
        n_steps=2048,
        batch_size=1024,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        learning_rate=2.5e-4,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        policy_kwargs={"net_arch": [128, 128]},
    )

    def save_cb(model, step):
        model_path = os.path.join(MODELS_DIR, f"ppo_parking_{step}.zip")
        model.save(model_path)
        venv.save(VECNORM_PATH)
        return True

    cb = EveryNSteps(100_000, save_cb)
    model.learn(total_timesteps=TOTAL_STEPS, callback=cb)

    model.save(os.path.join(MODELS_DIR, "ppo_parking_final.zip"))
    venv.save(VECNORM_PATH)


if __name__ == "__main__":
    main()


