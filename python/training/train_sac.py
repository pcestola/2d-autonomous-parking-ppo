import os
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from lib import Parking2DEnv

N_ENVS = 8
TOTAL_STEPS = 5_000_000
MODELS_DIR = "models_sac"
VECNORM_PATH = os.path.join(MODELS_DIR, "vecnorm.pkl")

def make_env():
    def _thunk():
        return Monitor(Parking2DEnv(rays=8, max_steps=400))
    return _thunk

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    venv = SubprocVecEnv([make_env() for _ in range(N_ENVS)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.995)

    model = SAC(
        "MlpPolicy", venv,
        learning_rate=3e-4,
        batch_size=256,
        gamma=0.995,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        buffer_size=1_000_000,
        target_update_interval=1,
        ent_coef="auto",
        verbose=1,
    )

    model.learn(total_timesteps=TOTAL_STEPS)
    venv.save(VECNORM_PATH)
    model.save(os.path.join(MODELS_DIR, "sac_parking_final.zip"))

if __name__ == "__main__":
    main()
