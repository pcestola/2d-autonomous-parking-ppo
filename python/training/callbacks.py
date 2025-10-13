from stable_baselines3.common.callbacks import BaseCallback

class EveryNSteps(BaseCallback):
    def __init__(self, n_steps: int, callback, verbose=0):
        super().__init__(verbose)
        self.n_steps = n_steps
        self.callback = callback
    def _on_step(self) -> bool:
        if self.num_timesteps % self.n_steps == 0:
            return self.callback(self.model, self.num_timesteps)
        return True
