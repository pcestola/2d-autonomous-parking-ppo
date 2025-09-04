extends SceneTree

## Demo PPO headless.
## Esegui con: godot4 --headless --path . --script res://scripts/rl/trainer_demo.gd

func _initialize() -> void:
	print("PPO demo starting...")
	var env: CarVirtualEnv = CarVirtualEnv.new()
	var obs_dim: int = env.obs_dim()
	var act_dim: int = env.act_dim()
	var ppo: PPO = PPO.new(obs_dim, act_dim)
	var iters: int = 30
	var steps: int = 2048
	var batch: int = 256
	var epochs: int = 5
	var dt: float = 0.05
	var k: int = 0
	while k < iters:
		var rollout: Dictionary = ppo.collect_rollout(env, steps, dt)
		ppo.update(rollout, batch, epochs)
		print("Iter ", k, " ok")
		k += 1
	var success_rate: float = _evaluate(env, ppo, 30, 400, dt)
	print("Evaluation success rate: ", success_rate * 100.0, "%")
	quit()

func _evaluate(env: CarVirtualEnv, ppo: PPO, episodes: int, max_steps: int, dt: float) -> float:
	var successes: int = 0
	var ep: int = 0
	while ep < episodes:
		var obs: PackedFloat32Array = env.reset(true)
		var t: int = 0
		var done: bool = false
		while t < max_steps and not done:
			var mu: PackedFloat32Array = ppo.actor.forward(obs)
			var d: Dictionary = env.step(Vector2(mu[0], mu[1]), dt)
			obs = d["obs"]
			done = d["done"]
			t += 1
		if done:
			var dx: float = obs[0]
			var dz: float = obs[1]
			var yaw_err: float = atan2(obs[2], obs[3])
			var v_now: float = obs[4]
			var dist: float = sqrt(dx * dx + dz * dz)
			if dist < env.parking_tol_pos and abs(yaw_err) < env.parking_tol_yaw and abs(v_now) < env.parking_tol_speed:
				successes += 1
		ep += 1
	return float(successes) / float(episodes)
