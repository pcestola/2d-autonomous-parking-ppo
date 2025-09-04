extends RefCounted
class_name PPO

var actor: MLP
var critic: MLP
var act_dim: int = 2
var log_std: PackedFloat32Array
var m_log_std: PackedFloat32Array
var v_log_std: PackedFloat32Array
var t_step: int = 0

var gamma: float = 0.99
var lam: float = 0.95
var clip_eps: float = 0.2
var ent_coef: float = 0.01
var vf_coef: float = 0.5
var lr: float = 0.0005
var beta1: float = 0.9
var beta2: float = 0.999
var adam_eps: float = 1e-8

func _init(obs_dim: int, action_dim: int) -> void:
	act_dim = action_dim
	actor = MLP.new(obs_dim, PackedInt32Array([64, 64]), action_dim, 1234)
	critic = MLP.new(obs_dim, PackedInt32Array([64, 64]), 1, 4321)
	log_std = PackedFloat32Array()
	log_std.resize(action_dim)
	m_log_std = PackedFloat32Array()
	m_log_std.resize(action_dim)
	v_log_std = PackedFloat32Array()
	v_log_std.resize(action_dim)
	var i: int = 0
	while i < action_dim:
		log_std[i] = -0.5
		m_log_std[i] = 0.0
		v_log_std[i] = 0.0
		i += 1
	t_step = 0

func gaussian_sample(mu: PackedFloat32Array) -> PackedFloat32Array:
	var a: PackedFloat32Array = PackedFloat32Array()
	a.resize(act_dim)
	var rng: RandomNumberGenerator = RandomNumberGenerator.new()
	rng.randomize()
	var i: int = 0
	while i < act_dim:
		var std_i: float = exp(log_std[i])
		a[i] = clamp(mu[i] + std_i * rng.randfn(), -1.0, 1.0)
		i += 1
	return a

func gaussian_logprob(mu: PackedFloat32Array, a: PackedFloat32Array) -> float:
	var s: float = 0.0
	var i: int = 0
	while i < act_dim:
		var std_i: float = exp(log_std[i])
		var z: float = (a[i] - mu[i]) / std_i
		s += -0.5 * (z * z + 2.0 * log_std[i] + log(2.0 * 3.14159265))
		i += 1
	return s

func value_forward(obs: PackedFloat32Array) -> float:
	var v_vec: PackedFloat32Array = critic.forward(obs)
	return v_vec[0]

func collect_rollout(env: CarVirtualEnv, steps: int, dt: float) -> Dictionary:
	var obs_buf: Array[PackedFloat32Array] = []
	var act_buf: Array[PackedFloat32Array] = []
	var rew_buf: PackedFloat32Array = PackedFloat32Array()
	rew_buf.resize(steps)
	var done_buf: PackedFloat32Array = PackedFloat32Array()
	done_buf.resize(steps)
	var val_buf: PackedFloat32Array = PackedFloat32Array()
	val_buf.resize(steps)
	var logp_buf: PackedFloat32Array = PackedFloat32Array()
	logp_buf.resize(steps)

	var obs: PackedFloat32Array = env.reset(true)
	var t: int = 0
	while t < steps:
		var mu: PackedFloat32Array = actor.forward(obs)
		var a: PackedFloat32Array = gaussian_sample(mu)
		var d: Dictionary = env.step(Vector2(a[0], a[1]), dt)
		var next_obs: PackedFloat32Array = d["obs"]
		var r: float = d["reward"]
		var done: bool = d["done"]
		obs_buf.append(obs)
		act_buf.append(a)
		rew_buf[t] = r
		done_buf[t] = 1.0 if done else 0.0
		val_buf[t] = value_forward(obs)
		logp_buf[t] = gaussian_logprob(mu, a)
		obs = next_obs
		if done:
			obs = env.reset(true)
		t += 1

	var adv_buf: PackedFloat32Array = PackedFloat32Array()
	adv_buf.resize(steps)
	var ret_buf: PackedFloat32Array = PackedFloat32Array()
	ret_buf.resize(steps)

	var lastgaelam: float = 0.0
	var next_value: float = 0.0
	var t2: int = steps - 1
	while t2 >= 0:
		var nonterminal: float = 1.0
		if done_buf[t2] > 0.5: nonterminal = 0.0
		var delta: float = rew_buf[t2] + gamma * next_value * nonterminal - val_buf[t2]
		lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
		adv_buf[t2] = lastgaelam
		next_value = val_buf[t2]
		t2 -= 1

	var mean_adv: float = 0.0
	var i: int = 0
	while i < steps:
		mean_adv += adv_buf[i]
		i += 1
	mean_adv /= float(steps)
	var var_adv: float = 0.0
	i = 0
	while i < steps:
		var_adv += (adv_buf[i] - mean_adv) * (adv_buf[i] - mean_adv)
		i += 1
	var_adv /= float(steps)
	var std_adv: float = sqrt(var_adv + 1e-8)
	i = 0
	while i < steps:
		adv_buf[i] = (adv_buf[i] - mean_adv) / std_adv
		ret_buf[i] = val_buf[i] + adv_buf[i]
		i += 1

	var out: Dictionary = {}
	out["obs"] = obs_buf
	out["act"] = act_buf
	out["rew"] = rew_buf
	out["done"] = done_buf
	out["val"] = val_buf
	out["logp"] = logp_buf
	out["adv"] = adv_buf
	out["ret"] = ret_buf
	return out

func _adam_update_log_std(grad: PackedFloat32Array) -> void:
	t_step += 1
	var i: int = 0
	while i < log_std.size():
		m_log_std[i] = beta1 * m_log_std[i] + (1.0 - beta1) * grad[i]
		v_log_std[i] = beta2 * v_log_std[i] + (1.0 - beta2) * grad[i] * grad[i]
		i += 1
	var t: float = float(t_step)
	var c1: float = 1.0 - pow(beta1, t)
	var c2: float = 1.0 - pow(beta2, t)
	i = 0
	while i < log_std.size():
		var m_hat: float = m_log_std[i] / c1
		var v_hat: float = v_log_std[i] / c2
		log_std[i] += lr * m_hat / (sqrt(v_hat) + adam_eps)
		i += 1

func update(rollout: Dictionary, batch_size: int, epochs: int) -> void:
	var obs_buf: Array[PackedFloat32Array] = rollout["obs"]
	var act_buf: Array[PackedFloat32Array] = rollout["act"]
	var old_logp: PackedFloat32Array = rollout["logp"]
	var adv_buf: PackedFloat32Array = rollout["adv"]
	var ret_buf: PackedFloat32Array = rollout["ret"]
	var n: int = old_logp.size()
	var idx: PackedInt32Array = PackedInt32Array()
	idx.resize(n)
	var i: int = 0
	while i < n:
		idx[i] = i
		i += 1
	var rng: RandomNumberGenerator = RandomNumberGenerator.new()
	rng.randomize()

	var e: int = 0
	while e < epochs:
		i = 0
		while i < n:
			var j: int = rng.randi_range(0, n - 1)
			var tmp: int = idx[i]
			idx[i] = idx[j]
			idx[j] = tmp
			i += 1
		var start: int = 0
		while start < n:
			var endi: int = min(n, start + batch_size)
			var k: int = start
			while k < endi:
				var id: int = idx[k]
				var obs: PackedFloat32Array = obs_buf[id]
				var act: PackedFloat32Array = act_buf[id]
				var mu: PackedFloat32Array = actor.forward(obs)
				var logp_new: float = gaussian_logprob(mu, act)
				var ratio: float = exp(logp_new - old_logp[id])
				var adv: float = adv_buf[id]
				var clipped_ratio: float = ratio
				if ratio > 1.0 + clip_eps: clipped_ratio = 1.0 + clip_eps
				elif ratio < 1.0 - clip_eps: clipped_ratio = 1.0 - clip_eps
				var use_clipped: bool = false
				if adv >= 0.0 and ratio > 1.0 + clip_eps: use_clipped = true
				elif adv < 0.0 and ratio < 1.0 - clip_eps: use_clipped = true
				var gcoeff: float = - (clipped_ratio * adv if use_clipped else ratio * adv)
				var grad_logp_mu: PackedFloat32Array = PackedFloat32Array()
				grad_logp_mu.resize(act_dim)
				var i2: int = 0
				while i2 < act_dim:
					var std_i: float = exp(log_std[i2])
					var denom: float = std_i * std_i
					grad_logp_mu[i2] = (act[i2] - mu[i2]) / denom
					i2 += 1
				var cache: Dictionary = actor.forward_cache(obs)
				var grad_mu: PackedFloat32Array = PackedFloat32Array()
				grad_mu.resize(act_dim)
				i2 = 0
				while i2 < act_dim:
					grad_mu[i2] = gcoeff * grad_logp_mu[i2]
					i2 += 1
				actor.backward_update(cache, grad_mu, lr, beta1, beta2, adam_eps)
				var grad_logstd: PackedFloat32Array = PackedFloat32Array()
				grad_logstd.resize(act_dim)
				i2 = 0
				while i2 < act_dim:
					var std_i2: float = exp(log_std[i2])
					var term_policy: float = gcoeff * ( ((act[i2] - mu[i2]) * (act[i2] - mu[i2])) / (std_i2 * std_i2) - 1.0 )
					var g_ent: float = ent_coef * 1.0
					grad_logstd[i2] = g_ent + term_policy
					i2 += 1
				_adam_update_log_std(grad_logstd)
				var vpred_vec: PackedFloat32Array = critic.forward(obs)
				var vpred: float = vpred_vec[0]
				var vtarget: float = ret_buf[id]
				var grad_v: PackedFloat32Array = PackedFloat32Array()
				grad_v.resize(1)
				grad_v[0] = vf_coef * 2.0 * (vpred - vtarget)
				var cache_v: Dictionary = critic.forward_cache(obs)
				critic.backward_update(cache_v, grad_v, lr, beta1, beta2, adam_eps)
				k += 1
			start = endi
		e += 1
