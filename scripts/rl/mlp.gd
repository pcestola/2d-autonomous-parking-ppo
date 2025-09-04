extends RefCounted
class_name MLP

var sizes: PackedInt32Array
var weights: Array[PackedFloat32Array] = []
var biases: Array[PackedFloat32Array] = []
var m_w: Array[PackedFloat32Array] = []
var v_w: Array[PackedFloat32Array] = []
var m_b: Array[PackedFloat32Array] = []
var v_b: Array[PackedFloat32Array] = []
var t_step: int = 0

func _init(input_dim: int = 1, hidden: PackedInt32Array = PackedInt32Array([8]), output_dim: int = 1, seed: int = 12345) -> void:
	sizes = PackedInt32Array()
	sizes.resize(hidden.size() + 2)
	sizes[0] = input_dim
	var i: int = 0
	while i < hidden.size():
		sizes[i + 1] = hidden[i]
		i += 1
	sizes[sizes.size() - 1] = output_dim
	_init_params(seed)

func _init_params(seed: int) -> void:
	weights.clear()
	biases.clear()
	m_w.clear()
	v_w.clear()
	m_b.clear()
	v_b.clear()
	var rng: RandomNumberGenerator = RandomNumberGenerator.new()
	rng.seed = seed
	var l: int = 0
	while l < sizes.size() - 1:
		var fan_in: int = sizes[l]
		var fan_out: int = sizes[l + 1]
		var w: PackedFloat32Array = PackedFloat32Array()
		w.resize(fan_out * fan_in)
		var b: PackedFloat32Array = PackedFloat32Array()
		b.resize(fan_out)
		var k: int = 0
		var limit: float = sqrt(6.0 / float(fan_in + fan_out))
		while k < w.size():
			w[k] = rng.randf_range(-limit, limit)
			k += 1
		k = 0
		while k < b.size():
			b[k] = 0.0
			k += 1
		weights.append(w)
		biases.append(b)
		var mw: PackedFloat32Array = PackedFloat32Array()
		mw.resize(w.size())
		var vw: PackedFloat32Array = PackedFloat32Array()
		vw.resize(w.size())
		var mb: PackedFloat32Array = PackedFloat32Array()
		mb.resize(b.size())
		var vb: PackedFloat32Array = PackedFloat32Array()
		vb.resize(b.size())
		m_w.append(mw)
		v_w.append(vw)
		m_b.append(mb)
		v_b.append(vb)
		l += 1
	t_step = 0

func forward(x: PackedFloat32Array) -> PackedFloat32Array:
	var acts: PackedFloat32Array = x
	var l: int = 0
	while l < weights.size():
		acts = _linear_layer(acts, weights[l], biases[l], sizes[l], sizes[l + 1])
		if l < weights.size() - 1:
			acts = _relu(acts)
		l += 1
	return acts

func forward_cache(x: PackedFloat32Array) -> Dictionary:
	var cache: Dictionary = {}
	var activations: Array[PackedFloat32Array] = []
	activations.append(x)
	var preacts: Array[PackedFloat32Array] = []
	var a: PackedFloat32Array = x
	var l: int = 0
	while l < weights.size():
		var z: PackedFloat32Array = _linear_layer(a, weights[l], biases[l], sizes[l], sizes[l + 1])
		preacts.append(z)
		if l < weights.size() - 1:
			a = _relu(z)
		else:
			a = z
		activations.append(a)
		l += 1
	cache["acts"] = activations
	cache["preacts"] = preacts
	return cache

func backward_update(cache: Dictionary, grad_out: PackedFloat32Array, lr: float, beta1: float, beta2: float, eps: float) -> void:
	t_step += 1
	var activations: Array[PackedFloat32Array] = cache["acts"]
	var preacts: Array[PackedFloat32Array] = cache["preacts"]
	var l: int = weights.size() - 1
	var grad_next: PackedFloat32Array = grad_out
	while l >= 0:
		var a_prev: PackedFloat32Array = activations[l]
		var in_dim: int = sizes[l]
		var out_dim: int = sizes[l + 1]
		var gw: PackedFloat32Array = PackedFloat32Array()
		gw.resize(out_dim * in_dim)
		var gb: PackedFloat32Array = PackedFloat32Array()
		gb.resize(out_dim)
		var i: int = 0
		while i < out_dim:
			gb[i] = grad_next[i]
			var j: int = 0
			while j < in_dim:
				gw[i * in_dim + j] = grad_next[i] * a_prev[j]
				j += 1
			i += 1
		var grad_in: PackedFloat32Array = PackedFloat32Array()
		grad_in.resize(in_dim)
		i = 0
		while i < in_dim:
			var s: float = 0.0
			var j2: int = 0
			while j2 < out_dim:
				s += weights[l][j2 * in_dim + i] * grad_next[j2]
				j2 += 1
			grad_in[i] = s
			i += 1
		if l > 0:
			var gz: PackedFloat32Array = _relu_grad(preacts[l - 1])
			var k2: int = 0
			while k2 < grad_in.size():
				grad_in[k2] *= gz[k2]
				k2 += 1
		_update_adam_layer(l, gw, gb, lr, beta1, beta2, eps)
		grad_next = grad_in
		l -= 1

func _linear_layer(x: PackedFloat32Array, w: PackedFloat32Array, b: PackedFloat32Array, in_dim: int, out_dim: int) -> PackedFloat32Array:
	var y: PackedFloat32Array = PackedFloat32Array()
	y.resize(out_dim)
	var i: int = 0
	while i < out_dim:
		var s: float = 0.0
		var j: int = 0
		while j < in_dim:
			s += w[i * in_dim + j] * x[j]
			j += 1
		y[i] = s + b[i]
		i += 1
	return y

func _relu(x: PackedFloat32Array) -> PackedFloat32Array:
	var y: PackedFloat32Array = PackedFloat32Array()
	y.resize(x.size())
	var i: int = 0
	while i < x.size():
		if x[i] > 0.0: y[i] = x[i]
		else: y[i] = 0.0
		i += 1
	return y

func _relu_grad(x: PackedFloat32Array) -> PackedFloat32Array:
	var g: PackedFloat32Array = PackedFloat32Array()
	g.resize(x.size())
	var i: int = 0
	while i < x.size():
		if x[i] > 0.0: g[i] = 1.0
		else: g[i] = 0.0
		i += 1
	return g

func _update_adam_layer(l: int, gw: PackedFloat32Array, gb: PackedFloat32Array, lr: float, beta1: float, beta2: float, eps: float) -> void:
	var mw: PackedFloat32Array = m_w[l]
	var vw: PackedFloat32Array = v_w[l]
	var mb: PackedFloat32Array = m_b[l]
	var vb: PackedFloat32Array = v_b[l]
	var i: int = 0
	while i < mw.size():
		mw[i] = beta1 * mw[i] + (1.0 - beta1) * gw[i]
		vw[i] = beta2 * vw[i] + (1.0 - beta2) * gw[i] * gw[i]
		i += 1
	i = 0
	while i < mb.size():
		mb[i] = beta1 * mb[i] + (1.0 - beta1) * gb[i]
		vb[i] = beta2 * vb[i] + (1.0 - beta2) * gb[i] * gb[i]
		i += 1
	var t: float = float(t_step)
	var c1: float = 1.0 - pow(beta1, t)
	var c2: float = 1.0 - pow(beta2, t)
	i = 0
	while i < weights[l].size():
		var m_hat: float = mw[i] / c1
		var v_hat: float = vw[i] / c2
		weights[l][i] -= lr * m_hat / (sqrt(v_hat) + eps)
		i += 1
	i = 0
	while i < biases[l].size():
		var m_hat_b: float = mb[i] / c1
		var v_hat_b: float = vb[i] / c2
		biases[l][i] -= lr * m_hat_b / (sqrt(v_hat_b) + eps)
		i += 1
	m_w[l] = mw
	v_w[l] = vw
	m_b[l] = mb
	v_b[l] = vb
