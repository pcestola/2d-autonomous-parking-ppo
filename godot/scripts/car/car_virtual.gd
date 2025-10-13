extends RefCounted
class_name CarVirtualEnv

## Osservazione: [dx, dz, sin(yaw_err), cos(yaw_err), v, steer, ray_0..ray_N]

var world_bounds: Rect2 = Rect2(Vector2(-12.0, -12.0), Vector2(24.0, 24.0))
var obstacles: Array[Rect2] = []
var rays_angles: PackedFloat32Array = PackedFloat32Array([-1.2, -0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9, 1.2])
var ray_max_dist: float = 10.0

var parking_pos: Vector2 = Vector2(4.0, -4.0)
var parking_yaw: float = 3.14159265
var parking_tol_pos: float = 0.8
var parking_tol_yaw: float = 0.2
var parking_tol_speed: float = 0.4

var car: CarPhysics = SimpleAckermannPhysics.new()

func obs_dim() -> int:
	return 6 + int(rays_angles.size())

func act_dim() -> int:
	return 2

func reset(rand: bool = true) -> PackedFloat32Array:
	var px: float = 0.0
	var pz: float = 0.0
	var yaw0: float = 0.0
	if rand:
		var rng: RandomNumberGenerator = RandomNumberGenerator.new()
		rng.randomize()
		px = lerp(parking_pos.x - 3.0, parking_pos.x + 3.0, rng.randf())
		pz = lerp(parking_pos.y - 3.0, parking_pos.y + 3.0, rng.randf())
		yaw0 = rng.randf_range(parking_yaw - 0.6, parking_yaw + 0.6)
	else:
		px = 0.0
		pz = 0.0
		yaw0 = 0.0
	car.reset({"pos": Vector3(px, 0.0, pz), "yaw": yaw0, "v": 0.0, "steer": 0.0})
	return _make_obs()

func step(action: Vector2, dt: float = 0.05) -> Dictionary:
	car.step(action, dt)
	var obs: PackedFloat32Array = _make_obs()
	var reward: float = _compute_reward(obs)
	var done: bool = false
	var dx: float = obs[0]
	var dz: float = obs[1]
	var yaw_err_now: float = atan2(obs[2], obs[3])
	var dist_slot_now: float = sqrt(dx * dx + dz * dz)
	var v_now: float = obs[4]
	if dist_slot_now < parking_tol_pos and abs(yaw_err_now) < parking_tol_yaw and abs(v_now) < parking_tol_speed:
		reward += 5.0
		done = true
	if _is_out_of_bounds(car.pos):
		done = true
		reward -= 5.0
	var d: Dictionary = {}
	d["obs"] = obs
	d["reward"] = reward
	d["done"] = done
	d["info"] = {}
	return d

func _make_obs() -> PackedFloat32Array:
	var to_slot: Vector2 = Vector2(parking_pos.x - car.pos.x, parking_pos.y - car.pos.z)
	var dx: float = to_slot.x
	var dz: float = to_slot.y
	var yaw_err: float = car.yaw - parking_yaw
	while yaw_err > 3.14159265: yaw_err -= 2.0 * 3.14159265
	while yaw_err < -3.14159265: yaw_err += 2.0 * 3.14159265
	var base: PackedFloat32Array = PackedFloat32Array([dx, dz, sin(yaw_err), cos(yaw_err), car.v, car.steer])
	var rays: PackedFloat32Array = _do_rays()
	var total: PackedFloat32Array = PackedFloat32Array()
	total.resize(base.size() + rays.size())
	var i: int = 0
	while i < base.size():
		total[i] = base[i]
		i += 1
	var j: int = 0
	while j < rays.size():
		total[i + j] = rays[j]
		j += 1
	return total

func _do_rays() -> PackedFloat32Array:
	var out: PackedFloat32Array = PackedFloat32Array()
	out.resize(rays_angles.size())
	var i: int = 0
	while i < rays_angles.size():
		var a: float = rays_angles[i] + car.yaw
		var dir2: Vector2 = Vector2(sin(a), cos(a))
		var dnorm: float = _raycast_dist_2d(Vector2(car.pos.x, car.pos.z), dir2, ray_max_dist)
		out[i] = dnorm
		i += 1
	return out

func _raycast_dist_2d(origin: Vector2, dir2: Vector2, max_dist: float) -> float:
	var best: float = max_dist
	var hit: bool = false
	var rects: Array[Rect2] = []
	rects.append(world_bounds)
	var k: int = 0
	while k < obstacles.size():
		rects.append(obstacles[k])
		k += 1
	var i: int = 0
	while i < rects.size():
		var r: Rect2 = rects[i]
		var dist: float = _ray_rect_intersect_dist(origin, dir2, r, max_dist)
		if dist >= 0.0:
			hit = true
			if dist < best:
				best = dist
		i += 1
	if hit:
		return clamp(best / max_dist, 0.0, 1.0)
	else:
		return 1.0

func _ray_rect_intersect_dist(origin: Vector2, dir2: Vector2, rect: Rect2, max_dist: float) -> float:
	var inv_dx: float = 0.0
	var inv_dz: float = 0.0
	if abs(dir2.x) > 1e-6: inv_dx = 1.0 / dir2.x
	else: inv_dx = 1e9
	if abs(dir2.y) > 1e-6: inv_dz = 1.0 / dir2.y
	else: inv_dz = 1e9
	var minx: float = rect.position.x
	var maxx: float = rect.position.x + rect.size.x
	var minz: float = rect.position.y
	var maxz: float = rect.position.y + rect.size.y
	var t1: float = (minx - origin.x) * inv_dx
	var t2: float = (maxx - origin.x) * inv_dx
	var t3: float = (minz - origin.y) * inv_dz
	var t4: float = (maxz - origin.y) * inv_dz
	var tmin: float = min(t1, t2)
	var tmax: float = max(t1, t2)
	var tzmin: float = min(t3, t4)
	var tzmax: float = max(t3, t4)
	if tzmax < tmin or tmax < tzmin: return -1.0
	tmin = max(tmin, tzmin)
	tmax = min(tmax, tzmax)
	if tmax < 0.0: return -1.0
	var t_hit: float = 0.0
	if tmin >= 0.0: t_hit = tmin
	else: t_hit = tmax
	if t_hit > max_dist: return -1.0
	return t_hit

func _compute_reward(obs: PackedFloat32Array) -> float:
	var dx: float = obs[0]
	var dz: float = obs[1]
	var sin_yerr: float = obs[2]
	var cos_yerr: float = obs[3]
	var v: float = obs[4]
	var dist_slot: float = sqrt(dx * dx + dz * dz)
	var yaw_err: float = atan2(sin_yerr, cos_yerr)
	var r: float = 0.0
	r -= 0.3 * dist_slot
	r -= 0.15 * abs(yaw_err)
	r -= 0.03 * abs(v)
	var i: int = 6
	var min_ray: float = 1.0
	while i < obs.size():
		if obs[i] < min_ray: min_ray = obs[i]
		i += 1
	r -= 0.2 * (1.0 - min_ray)
	if dist_slot < parking_tol_pos * 1.5 and abs(yaw_err) < parking_tol_yaw * 2.0:
		r += 0.05
	return r

func _is_out_of_bounds(p: Vector3) -> bool:
	var inx: bool = p.x >= world_bounds.position.x and p.x <= world_bounds.position.x + world_bounds.size.x
	var inz: bool = p.z >= world_bounds.position.y and p.z <= world_bounds.position.y + world_bounds.size.y
	return not (inx and inz)
