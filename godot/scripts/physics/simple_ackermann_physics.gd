extends CarPhysics
class_name SimpleAckermannPhysics

func step(action: Vector2, dt: float) -> void:
	var throttle: float = clamp(action.x, -1.0, 1.0)
	var steer_in: float = clamp(action.y, -1.0, 1.0)

	var target_steer: float = steer_in * max_steer
	var ds: float = target_steer - steer
	var max_ds: float = steer_rate * dt
	if ds > max_ds:
		ds = max_ds
	elif ds < -max_ds:
		ds = -max_ds
	steer += ds

	var acc_cmd: float = 0.0
	if throttle >= 0.0:
		acc_cmd = throttle * accel_rate
	else:
		acc_cmd = throttle * brake_rate

	var drag_force: float = drag * v * v
	if v < 0.0:
		drag_force = -drag_force
	var rolling: float = rolling_resistance * v

	var a: float = acc_cmd - drag_force - rolling
	v += a * dt
	if v > max_speed:
		v = max_speed
	elif v < -max_speed:
		v = -max_speed

	var yaw_rate: float = 0.0
	if abs(steer) > 0.0001:
		yaw_rate = (v / wheel_base) * tan(steer)
	else:
		yaw_rate = 0.0
	yaw += yaw_rate * dt

	var forward: Vector3 = Vector3(sin(yaw), 0.0, cos(yaw))
	pos += forward * v * dt
