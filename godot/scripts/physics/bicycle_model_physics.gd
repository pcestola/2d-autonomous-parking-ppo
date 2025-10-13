extends CarPhysics
class_name BicycleModelPhysics

@export var mass: float = 1200.0
@export var inertia_yaw: float = 1500.0
@export var brake_drag_scale: float = 1.4
@export var extra_damping: float = 1.2

var yaw_rate_state: float = 0.0

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

	var engine_acc: float = 0.0
	if throttle >= 0.0:
		engine_acc = throttle * accel_rate
	else:
		engine_acc = throttle * brake_rate * brake_drag_scale

	var drag_force: float = drag * v * v
	if v < 0.0:
		drag_force = -drag_force
	var roll: float = rolling_resistance * v * extra_damping
	var acc_long: float = engine_acc - drag_force - roll

	v += (acc_long) * dt
	if v > max_speed:
		v = max_speed
	elif v < -max_speed:
		v = -max_speed

	var cur_yaw_acc: float = 0.0
	if abs(steer) > 0.0001:
		cur_yaw_acc = ((v / wheel_base) * tan(steer) - yaw_rate_state) * (1.0 / max(0.001, dt))
	else:
		cur_yaw_acc = -yaw_rate_state * 0.5
	yaw_rate_state += cur_yaw_acc * dt
	yaw += yaw_rate_state * dt

	var forward: Vector3 = Vector3(sin(yaw), 0.0, cos(yaw))
	pos += forward * v * dt
