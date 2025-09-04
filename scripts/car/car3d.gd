extends Node3D
class_name Car3D

@export var use_bicycle_model: bool = false
@export var start_pos: Vector3 = Vector3(0.0, 0.6, 0.0)
@export var start_yaw: float = 3.14159265

var _physics: CarPhysics
var _initial_state: Dictionary = {}

func _ready() -> void:
	if use_bicycle_model:
		_physics = BicycleModelPhysics.new()
	else:
		_physics = SimpleAckermannPhysics.new()
	_initial_state = {"pos": start_pos, "yaw": start_yaw, "v": 0.0, "steer": 0.0}
	_physics.reset(_initial_state)
	_update_transform()

func _physics_process(delta: float) -> void:
	var throttle: float = Input.get_action_strength("move_forward") - Input.get_action_strength("move_back")
	var steer_in: float = Input.get_action_strength("steer_right") - Input.get_action_strength("steer_left")
	var action: Vector2 = Vector2(throttle, -steer_in)
	_physics.step(action, delta)
	_update_transform()

	if Input.is_action_just_pressed("reset"):
		_physics.reset(_initial_state)
		_update_transform()

func _update_transform() -> void:
	var b: Basis = Basis(Vector3.UP, _physics.yaw)
	var t: Transform3D = Transform3D(b, _physics.pos)
	global_transform = t
