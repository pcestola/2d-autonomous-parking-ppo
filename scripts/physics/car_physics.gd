extends RefCounted
class_name CarPhysics

@export var wheel_base: float = 2.6
@export var max_steer: float = 0.6
@export var max_speed: float = 18.0
@export var accel_rate: float = 8.0
@export var brake_rate: float = 10.0
@export var steer_rate: float = 2.5
@export var drag: float = 0.25
@export var rolling_resistance: float = 0.8

var pos: Vector3 = Vector3.ZERO
var yaw: float = 0.0
var v: float = 0.0
var steer: float = 0.0

func reset(state: Dictionary) -> void:
	var p: Vector3 = Vector3.ZERO
	var y: float = 0.0
	var vel: float = 0.0
	var st: float = 0.0
	if state.has("pos"):
		p = state["pos"]
	if state.has("yaw"):
		y = float(state["yaw"])
	if state.has("v"):
		vel = float(state["v"])
	if state.has("steer"):
		st = float(state["steer"])
	pos = p
	yaw = y
	v = vel
	steer = st

func get_state() -> Dictionary:
	var d: Dictionary = {}
	d["pos"] = pos
	d["yaw"] = yaw
	d["v"] = v
	d["steer"] = steer
	return d

func step(action: Vector2, dt: float) -> void:
	pass
