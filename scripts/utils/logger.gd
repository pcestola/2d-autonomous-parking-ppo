extends RefCounted
class_name Logger

static func info(msg: String) -> void:
	print("[INFO] ", msg)

static func warn(msg: String) -> void:
	print("[WARN] ", msg)

static func error(msg: String) -> void:
	push_error("[ERROR] " + msg)
