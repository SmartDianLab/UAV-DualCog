from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
COMMON_STAGE_CONFIG_PATH = WORKSPACE_ROOT / "configs" / "uav_dualcog" / "common_stage_configs.yaml"
BEHAVIOR_SHARED_CONFIG: dict[str, Any] = {}


def _normalize_camera_mode_value(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    if text in {"track_target", "landmark_track"}:
        return "landmark_track"
    if text in {"velocity_aligned", "look_forward"}:
        return "look_forward"
    return "landmark_track"


def _param_spec(
    *,
    label: str,
    minimum: float | int | None = None,
    maximum: float | int | None = None,
    default: float | int | str | bool | None = None,
    step: float | int | None = None,
    choices: list[Any] | None = None,
    description: str = "",
    auto_method_default: str = "random",
    auto_center: float | int | None = None,
    auto_std: float | int | None = None,
) -> dict[str, Any]:
    return {
        "label": label,
        "min": minimum,
        "max": maximum,
        "default": default,
        "step": step,
        "choices": list(choices or []),
        "description": description,
        "auto_method_default": str(auto_method_default or "random"),
        "auto_center": auto_center if auto_center is not None else default,
        "auto_std": auto_std,
    }


def _element_step(
    element_class: str,
    *,
    params: dict[str, Any] | None = None,
    target_binding: str = "primary",
) -> dict[str, Any]:
    return {
        "element_class": str(element_class),
        "params": dict(params or {}),
        "target_binding": str(target_binding or "primary"),
    }


def _deep_update_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update_dict(dict(out.get(key, {})), dict(value))
        else:
            out[key] = value
    return out


def _orbit_safety_distance_m() -> float:
    try:
        return float(BEHAVIOR_SHARED_CONFIG.get("safety_distance_m", 2.0) or 2.0)
    except Exception:
        return 2.0


def _orbit_scale_anchor_from_bbox(bbox: list[float]) -> float:
    sx = float(bbox[3]) if len(bbox) > 3 else 3.0
    sy = float(bbox[4]) if len(bbox) > 4 else 3.0
    half_xy_diag = 0.5 * float(math.sqrt(max(1e-6, sx * sx + sy * sy)))
    return max(0.5, half_xy_diag + _orbit_safety_distance_m())


def _load_behavior_library_config() -> dict[str, Any]:
    if not COMMON_STAGE_CONFIG_PATH.exists():
        raise RuntimeError(f"stage3_behavior_library_config_missing: {COMMON_STAGE_CONFIG_PATH}")
    try:
        payload = yaml.safe_load(COMMON_STAGE_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise RuntimeError(f"stage3_behavior_library_config_load_failed: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("invalid_stage3_behavior_library_config")
    block = dict(payload.get("stage3_behavior_library", {}) or {})
    if not block:
        raise RuntimeError("missing_stage3_behavior_library_block")
    return block


def _normalize_param_entry(param_key: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid_stage3_behavior_param_spec: {param_key}")
    out = dict(value)
    if str(param_key) == "camera_mode":
        choices = [_normalize_camera_mode_value(item) for item in list(out.get("choices", []) or [])]
        out["choices"] = [item for idx, item in enumerate(choices) if item and item not in choices[:idx]]
        out["default"] = _normalize_camera_mode_value(out.get("default", "landmark_track"))
        if out.get("auto_center", None) is not None:
            out["auto_center"] = _normalize_camera_mode_value(out.get("auto_center"))
    return out


def _normalize_element_spec(element_key: str, spec: Any) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise RuntimeError(f"invalid_stage3_behavior_element_spec: {element_key}")
    out = dict(spec)
    out["display_name"] = str(out.get("display_name", element_key) or element_key)
    out["family"] = str(out.get("family", "") or "")
    out["description"] = str(out.get("description", "") or "")
    out["camera_mode_default"] = _normalize_camera_mode_value(out.get("camera_mode_default", "landmark_track"))
    params = dict(out.get("params", {}) or {})
    out["params"] = {str(param_key): _normalize_param_entry(str(param_key), param_value) for param_key, param_value in params.items()}
    return out


def _normalize_set_spec(set_key: str, spec: Any, *, element_keys: set[str]) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise RuntimeError(f"invalid_stage3_behavior_set_spec: {set_key}")
    out = dict(spec)
    out["display_name"] = str(out.get("display_name", set_key) or set_key)
    out["scope"] = str(out.get("scope", "single-landmark") or "single-landmark")
    out["description"] = str(out.get("description", "") or "")
    out["landmark_count_default"] = int(out.get("landmark_count_default", 1) or 1)
    out["allow_revisit"] = bool(out.get("allow_revisit", False))
    if "multi_landmark_component" in out:
        out["multi_landmark_component"] = bool(out.get("multi_landmark_component", False))
    steps_out: list[dict[str, Any]] = []
    for idx, step in enumerate(list(out.get("element_steps", []) or [])):
        if not isinstance(step, dict):
            raise RuntimeError(f"invalid_stage3_behavior_step_spec: {set_key}[{idx}]")
        step_out = dict(step)
        element_class = str(step_out.get("element_class", "") or "").strip()
        if not element_class or element_class not in element_keys:
            raise RuntimeError(f"invalid_stage3_behavior_step_element: {set_key}[{idx}]={element_class}")
        step_out["element_class"] = element_class
        step_out["target_binding"] = str(step_out.get("target_binding", "primary") or "primary")
        params = dict(step_out.get("params", {}) or {})
        if "camera_mode" in params:
            params["camera_mode"] = _normalize_camera_mode_value(params.get("camera_mode"))
        step_out["params"] = params
        auto_rules = dict(step_out.get("auto_rules", {}) or {})
        if "camera_mode" in auto_rules and isinstance(auto_rules["camera_mode"], dict):
            auto_rules["camera_mode"] = dict(auto_rules["camera_mode"])
            if auto_rules["camera_mode"].get("mean", None) is not None:
                auto_rules["camera_mode"]["mean"] = _normalize_camera_mode_value(auto_rules["camera_mode"].get("mean"))
        step_out["auto_rules"] = auto_rules
        steps_out.append(step_out)
    out["element_steps"] = steps_out
    return out


def _load_behavior_libraries() -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    block = _load_behavior_library_config()
    shared_cfg = dict(block.get("shared", {}) or {})
    raw_elements = dict(block.get("elements", {}) or {})
    raw_sets = dict(block.get("sets", {}) or {})
    if not raw_elements or not raw_sets:
        raise RuntimeError("stage3_behavior_library_config_incomplete: missing elements or sets")
    elements = {str(key): _normalize_element_spec(str(key), value) for key, value in raw_elements.items()}
    element_keys = set(elements.keys())
    sets = {str(key): _normalize_set_spec(str(key), value, element_keys=element_keys) for key, value in raw_sets.items()}
    return shared_cfg, elements, sets


BEHAVIOR_SHARED_CONFIG, ELEMENT_LIBRARY, SET_LIBRARY = _load_behavior_libraries()

BEHAVIOR_SET = set(ELEMENT_LIBRARY.keys())
LOW_LEVEL_MODE_SPECS = ELEMENT_LIBRARY
MID_LEVEL_MODE_SPECS = {
    key: {
        "display_name": spec["display_name"],
        "behavior_id": key,
        "description": spec["description"],
        "param_options": dict(spec.get("params", {}) or {}),
    }
    for key, spec in ELEMENT_LIBRARY.items()
}


def _bbox_axes(bbox_3d: list[float]) -> tuple[float, float, float]:
    if len(bbox_3d) >= 6:
        sx = max(1e-3, float(bbox_3d[3]))
        sy = max(1e-3, float(bbox_3d[4]))
        sz = max(1e-3, float(bbox_3d[5]))
        return sx, sy, sz
    return 3.0, 3.0, 3.0


def _bbox_diag(bbox_3d: list[float]) -> float:
    sx, sy, sz = _bbox_axes(bbox_3d)
    return float(max(1.0, math.sqrt(sx * sx + sy * sy + sz * sz)))


def _bbox_xy_diag(bbox_3d: list[float]) -> float:
    sx, sy, _ = _bbox_axes(bbox_3d)
    return float(max(1.0, math.sqrt(sx * sx + sy * sy)))


def _yaw_rotation(yaw_deg: float) -> np.ndarray:
    yaw = math.radians(float(yaw_deg))
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.asarray([[c, -s], [s, c]], dtype=np.float32)


def _local_to_world(center: np.ndarray, yaw_deg: float, xy: np.ndarray, z: np.ndarray) -> np.ndarray:
    rot = _yaw_rotation(yaw_deg)
    xy_world = xy @ rot.T
    pts = np.zeros((xy.shape[0], 3), dtype=np.float32)
    pts[:, :2] = xy_world + center[:2].reshape(1, 2)
    pts[:, 2] = z.astype(np.float32)
    return pts


def _linspace_points(start: np.ndarray, end: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return end.reshape(1, 3)
    t = np.linspace(0.0, 1.0, num=n, dtype=np.float32).reshape(-1, 1)
    return (start.reshape(1, 3) * (1.0 - t) + end.reshape(1, 3) * t).astype(np.float32)


def _sample_discrete(value: Any, spec: dict[str, Any]) -> Any:
    if value is not None:
        return value
    if spec.get("choices"):
        return spec.get("default", list(spec["choices"])[0])
    default = spec.get("default")
    minimum = spec.get("min")
    maximum = spec.get("max")
    step = spec.get("step")
    if minimum is None or maximum is None or step is None:
        return default
    if isinstance(default, int) and isinstance(step, int):
        return int(default)
    return default


def _snap_numeric(value: float, spec: dict[str, Any]) -> float:
    minimum = spec.get("min")
    maximum = spec.get("max")
    step = spec.get("step")
    out = float(value)
    if minimum is not None:
        out = max(float(minimum), out)
    if maximum is not None:
        out = min(float(maximum), out)
    if step is not None and float(step) > 0:
        base = float(minimum if minimum is not None else 0.0)
        out = base + round((out - base) / float(step)) * float(step)
        if minimum is not None:
            out = max(float(minimum), out)
        if maximum is not None:
            out = min(float(maximum), out)
    return float(out)


def _apply_param_rules(element_key: str, landmark: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    bbox = list(landmark.get("bbox_3d", []) or [])
    diag_xy = _bbox_xy_diag(bbox)
    scale_anchor = _orbit_scale_anchor_from_bbox(bbox)
    out = dict(params)

    if element_key in {"circular_orbit", "square_orbit", "triangular_orbit", "figure8_orbit"}:
        extension = float(out.get("extension_m", 12.0) or 12.0)
        out["orbit_scale_anchor_m"] = float(scale_anchor)
        out["effective_radius_m"] = float(scale_anchor + extension)
        out["effective_side_m"] = float(diag_xy + 2.0 * extension)
    elif element_key == "spiral_orbit":
        start_ext = float(out.get("start_extension_m", 10.0) or 10.0)
        end_ext = float(out.get("end_extension_m", 24.0) or 24.0)
        end_ext = max(end_ext, start_ext + 2.0)
        out["orbit_scale_anchor_m"] = float(scale_anchor)
        out["effective_start_radius_m"] = float(scale_anchor + start_ext)
        out["effective_end_radius_m"] = float(scale_anchor + end_ext)
    elif element_key == "comet":
        extension = float(out.get("extension_m", 18.0) or 18.0)
        out["orbit_scale_anchor_m"] = float(scale_anchor)
        out["effective_entry_radius_m"] = float(scale_anchor + extension)
    elif element_key == "surface_mapping":
        sx = float(bbox[3]) if len(bbox) > 3 else 3.0
        sy = float(bbox[4]) if len(bbox) > 4 else 3.0
        ex = float(out.get("extension_x_m", 18.0) or 18.0)
        ey = float(out.get("extension_y_m", 18.0) or 18.0)
        out["effective_scan_width_m"] = float(max(4.0, sx + 2.0 * ex))
        out["effective_scan_height_m"] = float(max(4.0, sy + 2.0 * ey))

    return out


def build_element_instance(
    element_key: str,
    *,
    landmark: dict[str, Any],
    element_instance_id: str,
    seed: int,
    explicit_params: dict[str, Any] | None = None,
    target_binding: str = "primary",
    class_instance_index: int = 0,
) -> dict[str, Any]:
    if element_key not in ELEMENT_LIBRARY:
        raise ValueError(f"unsupported element: {element_key}")
    spec = ELEMENT_LIBRARY[element_key]
    params_spec = dict(spec.get("params", {}) or {})
    params: dict[str, Any] = {}
    raw_explicit = dict(explicit_params or {})
    for key, rule in params_spec.items():
        explicit = raw_explicit.get(key, None)
        if rule.get("choices"):
            params[key] = explicit if explicit is not None else rule.get("default")
        else:
            base = float(explicit if explicit is not None else rule.get("default", 0.0) or 0.0)
            params[key] = _snap_numeric(base, rule)
    for key, value in raw_explicit.items():
        if key not in params:
            params[key] = value
    params = _apply_param_rules(element_key, landmark, params)
    camera_mode = _normalize_camera_mode_value(params.get("camera_mode", spec.get("camera_mode_default", "landmark_track")))
    params["camera_mode"] = camera_mode
    display_name = str(spec.get("display_name", element_key))
    if camera_mode == "look_forward":
        display_name = f"{display_name}（向前看）"
    elif camera_mode == "landmark_track":
        display_name = f"{display_name}（跟随地标）"
    return {
        "element_instance_id": str(element_instance_id),
        "element_class": str(element_key),
        "element_display_name": display_name,
        "description": str(spec.get("description", "") or ""),
        "family": str(spec.get("family", "") or ""),
        "target_binding": str(target_binding or "primary"),
        "target_instance_id": str(landmark.get("instance_id", "") or ""),
        "params": params,
        "class_instance_index": int(class_instance_index),
        "seed": int(seed),
    }


def build_mid_level_event(
    mode_key: str,
    *,
    target_instance_id: str,
    event_index: int,
    seed: int,
    target_scale_m: float,
    explicit_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del target_scale_m
    dummy_landmark = {
        "instance_id": str(target_instance_id),
        "center_3d": [0.0, 0.0, 0.0],
        "bbox_3d": [0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 0.0],
    }
    element = build_element_instance(
        mode_key,
        landmark=dummy_landmark,
        element_instance_id=f"evt_{event_index:02d}",
        seed=seed,
        explicit_params=explicit_params,
        target_binding="primary",
        class_instance_index=event_index,
    )
    params = dict(element.get("params", {}) or {})
    display = str(element.get("element_display_name", mode_key))
    if "direction" in params:
        display += f"-{'顺时针' if str(params['direction']) == 'cw' else '逆时针'}"
    if "arc_deg" in params:
        display += f"-{int(round(float(params['arc_deg'])))}°"
    if "edge_count" in params:
        display += f"-{int(params['edge_count'])}边"
    return {
        "event_id": f"evt_{event_index:02d}",
        "event_label": display,
        "mode_key": mode_key,
        "mode_name": str(element.get("element_display_name", mode_key)),
        "behavior_id": mode_key,
        "target_instance_id": str(target_instance_id),
        "description": str(element.get("description", "") or ""),
        "params": params,
        "element_instance_id": str(element.get("element_instance_id", "")),
        "element_class": str(mode_key),
    }


def _project_velocity_aligned(points_xyz: np.ndarray, gaze_pitch_deg: float) -> np.ndarray:
    n = int(points_xyz.shape[0])
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    out = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        if i == 0 and n > 1:
            vec = points_xyz[1] - points_xyz[0]
        elif i == n - 1 and n > 1:
            vec = points_xyz[-1] - points_xyz[-2]
        else:
            vec = points_xyz[min(n - 1, i + 1)] - points_xyz[max(0, i - 1)]
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            vec = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            vec = (vec / norm).astype(np.float32)
        xy_norm = float(np.linalg.norm(vec[:2]))
        if xy_norm < 1e-6:
            xy = np.asarray([1.0, 0.0], dtype=np.float32)
        else:
            xy = (vec[:2] / xy_norm).astype(np.float32)
        pitch = math.radians(float(gaze_pitch_deg))
        out[i] = np.asarray(
            [
                float(math.cos(pitch)) * float(xy[0]),
                float(math.cos(pitch)) * float(xy[1]),
                float(math.sin(pitch)),
            ],
            dtype=np.float32,
        )
    return out


def _project_track_target(points_xyz: np.ndarray, target_center: np.ndarray, gaze_pitch_deg: float) -> np.ndarray:
    out = target_center.reshape(1, 3) - points_xyz
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    safe = np.where(norms < 1e-6, 1.0, norms)
    out = (out / safe).astype(np.float32)
    if abs(float(gaze_pitch_deg)) > 1e-6:
        desired_pitch = math.sin(math.radians(float(gaze_pitch_deg)))
        out[:, 2] = np.clip(out[:, 2] + float(desired_pitch) * 0.25, -0.98, 0.98)
        renorm = np.linalg.norm(out, axis=1, keepdims=True)
        out = (out / np.where(renorm < 1e-6, 1.0, renorm)).astype(np.float32)
    return out


def _sample_circle(
    center: np.ndarray,
    radius: float,
    z: np.ndarray,
    arc_deg: float,
    direction: str,
    n: int,
    start_phase_rad: float = 0.0,
) -> np.ndarray:
    sign = -1.0 if str(direction) == "cw" else 1.0
    phase = float(start_phase_rad) + np.linspace(0.0, math.radians(float(arc_deg)) * sign, num=n, dtype=np.float32)
    x = center[0] + float(radius) * np.cos(phase)
    y = center[1] + float(radius) * np.sin(phase)
    return np.stack([x, y, z.astype(np.float32)], axis=1).astype(np.float32)


def _sample_polygon(center: np.ndarray, side_m: float, edge_count: int, edges_per_loop: int, z: np.ndarray, rotation_deg: float, direction: str) -> np.ndarray:
    sides = max(3, int(edges_per_loop))
    radius = float(side_m) / max(1e-6, 2.0 * math.sin(math.pi / float(sides)))
    phase0 = math.radians(float(rotation_deg))
    idxs = list(range(sides))
    if str(direction) == "cw":
        idxs = list(reversed(idxs))
    vertices = []
    for idx in idxs:
        ang = phase0 + (2.0 * math.pi * float(idx) / float(sides))
        vertices.append([center[0] + radius * math.cos(ang), center[1] + radius * math.sin(ang)])
    vertices.append(vertices[0])
    total_edges = max(1, int(edge_count))
    pts_xy: list[np.ndarray] = []
    per_edge = max(4, int(math.ceil(float(z.shape[0]) / float(total_edges))))
    cur_edge = 0
    while cur_edge < total_edges:
        a = np.asarray(vertices[cur_edge % sides], dtype=np.float32)
        b = np.asarray(vertices[(cur_edge + 1) % sides], dtype=np.float32)
        t = np.linspace(0.0, 1.0, num=per_edge, dtype=np.float32).reshape(-1, 1)
        seg = a.reshape(1, 2) * (1.0 - t) + b.reshape(1, 2) * t
        if pts_xy:
            seg = seg[1:]
        pts_xy.append(seg)
        cur_edge += 1
    xy = np.vstack(pts_xy)
    if xy.shape[0] > z.shape[0]:
        keep = np.linspace(0, xy.shape[0] - 1, num=z.shape[0], endpoint=True).astype(np.int64)
        xy = xy[keep]
    if xy.shape[0] < z.shape[0]:
        pad = np.repeat(xy[-1:].copy(), repeats=(z.shape[0] - xy.shape[0]), axis=0)
        xy = np.vstack([xy, pad])
    return np.column_stack([xy, z.astype(np.float32)]).astype(np.float32)


def _roll_closed_path_to_start(pts: np.ndarray, start_pos: np.ndarray) -> np.ndarray:
    arr = np.asarray(pts, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] <= 1:
        return arr
    start = np.asarray(start_pos, dtype=np.float32).reshape(1, 3)
    idx = int(np.argmin(np.linalg.norm(arr - start, axis=1)))
    if idx <= 0:
        return arr
    return np.vstack([arr[idx:], arr[:idx]]).astype(np.float32)


def sample_waypoints(
    element_instance: dict[str, Any],
    landmark: dict[str, Any],
    start_pos: np.ndarray,
    num_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    element_key = str(element_instance.get("element_class", "") or "")
    if element_key not in ELEMENT_LIBRARY:
        raise ValueError(f"unsupported element: {element_key}")
    params = dict(element_instance.get("params", {}) or {})
    center = np.asarray(landmark.get("center_3d", [0.0, 0.0, 0.0]), dtype=np.float32)
    bbox = list(landmark.get("bbox_3d", []) or [0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 0.0])
    sx, sy, sz = _bbox_axes(bbox)
    yaw_deg = float(bbox[6]) if len(bbox) > 6 else 0.0
    n = max(16, int(num_points))
    p0 = start_pos.astype(np.float32)
    adaptive_altitude = params.get("adaptive_altitude_m", None)
    if adaptive_altitude is None:
        z_base = float(center[2] + 0.8 * sz + float(params.get("altitude_offset_m", 0.0) or 0.0))
    else:
        z_base = float(adaptive_altitude)
    start_vec_xy = p0[:2] - center[:2]
    start_phase = float(math.atan2(float(start_vec_xy[1]), float(start_vec_xy[0]))) if float(np.linalg.norm(start_vec_xy)) > 1e-4 else 0.0

    if element_key == "gradual_approach":
        dist = float(params["travel_distance_m"])
        descent = float(params["descent_m"])
        yaw_offset_deg = float(params["yaw_offset_deg"])
        adaptive_heading = params.get("adaptive_heading_deg", None)
        heading = math.radians(float(adaptive_heading if adaptive_heading is not None else (yaw_deg + yaw_offset_deg)))
        end_radius = float(params.get("adaptive_end_radius_m", max(8.0, dist * 0.25)) or max(8.0, dist * 0.25))
        end_altitude = float(params.get("adaptive_end_altitude_m", center[2] + 0.5 * sz) or (center[2] + 0.5 * sz))
        end = np.asarray([center[0] + math.cos(heading) * end_radius, center[1] + math.sin(heading) * end_radius, end_altitude], dtype=np.float32)
        pts = _linspace_points(p0, end, n)
    elif element_key == "gradual_depart":
        dist = float(params["travel_distance_m"])
        rise = float(params["rise_m"])
        yaw_offset_deg = float(params["yaw_offset_deg"])
        adaptive_heading = params.get("adaptive_heading_deg", None)
        heading = math.radians(float(adaptive_heading if adaptive_heading is not None else (yaw_deg + yaw_offset_deg + 180.0)))
        near_radius = float(params.get("start_radius_m", max(8.0, dist * 0.2)) or max(8.0, dist * 0.2))
        near = center + np.asarray([math.cos(heading) * near_radius, math.sin(heading) * near_radius, 0.6 * sz], dtype=np.float32)
        far = center + np.asarray([math.cos(heading) * dist, math.sin(heading) * dist, 0.8 * sz + rise], dtype=np.float32)
        if bool(params.get("adaptive_start_from_current", False)):
            pts = _linspace_points(p0, far, n)
        else:
            pts = _linspace_points(near if np.linalg.norm(p0 - near) > 3.0 else p0, far, n)
    elif element_key == "circular_orbit":
        radius = float(params["effective_radius_m"])
        z = np.full((n,), z_base, dtype=np.float32)
        pts = _sample_circle(
            center=center,
            radius=radius,
            z=z,
            arc_deg=float(params["arc_deg"]),
            direction=str(params["direction"]),
            n=n,
            start_phase_rad=start_phase,
        )
    elif element_key == "spiral_orbit":
        r0 = float(params["effective_start_radius_m"])
        r1 = float(params["effective_end_radius_m"])
        arc_deg = float(params["arc_deg"])
        direction = str(params["direction"])
        sign = -1.0 if direction == "cw" else 1.0
        phase = start_phase + np.linspace(0.0, math.radians(arc_deg) * sign, num=n, dtype=np.float32)
        radii = np.linspace(r0, r1, num=n, dtype=np.float32)
        z = np.linspace(z_base, z_base + float(params["rise_m"]), num=n, dtype=np.float32)
        x = center[0] + radii * np.cos(phase)
        y = center[1] + radii * np.sin(phase)
        pts = np.stack([x, y, z], axis=1).astype(np.float32)
    elif element_key == "square_orbit":
        z = np.full((n,), z_base, dtype=np.float32)
        pts = _sample_polygon(center=center, side_m=float(params["effective_side_m"]), edge_count=int(params["edge_count"]), edges_per_loop=4, z=z, rotation_deg=float(yaw_deg), direction=str(params["direction"]))
        pts = _roll_closed_path_to_start(pts, p0)
    elif element_key == "triangular_orbit":
        z = np.full((n,), z_base, dtype=np.float32)
        pts = _sample_polygon(center=center, side_m=float(params["effective_side_m"]), edge_count=int(params["edge_count"]), edges_per_loop=3, z=z, rotation_deg=float(params.get("rotation_deg", 0.0)), direction=str(params["direction"]))
        pts = _roll_closed_path_to_start(pts, p0)
    elif element_key == "figure8_orbit":
        radius = float(params["effective_radius_m"])
        cycles = max(1, int(params["cycles"]))
        phase = np.linspace(0.0, 2.0 * math.pi * float(cycles), num=n, dtype=np.float32)
        x = center[0] + radius * np.sin(phase)
        y = center[1] + 0.5 * radius * np.sin(2.0 * phase)
        z = np.full((n,), z_base, dtype=np.float32)
        pts = np.stack([x, y, z], axis=1).astype(np.float32)
        pts = _roll_closed_path_to_start(pts, p0)
    elif element_key == "sky_rise":
        radius = float(params.get("top_extension_m", 2.0) or 2.0)
        scale_anchor = _orbit_scale_anchor_from_bbox(bbox)
        dir_xy = start_vec_xy.copy()
        dir_xy_norm = float(np.linalg.norm(dir_xy))
        if dir_xy_norm < 1e-4:
            dir_xy = np.asarray([1.0, 0.0], dtype=np.float32)
        else:
            dir_xy = (dir_xy / dir_xy_norm).astype(np.float32)
        near_radius = max(3.0, float(scale_anchor) + 1.5)
        near_xy = center[:2] + dir_xy * float(near_radius)
        near = np.asarray(
            [
                float(near_xy[0]),
                float(near_xy[1]),
                float(center[2] + 0.6 * sz),
            ],
            dtype=np.float32,
        )
        top_xy = center[:2] + dir_xy * float(radius)
        top = np.asarray(
            [
                float(top_xy[0]),
                float(top_xy[1]),
                float(center[2] + 0.9 * sz + float(params["rise_m"])),
            ],
            dtype=np.float32,
        )
        bridge_n = max(8, n // 4)
        ascent_n = max(12, n - bridge_n + 1)
        bridge = _linspace_points(p0, near, bridge_n)
        ascent = _linspace_points(near, top, ascent_n)
        pts = np.vstack([bridge[:-1], ascent]).astype(np.float32)
        if pts.shape[0] > n:
            keep = np.linspace(0, pts.shape[0] - 1, num=n, endpoint=True).astype(np.int64)
            pts = pts[keep]
    elif element_key == "comet":
        semi_major = float(params["semi_major_m"])
        eccentricity = float(params["eccentricity"])
        entry_r = float(params["effective_entry_radius_m"])
        arc_deg = float(params["arc_deg"])
        theta = np.linspace(0.0, math.radians(arc_deg), num=n, dtype=np.float32)
        semi_minor = semi_major * max(0.1, math.sqrt(max(0.01, 1.0 - eccentricity * eccentricity)))
        x = center[0] + (entry_r + semi_major) * np.cos(theta)
        y = center[1] + semi_minor * np.sin(theta)
        z = np.linspace(z_base, z_base + 0.2 * sz, num=n, dtype=np.float32)
        half = np.stack([x, y, z], axis=1).astype(np.float32)
        if bool(params.get("adaptive_start_from_current", False)):
            bridge = _linspace_points(p0, half[0], max(8, n // 4))
            half = np.vstack([bridge[:-1], half]).astype(np.float32)
        return_path = _linspace_points(half[-1], p0, max(8, n // 3))
        pts = np.vstack([half, return_path[1:]]).astype(np.float32)
        if pts.shape[0] > n:
            keep = np.linspace(0, pts.shape[0] - 1, num=n, endpoint=True).astype(np.int64)
            pts = pts[keep]
    elif element_key == "surface_mapping":
        width_m = float(params["effective_scan_width_m"])
        height_m = float(params["effective_scan_height_m"])
        lane_count = max(3, int(params["lane_count"]))
        x_vals = np.linspace(-0.5 * width_m, 0.5 * width_m, num=max(12, n // max(1, lane_count)), dtype=np.float32)
        y_vals = np.linspace(-0.5 * height_m, 0.5 * height_m, num=lane_count, dtype=np.float32)
        pts_local: list[list[float]] = []
        for idx, y in enumerate(y_vals):
            xs = x_vals if (idx % 2 == 0) else x_vals[::-1]
            for x in xs:
                pts_local.append([float(x), float(y), float(z_base)])
        pts_local_arr = np.asarray(pts_local, dtype=np.float32)
        if pts_local_arr.shape[0] > n:
            keep = np.linspace(0, pts_local_arr.shape[0] - 1, num=n, endpoint=True).astype(np.int64)
            pts_local_arr = pts_local_arr[keep]
        elif pts_local_arr.shape[0] < n and pts_local_arr.shape[0] > 1:
            keep = np.linspace(0, pts_local_arr.shape[0] - 1, num=n, endpoint=True).astype(np.float32)
            interp = np.zeros((n, 3), dtype=np.float32)
            for axis in range(3):
                interp[:, axis] = np.interp(keep, np.arange(pts_local_arr.shape[0], dtype=np.float32), pts_local_arr[:, axis])
            pts_local_arr = interp
        rot = _yaw_rotation(yaw_deg)
        xy = pts_local_arr[:, :2] @ rot.T
        pts = np.zeros((pts_local_arr.shape[0], 3), dtype=np.float32)
        pts[:, :2] = xy + center[:2].reshape(1, 2)
        pts[:, 2] = pts_local_arr[:, 2]
    else:
        raise ValueError(f"unsupported element: {element_key}")

    camera_mode = _normalize_camera_mode_value(params.get("camera_mode", ELEMENT_LIBRARY[element_key].get("camera_mode_default", "landmark_track")))
    gaze_pitch_deg = float(params.get("gaze_pitch_deg", -15.0) or -15.0)
    if camera_mode == "look_forward":
        forwards = _project_velocity_aligned(pts, gaze_pitch_deg=gaze_pitch_deg)
    else:
        forwards = _project_track_target(pts, target_center=center, gaze_pitch_deg=gaze_pitch_deg)
    return pts.astype(np.float32), forwards.astype(np.float32)
