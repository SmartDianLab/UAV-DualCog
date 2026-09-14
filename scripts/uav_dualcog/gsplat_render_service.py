#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _require_runtime() -> tuple[Any, Any, Any]:
    try:
        import torch
        from gsplat.rendering import rasterization
        from plyfile import PlyData
    except ImportError as exc:
        raise RuntimeError(
            "gsplat service requires torch, gsplat, and plyfile; install the isolated server environment first"
        ) from exc
    return torch, rasterization, PlyData


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _numeric_suffix(name: str) -> int:
    try:
        return int(name.rsplit("_", 1)[-1])
    except ValueError:
        return 0


def _euler_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    yaw, pitch, roll = np.radians([yaw_deg, pitch_deg, roll_deg])
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float32)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float32)
    return rz @ ry @ rx


def _pose_to_c2w(pose: dict[str, Any]) -> np.ndarray:
    # UAV body: x forward, y left, z up. Camera: x right, y down, z forward.
    body_to_camera = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float32)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = _euler_matrix(
        float(pose.get("yaw", 0.0)),
        float(pose.get("pitch", 0.0)),
        float(pose.get("roll", 0.0)),
    ) @ body_to_camera
    c2w[:3, 3] = [float(pose["x"]), float(pose["y"]), float(pose["z"])]
    return c2w


def _load_geometry(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        points = np.load(path)
    elif path.suffix.lower() == ".npz":
        with np.load(path) as payload:
            key = "points" if "points" in payload else payload.files[0]
            points = payload[key]
    else:
        _, _, PlyData = _require_runtime()
        vertex = PlyData.read(str(path))["vertex"]
        points = np.column_stack([vertex[axis] for axis in ("x", "y", "z")])
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"invalid geometry shape: {points.shape}")
    return points[:, :3]


@dataclass
class LoadedScene:
    scene_id: str
    manifest_path: Path
    manifest: dict[str, Any]
    backend: str
    geometry: np.ndarray
    world_to_gs: np.ndarray
    gaussian_count: int
    means: Any = None
    quats: Any = None
    scales: Any = None
    opacities: Any = None
    colors: Any = None
    sh_degree: int | None = None
    native_model: Any = None
    native_renderer: Any = None
    native_cameras_type: Any = None
    native_background: Any = None


def _resolve_manifest_path(root: Path, value: str, *, asset_root: Path | None = None) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if expanded.is_absolute():
        return expanded.resolve()
    return ((asset_root or root) / expanded).resolve()


def _camera_pose_in_gs_frame(c2w_world: np.ndarray, world_to_gs: np.ndarray) -> np.ndarray:
    linear = np.asarray(world_to_gs[:3, :3], dtype=np.float32)
    determinant = float(np.linalg.det(linear))
    scale = abs(determinant) ** (1.0 / 3.0)
    if scale <= 1e-8:
        raise ValueError("T_world_to_gs has a singular linear transform")
    rotation = linear / scale
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3):
        raise ValueError("T_world_to_gs must be a rigid similarity transform")
    c2w_gs = np.eye(4, dtype=np.float32)
    c2w_gs[:3, :3] = rotation @ np.asarray(c2w_world[:3, :3], dtype=np.float32)
    c2w_gs[:3, 3] = linear @ np.asarray(c2w_world[:3, 3], dtype=np.float32) + world_to_gs[:3, 3]
    return c2w_gs


def _world_to_gs_scale(world_to_gs: np.ndarray) -> float:
    linear = np.asarray(world_to_gs[:3, :3], dtype=np.float64)
    scale = abs(float(np.linalg.det(linear))) ** (1.0 / 3.0)
    if scale <= 1e-8:
        raise ValueError("T_world_to_gs has a singular linear transform")
    return scale


class SceneStore:
    def __init__(self, device: str) -> None:
        self.device_name = device
        self.scene: LoadedScene | None = None

    def load(self, manifest_path: Path) -> LoadedScene:
        manifest_path = manifest_path.expanduser().resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        root = manifest_path.parent
        asset_root_value = str(manifest.get("asset_root", "") or "")
        asset_root = _resolve_manifest_path(root, asset_root_value) if asset_root_value else root
        splat_path = _resolve_manifest_path(root, str(manifest["gs_checkpoint"]), asset_root=asset_root)
        geometry_path = _resolve_manifest_path(root, str(manifest["geometry_path"]), asset_root=asset_root)
        backend = str(manifest.get("renderer_backend", "standard_3dgs_ply") or "standard_3dgs_ply").strip().lower()
        world_to_gs = np.asarray(manifest.get("T_world_to_gs", np.eye(4)), dtype=np.float32)
        if world_to_gs.shape != (4, 4):
            raise ValueError("T_world_to_gs must be a 4x4 matrix")
        geometry = _load_geometry(geometry_path)
        if backend == "citygaussian_v2_checkpoint":
            return self._load_citygaussian(
                manifest=manifest,
                manifest_path=manifest_path,
                root=root,
                splat_path=splat_path,
                geometry=geometry,
                world_to_gs=world_to_gs,
            )
        if backend != "standard_3dgs_ply":
            raise ValueError(f"unsupported renderer_backend: {backend}")

        torch, _, PlyData = _require_runtime()
        if splat_path.suffix.lower() != ".ply":
            raise ValueError("gs_checkpoint must be a standard 3DGS PLY export")
        vertex = PlyData.read(str(splat_path))["vertex"]
        names = set(vertex.data.dtype.names or ())
        required = {"x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"}
        missing = sorted(required - names)
        if missing:
            raise ValueError(f"3DGS PLY is missing properties: {missing}")

        means_np = np.column_stack([vertex[axis] for axis in ("x", "y", "z")]).astype(np.float32)
        scales_np = np.exp(np.column_stack([vertex[f"scale_{i}"] for i in range(3)])).astype(np.float32)
        quats_np = np.column_stack([vertex[f"rot_{i}"] for i in range(4)]).astype(np.float32)
        quats_np /= np.maximum(np.linalg.norm(quats_np, axis=1, keepdims=True), 1e-8)
        opacities_np = _sigmoid(np.asarray(vertex["opacity"], dtype=np.float32))

        dc_names = sorted((name for name in names if name.startswith("f_dc_")), key=_numeric_suffix)
        rest_names = sorted((name for name in names if name.startswith("f_rest_")), key=_numeric_suffix)
        if len(dc_names) == 3:
            dc = np.column_stack([vertex[name] for name in dc_names]).astype(np.float32)[:, None, :]
            if rest_names and len(rest_names) % 3 == 0:
                rest = np.column_stack([vertex[name] for name in rest_names]).astype(np.float32)
                rest = rest.reshape(len(rest), 3, -1).transpose(0, 2, 1)
                colors_np = np.concatenate([dc, rest], axis=1)
            else:
                colors_np = dc
            sh_degree = int(round(math.sqrt(colors_np.shape[1]) - 1))
        elif {"red", "green", "blue"}.issubset(names):
            colors_np = np.column_stack([vertex[name] for name in ("red", "green", "blue")]).astype(np.float32) / 255.0
            sh_degree = None
        else:
            raise ValueError("3DGS PLY must contain f_dc_* coefficients or RGB colors")

        device = torch.device(self.device_name if torch.cuda.is_available() else "cpu")
        scene = LoadedScene(
            scene_id=str(manifest["scene_id"]),
            manifest_path=manifest_path,
            manifest=manifest,
            backend=backend,
            geometry=geometry,
            world_to_gs=world_to_gs,
            gaussian_count=int(len(means_np)),
            means=torch.from_numpy(means_np).to(device),
            quats=torch.from_numpy(quats_np).to(device),
            scales=torch.from_numpy(scales_np).to(device),
            opacities=torch.from_numpy(opacities_np).to(device),
            colors=torch.from_numpy(colors_np).to(device),
            sh_degree=sh_degree,
        )
        self.scene = scene
        return scene

    def _load_citygaussian(
        self,
        *,
        manifest: dict[str, Any],
        manifest_path: Path,
        root: Path,
        splat_path: Path,
        geometry: np.ndarray,
        world_to_gs: np.ndarray,
    ) -> LoadedScene:
        citygaussian_value = str(manifest.get("citygaussian_root", "") or "")
        if not citygaussian_value:
            raise ValueError("citygaussian_root is required for citygaussian_v2_checkpoint")
        citygaussian_root = _resolve_manifest_path(root, citygaussian_value)
        if not (citygaussian_root / "internal").is_dir():
            raise FileNotFoundError(f"invalid CityGaussianV2 root: {citygaussian_root}")
        if str(citygaussian_root) not in sys.path:
            sys.path.insert(0, str(citygaussian_root))
        try:
            import torch
            from internal.cameras.cameras import Cameras
            from internal.utils.gaussian_model_loader import GaussianModelLoader
        except ImportError as exc:
            raise RuntimeError("failed to import the configured CityGaussianV2 runtime") from exc

        device = torch.device(self.device_name if torch.cuda.is_available() else "cpu")
        loadable = GaussianModelLoader.search_load_file(str(splat_path))
        if not str(loadable).lower().endswith(".ckpt"):
            raise ValueError(f"CityGaussianV2 asset must resolve to a .ckpt file: {loadable}")
        model, renderer, checkpoint = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
            str(loadable),
            device=device,
            eval_mode=True,
            pre_activate=True,
        )
        model.freeze()
        background = checkpoint.get("hyper_parameters", {}).get("background_color", [0.0, 0.0, 0.0])
        background_tensor = torch.tensor(background, dtype=torch.float32, device=device)
        xyz = getattr(model, "get_xyz", None)
        if callable(xyz):
            xyz = xyz()
        gaussian_count = int(xyz.shape[0]) if xyz is not None else int(manifest.get("gaussian_count", 0) or 0)
        scene = LoadedScene(
            scene_id=str(manifest["scene_id"]),
            manifest_path=manifest_path,
            manifest=manifest,
            backend="citygaussian_v2_checkpoint",
            geometry=geometry,
            world_to_gs=world_to_gs,
            gaussian_count=gaussian_count,
            native_model=model,
            native_renderer=renderer,
            native_cameras_type=Cameras,
            native_background=background_tensor,
        )
        self.scene = scene
        return scene


def _native_camera(scene: LoadedScene, c2w_gs: np.ndarray, intrinsics: np.ndarray, width: int, height: int) -> Any:
    import torch

    w2c = np.linalg.inv(c2w_gs).astype(np.float32)
    device = scene.native_background.device
    appearance_id = int(scene.manifest.get("appearance_id", 0) or 0)
    cameras = scene.native_cameras_type(
        R=torch.from_numpy(w2c[None, :3, :3]),
        T=torch.from_numpy(w2c[None, :3, 3]),
        fx=torch.tensor([float(intrinsics[0, 0])], dtype=torch.float32),
        fy=torch.tensor([float(intrinsics[1, 1])], dtype=torch.float32),
        cx=torch.tensor([float(intrinsics[0, 2])], dtype=torch.float32),
        cy=torch.tensor([float(intrinsics[1, 2])], dtype=torch.float32),
        width=torch.tensor([int(width)], dtype=torch.int16),
        height=torch.tensor([int(height)], dtype=torch.int16),
        appearance_id=torch.tensor([appearance_id], dtype=torch.long),
        normalized_appearance_id=torch.tensor([float(scene.manifest.get("normalized_appearance_id", 0.0) or 0.0)]),
        distortion_params=None,
        camera_type=torch.zeros(1, dtype=torch.int32),
    )
    return cameras[0].to_device(device)


def _native_render(scene: LoadedScene, c2w_gs: np.ndarray, intrinsics: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    import torch

    camera = _native_camera(scene, c2w_gs, intrinsics, width, height)
    with torch.inference_mode():
        outputs = scene.native_renderer(camera, scene.native_model, scene.native_background)
    if not isinstance(outputs, dict) or "render" not in outputs:
        raise RuntimeError("CityGaussianV2 renderer did not return an RGB 'render' output")

    def as_image(value: Any) -> np.ndarray:
        array = value.detach().float().cpu().numpy()
        if array.ndim == 3 and array.shape[0] in {1, 3, 4}:
            array = np.moveaxis(array, 0, -1)
        return np.asarray(array)

    rgb = as_image(outputs["render"])[..., :3]
    depth = None
    for key in ("surf_depth", "depth", "expected_depth", "rendered_depth", "acc_depth"):
        if key in outputs:
            depth = np.squeeze(as_image(outputs[key])).astype(np.float32)
            break
    alpha = None
    for key in ("alpha", "rend_alpha", "accumulation", "opacity"):
        if key in outputs:
            alpha = np.squeeze(as_image(outputs[key])).astype(np.float32)
            break
    return rgb, depth, alpha


def create_app(default_manifest: str = "", device: str = "cuda:0") -> Any:
    try:
        from flask import Flask, Response, jsonify, request
    except ImportError as exc:
        raise RuntimeError("gsplat render service requires Flask") from exc
    app = Flask(__name__)
    store = SceneStore(device=device)
    if default_manifest:
        store.load(Path(default_manifest))

    @app.get("/health")
    def health() -> Any:
        return jsonify({"ok": True, "scene_id": store.scene.scene_id if store.scene else None})

    @app.post("/scene/load")
    def load_scene() -> Any:
        payload = request.get_json(force=True) or {}
        path = str(payload.get("manifest_path", default_manifest) or "")
        if not path:
            return jsonify({"ok": False, "error": "manifest_path is required"}), 400
        try:
            scene = store.load(Path(path))
            return jsonify(
                {
                    "ok": True,
                    "scene_id": scene.scene_id,
                    "backend": scene.backend,
                    "gaussians": int(scene.gaussian_count),
                }
            )
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.get("/geometry")
    def geometry() -> Any:
        if store.scene is None:
            return jsonify({"ok": False, "error": "no scene loaded"}), 409
        buffer = io.BytesIO()
        np.savez_compressed(buffer, points=store.scene.geometry)
        return Response(buffer.getvalue(), mimetype="application/octet-stream")

    @app.post("/render")
    def render() -> Any:
        if store.scene is None:
            return jsonify({"ok": False, "error": "no scene loaded"}), 409
        payload = request.get_json(force=True) or {}
        camera = payload.get("camera", {}) or {}
        width, height = int(camera["width"]), int(camera["height"])
        intrinsics = np.array(
            [[float(camera["fx"]), 0, float(camera["cx"])], [0, float(camera["fy"]), float(camera["cy"])], [0, 0, 1]],
            dtype=np.float32,
        )
        if "camera_to_world" in payload:
            c2w_world = np.asarray(payload["camera_to_world"], dtype=np.float32)
        else:
            c2w_world = _pose_to_c2w(payload["pose_enu"])
        if c2w_world.shape != (4, 4):
            return jsonify({"ok": False, "error": "camera_to_world must be a 4x4 matrix"}), 400
        c2w_gs = _camera_pose_in_gs_frame(c2w_world, store.scene.world_to_gs)
        if store.scene.backend == "citygaussian_v2_checkpoint":
            rgb_np, depth_np, alpha_np = _native_render(store.scene, c2w_gs, intrinsics, width, height)
        else:
            torch, rasterization, _ = _require_runtime()
            viewmat = np.linalg.inv(c2w_gs).astype(np.float32)
            device_obj = store.scene.means.device
            renders, alphas, _ = rasterization(
                store.scene.means,
                store.scene.quats,
                store.scene.scales,
                store.scene.opacities,
                store.scene.colors,
                torch.from_numpy(viewmat[None]).to(device_obj),
                torch.from_numpy(intrinsics[None]).to(device_obj),
                width,
                height,
                sh_degree=store.scene.sh_degree,
                render_mode="RGB+ED",
            )
            render_np = renders[0].detach().float().cpu().numpy()
            rgb_np = render_np[..., :3]
            depth_np = render_np[..., 3].astype(np.float32)
            alpha_np = alphas[0, ..., 0].detach().float().cpu().numpy()
        mode = str(payload.get("mode", "rgb")).lower()
        if mode == "depth":
            if depth_np is None:
                return jsonify({"ok": False, "error": "renderer does not expose a metric depth output"}), 501
            buffer = io.BytesIO()
            # Rasterizers return depth in the Gaussian coordinate frame. Restore
            # metric-world units so back-projection agrees with GauUscene V2.1.
            depth_world = np.asarray(depth_np, dtype=np.float32) / _world_to_gs_scale(store.scene.world_to_gs)
            np.savez_compressed(buffer, depth=depth_world)
            return Response(buffer.getvalue(), mimetype="application/octet-stream")
        if mode == "seg":
            if alpha_np is None:
                alpha_np = np.ones((height, width), dtype=np.float32)
            mask = np.where(alpha_np >= float(store.scene.manifest.get("alpha_threshold", 0.05)), 255, 0).astype(np.uint8)
            ok, encoded = cv2.imencode(".png", mask)
        else:
            rgb = np.clip(rgb_np * 255.0, 0, 255).astype(np.uint8)
            ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            return jsonify({"ok": False, "error": "image encoding failed"}), 500
        return Response(encoded.tobytes(), mimetype="image/png" if mode == "seg" else "image/jpeg")

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Headless gsplat render service for UAV-DualCog")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    create_app(default_manifest=args.manifest, device=args.device).run(host=args.host, port=args.port, threaded=False)


if __name__ == "__main__":
    main()
