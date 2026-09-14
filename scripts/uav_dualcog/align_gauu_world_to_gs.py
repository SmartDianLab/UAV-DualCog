#!/usr/bin/env python3
"""Estimate the GauUscene metric-world to CityGaussian similarity transform."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def _load_points(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npz":
        payload = np.load(path)
        key = "points" if "points" in payload else payload.files[0]
        points = payload[key]
    else:
        points = np.load(path, mmap_mode="r")
    points = np.asarray(points[:, :3], dtype=np.float64)
    return points[np.isfinite(points).all(axis=1)]


def _sample(points: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    if len(points) <= count:
        return np.asarray(points)
    return np.asarray(points[rng.choice(len(points), count, replace=False)])


def _robust_core(points: np.ndarray, quantile: float = 0.005) -> np.ndarray:
    lo = np.quantile(points, quantile, axis=0)
    hi = np.quantile(points, 1.0 - quantile, axis=0)
    keep = ((points >= lo) & (points <= hi)).all(axis=1)
    return points[keep]


def _principal_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.median(points, axis=0)
    centered = points - center
    covariance = np.cov(centered.T)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    return center, vectors[:, order], values[order]


def _apply(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return scale * (points @ rotation.T) + translation


def _umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ correction @ vt
    source_variance = np.mean(np.sum(source_centered**2, axis=1))
    scale = float(np.trace(np.diag(singular) @ correction) / max(source_variance, 1e-12))
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def _score(points: np.ndarray, tree: cKDTree) -> tuple[float, float, float]:
    distances, _ = tree.query(points, k=1, workers=-1)
    return tuple(float(v) for v in np.quantile(distances, [0.5, 0.8, 0.95]))


def estimate_transform(
    world: np.ndarray,
    gaussians: np.ndarray,
    *,
    seed: int,
    sample_count: int,
    target_count: int,
    iterations: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    source = _robust_core(_sample(world, sample_count * 3, rng))
    source = _sample(source, sample_count, rng)
    target = _robust_core(_sample(gaussians, target_count * 3, rng))
    target = _sample(target, target_count, rng)
    source_center, source_axes, source_values = _principal_frame(source)
    target_center, target_axes, target_values = _principal_frame(target)
    initial_scale = float(np.median(np.sqrt(target_values / source_values)))
    target_tree = cKDTree(target)

    candidates: list[tuple[tuple[float, float, float], float, np.ndarray, np.ndarray]] = []
    for signs in itertools.product((-1.0, 1.0), repeat=3):
        sign_matrix = np.diag(signs)
        rotation = target_axes @ sign_matrix @ source_axes.T
        if np.linalg.det(rotation) < 0:
            continue
        translation = target_center - initial_scale * (rotation @ source_center)
        transformed = _apply(source, initial_scale, rotation, translation)
        candidates.append((_score(transformed, target_tree), initial_scale, rotation, translation))
    candidates.sort(key=lambda item: item[0][1])
    initial_score, scale, rotation, translation = candidates[0]

    history: list[dict[str, object]] = []
    for iteration in range(iterations):
        transformed = _apply(source, scale, rotation, translation)
        distances, indices = target_tree.query(transformed, k=1, workers=-1)
        threshold = float(np.quantile(distances, max(0.35, 0.80 - 0.04 * iteration)))
        keep = distances <= threshold
        delta_scale, delta_rotation, delta_translation = _umeyama(transformed[keep], target[indices[keep]])
        scale = delta_scale * scale
        rotation = delta_rotation @ rotation
        translation = delta_scale * (delta_rotation @ translation) + delta_translation
        current_score = _score(_apply(source, scale, rotation, translation), target_tree)
        history.append(
            {
                "iteration": iteration + 1,
                "pair_count": int(keep.sum()),
                "threshold_gs": threshold,
                "median_p80_p95_gs": list(current_score),
            }
        )

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = translation
    final_score = _score(_apply(source, scale, rotation, translation), target_tree)
    return {
        "method": "robust_pca_similarity_then_trimmed_icp",
        "seed": seed,
        "world_sample_count": int(len(source)),
        "gaussian_sample_count": int(len(target)),
        "scale_world_to_gs": scale,
        "rotation_world_to_gs": rotation.tolist(),
        "translation_world_to_gs": translation.tolist(),
        "T_world_to_gs": matrix.tolist(),
        "initial_median_p80_p95_gs": list(initial_score),
        "final_median_p80_p95_gs": list(final_score),
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-points", required=True, type=Path)
    parser.add_argument("--gaussian-points", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--sample-count", type=int, default=120_000)
    parser.add_argument("--target-count", type=int, default=300_000)
    parser.add_argument("--iterations", type=int, default=14)
    args = parser.parse_args()

    result = estimate_transform(
        _load_points(args.world_points),
        _load_points(args.gaussian_points),
        seed=args.seed,
        sample_count=args.sample_count,
        target_count=args.target_count,
        iterations=args.iterations,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("scale_world_to_gs", "initial_median_p80_p95_gs", "final_median_p80_p95_gs")}, indent=2))


if __name__ == "__main__":
    main()
