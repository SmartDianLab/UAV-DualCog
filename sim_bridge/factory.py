from __future__ import annotations

from typing import Optional

from .base import BaseSimBridge


def create_bridge(engine: str, scene_id: str, config: Optional[dict] = None) -> BaseSimBridge:
    engine_normalized = engine.lower()

    if engine_normalized == "airsim":
        from .airsim_bridge import AirSimBridge

        return AirSimBridge(scene_id=scene_id, config=config)
    if engine_normalized in {"unrealcv", "ue", "ue5"}:
        from .unrealcv_bridge import UnrealCVBridge

        return UnrealCVBridge(scene_id=scene_id, config=config)
    if engine_normalized in {"sibr", "gs", "3dgs"}:
        from .sibr_bridge import SIBRBridge

        return SIBRBridge(scene_id=scene_id, config=config)
    if engine_normalized == "carla":
        from .carla_bridge import CarlaBridge

        return CarlaBridge(scene_id=scene_id, config=config)

    raise ValueError(f"Unsupported engine: {engine}")
