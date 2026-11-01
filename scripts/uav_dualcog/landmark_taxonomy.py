from __future__ import annotations

from collections.abc import Mapping, Sequence


LANDMARK_TAXONOMY: Mapping[str, tuple[str, ...]] = {
    "building": (
        "low-rise building",
        "mid-rise building",
        "high-rise building",
        "warehouse",
        "pagoda",
        "chapel",
        "factory shed",
        "rural farmhouse",
        "glass skyscraper",
        "brick schoolhouse",
    ),
    "vehicle": (
        "sedan car",
        "delivery van",
        "city bus",
        "cargo truck",
        "motorcycle",
        "construction excavator",
    ),
    "public_facility": (
        "street lamp post",
        "bus shelter",
        "public bench",
        "trash bin",
        "fire hydrant",
        "antenna",
    ),
    "urban_landscape": (
        "sculpture",
        "plaza",
        "fountain",
        "city square",
        "urban garden",
        "landscape installation",
        "signboard",
        "billboard",
        "advertising board",
        "wayfinding sign",
        "landmark signage",
    ),
    "transport_infrastructure": (
        "arch bridge",
        "overpass",
        "railway track",
        "tunnel entrance",
        "roundabout",
        "pedestrian crosswalk",
        "subway station entrance",
        "traffic island",
        "railway platform",
    ),
    "industrial_infrastructure": (
        "shipping container stack",
        "oil storage tank",
        "grain silo",
        "crane tower",
        "pipeline",
    ),
    "vegetation": (
        "deciduous tree",
        "coniferous tree",
        "palm tree",
        "shrub cluster",
        "hedge row",
        "grassy lawn",
    ),
    "other": (
        "temporary tent",
        "rubble pile",
        "playground slide",
        "inflatable archway",
        "construction barrier",
    ),
}

CAMPUS_CATEGORY_PRIORITY: tuple[str, ...] = (
    "building",
    "public_facility",
    "urban_landscape",
    "transport_infrastructure",
    "vegetation",
    "industrial_infrastructure",
    "vehicle",
    "other",
)

STABILITY_SENSITIVE_CATEGORIES: frozenset[str] = frozenset({"vehicle", "vegetation", "other"})


def category_names() -> tuple[str, ...]:
    return tuple(LANDMARK_TAXONOMY.keys())


def normalize_category(value: object) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "public_facilities": "public_facility",
        "urban_landscapes": "urban_landscape",
        "transportation_infrastructure": "transport_infrastructure",
        "industrial_facility": "industrial_infrastructure",
        "industrial_facilities": "industrial_infrastructure",
        "plant": "vegetation",
        "plants": "vegetation",
    }
    return aliases.get(text, text)


def format_category_candidates() -> str:
    return "[" + ", ".join(category_names()) + "]"


def format_subcategory_examples(*, categories: Sequence[str] | None = None) -> str:
    selected = tuple(categories or category_names())
    lines = []
    for category in selected:
        examples = LANDMARK_TAXONOMY.get(category, ())
        lines.append(f"  • {category}: {', '.join(examples)}")
    return "\n".join(lines)
