#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_PYTHON_BIN = "/home/junhuan/anaconda3/envs/flight-mvstg/bin/python"
DEFAULT_CONFIG_DIR = ROOT_DIR / "configs" / "flightmvstg"
DEFAULT_CLEANUP_PATTERNS: list[str] = []
DEFAULT_PROBE_SOURCE = "hybrid"
DEFAULT_PROBE_WORKERS = 6
DEFAULT_PROBE_HALF_SPAN = 40.0
DEFAULT_PROBE_LATERAL_STEP = 40.0
DEFAULT_PROBE_COARSE_STEP = 50.0
DEFAULT_PROBE_REFINE_TOL = 2.0
DEFAULT_PROBE_MAX_STEPS = 50
DEFAULT_PROBE_SETTLE_SEC = 0.02
DEFAULT_PROBE_SURFACE_PERCENTILE = 5.0
DEFAULT_PROBE_SURFACE_LOCAL_XY_RADIUS = 120.0
DEFAULT_PROBE_SURFACE_Z_MARGIN = 8.0
DEFAULT_PROBE_SURFACE_ROBUST_LOW_PERCENTILE = 35.0
DEFAULT_PROBE_SURFACE_CENTER_BIAS_MAX_ABOVE_LOW = 3.0
DEFAULT_PROBE_BOUNDARY_LIDAR_MAX_RANGE = 50.0
DEFAULT_PROBE_BOUNDARY_MIN_POINTS_PER_YAW = 250
DEFAULT_PROBE_BOUNDARY_MIN_VALID_YAWS = 2
DEFAULT_PROBE_BOUNDARY_SEG_MIN_POINTS_PER_ID = 50
DEFAULT_PROBE_BOUNDARY_SEG_STOP_MAX_DISTINCT_IDS = 4
DEFAULT_PROBE_BOUNDARY_SEG_STOP_MAX_TOTAL_POINTS = 800
DEFAULT_PROBE_XY_BOUNDARY_QUANTILE = 0.2
DEFAULT_STAGE_HEARTBEAT_SEC = 15.0


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_scene_id(raw: str) -> str:
    value = str(raw or "").strip()
    if not value:
        raise ValueError("scene id is empty")
    if value.startswith("env_"):
        return value
    if value.isdigit():
        return f"env_{int(value)}"
    return value


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(item, ensure_ascii=False) + "\n")


def _write_process_snapshot(log_file: Any, pid: int) -> None:
    try:
        ps_proc = subprocess.run(
            ["ps", "-fp", str(int(pid))],
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
            check=False,
        )
        output = (ps_proc.stdout or ps_proc.stderr or "").strip()
        if output:
            for line in output.splitlines():
                log_file.write(f"[{utc_now_iso()}] process_snapshot {line}\n")
    except Exception as exc:
        log_file.write(f"[{utc_now_iso()}] process_snapshot_error={exc}\n")


def run_command(
    cmd: list[str],
    *,
    cwd: Path,
    log_path: Path,
    extra_env: dict[str, str] | None = None,
    heartbeat_sec: float = DEFAULT_STAGE_HEARTBEAT_SEC,
    scene_id: str = "",
    engine: str = "",
    stage_name: str = "",
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if extra_env:
        env.update({str(k): str(v) for k, v in extra_env.items()})
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(
            f"[{utc_now_iso()}] context engine={engine or '-'} scene_id={scene_id or '-'} stage={stage_name or '-'}\n"
        )
        log_file.write(f"[{utc_now_iso()}] cwd={cwd}\n")
        log_file.write(f"[{utc_now_iso()}] cmd={shlex.join(cmd)}\n")
        if extra_env:
            log_file.write(
                f"[{utc_now_iso()}] env_override="
                f"{json.dumps(extra_env, ensure_ascii=False, sort_keys=True)}\n"
            )
        log_file.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        start_ts = time.monotonic()
        next_heartbeat = start_ts + max(1.0, float(heartbeat_sec))
        log_file.write(f"[{utc_now_iso()}] pid={proc.pid}\n")
        _write_process_snapshot(log_file, proc.pid)
        log_file.flush()
        print(f"[PID][{engine or '-'}][{scene_id}] {stage_name} pid={proc.pid}")
        while True:
            returncode = proc.poll()
            if returncode is not None:
                log_file.write(f"[{utc_now_iso()}] returncode={returncode}\n")
                return int(returncode)
            now = time.monotonic()
            if now >= next_heartbeat:
                elapsed_sec = now - start_ts
                log_file.write(
                    f"[{utc_now_iso()}] heartbeat engine={engine or '-'} scene_id={scene_id} stage={stage_name} "
                    f"pid={proc.pid} elapsed_sec={elapsed_sec:.1f}\n"
                )
                _write_process_snapshot(log_file, proc.pid)
                log_file.flush()
                print(f"[HEARTBEAT][{engine or '-'}][{scene_id}] {stage_name} pid={proc.pid} elapsed={elapsed_sec:.1f}s")
                next_heartbeat = now + max(1.0, float(heartbeat_sec))
            time.sleep(1.0)


def load_scene_cleanup_patterns(config_path: Path, scene_id: str) -> list[str]:
    patterns: list[str] = []
    try:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        engine_cfg = ((cfg.get("engine_params", {}) or {}).get("airsim", {}) or {})
        sim_port = int(engine_cfg.get("sim_port", 0) or 0)
        if sim_port > 0:
            token = f"{scene_id}_{sim_port}.json"
            patterns.append(token)
            patterns.append(f".airsim_runtime/settings/{token}")
    except Exception:
        pass
    if not patterns:
        patterns.append(str(scene_id))
    deduped: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        key = str(pattern).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped


def load_scene_sim_port(config_path: Path) -> int | None:
    try:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        engine_cfg = ((cfg.get("engine_params", {}) or {}).get("airsim", {}) or {})
        sim_port = int(engine_cfg.get("sim_port", 0) or 0)
        return sim_port if sim_port > 0 else None
    except Exception:
        return None


def is_tcp_port_open(host: str, port: int, timeout_sec: float = 0.5) -> bool:
    try:
        with socket.create_connection((str(host), int(port)), timeout=float(timeout_sec)):
            return True
    except Exception:
        return False


def wait_for_port_release(
    *,
    log_path: Path,
    sim_port: int | None,
    scene_id: str = "",
    engine: str = "",
    stage_name: str = "",
    timeout_sec: float = 10.0,
    poll_sec: float = 0.5,
) -> None:
    if sim_port is None or int(sim_port) <= 0:
        return
    deadline = time.monotonic() + max(1.0, float(timeout_sec))
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(
            f"[{utc_now_iso()}] wait_port_release_start engine={engine or '-'} scene_id={scene_id or '-'} "
            f"stage={stage_name or '-'} sim_port={int(sim_port)} timeout_sec={float(timeout_sec):.1f}\n"
        )
        while True:
            open_now = is_tcp_port_open("127.0.0.1", int(sim_port))
            if not open_now:
                log_file.write(
                    f"[{utc_now_iso()}] wait_port_release_done engine={engine or '-'} scene_id={scene_id or '-'} "
                    f"stage={stage_name or '-'} sim_port={int(sim_port)} status=closed\n"
                )
                log_file.flush()
                return
            if time.monotonic() >= deadline:
                log_file.write(
                    f"[{utc_now_iso()}] wait_port_release_timeout engine={engine or '-'} scene_id={scene_id or '-'} "
                    f"stage={stage_name or '-'} sim_port={int(sim_port)} status=still_open\n"
                )
                log_file.flush()
                return
            time.sleep(max(0.1, float(poll_sec)))


def run_cleanup(
    *,
    log_path: Path,
    patterns: list[str],
    sleep_sec: float,
    scene_id: str = "",
    engine: str = "",
    stage_name: str = "",
) -> None:
    with log_path.open("a", encoding="utf-8") as log_file:
        # 精确查找包含 scene_id 和 sim_port 的进程
        match_keys = []
        if scene_id:
            match_keys.append(str(scene_id))
        # 尝试从 patterns 中提取端口号
        sim_port = None
        for p in patterns:
            if p.isdigit() and int(p) > 1000:
                sim_port = p
                break
        if sim_port:
            match_keys.append(str(sim_port))
        if not match_keys:
            log_file.write(f"[{utc_now_iso()}] cleanup_skipped reason=no_match_keys\n")
            if sleep_sec > 0:
                time.sleep(sleep_sec)
            return

        # ps aux | grep key1 | grep key2 | grep -v grep | awk '{print $2}'
        ps_cmd = ["ps", "aux"]
        grep_cmds = []
        for key in match_keys:
            # 对 scene_id 做全词匹配，避免 env_1 误杀 env_11
            if key.startswith("env_"):
                grep_cmds.append(["grep", "-w", key])
            else:
                grep_cmds.append(["grep", key])
        awk_cmd = ["awk", "{print $2}"]

        # 组合管道
        ps_proc = subprocess.Popen(ps_cmd, stdout=subprocess.PIPE, cwd=str(ROOT_DIR))
        prev_proc = ps_proc
        for grep_cmd in grep_cmds:
            proc = subprocess.Popen(grep_cmd, stdin=prev_proc.stdout, stdout=subprocess.PIPE, cwd=str(ROOT_DIR))
            prev_proc.stdout.close()
            prev_proc = proc
        # 排除grep自身
        proc = subprocess.Popen(["grep", "-v", "grep"], stdin=prev_proc.stdout, stdout=subprocess.PIPE, cwd=str(ROOT_DIR))
        prev_proc.stdout.close()
        prev_proc = proc
        # 获取pid
        awk_proc = subprocess.Popen(awk_cmd, stdin=prev_proc.stdout, stdout=subprocess.PIPE, cwd=str(ROOT_DIR), text=True)
        prev_proc.stdout.close()
        pids = awk_proc.communicate()[0].strip().splitlines()
        log_file.write(f"[{utc_now_iso()}] cleanup_ps_match_keys={match_keys} found_pids={pids}\n")
        # kill
        this_pid = os.getpid()
        parent_pid = os.getppid()
        for pid in pids:
            if not pid or not pid.isdigit() or int(pid) <= 1:
                continue
            ipid = int(pid)
            if ipid == this_pid:
                log_file.write(f"[{utc_now_iso()}] cleanup_skip_pid={pid} reason=self_pid\n")
                continue
            if ipid == parent_pid:
                log_file.write(f"[{utc_now_iso()}] cleanup_skip_pid={pid} reason=parent_pid\n")
                continue
            kill_cmd = ["kill", "-9", pid]
            log_file.write(f"[{utc_now_iso()}] cleanup_kill_cmd={shlex.join(kill_cmd)}\n")
            proc = subprocess.run(kill_cmd, cwd=str(ROOT_DIR), stdout=log_file, stderr=subprocess.STDOUT, check=False)
            log_file.write(f"[{utc_now_iso()}] cleanup_kill_returncode={proc.returncode} pid={pid}\n")
        if sleep_sec > 0:
            log_file.write(
                f"[{utc_now_iso()}] cleanup_sleep_sec={sleep_sec} engine={engine or '-'} "
                f"scene_id={scene_id or '-'} stage={stage_name or '-'}\n"
            )
            log_file.flush()
            time.sleep(sleep_sec)


def build_stage_commands(
    *,
    scene_id: str,
    config_path: Path,
    python_bin: str,
    engine: str,
    scene_log_dir: Path,
    probe_source: str,
    probe_workers: int,
    probe_half_span: float,
    probe_lateral_step: float,
    probe_coarse_step: float,
    probe_refine_tol: float,
    probe_max_steps: int,
    probe_settle_sec: float,
    probe_surface_percentile: float,
    probe_surface_local_xy_radius: float,
    probe_surface_z_margin: float,
    probe_surface_robust_low_percentile: float,
    probe_surface_center_bias_max_above_low: float,
    probe_boundary_lidar_max_range: float,
    probe_boundary_min_points_per_yaw: int,
    probe_boundary_min_valid_yaws: int,
    probe_boundary_seg_min_points_per_id: int,
    probe_boundary_seg_stop_max_distinct_ids: int,
    probe_boundary_seg_stop_max_total_points: int,
    probe_xy_boundary_quantile: float,
    probe_min_map_bound: list[float] | None,
    probe_hard_map_bound: list[float] | None,
    probe_write_back: bool,
    skip_probe: bool = False,
) -> list[tuple[str, list[str]]]:
    probe_output_path = scene_log_dir / f"{scene_id}.probe_result.json"
    probe_cmd = [
        python_bin,
        "scripts/flightmvstg/probe_airsim_mapbound.py",
        "--config",
        str(config_path),
        "--scene-id",
        scene_id,
        "--probe-source",
        str(probe_source),
        "--workers",
        str(int(probe_workers)),
        "--sweep-half-span",
        str(float(probe_half_span)),
        "--sweep-lateral-step",
        str(float(probe_lateral_step)),
        "--coarse-step",
        str(float(probe_coarse_step)),
        "--refine-tol",
        str(float(probe_refine_tol)),
        "--max-steps",
        str(int(probe_max_steps)),
        "--settle-sec",
        str(float(probe_settle_sec)),
        "--surface-percentile",
        str(float(probe_surface_percentile)),
        "--surface-local-xy-radius",
        str(float(probe_surface_local_xy_radius)),
        "--surface-z-margin",
        str(float(probe_surface_z_margin)),
        "--surface-robust-low-percentile",
        str(float(probe_surface_robust_low_percentile)),
        "--surface-center-bias-max-above-low",
        str(float(probe_surface_center_bias_max_above_low)),
        "--boundary-lidar-max-range",
        str(float(probe_boundary_lidar_max_range)),
        "--boundary-min-points-per-yaw",
        str(int(probe_boundary_min_points_per_yaw)),
        "--boundary-min-valid-yaws",
        str(int(probe_boundary_min_valid_yaws)),
        "--boundary-seg-min-points-per-id",
        str(int(probe_boundary_seg_min_points_per_id)),
        "--boundary-seg-stop-max-distinct-ids",
        str(int(probe_boundary_seg_stop_max_distinct_ids)),
        "--boundary-seg-stop-max-total-points",
        str(int(probe_boundary_seg_stop_max_total_points)),
        "--xy-boundary-quantile",
        str(float(probe_xy_boundary_quantile)),
        "--output",
        str(probe_output_path),
    ]
    if isinstance(probe_min_map_bound, list) and len(probe_min_map_bound) == 6:
        probe_cmd.extend(["--min-map-bound", *[str(float(v)) for v in probe_min_map_bound]])
    if isinstance(probe_hard_map_bound, list) and len(probe_hard_map_bound) == 6:
        probe_cmd.extend(["--hard-map-bound", *[str(float(v)) for v in probe_hard_map_bound]])
    probe_cmd.append("--write-back" if probe_write_back else "--no-write-back")

    stages: list[tuple[str, list[str]]] = []
    if not skip_probe:
        stages.append(("probe_mapbound", probe_cmd))
    stages.extend(
        [
            (
                "stage1",
                [
                    python_bin,
                    "scripts/flightmvstg/stage1_collect_pcd.py",
                    "--config",
                    str(config_path),
                    "--mode",
                    "all",
                    "--scene-id",
                    scene_id,
                    "--engine",
                    engine,
                ],
            ),
            (
                "stage2_collect_instances",
                [
                    python_bin,
                    "scripts/flightmvstg/stage2_landmark_label.py",
                    "--config",
                    str(config_path),
                    "--scene-id",
                    scene_id,
                    "--mode",
                    "collect_instances",
                ],
            ),
        ]
    )
    return stages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run probe, Stage 1, and Stage 2 Step 1 serially for specified scenes."
    )
    parser.add_argument("scene_ids", nargs="*", help="Scene ids like env_1 env_7, or numeric ids like 1 7")
    parser.add_argument(
        "--no-probe",
        action="store_true",
        help="Skip probe_mapbound stage and run Stage 1 + Stage 2 directly",
    )
    parser.add_argument(
        "--scene",
        dest="scene_opts",
        action="append",
        default=[],
        help="Repeatable scene id option, e.g. --scene env_1 --scene env_7",
    )
    parser.add_argument("--python-bin", default=DEFAULT_PYTHON_BIN, help="Python executable used to run stage scripts")
    parser.add_argument("--engine", default="airsim", help="Engine name passed to Stage 1")
    parser.add_argument(
        "--gpu",
        default="",
        help="GPU id or CUDA_VISIBLE_DEVICES value passed to child processes, e.g. --gpu 0 or --gpu 1,2",
    )
    parser.add_argument(
        "--config-dir",
        default=str(DEFAULT_CONFIG_DIR),
        help="Directory containing task_airsim_<scene>.yaml files",
    )
    parser.add_argument(
        "--log-root",
        default="",
        help="Optional log root directory; default is .runtime/run_stage1_stage2_collect_serial_<timestamp>",
    )
    parser.add_argument(
        "--cleanup-pattern",
        dest="cleanup_patterns",
        action="append",
        default=[],
        help="Repeatable pkill -f pattern run after each stage; defaults include AirVLN",
    )
    parser.add_argument(
        "--cleanup-sleep-sec",
        type=float,
        default=2.0,
        help="Sleep time after cleanup commands",
    )
    parser.add_argument("--probe-source", default=DEFAULT_PROBE_SOURCE, choices=["depth", "lidar", "hybrid"])
    parser.add_argument("--probe-workers", type=int, default=DEFAULT_PROBE_WORKERS)
    parser.add_argument("--probe-half-span", type=float, default=DEFAULT_PROBE_HALF_SPAN)
    parser.add_argument("--probe-lateral-step", type=float, default=DEFAULT_PROBE_LATERAL_STEP)
    parser.add_argument("--probe-coarse-step", type=float, default=DEFAULT_PROBE_COARSE_STEP)
    parser.add_argument("--probe-refine-tol", type=float, default=DEFAULT_PROBE_REFINE_TOL)
    parser.add_argument("--probe-max-steps", type=int, default=DEFAULT_PROBE_MAX_STEPS)
    parser.add_argument("--probe-settle-sec", type=float, default=DEFAULT_PROBE_SETTLE_SEC)
    parser.add_argument("--probe-surface-percentile", type=float, default=DEFAULT_PROBE_SURFACE_PERCENTILE)
    parser.add_argument("--probe-surface-local-xy-radius", type=float, default=DEFAULT_PROBE_SURFACE_LOCAL_XY_RADIUS)
    parser.add_argument("--probe-surface-z-margin", type=float, default=DEFAULT_PROBE_SURFACE_Z_MARGIN)
    parser.add_argument(
        "--probe-surface-robust-low-percentile",
        type=float,
        default=DEFAULT_PROBE_SURFACE_ROBUST_LOW_PERCENTILE,
    )
    parser.add_argument(
        "--probe-surface-center-bias-max-above-low",
        type=float,
        default=DEFAULT_PROBE_SURFACE_CENTER_BIAS_MAX_ABOVE_LOW,
    )
    parser.add_argument("--probe-boundary-lidar-max-range", type=float, default=DEFAULT_PROBE_BOUNDARY_LIDAR_MAX_RANGE)
    parser.add_argument("--probe-boundary-min-points-per-yaw", type=int, default=DEFAULT_PROBE_BOUNDARY_MIN_POINTS_PER_YAW)
    parser.add_argument("--probe-boundary-min-valid-yaws", type=int, default=DEFAULT_PROBE_BOUNDARY_MIN_VALID_YAWS)
    parser.add_argument("--probe-boundary-seg-min-points-per-id", type=int, default=DEFAULT_PROBE_BOUNDARY_SEG_MIN_POINTS_PER_ID)
    parser.add_argument("--probe-boundary-seg-stop-max-distinct-ids", type=int, default=DEFAULT_PROBE_BOUNDARY_SEG_STOP_MAX_DISTINCT_IDS)
    parser.add_argument("--probe-boundary-seg-stop-max-total-points", type=int, default=DEFAULT_PROBE_BOUNDARY_SEG_STOP_MAX_TOTAL_POINTS)
    parser.add_argument("--probe-xy-boundary-quantile", type=float, default=DEFAULT_PROBE_XY_BOUNDARY_QUANTILE)
    parser.add_argument(
        "--probe-min-map-bound",
        nargs=6,
        type=float,
        default=None,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="minimum allowed probe MapBound envelope, e.g. --probe-min-map-bound -500 500 -500 500 -50 100",
    )
    parser.add_argument(
        "--probe-hard-map-bound",
        nargs=6,
        type=float,
        default=None,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="hard probe bounds applied during actual x/y/z scans; an axis is enabled only when MIN < MAX",
    )
    parser.add_argument("--probe-no-write-back", dest="probe_write_back", action="store_false")
    parser.set_defaults(probe_write_back=True)
    parser.add_argument("--heartbeat-sec", type=float, default=DEFAULT_STAGE_HEARTBEAT_SEC, help="Heartbeat interval for child process status logs")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scenes_raw = list(args.scene_opts) + list(args.scene_ids)
    if not scenes_raw:
        print("No scenes specified. Example: --scene env_1 --scene env_7", file=sys.stderr)
        return 2

    scenes: list[str] = []
    seen: set[str] = set()
    for raw in scenes_raw:
        scene_id = normalize_scene_id(raw)
        if scene_id in seen:
            continue
        seen.add(scene_id)
        scenes.append(scene_id)

    config_dir = Path(args.config_dir)
    if not config_dir.is_absolute():
        config_dir = (ROOT_DIR / config_dir).resolve()
    run_stamp = time.strftime('%Y%m%d_%H%M%S')
    log_base_dir = Path(args.log_root) if args.log_root else (ROOT_DIR / ".runtime")
    if not log_base_dir.is_absolute():
        log_base_dir = (ROOT_DIR / log_base_dir).resolve()
    cleanup_patterns = list(args.cleanup_patterns) if args.cleanup_patterns else list(DEFAULT_CLEANUP_PATTERNS)
    child_env: dict[str, str] = {}
    if str(args.gpu or "").strip():
        child_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu).strip()
    summary_label = f"{args.engine}_multi" if len(scenes) > 1 else f"{args.engine}_{scenes[0]}"
    summary_path = log_base_dir / f"{summary_label}_run_stage1_stage2_collect_serial_{run_stamp}_summary.jsonl"
    log_base_dir.mkdir(parents=True, exist_ok=True)

    append_jsonl(
        summary_path,
        {
            "ts": utc_now_iso(),
            "event": "run_started",
            "scenes": scenes,
            "python_bin": args.python_bin,
            "engine": args.engine,
            "gpu": str(args.gpu or "").strip() or None,
            "child_env": child_env,
            "config_dir": str(config_dir),
            "log_base_dir": str(log_base_dir),
            "cleanup_patterns": cleanup_patterns,
            "cleanup_sleep_sec": args.cleanup_sleep_sec,
            "dry_run": bool(args.dry_run),
        },
    )

    success_count = 0
    failed_scenes: list[str] = []

    for scene_id in scenes:
        config_path = config_dir / f"task_airsim_{scene_id}.yaml"
        scene_run_label = f"{args.engine}_{scene_id}_run_stage1_stage2_collect_serial_{run_stamp}"
        scene_log_dir = log_base_dir / scene_run_label
        scene_log_dir.mkdir(parents=True, exist_ok=True)
        sim_port = load_scene_sim_port(config_path)

        if not config_path.exists():
            print(f"[SKIP][{scene_id}] missing config: {config_path}")
            failed_scenes.append(scene_id)
            append_jsonl(
                summary_path,
                {
                    "ts": utc_now_iso(),
                    "event": "scene_skipped",
                    "scene_id": scene_id,
                    "reason": "missing_config",
                    "config_path": str(config_path),
                },
            )
            continue

        print(f"[SCENE][{args.engine}][{scene_id}] start")
        scene_cleanup_patterns = load_scene_cleanup_patterns(config_path=config_path, scene_id=scene_id)
        if cleanup_patterns:
            scene_cleanup_patterns.extend(list(cleanup_patterns))
        append_jsonl(
            summary_path,
            {
                "ts": utc_now_iso(),
                "event": "scene_started",
                "scene_id": scene_id,
                "config_path": str(config_path),
                "scene_log_dir": str(scene_log_dir),
                "sim_port": sim_port,
                "cleanup_patterns": scene_cleanup_patterns,
            },
        )

        scene_failed = False
        for stage_index, (stage_name, cmd) in enumerate(
            build_stage_commands(
                scene_id=scene_id,
                config_path=config_path,
                python_bin=args.python_bin,
                engine=args.engine,
                scene_log_dir=scene_log_dir,
                probe_source=args.probe_source,
                probe_workers=args.probe_workers,
                probe_half_span=args.probe_half_span,
                probe_lateral_step=args.probe_lateral_step,
                probe_coarse_step=args.probe_coarse_step,
                probe_refine_tol=args.probe_refine_tol,
                probe_max_steps=args.probe_max_steps,
                probe_settle_sec=args.probe_settle_sec,
                probe_surface_percentile=args.probe_surface_percentile,
                probe_surface_local_xy_radius=args.probe_surface_local_xy_radius,
                probe_surface_z_margin=args.probe_surface_z_margin,
                probe_surface_robust_low_percentile=args.probe_surface_robust_low_percentile,
                probe_surface_center_bias_max_above_low=args.probe_surface_center_bias_max_above_low,
                probe_boundary_lidar_max_range=args.probe_boundary_lidar_max_range,
                probe_boundary_min_points_per_yaw=args.probe_boundary_min_points_per_yaw,
                probe_boundary_min_valid_yaws=args.probe_boundary_min_valid_yaws,
                probe_boundary_seg_min_points_per_id=args.probe_boundary_seg_min_points_per_id,
                probe_boundary_seg_stop_max_distinct_ids=args.probe_boundary_seg_stop_max_distinct_ids,
                probe_boundary_seg_stop_max_total_points=args.probe_boundary_seg_stop_max_total_points,
                probe_xy_boundary_quantile=args.probe_xy_boundary_quantile,
                probe_min_map_bound=args.probe_min_map_bound,
                probe_hard_map_bound=args.probe_hard_map_bound,
                probe_write_back=args.probe_write_back,
                skip_probe=bool(args.no_probe),
            ),
            start=1,
        ):
            log_path = scene_log_dir / f"{args.engine}_{scene_id}_{stage_index:02d}_{stage_name}.log"
            print(f"[RUN][{args.engine}][{scene_id}] {stage_name}")
            append_jsonl(
                summary_path,
                {
                    "ts": utc_now_iso(),
                    "event": "stage_started",
                    "engine": args.engine,
                    "scene_id": scene_id,
                    "stage": stage_name,
                    "log_path": str(log_path),
                    "cmd": cmd,
                    "child_env": child_env,
                },
            )

            if args.dry_run:
                returncode = 0
                with log_path.open("a", encoding="utf-8") as log_file:
                    log_file.write(f"[{utc_now_iso()}] dry_run cmd={shlex.join(cmd)}\n")
                    if child_env:
                        log_file.write(
                            f"[{utc_now_iso()}] dry_run env_override="
                            f"{json.dumps(child_env, ensure_ascii=False, sort_keys=True)}\n"
                        )
                    for pattern in scene_cleanup_patterns:
                        dry_cleanup_cmd = ["pkill", "-f", "--", pattern]
                        log_file.write(f"[{utc_now_iso()}] dry_run cleanup_cmd={shlex.join(dry_cleanup_cmd)}\n")
                # dry_run模式下也sleep，模拟真实间隔
                time.sleep(10.0)
            else:
                returncode = run_command(
                    cmd,
                    cwd=ROOT_DIR,
                    log_path=log_path,
                    extra_env=child_env,
                    heartbeat_sec=float(args.heartbeat_sec),
                    scene_id=scene_id,
                    engine=args.engine,
                    stage_name=stage_name,
                )
                run_cleanup(
                    log_path=log_path,
                    patterns=scene_cleanup_patterns,
                    sleep_sec=10.0,
                    scene_id=scene_id,
                    engine=args.engine,
                    stage_name=stage_name,
                )
                wait_for_port_release(
                    log_path=log_path,
                    sim_port=sim_port,
                    scene_id=scene_id,
                    engine=args.engine,
                    stage_name=stage_name,
                )
                # 每个stage之间sleep 20秒，确保彻底释放
                time.sleep(10.0)

            status = "ok" if returncode == 0 else "failed"
            append_jsonl(
                summary_path,
                {
                    "ts": utc_now_iso(),
                    "event": "stage_finished",
                    "engine": args.engine,
                    "scene_id": scene_id,
                    "stage": stage_name,
                    "status": status,
                    "returncode": returncode,
                    "log_path": str(log_path),
                },
            )

            if returncode != 0:
                print(f"[SKIP][{args.engine}][{scene_id}] {stage_name} failed, skip remaining stages")
                failed_scenes.append(scene_id)
                scene_failed = True
                break

        if scene_failed:
            append_jsonl(
                summary_path,
                {
                    "ts": utc_now_iso(),
                    "event": "scene_finished",
                    "scene_id": scene_id,
                    "status": "failed",
                },
            )
            continue

        success_count += 1
        print(
            f"[DONE][{args.engine}][{scene_id}] stage1 + stage2 collect_instances finished "
            f"(probe_skipped={bool(args.no_probe)})"
        )
        append_jsonl(
            summary_path,
            {
                "ts": utc_now_iso(),
                "event": "scene_finished",
                "scene_id": scene_id,
                "status": "ok",
            },
        )

    append_jsonl(
        summary_path,
        {
            "ts": utc_now_iso(),
            "event": "run_finished",
            "success_count": success_count,
            "failed_scenes": failed_scenes,
            "log_base_dir": str(log_base_dir),
            "summary_jsonl": str(summary_path),
        },
    )

    print(f"[SUMMARY] success={success_count} failed={len(failed_scenes)} log_base_dir={log_base_dir}")
    if failed_scenes:
        print(f"[SUMMARY] failed_scenes={', '.join(failed_scenes)}")
    print(f"[SUMMARY] summary_jsonl={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
