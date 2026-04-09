# UAV-DualCog

This folder is the official code of UAV-DualCog for reviewers and external users.

It keeps the implemented Stage 1-4 entrypoints, shared runtime modules, simulator and trajectory
adapters, prompt templates, and sanitized configuration examples needed to inspect or reproduce the
pipeline logic of **UAV-DualCog** without shipping environment files, cached artifacts, generated
datasets, media outputs, or any private credentials.

For benchmark definitions, task examples, leaderboard views, and detailed release notes, please
refer to the public website and dataset pages first:

- Website: https://uav-dualcog.lozumi.com/
- Dataset: https://www.modelscope.cn/datasets/Lozumi/UAV-DualCog
- Code repository: https://github.com/SmartDianLab/UAV-DualCog
- AerialVLN simulator: https://www.kaggle.com/datasets/shuboliu/aerialvln-simulators

Detailed usage guidance and dataset download instructions should be treated as external-release
documentation and should be read from those pages.

## Included

- `scripts/uav_dualcog/`
  - Stage 1 point-cloud collection and fusion
  - Stage 2 landmark construction, review, and auto-labeling
  - Stage 3 trajectory, mission, and video-task generation
  - Stage 4 image-QA generation and evaluation
  - pipeline orchestration and shared runtime helpers
- `sim_bridge/`
  - simulator bridge factory and engine adapters
- `trajectory/`
  - hierarchical flight-behavior library and trajectory composition helpers
- `configs/uav_dualcog/`
  - sanitized runnable examples for 18 released AirSim scenes together with common-stage, API, and task-pipeline configs
  - template copies for customization
- `configs/prompts/`
  - prompt package used by Stage 2-4

## Not Included

- any `.env`, Conda bootstrap, or local machine setup files
- any API keys, private endpoints, or internal routing settings
- any `scene_data/`, `task_pipeline_data/`, website outputs, logs, or cached artifacts
- any one-off repair scripts, backups, or experimental branches not needed for review

## Package Layout

```text
reviewer_code_repo/
├── configs/
│   ├── prompts/
│   │   ├── templates/
│   │   │   └── uav_dualcog_prompts.template.yaml
│   │   └── uav_dualcog_prompts.yaml
│   └── uav_dualcog/
│       ├── common_api_runtime.yaml
│       ├── common_stage_configs.yaml
│       ├── task_airsim_env_7.yaml
│       ├── task_pipeline/
│       │   └── task_pipeline_uav_dualcog_v1.yaml
│       └── templates/
│           ├── common_api_runtime.template.yaml
│           ├── common_stage_configs.template.yaml
│           ├── scene_config.template.yaml
│           └── task_pipeline.template.yaml
├── deps/
│   └── msgpack_rpc_python-0.4-py3-none-any.whl
├── scripts/
│   └── uav_dualcog/
│       ├── stage1_collect_pcd.py
│       ├── stage2_landmark_label.py
│       ├── stage3_generate_traj.py
│       ├── stage3_task_suite.py
│       ├── stage4_qa_generate_and_eval.py
│       ├── task_pipeline.py
│       ├── pipeline_common.py
│       └── probe_airsim_mapbound.py
├── sim_bridge/
├── trajectory/
├── coord_transform_utils.py
├── environment.yml
├── requirements.txt
└── README.md
```

## Expected External Workspace

When reproducing the released benchmark, the repository is typically paired with two external asset
roots:

```text
UAV-DualCog/
├── scene_data/
│   └── airsim_env_*/
│       ├── pcd_map/
│       ├── landmarks_raw/
│       └── landmarks_review/
└── task_pipeline_data/
    └── UAV-DualCog-V1/
        ├── airsim_env_*/
        │   ├── video_tasks/
        │   └── image_tasks/
        └── task_pipeline/
            ├── dataset_stats/
            ├── exports/
            └── landmark_lists/
```

The released public dataset uses `video_tasks/` and `image_tasks/` as the benchmark-facing folder
names. Internally, some construction scripts still generate intermediate artifacts under
`stage3_tasks/` and `qa/` before release packaging.

## Shipped Config Examples

The following sanitized example files are already included at their runtime paths:

- `configs/uav_dualcog/task_airsim_env_1.yaml`
- `configs/uav_dualcog/task_airsim_env_3.yaml`
- `configs/uav_dualcog/task_airsim_env_5.yaml`
- `configs/uav_dualcog/task_airsim_env_7.yaml`
- `configs/uav_dualcog/task_airsim_env_8.yaml`
- `configs/uav_dualcog/task_airsim_env_9.yaml`
- `configs/uav_dualcog/task_airsim_env_10.yaml`
- `configs/uav_dualcog/task_airsim_env_11.yaml`
- `configs/uav_dualcog/task_airsim_env_13.yaml`
- `configs/uav_dualcog/task_airsim_env_15.yaml`
- `configs/uav_dualcog/task_airsim_env_16.yaml`
- `configs/uav_dualcog/task_airsim_env_17.yaml`
- `configs/uav_dualcog/task_airsim_env_18.yaml`
- `configs/uav_dualcog/task_airsim_env_20.yaml`
- `configs/uav_dualcog/task_airsim_env_21.yaml`
- `configs/uav_dualcog/task_airsim_env_22.yaml`
- `configs/uav_dualcog/task_airsim_env_23.yaml`
- `configs/uav_dualcog/task_airsim_env_24.yaml`
- `configs/uav_dualcog/common_stage_configs.yaml`
- `configs/uav_dualcog/common_api_runtime.yaml`
- `configs/uav_dualcog/task_pipeline/task_pipeline_uav_dualcog_v1.yaml`
- `configs/prompts/uav_dualcog_prompts.yaml`
- `environment.yml`

These files are parseable examples and can be copied or edited directly for local reproduction.

## Core Config Roles

### 1. Scene Config

`configs/uav_dualcog/task_airsim_env_7.yaml`

Use this file to define:

- scene identity and `scene_data` location
- Stage 1 capture resolution and LiDAR collection settings
- Stage 2 landmark-view collection, review, and auto-label defaults
- Stage 3 rendering defaults and simulator connection parameters

### 2. Common Stage Config

`configs/uav_dualcog/common_stage_configs.yaml`

Use this file to define:

- the hierarchical Stage 3 behavior library
- shared atomic and composite flight-mode parameter ranges
- behavior-set composition rules reused by Stage 3 tasks

### 3. API Runtime Config

`configs/uav_dualcog/common_api_runtime.yaml`

Use this file to define:

- default models for Stage 2, Stage 3, and Stage 4
- model-specific API base URLs, keys, and rate limits
- Stage 3 temporal-upload settings

### 4. Task Pipeline Spec

`configs/uav_dualcog/task_pipeline/task_pipeline_uav_dualcog_v1.yaml`

Use this file to define:

- benchmark name and artifact root
- selected scene list
- Stage 3 task-generation scope and render settings
- Stage 4 task-generation scope and difficulty settings
- pipeline-wide experiment scheduling parameters

### 5. Prompt Package

`configs/prompts/uav_dualcog_prompts.yaml`

This file is the prompt contract used by Stage 2 auto-labeling and by Stage 3/4 benchmark-facing
task generation and evaluation.

## Stage 1 Command

Entrypoint: `python scripts/uav_dualcog/stage1_collect_pcd.py`

Main options:

- `--config`: scene config yaml
- `--mode`: `collect_raw` | `fuse` | `all`
- `--engine`: simulator backend, usually `airsim`
- `--scene-id`: override scene id such as `7` or `env_7`
- `--workers`: override worker count
- `--max-frames`: cap raw collection frames
- `--control-port`: override runtime control port
- `--output-root`: override output directory
- `--voxel-size`: override fusion voxel size
- `--use-sor`: enable statistical outlier removal
- `--no-dedup`: disable fusion deduplication
- `--stop-on-error`: fail fast on worker errors
- `--multimodal-fusion`: enable multimodal fusion branch
- `--lidar-segmentation`: enable segmented LiDAR branch

Commands:

```bash
# Stage 1-A: raw collection only
python scripts/uav_dualcog/stage1_collect_pcd.py \
  --config configs/uav_dualcog/task_airsim_env_7.yaml \
  --scene-id 7 \
  --mode collect_raw \
  --engine airsim

# Stage 1-B: fusion only
python scripts/uav_dualcog/stage1_collect_pcd.py \
  --config configs/uav_dualcog/task_airsim_env_7.yaml \
  --scene-id 7 \
  --mode fuse

# Stage 1 full run
python scripts/uav_dualcog/stage1_collect_pcd.py \
  --config configs/uav_dualcog/task_airsim_env_7.yaml \
  --scene-id 7 \
  --mode all
```

## Stage 2 Command

Entrypoint: `python scripts/uav_dualcog/stage2_landmark_label.py`

Main options:

- `--config`: scene config yaml
- `--scene-id`: override scene id
- `--mode`: `collect_instances` | `review_instances` | `review_instances_web` | `auto_label` | `all`
- `--pcd`: override point-cloud path
- `--min-points`: minimum instance point threshold
- `--review-decisions`: path to review decision file
- `--host`, `--port`: web review host and port
- `--auto-label-sample-size`: debug/sample subset size for auto-labeling
- `--auto-label-sample-seed`: seed for sampled auto-labeling
- `--auto-label-debug-save-bbox-img`: save bbox-debug images during auto-labeling

Commands:

```bash
# Stage 2-A: candidate aggregation and multiview capture
python scripts/uav_dualcog/stage2_landmark_label.py \
  --config configs/uav_dualcog/task_airsim_env_7.yaml \
  --scene-id 7 \
  --mode collect_instances

# Stage 2-B: browser-based review
python scripts/uav_dualcog/stage2_landmark_label.py \
  --config configs/uav_dualcog/task_airsim_env_7.yaml \
  --scene-id 7 \
  --mode review_instances_web \
  --host 0.0.0.0 \
  --port 20261

# Stage 2-C: semantic auto-labeling
python scripts/uav_dualcog/stage2_landmark_label.py \
  --config configs/uav_dualcog/task_airsim_env_7.yaml \
  --scene-id 7 \
  --mode auto_label

# Stage 2 full run
python scripts/uav_dualcog/stage2_landmark_label.py \
  --config configs/uav_dualcog/task_airsim_env_7.yaml \
  --scene-id 7 \
  --mode all
```

## Stage 3 Command

Entrypoint: `python scripts/uav_dualcog/stage3_generate_traj.py`

Main options:

- `--config`: scene config yaml
- `--mode`: `generate` | `generate_mission` | `generate_dataset` | `run_experiment` | `record_scene_videos` | `web`
- `--scene-id`: override scene id
- `--landmark-id`: one Stage 2 `instance_id`
- `--traj-id`: explicit trajectory id
- `--behavior-sequence`: explicit behavior sequence
- `--mission-type`: mission-type override
- `--generation-kind`: `auto` | `atomic-only` | `composite-driven`
- `--mission-mode`: `single-landmark` | `multi-landmark`
- `--seed`: random seed
- `--sample-count`: generated sample count
- `--forms`: comma-separated task forms
- `--task-group`: `all` | `self-state` | `environmental`
- `--approved-only`: use only reviewed inputs
- `--manifest-path`: explicit manifest path
- `--limit`: experiment limit
- `--model`: experiment model name
- `--traj-ids`: comma-separated trajectory ids for recording
- `--record-parallel-workers`: parallel render workers
- `--record-reuse-worker-connections`: reuse render worker connections
- `--rerender-existing`: rerender existing outputs
- `--ignore-waypoint-forwards`: rebuild forward vectors
- `--provide-flight-description`: attach text description
- `--include-keyframes`: export keyframe boards

Commands:

```bash
# Stage 3-A: generate missions from reviewed landmarks
python scripts/uav_dualcog/stage3_generate_traj.py \
  --config configs/uav_dualcog/task_airsim_env_7.yaml \
  --scene-id 7 \
  --mode generate_mission \
  --generation-kind auto \
  --mission-mode single-landmark

# Stage 3-B: build dataset manifests from generated missions
python scripts/uav_dualcog/stage3_generate_traj.py \
  --config configs/uav_dualcog/task_airsim_env_7.yaml \
  --scene-id 7 \
  --mode generate_dataset \
  --forms self_instance_recognition_joint,env_visibility_reasoning

# Stage 3-C: record benchmark-facing task videos
python scripts/uav_dualcog/stage3_generate_traj.py \
  --config configs/uav_dualcog/task_airsim_env_7.yaml \
  --scene-id 7 \
  --mode record_scene_videos \
  --record-parallel-workers 8

# Stage 3-D: run one experiment pass
python scripts/uav_dualcog/stage3_generate_traj.py \
  --config configs/uav_dualcog/task_airsim_env_7.yaml \
  --scene-id 7 \
  --mode run_experiment \
  --model openai/gpt-5.3-chat
```

## Stage 4 Command

Entrypoint: `python scripts/uav_dualcog/stage4_qa_generate_and_eval.py`

Main options:

- `--config`: scene config yaml
- `--scene-id`: scene id
- `--engine`: simulator backend
- `--mode`: `generate` | `experiment` | `web` | `all`
- `--sample-count`: generation sample count
- `--seed`: random seed
- `--reference-main-only`: use only reviewed main views as references
- `--allow-non-main-reference`: allow non-main references
- `--manifest-path`: explicit manifest path
- `--model`: experiment model name
- `--limit`: experiment limit
- `--port`: web workbench port
- `--landmark-category`: repeatable category filter

Commands:

```bash
# Stage 4-A: generate image-task manifests
python scripts/uav_dualcog/stage4_qa_generate_and_eval.py \
  --config configs/uav_dualcog/task_airsim_env_7.yaml \
  --scene-id 7 \
  --mode generate \
  --sample-count 10 \
  --seed 7

# Stage 4-B: run one experiment pass
python scripts/uav_dualcog/stage4_qa_generate_and_eval.py \
  --config configs/uav_dualcog/task_airsim_env_7.yaml \
  --scene-id 7 \
  --mode experiment \
  --model openai/gpt-5.3-chat

# Stage 4-C: open the local QA workbench
python scripts/uav_dualcog/stage4_qa_generate_and_eval.py \
  --config configs/uav_dualcog/task_airsim_env_7.yaml \
  --scene-id 7 \
  --mode web \
  --port 20264
```

## Cross-Scene Pipeline Orchestration

Entrypoint: `python scripts/uav_dualcog/task_pipeline.py`

Main options:

- `--spec`: yaml task-pipeline spec
- `--spec-json`: json task-pipeline spec
- `--config`: optional fallback scene config
- `--stage`: `both` | `stage3` | `stage4`
- `--phase`: `both` | `selection` | `data` | `render` | `experiment` | `analyze`
- `--clear-stage3`, `--clear-stage4`: clear target-stage artifacts before rerun
- `--landmark-count`: default fallback landmark count
- `--task-name`: override task name
- `--experiment-models`: one or more experiment models

Commands:

```bash
# Stage 3 selection
python scripts/uav_dualcog/task_pipeline.py \
  --spec configs/uav_dualcog/task_pipeline/task_pipeline_uav_dualcog_v1.yaml \
  --stage stage3 \
  --phase selection

# Stage 3 data assembly
python scripts/uav_dualcog/task_pipeline.py \
  --spec configs/uav_dualcog/task_pipeline/task_pipeline_uav_dualcog_v1.yaml \
  --stage stage3 \
  --phase data

# Stage 3 rendering
python scripts/uav_dualcog/task_pipeline.py \
  --spec configs/uav_dualcog/task_pipeline/task_pipeline_uav_dualcog_v1.yaml \
  --stage stage3 \
  --phase render

# Stage 3 experiments
python scripts/uav_dualcog/task_pipeline.py \
  --spec configs/uav_dualcog/task_pipeline/task_pipeline_uav_dualcog_v1.yaml \
  --stage stage3 \
  --phase experiment \
  --experiment-models openai/gpt-5.3-chat Qwen/Qwen3.5-9B

# Stage 3 report analysis
python scripts/uav_dualcog/task_pipeline.py \
  --spec configs/uav_dualcog/task_pipeline/task_pipeline_uav_dualcog_v1.yaml \
  --stage stage3 \
  --phase analyze

# Stage 4 selection
python scripts/uav_dualcog/task_pipeline.py \
  --spec configs/uav_dualcog/task_pipeline/task_pipeline_uav_dualcog_v1.yaml \
  --stage stage4 \
  --phase selection

# Stage 4 data assembly
python scripts/uav_dualcog/task_pipeline.py \
  --spec configs/uav_dualcog/task_pipeline/task_pipeline_uav_dualcog_v1.yaml \
  --stage stage4 \
  --phase data

# Stage 4 rendering
python scripts/uav_dualcog/task_pipeline.py \
  --spec configs/uav_dualcog/task_pipeline/task_pipeline_uav_dualcog_v1.yaml \
  --stage stage4 \
  --phase render

# Stage 4 experiments
python scripts/uav_dualcog/task_pipeline.py \
  --spec configs/uav_dualcog/task_pipeline/task_pipeline_uav_dualcog_v1.yaml \
  --stage stage4 \
  --phase experiment \
  --experiment-models openai/gpt-5.3-chat Qwen/Qwen3.5-4B

# Stage 4 report analysis
python scripts/uav_dualcog/task_pipeline.py \
  --spec configs/uav_dualcog/task_pipeline/task_pipeline_uav_dualcog_v1.yaml \
  --stage stage4 \
  --phase analyze
```

## Notes

- `requirements.txt` is copied from the AirSim-oriented dependency list used by this repo.
- `environment.yml` is the current exported Conda environment file from the active development
  workspace, and the referenced local wheel is included under `deps/`.
- `configs/uav_dualcog/*.yaml` and `configs/prompts/uav_dualcog_prompts.yaml` are sanitized but
  parseable examples; replace placeholder API values before running model-backed stages.
- `common_stage_configs.yaml` and `uav_dualcog_prompts.yaml` preserve the implemented structure so
  reviewers can inspect the exact behavior-library and prompt interfaces.
