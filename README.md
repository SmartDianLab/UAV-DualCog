# UAV-DualCog Reviewer Code Package

This folder is a cleaned, reviewer-facing code snapshot for **UAV-DualCog**.

- Website: https://uav-dualcog.lozumi.com/
- Code repo: https://github.com/SmartDianLab/UAV-DualCog
- Dataset (ModelScope): https://www.modelscope.cn/datasets/Lozumi/UAV-DualCog
- Dataset (Hugging Face): https://huggingface.co/datasets/Lozumi/UAV-DualCog (preparing)
- AerialVLN simulator: https://www.kaggle.com/datasets/shuboliu/aerialvln-simulators

For benchmark definitions, leaderboard interpretation, and detailed supplementary explanations, please read the website pages in order: Home -> Benchmark -> Construction -> Evaluation -> Leaderboard -> Analysis -> Usage.

## 1) What Is Included

- `scripts/uav_dualcog/`: Stage 1-4 entrypoints, pipeline orchestrator, shared utilities.
- `trajectory/`: hierarchical atomic/composite behavior library and composition logic.
- `sim_bridge/`: simulator abstraction and AirSim bridge.
- `configs/uav_dualcog/`: 18 runnable per-scene configs + shared configs + task-pipeline spec.
- `configs/uav_dualcog/templates/`: fully commented templates for customization.
- `configs/prompts/uav_dualcog_prompts.yaml`: Stage 2-4 prompt package.
- `environment.yml`, `requirements.txt`, `deps/`: environment/dependency references.

## 2) What Is Excluded

- No private keys or private endpoints.
- No generated artifacts (`scene_data/`, `task_pipeline_data/`, logs, caches, media outputs).
- No internal notes/workflow docs outside public release scope.

## 3) Full Workspace Structure (Code + Env + Data + Outputs)

```text
reviewer_code_repo/
├── scripts/uav_dualcog/                        # Stage 1-4 + task_pipeline entrypoints
├── trajectory/                                  # behavior elements/sets and composition
├── sim_bridge/                                  # AirSim bridge and engine adapter layer
├── configs/
│   ├── uav_dualcog/
│   │   ├── task_airsim_env_<id>.yaml           # runnable scene configs (18 scenes)
│   │   ├── common_stage_configs.yaml            # behavior library and shared stage defaults
│   │   ├── common_api_runtime.yaml              # model routing (API + local deployment)
│   │   ├── task_pipeline/
│   │   │   └── task_pipeline_uav_dualcog_v1.yaml
│   │   └── templates/                           # fully-commented config templates
│   └── prompts/
│       └── uav_dualcog_prompts.yaml
├── envs/                                        # simulator env assets (download separately)
│   └── airsim/
│       └── env_7/
├── scene_data/                                  # Stage 1-2 outputs
│   └── airsim_env_7/
│       ├── pcd_map/
│       ├── landmarks_raw/
│       └── landmarks_review/
├── task_pipeline_data/                          # Stage 3-4 outputs
│   └── UAV-DualCog-V1/
│       ├── airsim_env_7/
│       │   ├── video_tasks/
│       │   └── image_tasks/
│       └── task_pipeline/
│           ├── dataset_stats/
│           ├── exports/
│           └── landmark_lists/
├── environment.yml                              # conda environment reference
├── requirements.txt
└── deps/
```

## 4) Two Reproduction Modes

### Mode A: Data Construction (Stage 1-4)

Use this when reproducing benchmark construction from scene/simulator inputs.

Requires:
- `envs/airsim/env_*` simulator files.
- writeable `scene_data/` and `task_pipeline_data/`.
- stage configs + prompt package.

Recommended workflow:
1. Stage 1 collects and fuses scene point clouds.
2. Stage 2 collects landmark candidates and performs review/auto-labeling.
3. Stage 3 generates behavior-driven video tasks.
4. Stage 4 generates image tasks and evaluation manifests.

Important operational notes:
- Stage 2 Step 2-4 are completed in the internal review web (`review_instances_web` + auto-label flow).
- Stage 3 and Stage 4 both provide internal web workbenches for inspection (behavior library, landmark/task previews, experiment outputs), but for released split generation we recommend `task_pipeline.py` batch phases.

### Mode B: Experiment Only (No Scene Reconstruction)

Use this when you only evaluate models on released benchmark assets.

Requires:
- downloaded `task_pipeline_data/UAV-DualCog-V1` release.
- no simulator environment files needed.
- `common_api_runtime.yaml` configured (API or local).

## 5) Model Invocation Methods

`configs/uav_dualcog/common_api_runtime.yaml` supports:

1. **API routing** (`api_source: cloud/openrouter/...`)  
   Call remote OpenAI-compatible endpoints.
2. **Local deployment** (`api_source: local`)  
   Call local OpenAI-compatible serving endpoints.  
   The release package assumes local models are used as deployed, with no additional quantization handling logic in this code package.

For safe dry checks (no real model calls), run:

```bash
python scripts/uav_dualcog/api_common.py --help
python scripts/uav_dualcog/mock_api_runtime_check.py --config configs/uav_dualcog/common_api_runtime.yaml
```

## 6) Config Files You Should Edit

All templates below are fully commented:

- `configs/uav_dualcog/templates/scene_config.template.yaml`
- `configs/uav_dualcog/templates/common_stage_configs.template.yaml`
- `configs/uav_dualcog/templates/common_api_runtime.template.yaml`
- `configs/uav_dualcog/templates/task_pipeline.template.yaml`

Runnable examples are already provided under:

- `configs/uav_dualcog/task_airsim_env_*.yaml` (18 scenes)
- `configs/uav_dualcog/common_stage_configs.yaml`
- `configs/uav_dualcog/common_api_runtime.yaml`
- `configs/uav_dualcog/task_pipeline/task_pipeline_uav_dualcog_v1.yaml`

## 7) Executable Steps (Mode A: Construction)

Below uses `env_7` as example scene.

### Step 0. Environment

```bash
conda env create -f environment.yml
conda activate uav-dualcog
```

### Step 1. Stage 1 (Point Cloud Collection + Fusion)

Purpose: build segmented/fused scene cloud for landmark construction.

```bash
python scripts/uav_dualcog/stage1_collect_pcd.py \
  --config configs/uav_dualcog/task_airsim_env_7.yaml \
  --scene-id 7 \
  --mode all \
  --engine airsim
```

### Step 2. Stage 2 (Landmark Construction + Review + Auto-Label)

Purpose: construct landmark instances and finalize reviewed semantic annotations.

2.1 Collect candidates and multiview evidence:
```bash
python scripts/uav_dualcog/stage2_landmark_label.py \
  --config configs/uav_dualcog/task_airsim_env_7.yaml \
  --scene-id 7 \
  --mode collect_instances
```

2.2 Open review web (Step 2-4 are web-centered in practice):
```bash
python scripts/uav_dualcog/stage2_landmark_label.py \
  --config configs/uav_dualcog/task_airsim_env_7.yaml \
  --scene-id 7 \
  --mode review_instances_web \
  --host 0.0.0.0 \
  --port 20261
```

2.3 Auto-label reviewed instances:
```bash
python scripts/uav_dualcog/stage2_landmark_label.py \
  --config configs/uav_dualcog/task_airsim_env_7.yaml \
  --scene-id 7 \
  --mode auto_label
```

### Step 3. Stage 3 (Video Task Construction)

Purpose: generate missions/trajectories, render videos, build stage3 manifests.

Direct entrypoint:
```bash
python scripts/uav_dualcog/stage3_generate_traj.py --help
```

Recommended (batch/reproducible) pipeline phases:
```bash
python scripts/uav_dualcog/task_pipeline.py \
  --spec configs/uav_dualcog/task_pipeline/task_pipeline_uav_dualcog_v1.yaml \
  --stage stage3 --phase selection

python scripts/uav_dualcog/task_pipeline.py \
  --spec configs/uav_dualcog/task_pipeline/task_pipeline_uav_dualcog_v1.yaml \
  --stage stage3 --phase data

python scripts/uav_dualcog/task_pipeline.py \
  --spec configs/uav_dualcog/task_pipeline/task_pipeline_uav_dualcog_v1.yaml \
  --stage stage3 --phase render
```

Optional internal web workbench:
```bash
python scripts/uav_dualcog/stage3_generate_traj.py \
  --config configs/uav_dualcog/task_airsim_env_7.yaml \
  --scene-id 7 \
  --mode web
```

### Step 4. Stage 4 (Image Task Construction)

Purpose: sample image QA tasks, render assets, export stage4 manifests.

Recommended pipeline phases:
```bash
python scripts/uav_dualcog/task_pipeline.py \
  --spec configs/uav_dualcog/task_pipeline/task_pipeline_uav_dualcog_v1.yaml \
  --stage stage4 --phase selection

python scripts/uav_dualcog/task_pipeline.py \
  --spec configs/uav_dualcog/task_pipeline/task_pipeline_uav_dualcog_v1.yaml \
  --stage stage4 --phase data

python scripts/uav_dualcog/task_pipeline.py \
  --spec configs/uav_dualcog/task_pipeline/task_pipeline_uav_dualcog_v1.yaml \
  --stage stage4 --phase render
```

Optional internal web workbench:
```bash
python scripts/uav_dualcog/stage4_qa_generate_and_eval.py \
  --config configs/uav_dualcog/task_airsim_env_7.yaml \
  --scene-id 7 \
  --mode web \
  --port 20264
```

## 8) Executable Steps (Mode B: Experiment)

Purpose: run model evaluation on released task manifests without redoing scene construction.

```bash
# Stage 3 experiments
python scripts/uav_dualcog/task_pipeline.py \
  --spec configs/uav_dualcog/task_pipeline/task_pipeline_uav_dualcog_v1.yaml \
  --stage stage3 --phase experiment \
  --experiment-models openai/gpt-5.3-chat Qwen/Qwen3.5-9B

# Stage 4 experiments
python scripts/uav_dualcog/task_pipeline.py \
  --spec configs/uav_dualcog/task_pipeline/task_pipeline_uav_dualcog_v1.yaml \
  --stage stage4 --phase experiment \
  --experiment-models openai/gpt-5.3-chat Qwen/Qwen3.5-4B
```

If you only want to verify interface wiring (without real model calls), use `--help` on stage/pipeline scripts and validate config parsing paths first.

## 9) Smoke-Test Commands (Reviewer Quick Check)

```bash
python scripts/uav_dualcog/stage1_collect_pcd.py --help
python scripts/uav_dualcog/stage2_landmark_label.py --help
python scripts/uav_dualcog/stage3_generate_traj.py --help
python scripts/uav_dualcog/stage4_qa_generate_and_eval.py --help
python scripts/uav_dualcog/task_pipeline.py --help
python scripts/uav_dualcog/mock_api_runtime_check.py --config configs/uav_dualcog/common_api_runtime.yaml
```

These checks confirm runnable CLI interfaces before launching long construction or experiment jobs.
