# FlightMVSTG Reviewer Code Package

This folder is a cleaned code-only snapshot for reviewers.

It keeps the Stage 1-4 entrypoints, shared runtime modules, simulator/trajectory adapters, and template configs that are sufficient to inspect the implemented pipeline logic without shipping any environment files, cached artifacts, generated datasets, media outputs, or private API credentials.

## Included

- `scripts/flightmvstg/`
  - Stage 1: point-cloud collection and fusion
  - Stage 2: landmark collection, review, and auto-labeling
  - Stage 3: trajectory generation, mission/task generation, and temporal evaluation support
  - Stage 4: QA generation and evaluation
  - shared helpers used by these stages
- `sim_bridge/`
  - engine bridge factory and adapters
- `trajectory/`
  - low-level behavior library and trajectory composition helpers
- `configs/flightmvstg/templates/`
  - scene config template
  - common stage config template
  - API runtime template
  - task pipeline template
- `configs/prompts/templates/`
  - prompt configuration template
- `docs/FlightMVSTG工具链设计.md`
  - design reference aligned with the current stage structure

## Not Included

- any `.env` or environment bootstrap files
- any API keys, internal endpoints, or private model-routing config
- `scene_data/`, `task_pipeline_data/`, website outputs, logs, or cached artifacts
- legacy backup scripts and one-off repair utilities

## Expected Layout

```text
reviewer_code_repo/
├── configs/
│   ├── flightmvstg/templates/
│   └── prompts/templates/
├── docs/
├── scripts/flightmvstg/
├── sim_bridge/
├── trajectory/
├── coord_transform_utils.py
├── requirements.txt
└── README.md
```

## Template Setup

The code expects non-template config paths at runtime. Before running, copy the templates into working config filenames.

```bash
mkdir -p configs/flightmvstg configs/prompts

cp configs/flightmvstg/templates/scene_config.template.yaml \
  configs/flightmvstg/task_airsim_env_demo.yaml

cp configs/flightmvstg/templates/common_stage_configs.template.yaml \
  configs/flightmvstg/common_stage_configs.yaml

cp configs/flightmvstg/templates/common_api_runtime.template.yaml \
  configs/flightmvstg/common_api_runtime.yaml

cp configs/flightmvstg/templates/task_pipeline.template.yaml \
  configs/flightmvstg/task_pipeline_demo.yaml

cp configs/prompts/templates/flightmvstg_prompts.template.yaml \
  configs/prompts/flightmvstg_prompts.yaml
```

Then fill in:

- the real scene id / scene directory in the scene config
- simulator connection fields in `engine_params`
- model routing and `${ENV_VAR}` tokens in `common_api_runtime.yaml`
- the pipeline spec paths and task counts in `task_pipeline_demo.yaml`

## Main Entrypoints

```bash
# Stage 1
python scripts/flightmvstg/stage1_collect_pcd.py \
  --config configs/flightmvstg/task_airsim_env_demo.yaml \
  --scene-id env_demo \
  --mode all \
  --engine airsim

# Stage 2
python scripts/flightmvstg/stage2_landmark_label.py \
  --config configs/flightmvstg/task_airsim_env_demo.yaml \
  --scene-id env_demo \
  --mode collect_instances

# Stage 3
python scripts/flightmvstg/stage3_generate_traj.py \
  --config configs/flightmvstg/task_airsim_env_demo.yaml \
  --scene-id env_demo \
  --mode generate_mission

# Stage 4
python scripts/flightmvstg/stage4_qa_generate_and_eval.py \
  --config configs/flightmvstg/task_airsim_env_demo.yaml \
  --scene-id env_demo \
  --mode generate
```

## Notes

- `requirements.txt` is copied from the AirSim-oriented dependency list already used in this repo.
- `common_stage_configs.template.yaml` and `flightmvstg_prompts.template.yaml` preserve the implemented structure so reviewers can inspect the exact behavior/prompt interfaces.
- `common_api_runtime.template.yaml` is intentionally sanitized and uses environment-variable placeholders instead of real keys or internal endpoints.
