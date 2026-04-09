# FlightMVSTG 工具链设计（实现对齐完整版）

> 更新时间：2026-03-31
> 对齐代码：
> - Stage 1：`scripts/flightmvstg/stage1_collect_pcd.py`
> - Stage 2：`scripts/flightmvstg/stage2_landmark_label.py`
> - Stage 3 任务/实验/工作台：`scripts/flightmvstg/stage3_generate_traj.py`、`scripts/flightmvstg/stage3_task_suite.py`
> - Stage 4 图像 QA：`scripts/flightmvstg/stage4_qa_generate_and_eval.py`
> - 流水线编排：`scripts/flightmvstg/task_pipeline.py`
> - 统一桥接与运行时：`scripts/flightmvstg/pipeline_common.py`、`sim_bridge/*`

---

## 1. 设计目标与总体定位

FlightMVSTG 的目标不是只做一份静态数据集，而是提供一条**从仿真场景、语义点云、地标资产、多视角图像、连续飞行视频，到结构化 QA 与统一评测**的完整自动化工具链。其核心关注点是：

1. 在**空中视角**下评测多模态模型的空间理解与时空推理能力。
2. 同时覆盖**主体自我状态理解**与**环境状态/探索决策理解**。
3. 同时覆盖**图像任务**与**视频任务**，并保持任务、资产、实验与 Web 工作台的统一组织。
4. 支持从场景级原始几何到 benchmark 样本的**可复现、可扩展、可审计**生成流程。

从实现上，当前系统已形成以下闭环：

- Stage 1：构建带语义/实例属性的场景级点云。
- Stage 2：生成并复核有效地标实例，形成带多视角资产和结构化描述的 `valid_instances.json`。
- Stage 3：定义参数化飞行行为，生成视频任务，组织时序 QA 与实验评测。
- Stage 4：生成图像 QA，组织实验与统一结果汇总。
- Task Pipeline：负责跨场景 selection / data / render / experiment / analyze 的统一编排。

---

## 2. Benchmark 能力空间与任务体系

FlightMVSTG 从能力上分为两条主线：

### 2.1 Self-aware reasoning

回答主体关于“我”的问题，关注 UAV 对自身相对状态和当前行为的理解。

- 图像任务：`self_where`、`self_what`
- 视频任务：`self_instance_recognition_joint`
- 统一问题形式：
  - 我相对地标在哪里？
  - 我当前在执行什么动作或任务？
  - 如果继续执行该动作，我接下来会看到什么？

### 2.2 Environment-aware reasoning

回答主体关于“环境”的问题，关注 UAV 对目标位置、可见性与下一步探索行为的理解。

- 图像任务：`env_where`、`env_how`
- 视频任务：`env_visibility_reasoning`
- 统一问题形式：
  - 目标相对我在哪里？
  - 为了接近目标，我应该朝哪个方向飞？
  - 目标在整个视频中何时出现、出现了几次？

### 2.3 当前核心发布集与扩展能力

按当前实现与任务配置，建议将 benchmark 分成：

#### 2.3.1 核心发布集（当前最完整）

- **Stage 4 单地标图像集**
  - 12 个测试场景
  - 512 个有效地标
  - 四类任务：`self_where`、`self_what`、`env_where`、`env_how`
  - 两档难度：`4way` / `8way`
  - 总样本数：`512 x 4 x 2 = 4096`

- **Stage 3 单地标视频集**
  - 12 个测试场景
  - 同样基于 512 个有效地标
  - 两类视频 QA 任务：`self_instance_recognition_joint`、`env_visibility_reasoning`
  - 每个地标 1 个 atomic mission + 1 个 composite mission
  - 总 QA 数约为：`512 x 2 mission types x 2 task forms = 2048`

#### 2.3.2 扩展任务（工具链已支持）

- Stage 3 多地标扩展：
  - `pair` / `triple` 组任务
  - `multi-landmark` joint 行为识别
  - 多目标环境可见性推理
- 更丰富的 composite 行为族与多目标调度
- 更大规模跨模型实验矩阵与云 API 限流实验

因此，从论文角度更准确的说法应当是：

- **当前工具链已完整支持单地标核心 benchmark，并支持多地标视频扩展。**
- **当前最稳定、最完整的测试集是单地标图像 + 单地标视频。**

---

## 3. 图像任务体系（Stage 4）

Stage 4 当前实现的图像任务为四类，统一由 `stage4_qa_generate_and_eval.py` 生成、实验和汇总。

### 3.1 方向标签空间与参照系

#### 3.1.1 难度标签空间

当前方向标签采用两档粒度：

- `4way`
  - `Front`
  - `Back`
  - `Left`
  - `Right`

- `8way`
  - `Front`
  - `Front-Left`
  - `Front-Right`
  - `Left`
  - `Right`
  - `Back`
  - `Back-Left`
  - `Back-Right`

#### 3.1.2 参照系语义

Stage 4 当前使用两类核心语义：

- `Object-Centric View`
  - 地标自身坐标系下的朝向标签，即“看到的是地标的哪一侧”
- `Observer-Centric View`
  - 相机相对地标的观察侧标签，即“相机从地标的哪一侧在看它”

在 prompt 中，系统只解释当前任务所需的那一种定义，而不是把两种定义同时写进 system prompt。

### 3.2 Self-Where

**任务目标**：判断 UAV 当前位于地标的哪个方向，并输出地标在 query image 中的 bbox。

**输入**：
- 一张参考图 `reference_image_with_bbox`
- 一张当前图 `target_image`
- 与 `difficulty` 对应的方向选项集合（`4way` 或 `8way`）

**输出**：
- `answer_option_id`
- `bbox_xyxy_norm`

**考察能力**：
- 观察视角与地标相对位置判断
- 从参考视角迁移到当前视角的 allocentric/self-aware 推理
- 目标定位

### 3.3 Self-What

**任务目标**：给定当前参考图和一个 orbit 行为实例，从 4 张候选未来图中选出正确的未来观测。

**输入**：
- 参考图（带 bbox）
- 4 张候选未来图
- 行为实例描述（当前来自 `behavior_templates.stage4.orbit_action`）

**输出**：
- `answer_option_id`
- `bbox_xyxy_norm`

**考察能力**：
- 当前姿态与未来观测之间的动作后果推理
- 视角变化与行为模式映射
- 对 orbit 动作的未来画面预测

### 3.4 Env-Where

**任务目标**：在 observation 图中判断目标地标相对 UAV 前向位于哪个方向，并输出 bbox。

**输入**：
- 一张 object-centric 参考图（带 bbox）
- 一张共享 observation 图
- 与 `difficulty` 对应的方向选项集合（`4way` 或 `8way`）

**输出**：
- `answer_option_id`
- `bbox_xyxy_norm`

**考察能力**：
- 目标相对前向方向判断
- 目标定位
- 从 object-centric reference 到当前视角的环境推理

### 3.5 Env-How

**任务目标**：给定当前 observation 图，判断为了接近目标地标，UAV 应朝哪个方向飞行。

**输入**：
- 一张 object-centric 参考图（带 bbox）
- 一张共享 observation 图
- 与 `difficulty` 对应的方向选项集合（`4way` 或 `8way`）

**输出**：
- `answer_option_id`
- `bbox_xyxy_norm`

**考察能力**：
- 环境状态到动作方向的决策映射
- 目标位置理解与下一步行为选择
- “看到什么”到“怎么飞”的 policy-level reasoning

### 3.6 Env 图像生成逻辑（当前实现）

当前 `env_where` / `env_how` 的 observation 图采用统一生成逻辑：

1. 先选参考图，优先主视图；
2. 再从地标周围重新选择一个新的 orbit 观察位姿；
3. 在该位姿基础上，对相机朝向做：
   - 左转 `30°`
   - 不转
   - 右转 `30°`
4. 先尝试得到 **bbox 完整落入视野** 的 observation 图；
5. 若初始朝向不满足，则允许小幅朝向微调；
6. 若仍不满足，则换新的观察位姿重试；
7. 最终实在无法做到整框可见时，退化到最佳部分可见方案，但不放弃该请求；
8. 在**最终确定的方案之后**，才计算最终 bbox 与方向标签。

这保证了当前 env 任务中：

- `env_where` 与 `env_how` 共享同一 observation 图；
- 答案来自最终确认后的相机位姿与朝向；
- bbox 不是中间方案的副产物，而是最终方案的结果。

### 3.7 Stage 4 指标体系

当前 Stage 4 report 的统一指标包括：

- `parse_success_rate`
- `option_accuracy`
- `bbox_acc@50iou`
- `bbox_mean_iou`
- `avg_latency_ms`

并支持按以下维度聚合：

- `view_definition`
- `task_type`
- `difficulty`
- `combo(view_definition|task_type|difficulty)`

---

## 4. 视频任务体系（Stage 3）

Stage 3 当前是一个统一工作台，覆盖：

- mission generation
- final task video generation
- candidate review
- dataset manifest generation
- experiment execution
- result aggregation
- unified web workspace

### 4.1 当前 Stage 3 任务形式

#### 4.1.1 Self-state 视频任务

当前主任务是：

- `self_instance_recognition_joint`

该任务要求模型观看第一人称飞行视频，识别视频中出现了哪些行为，并返回对应区间。

从评测层面，系统进一步从 joint 任务中派生出两个细粒度指标：

- `self_composite_instance_recognition`
- `self_atomic_instance_recognition`

也就是说：

- manifest / prompt 层以 joint 任务为主
- report / metrics 层会进一步拆成 composite-level 与 atomic-level 指标

#### 4.1.2 Environmental 视频任务

当前环境任务是：

- `env_visibility_reasoning`

该任务要求模型判断目标地标：

- 在视频中出现了几次
- 分别出现在哪些时间区间
- 并按需要给出 whole-second keyframes 对应的 bbox

### 4.2 Stage 3 任务字段

当前 Stage 3 manifest 样本中，核心字段包括：

- 任务身份：
  - `sample_id`
  - `task_name`
  - `mode(single-landmark|multi-landmark)`
- 任务语义：
  - `mission_id`
  - `mission_family`
  - `set_id`
  - `set_name`
  - `element_instances`
  - `element_sequence`
  - `behavior_intervals_sec`
- 地标信息：
  - `landmark_id`
  - `landmark_ids`
  - `landmark_category`
  - `landmark_subcategory`
  - `landmark_description`
  - `landmark_descriptions`
  - `landmark_set_map`
- 媒体资产：
  - `reference_image_with_bbox`
  - `overview_image`
  - `keyframe_board_image`
  - `video_path`
  - `video_web_path`
- 监督信号：
  - `visible_count`
  - `visible_intervals_sec`
  - `keyframe_gt_dense`
  - `self_state_keyframe_gt_dense`
- 难度：
  - `difficulty_score`
  - `difficulty_band`

### 4.3 Stage 3 遗留任务模板

当前 Stage 3 prompt 配置中还保留了以下旧模板：

- `self_set_instance_recognition`
- `self_element_instance_recognition`
- `env_visibility_reasoning_multi`

其中：

- 前两者在当前主数据链中已基本由 `self_instance_recognition_joint` 取代；
- `env_visibility_reasoning_multi` 当前标注为 deprecated。

因此从论文叙述上，更建议将当前主评测任务描述为：

- 一个 joint self-state 视频任务，派生两类细粒度指标；
- 一个 environmental visibility 视频任务。

### 4.4 Stage 3 指标体系

#### 4.4.1 Self-state 指标

- `set_instance_acc` / `main_metric`（composite-level）
- `element_instance_precision`
- `element_instance_recall`
- `element_instance_f1`
- `self_temporal_loc_f1@0.5`
- `self_temporal_loc_mean_tIoU`

#### 4.4.2 Environmental 指标

- `count_exact_acc`
- `count_within1_acc`
- `count_mae`
- `segment_f1@0.3`
- `segment_f1@0.5`
- `mean_best_tiou`

#### 4.4.3 大表展示规则

当前 Stage 3 实验大表已按任务性质做了列裁剪：

- `self_*`
  - 只显示 `Main`、`SelfF1@0.5`、`tIoU`
- `env_visibility_reasoning`
  - 只显示 `Main`、`SegF1@0.5`、`tIoU`

因此不会再把不适用指标显示成 `-`。

补充说明：

- Stage 3 的 CSV 导出仍会保留 `bbox_acc@50iou` 等兼容字段，用于关键帧可视化评估与调试；
- 但当前 Web 指标大表的主展示口径中，`env_visibility_reasoning` 的第三列是 `mean_best_tIoU`，不是 `BBox`。

---

## 5. 飞行体系设计：严格两层结构

从当前实现出发，Stage 3 的 flight system 应严格表述为**两层结构**：

1. **Atomic layer**：最小可执行飞行动作单元；
2. **Composite layer**：由多个 atomic 元素按顺序组织而成的完整任务流程。

这里需要特别强调：

- `single-landmark` 与 `multi-landmark` **不是与 atomic / composite 并列的第三层或第四层**；
- 它们更准确地说是 **composite layer 的两种应用形态**；
- 即：同一套 composite 设计既可以围绕一个目标地标展开，也可以扩展为多个目标地标的串联巡检任务。

这一点对于论文表述非常重要，因为如果把 `single / multi` 与 `atomic / composite` 写成同层级概念，审稿人很容易认为 flight taxonomy 不够严格，层次设计存在混杂。

### 5.1 Atomic layer

当前实现中的 atomic 行为库主要围绕“接近、离开、环绕、扫描、抬升”等基本飞行模式构建。代码中已明确出现或被稳定使用的元素包括：

- `gradual_approach`
- `gradual_depart`
- `circular_orbit`
- `spiral_orbit`
- `square_orbit`
- `triangular_orbit`
- `figure8_orbit`
- `surface_mapping`
- `sky_rise`
- `comet`

这些 atomic 行为一般具有以下参数：

- 平面延展尺度（如 `extension_m`、`extension_x_m`、`extension_y_m`）
- 高度偏移（如 `altitude_offset_m`）
- 旋转方向（clockwise / counterclockwise）
- 扫描/环绕角度或周期数
- 相机模式与朝向偏移（如跟踪地标、向前看等）

### 5.2 Composite layer

在 atomic 之上，系统定义了更高层的 composite 任务结构，用于表达更接近真实巡检任务的完整 flight workflow。当前工具链中稳定支持或明确出现的 composite 任务族包括：

- `circular_inspection`
- `spiral_inspection`
- `square_inspection`
- `triangular_inspection`
- `surface_mapping`
- `multi_landmark_composite_inspection`

这些 composite 任务通常不是单段环绕，而是一个**多步流程**：

- 从远处靠近目标
- 进入主体观测阶段
- 在观测阶段执行若干 atomic 子动作
- 中间可能发生 camera mode 切换
- 最后离开目标或转向下一个目标

### 5.3 Composite layer 的两种应用形态

#### 5.3.1 Single-landmark composite application

当前单地标任务是最稳定的数据主干。这里的 single-landmark 并不是新的层，而是 composite layer 的一种最基本应用方式：

- 一个 composite mission 围绕单个目标地标展开；
- 同时配套一个 atomic mission 作为对照；
- 每个 mission 后续派生 self-state 与 environmental 两类视频 QA。

#### 5.3.2 Multi-landmark composite application

当前多地标扩展则是 composite layer 的进一步应用形式。当前工具链支持：

- `pair`
- `triple`
- `multi_landmark_composite_inspection`

系统会基于多个目标地标构造一个长任务序列，并通过 `landmark_set_map` 将复合任务与具体目标关联起来。当前工具链中，multi-landmark 更适合做扩展实验、难度分析或后续增量发布。

### 5.4 障碍物与轨迹修复

当前 Stage 3 在生成行为参数和视频轨迹时，已经接入障碍物感知与轨迹修复机制：

- 对 obstacle-sensitive elements，会生成一组候选参数变体；
- 基于局部障碍点云和 keepout boxes 评估 collision-free 性；
- 对 orbit / mapping 等高风险动作，会调整半径、扫描范围或高度；
- 若轨迹仍然不安全，则使用 repair-lift 等方式尝试抬升修复；
- 失败样本会记录到 task pipeline 的 `failed_landmarks.jsonl` 中。

### 5.5 并行视频渲染与空间调度

为降低并行渲染时多架无人机互相进入画面的问题，当前 `record_scene_videos_cli(...)` 已加入专门调度策略：

- 同一批并行任务的目标地标尽量空间分散；
- 同一批任务的采样点数尽量接近；
- 同一批内尽量避免同一地标的 atomic/composite 同时出现；
- Stage 3 render 默认支持断点续跑，已完成的视频不会重复录制。

### 5.6 当前 Stage 3 渲染规格更新

当前实现中，Stage 3 final-task 渲染已经明确区分“归档帧图”和“编码视频”：

- `final_task/frames/*.jpg`
  - 由 capture-resolution 图像写出
  - 当前目标规格为 `4096x3072`（4:3）
- `final_task/task_rgb.mp4`
  - 由独立的视频画布编码
  - 当前目标规格为 `1440x1080`（4:3）
  - 默认编码为 `H.264`
  - 默认目标码率为 `10Mbps`

当前 `task_data.json` 中已同时写出：

- `video.width / video.height`：编码视频分辨率
- `video.frame_width / video.frame_height`：归档帧图分辨率

当前图像压缩逻辑也已收敛为：

- 默认目标大小 `1MB / 张`
- 只调 JPEG 质量，不再通过缩小分辨率满足目标大小
- Stage 3 / Stage 4 共用 `image_compression_utils.py`

---

## 6. Stage 1：场景点云采集与融合

### 6.1 输入

Stage 1 输入来自场景配置 `task_airsim_env_*.yaml`，核心包括：

- `task.scene_id`
- `task.engine`
- `traj_map.MapBound`
- `traj_map.LidarDelta`
- `engine_params.<engine>.sim_port`
- `engine_params.airsim.lidar_range_mode / lidar_range`
- `parallel.workers`
- `merge.voxel_size / sor_*`

### 6.2 主要技术

- 多位姿 LiDAR / RGB / segmentation 采集
- 分片原始数据组织
- 融合去重与体素化
- 语义属性与实例属性保留
- 统一坐标系映射（尤其 AirSim ENU/NED 对齐）

### 6.3 主要产物

默认目录：`scene_data/<engine>_<scene_id>/pcd_map/`

- `raw/`：分片采集结果
- `<scene_id>.semantic_lidar.pcd/.ply`
- `<scene_id>.raw.pcd`
- `semantic_lidar_raw.npy`
- `semantic_lidar_compact.npy`
- `semantic_lidar_instance.npy`
- `<scene_id>.meta.json`

### 6.4 设计要点

- 当前 Stage 2 的主下游输入是带实例/语义属性的 `semantic_lidar.pcd`
- `probe_airsim_mapbound.py` 可先自动估计有效场景边界
- `run_stage1_stage2_collect_serial.py` 可串行执行 `probe -> stage1 -> stage2 collect_instances`

---

## 7. Stage 2：地标资产构建与语义标注

### 7.1 目标

Stage 2 负责把 Stage 1 的场景级点云转成**有效地标资产库**，形成下游 Stage 3 和 Stage 4 的共同输入。

### 7.2 主要步骤

1. 按 `class_id + instance_id` 聚合语义点云，形成候选地标实例；
2. 为每个实例生成多视角 RGB / bbox / 几何摘要；
3. 在 Web 工作台中进行人工复核：
   - 保留/剔除实例
   - 修正主视图与方向定义
4. 调用 VLM 自动生成三级结构化语义：
   - `category`
   - `subcategory`
   - `description`
5. 将复核与自动标注结果写回最终 `valid_instances.json`。

### 7.3 自动标注设计

当前 Stage 2 自动标注链具有如下特点：

- 输入为多视角图像与几何上下文；
- 输出结构化字段：
  - `landmark_category`
  - `landmark_subcategory`
  - `landmark_description`
- 支持：
  - 上传图像压缩
  - bbox 叠加
  - VLM 置信度阈值过滤
  - RPM / TPM 限速
  - 自动重试与 debug 图保存

### 7.4 主要产物

默认目录：

- `landmarks_raw/`
- `landmarks_review/`
- `landmarks/`

关键文件：

- `<scene_id>.valid_instances.json`
- review index / log
- auto-label 调试与结果文件

### 7.5 Stage 2 -> Stage 3/4 契约

`valid_instances.json` 是当前后续阶段最重要的 handoff 文件，至少包含：

- `instance_id`
- `class_id`
- `class_name`
- `center_3d`
- `bbox_3d`
- 多视角图像与方向标注
- `landmark_category`
- `landmark_subcategory`
- `landmark_description`

---

## 8. Stage 3：视频任务生成、复核、实验与工作台

### 8.1 子流程结构

Stage 3 当前可理解为一个统一的“视频 benchmark 工作台”，内部子流程包括：

1. **Mission generation**
   - 基于目标地标和行为库生成高层 mission
2. **Final task generation**
   - 输出最终视频、时序标注、关键帧 GT 和 `task_data.json`
3. **Candidate discovery / review**
   - 从 mission 目录中发现已完成任务，组织候选与人工复核
4. **Manifest generation**
   - 将视频候选转成结构化 QA manifest
5. **Experiment**
   - 以结构化 prompt 对模型发起视频实验
6. **Metrics / Web**
   - 汇总 report、矩阵指标、逐样本详情和全局跨场景视图

### 8.2 Mission 与 final task 产物

默认目录：`scene_data/<scene>/stage3_tasks/`

- `missions/<traj_id>/`
  - `mission.json`
  - waypoints / preview / metadata
  - `final_task/task_data.json`
  - `final_task/task_rgb.mp4` 或同类视频资产
- `review/`
  - candidate review index / log
- `datasets/`
  - stage3 manifests
- `experiments/`
  - requests / responses / parsed / report / failed indices
- `reports/`
  - latest report summaries
- `cache/`
  - keyframe eval / web cache

### 8.3 Candidate 组织方式

当前 Stage 3 候选来自 `missions/*/final_task/task_data.json`，系统会从中提取：

- mission / set / element 元信息
- 参考图与 bbox
- 视频路径
- visible intervals
- keyframe GT
- behavior intervals
- review status
- render status

只有满足 `require_final_task=True` 的候选，才会进入正式 manifest 生成与实验。

### 8.4 Stage 3 Web 工作台

当前网页支持：

- 行为库查看
- mission 生成与预览
- 候选复核
- manifest 查看
- 实验执行
- 结果查看
- 指标汇总
- **ALL scenes** 全局聚合查看
- 新增的**实验进度大表**（模型 x 场景）

全局模式下：

- 结果查看、manifest/report/metrics 汇总可跨场景聚合
- 写操作为只读，不允许直接在 `ALL scenes` 下启动生成或写回复核

---

## 9. Stage 4：图像 QA 生成、实验与工作台

### 9.1 核心流程

Stage 4 的主流程为：

1. 从 `valid_instances.json` 选择有效地标；
2. 为每个地标准备四类 QA 样本；
3. 生成并写出 manifest；
4. 调用 API / 本地模型执行实验；
5. 写出 requests / responses / parsed / report；
6. 在 Web 工作台中查看任务、结果、指标与实验进度。

### 9.2 Manifest 构建方式

当前 `generate_manifest(...)` 支持：

- 指定 `sample_count`
- 指定 `reference_main_only`
- 指定 `difficulties`
- 指定 `task_types`
- 指定 `landmark_categories`
- 指定 `selected_landmark_ids`

并支持多进程数据准备：

- `data_prepare_parallel_workers`

### 9.2.1 Stage 4 render-only 资产重采

当前实现中，Stage 4 已支持独立的 render-only 资产重采路径：

- `task_pipeline.py --stage stage4 --phase render`

该路径的语义是：

- 读取现有 manifest
- 重绘 `reference_bbox`
- 重采 `env_observations`
- 不改动样本 ID、选项、答案等任务语义字段

为兼容旧 manifest，当前实现还支持：

- `qa/render_requests/*.json`
- 对历史 manifest 做 backfill
- 基于 `answer_bbox_xyxy_norm` 的近似拟合反推 env render request

### 9.2.2 当前 Stage 4 4K 重采实现要点

当前 Stage 4 4K 重采链路已经补齐以下稳定实现：

- `stage4.*` task-pipeline 配置可直接下发到 Stage 4 runtime
- env capture 请求会按 `request_id` 去重，避免 `env_where` / `env_how` 重复采同一张图
- env capture 当前按连续区间切分给 worker，worker 生命周期更接近 Stage 3 的稳定实现
- Stage 4 已补齐运行期日志：
  - preparing runtime
  - startup ports
  - worker connected
  - capture progress
- 出于 4K 并行稳定性考虑，Stage 4 当前 AirSim capture 已改为 RGB-only，而不是默认启用 depth / segmentation

### 9.3 Stage 4 工作台

当前 Stage 4 网页支持：

- 任务生成
- manifest 查看
- 多模型实验
- 后台 job 管理
- report 查看
- metrics matrix
- **ALL scenes** 全局聚合
- 新增的**实验进度大表**

当前进度大表规则为：

- 只统计当前 task 对应的场景（例如测试集 12 个场景）
- 只统计已有实验痕迹的模型
- `completed` 按 `parsed_predictions.jsonl` 的唯一 `sample_id` 数统计
- `total` 按当前场景 latest manifest 的样本总数统计

---

## 10. 实验运行时设计

### 10.1 API 路由与模型名规则

当前 API 源匹配与请求格式处理是分开的：

#### 10.1.1 API 源匹配

- 按基础模型名**精确匹配** `common_api_runtime.yaml` 中的 key
- 大小写、前缀、下划线/点号不一致，就不会命中 API 源

#### 10.1.2 请求格式匹配

- `Qwen` 与 `InternVL` 的 `-Instant / -Thinking / -Reasoning` 后缀按宽松规则识别
- 即使大小写或前缀不同，请求格式匹配仍可识别 `qwen` / `internvl` 家族

### 10.2 Qwen / InternVL 特殊请求格式

#### Qwen

- `-Thinking`
  - 直接正常请求
- `-Instant`
  - 在 `extra_body.chat_template_kwargs.enable_thinking=false`

#### InternVL

- `-Thinking`
  - 使用官方 R1 风格 thinking system prompt
  - system message 改成 content blocks 形式
- `-Instant`
  - assistant 预填 `<think>\n</think>\n`
  - `extra_body.continue_final_message=true`
  - `extra_body.add_generation_prompt=false`

### 10.3 请求记录

当前 requests 记录已扩展为可复现实验格式，包含：

- `request_model`
- `api_source`
- `api_base`
- `reasoning_mode`
- `assistant_prefill`
- `system_prompt_prefix`
- `system_prompt_as_blocks`
- `request_extra_body`
- `messages_preview`

其中 `messages_preview` 用于避免 prompt 字段重复记录。

### 10.4 Retry 与失败样本索引

当前实验链支持：

- 请求异常 / 空回复时自动重试
- 解析失败不重试
- 非 retryable 的模型错误直接抛出
- 每个实验目录维护：
  - `failed_request_indices.json`

而且在 `unique_experiment=true` 的续跑模式下：

- 已成功补跑的失败样本会自动从 `failed_request_indices.json` 中移除
- 该文件始终表示“当前最新状态下仍然失败的样本集合”

### 10.5 Unique experiment 与断点续跑

当前 Stage 3 / Stage 4 experiment 都支持：

- `unique_experiment: true`

含义：

- `scene + model + manifest` 固定使用唯一实验目录
- 已完成样本从 `parsed_predictions.jsonl` 中恢复
- 续跑时只提交剩余样本
- report / failed indices 会按当前目录的最新状态重写

### 10.6 限流与并发

当前系统支持：

- 在 `common_api_runtime.yaml` 的模型项里写：
  - `rpm_limit`
  - `tpm_limit`
  - `rate_limit_reserve_ratio`
- 系统会自动：
  - 对当前待跑样本动态估算 token
  - 留出 10% 冗余
  - 把实际并发压到不超过任务配置并且不超过额度的范围
- 即使某些云模型单请求频率都太高，限流器也会自动 `sleep` 等待，而不是持续硬打 API

### 10.7 跨场景实验调度

当前 `task_pipeline` 在多场景 experiment 模式下，采用：

- **模型并行**
- **每个模型跨场景连续跑**

并支持：

- 场景级 completeness 检查
- 不完整场景跳过
- 完整场景优先实验
- 跨场景总体进度条
- 跨场景按模型的全局进度条

---

## 11. Task Pipeline：统一编排层

`scripts/flightmvstg/task_pipeline.py` 是当前测试集与批量实验的核心编排入口。

### 11.1 支持阶段

- `selection`
- `data`
- `render`
- `experiment`
- `analyze`
- `both`

### 11.2 支持能力

- 跨场景 landmark list 生成
- Stage 3 / Stage 4 数据构建
- Stage 3 render
- Stage 4 render-only 资产重采
- 多模型实验
- 场景级断点续跑
- unique experiment 续跑
- 多场景全局进度条
- CLI 直接覆盖 `--experiment-models`
- CLI 直接覆盖 `timeout_s` 等实验参数

### 11.2.1 当前 task pipeline 与 rerender 行为约定

当前 task pipeline 中与 rerender 相关的稳定行为包括：

- `stage3.rerender_existing: false`
  - 只补缺失/损坏 Stage 3 render 产物
- `stage4.rerender_existing: false`
  - 只补缺失/被手动删除的 Stage 4 资产
  - 不会主动覆盖已经存在的旧图片
- Stage 4 render-only 多场景 sweep 已补齐 render-only 收尾逻辑
  - 已完成场景可安全跳过
  - 不应再因 `stage4_manifest` 为空而在切换下一个场景前崩溃

### 11.3 当前常用测试集配置

- `task_pipeline_airmultiviewst_test_single.yaml`
  - 单地标核心测试集
  - Stage 4 全量图像任务
  - Stage 3 单地标视频任务
- `task_pipeline_airmultiviewst_test_multi.yaml`
  - 多地标 Stage 3 扩展任务

---

## 12. 统一目录、日志与 Web 产物

### 12.1 场景目录

默认：

- `scene_data/<engine>_<scene_id>/`

任务流水线输出：

- `task_pipeline_data/<task_name>/<engine>_<scene_id>/...`
- 元数据目录：
  - `task_pipeline_data/<task_name>/task_pipeline/`

### 12.2 Web / 实验资产

Stage 3：

- `stage3_tasks/datasets/`
- `stage3_tasks/experiments/`
- `stage3_tasks/reports/`
- `stage3_tasks/cache/`

Stage 4：

- `qa/manifests/`
- `qa/experiments/`
- `qa/assets/`

### 12.3 结果文件

统一实验目录下常见文件：

- `requests.jsonl`
- `responses.jsonl`
- `parsed_predictions.jsonl`
- `requests.txt`
- `responses.txt`
- `report.json`
- `failed_request_indices.json`

### 12.4 公开展示网站与内部工作台的边界

当前仓库里已经存在两类面向“网页”的能力，但它们定位不同：

1. **内部工作台（已实现）**
   - Stage 2 `review_instances_web`
   - Stage 3 mission / review / manifest / experiment / metrics 工作台
   - Stage 4 generate / dataset / experiments / results / metrics 工作台
2. **公开展示网站（建议新建）**
   - 面向论文读者、benchmark 使用者与 leaderboard 浏览者
   - 用作 supplementary material、方法展示、任务定义说明、样例浏览与结果发布页

二者不要混写为同一个系统：

- **内部工作台**偏生产与运营，强调任务生成、人工复核、实验调度、跨场景汇总；
- **公开展示网站**偏发布与传播，强调叙事结构、可读性、稳定外链、静态资源部署与可引用页面。

因此更合理的工程关系是：

- 继续保留现有 Flask 工作台作为内部 benchmark construction / experiment workspace；
- 从 `task_pipeline_data/`、`scene_data/`、`report.json`、`latest_manifest.json` 中导出稳定的 JSON / CSV / 压缩媒体资源；
- 由独立的公开网站前端消费这些导出产物，生成 Home / Tasks / Dataset / Evaluation / Leaderboard / Analysis / Supplementary 页面。

当前仓库里已经可直接复用为公开网站数据源的脚本包括：

- `scripts/flightmvstg/export_dualcog_dataset_stats.py`
- `scripts/flightmvstg/export_dualcog_benchmark_figures.py`
- `scripts/flightmvstg/export_metrics_matrix_csv.py`

这意味着：**实现上，公开网站不需要直接耦合 Stage 3 / Stage 4 的在线 Flask API，而更适合构建为一个静态优先、数据导出驱动的网站。**

---

## 13. 当前最适合写论文的表述方式

基于当前实现，论文中最准确的写法建议是：

1. **FlightMVSTG 是一条从语义点云到多视角图像与时序视频 QA 的自动化 benchmark construction pipeline。**
2. **当前核心 benchmark 由单地标图像任务和单地标视频任务构成。**
3. **工具链同时支持多地标视频扩展、云 API 大规模实验、跨场景汇总和 Web 工作台。**
4. **Self-aware / Environment-aware 是能力维度，Image / Video 是任务载体维度。**

不要把当前系统表述成“已经完整发布所有 single + multi image/video 组合的最终 benchmark”；更准确的是：

- **核心发布集已稳定；多地标视频扩展已由工具链支持并可增量发布。**

---

## 14. 论文写作建议（与实现一致）

如果用于论文正文，推荐这样组织：

1. 问题定义：空中多视角时空推理 benchmark
2. 能力划分：self-aware vs environment-aware
3. 任务划分：image vs video
4. 飞行体系：two-layer atomic / composite taxonomy，以及 composite 在 single- / multi-landmark 场景下的应用
5. 自动化构建：Stage 1–4 + task pipeline
6. 统一实验：结构化 JSON 输出、bbox、时序区间、跨模型实验
7. 数据统计：当前核心 split 与扩展 split

---

## 15. 一句话总结

FlightMVSTG 当前不是单一脚本，而是一套**面向空中多视角时空推理 benchmark 的完整工具链**：

- Stage 1 构建可计算的场景几何与语义基础；
- Stage 2 构建可复用、可语言引用的地标资产；
- Stage 3 基于参数化飞行体系生成视频任务与时序评测；
- Stage 4 生成图像任务与多模型评测；
- Task Pipeline 负责测试集生产、跨场景实验与断点续跑；
- Web 工作台负责人工复核、任务查看、结果分析和全局汇总。
