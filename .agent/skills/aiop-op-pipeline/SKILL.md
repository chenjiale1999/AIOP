---
name: aiop-op-pipeline
description: "昇腾 A5（Ascend 950 / dav-3510）+ TileLang 算子端到端生成流水线：需求解析、知识召回、设计、生成、静态/编译/仿真/上板门禁、失败修复与报告归档。"
---

# aiop A5 算子生成流水线

本 skill 是**可执行流水线的操作手册**。所有步骤都有对应的 `aiop` 命令与产物；
不要凭记忆手写 API，事实以 `knowledge/`、`tilelang/ascend/README.md` 和源码为准。

## 适用场景

- 用 TileLang Ascend dialect 新增一个算子（elementwise / reduce / softmax / norm / GEMM）。
- 需要从自然语言或 `TaskSpec` 一路走到 A5 设备编译、仿真、上板实测。
- 生成失败时按固定分类修复，而不是盲目重写。

不适用：编译器 pass 开发、TileLang 运行时/C++ 修改、非 A5 目标。

## 0. 环境事实（先确认）

| 项 | 值 | 检查 |
|---|---|---|
| 目标 | Ascend 950PR/DT，`dav-3510`，NpuArch `3510` | `python -m aiop env` |
| A5 编译 | CANN **9.2+**（9.1 缺 A5 类型/指令） | `bisheng --version` |
| 模拟器 | `npusim record -s Ascend950` | `npusim --help` |
| 数值仿真 | 需匹配 CANN 的 `torch_npu` + 可见 NPU | `python -c "import torch_npu; print(torch.npu.is_available())"` |
| 上板 | SSH 公钥 + 远端 CANN 9.2 + TileLang 同 revision | `ssh -o BatchMode=yes <host> true` |

缺失的能力必须如实标记为 `unavailable`/`skipped`，**禁止伪造通过**。

## 0.1 双基准强制规则（所有生成与修复必须遵守）

1. **同时参考两类实现**：
   - CUDA 对等实现：`ref_reops/TileOPs/src/tileops/kernels/<family>/` 中同名算子的 `*Kernel`，
     决定功能语义、输入输出定义、计算顺序与精度预期；
   - Ascend A5 已验证实现：`ref_reops/TileOPs/src/tileops/kernels/ascend/` 中的
     `Ascend*Kernel`，决定代码结构、命名、注释、注册方式与硬件适配。
   检索结果见任务目录 `references.json` 与 `reference_context.md`（`aiop` 自动生成）。
2. **A5 适配**：只用 TileLang Ascend A5 API；遵守 UB 208 KiB / L1 512 KiB / L0C 256 KiB、
   一维 `T.Kernel` grid、`T.SimtVF`/`T.SimdVF` 线程域。
3. **精度**：输出与 CUDA 语义基准的 fp32 绝对误差 ≤ **1e-5**；fp16/bf16 以 dtype 分辨率
   为下限（脚本记录 `policy=dtype_floor` 与 `declared_atol=1e-5`）。
4. **集成**：模块同时提供 `def build()`、`class Ascend<Stem>Kernel` 与
   `register_kernel_type("<Op>", "ascend", <class>)`；目标路径由 `references.json` 给出，
   `python -m aiop integrate <task> [--apply]` 校验/安装并跑上游测试。
5. **门禁**：`blocked_reference`（缺基准）、`blocked_parity`（1e-5 未过）、
   `blocked_performance`（`perf.target` 未达）都必须如实上报，不得绕过。

## 1. 流水线总览

```text
需求 (自然语言 / TaskSpec / TileOPs manifest)
  ↓  aiop generate --spec / --requirement / --op <ManifestOp> --workload N
Stage 1 Design        规格校验 + 硬件预算 + 知识召回 → DESIGN.md
Stage 2 Generate      模板/LLM 生成 → {op}.py + test_{op}.py
Stage 3 Static        AST / API 白名单 / 约束 / 语义 → static_report.json
Stage 4 Compile       tilelang.lower + bisheng → compile_result.cce.c
Stage 5 Simulate      npusim/local NPU + golden → simulation_report.json
Stage 6 Board         SSH 部署 + 实测 → board_report.json
Stage 7 Report        report.md / result.json / manifest.json
```

门禁：Stage 3/4 必须通过才能算“生成成功”；Stage 5 通过才算“验证通过”；
Stage 6 只在配置了远端主机时执行。

### 输入来自 TileOPs manifest 时

```bash
python -m aiop manifest list                       # 条目 + aiop family 映射
python -m aiop manifest show MLAPagedFwdOp         # 签名与所有 workload 行
python -m aiop manifest convert RMSNormFwdOp --workload 0 --dtype-case 1 --out spec.yaml
python -m aiop generate --op GemmFwdOp --workload 1 --dtype-case 0
```

- 契约（`forall`/`params`/`optional`/`values`/`requires`/`mutated`/`let`/
  `shape_rules`/type family/`dtype_cases`）按 manifest 语义绑定，规则见
  `knowledge/10-manifest-spec.md`。
- 能映射到模板 family 的条目正常走 Stage 1–7。
- 不能映射的条目（attention/MoE/conv/...）在 Stage 1 抛 `UnsupportedOperator`：
  - **未配置 LLM** → 返回 `blocked_unsupported`，只写 `DESIGN.md`（契约表 + 元数据张量 +
    不支持原因 + 补齐路径），**不生成 kernel、不报任何测量值**；
  - **已配置 LLM** → 进入 open 模式（`Plan.variant="llm_open"`）：LLM 起草 kernel，
    tiling/UB 不做断言，但 L0–L2 门禁一条不放松；编译/静态失败带错误回到
    `repair_kernel` 循环；golden 若也由 LLM 提供则标记 `llm-self-referenced`，
    `verified` 恒为 `False`，状态最高 `success_static_only`。

```bash
export AIOP_LLM_BASE_URL=https://api.deepseek.com/v1
export AIOP_LLM_MODEL=<model id>
export AIOP_LLM_API_KEY=<key>
python -m aiop generate --op GQADenseFwdOp --workload 0
```

`--strategy auto`（默认）只对**没有模板**的 family 用 LLM；已有模板的仍走确定性模板。
**严禁**把 open 模式的自指 golden 当作独立验证证据。

## 2. 输入契约（TaskSpec）

```yaml
name: softmax
family: softmax          # elementwise_unary|elementwise_binary|row_reduce|softmax|rmsnorm|layernorm|gemm
formula: "softmax(x)=exp(x-max)/sum(exp(x-max))"
inputs:
  - {name: X, shape: [128, 1000], dtype: float32}
outputs:
  - {name: Y, shape: [128, 1000], dtype: float32}
precision: {atol: 1e-5, rtol: 1e-5, max_abs_error: 1e-4, matched_ratio: 0.9999}
perf: {metric: latency, target: null, unit: ms}
attrs: {eps: 1e-6}
```

- shape 必须为正静态整数；动态轴当前不支持，需先特化。
- `gemm` 约定 `A[M,K]`、`B[N,K]`（`transpose_B=True`）、`C[M,N] float32`。
- `row_reduce` 输出 shape 为输入去掉最后一维。
- dtype 必须在 A5 支持集合内（见 `knowledge/01`）。

## 3. 设计阶段（必须产出 DESIGN.md）

按顺序：

1. **瓶颈分类**：算 Bytes/FLOPs，判定 memory/compute/latency bound（`knowledge/06` §1）。
2. **骨架选型**：`knowledge/04` 模式 A–F；记录参考示例路径。
3. **容量预算**：UB 208KB / L1 512KB / L0C 256KB / L0A/B 64KB；含 buffer 版本与 padding。
4. **核与并行度**：Vector 64 block，Cube 32 block；`T.Kernel` 一维；线程域只在 `T.SimtVF`。
5. **尾部策略**：整 tile 整除 → UB-staged；否则 guarded 标量路径并标注性能风险。
6. **验证方案**：golden、dtype 精度标准、L0–L4 门禁。

DESIGN.md 必须包含：概述、编程模式、API 映射、Tiling、内存层级、循环/调度、同步、
验证方案、精度标准、风险点、知识引用。

## 4. 生成阶段

- 默认使用**已验证模板**（`aiop/generation/templates.py`）；模板结构与
  `examples/ascend/` 对齐。
- 配置 LLM（`AIOP_LLM_BASE_URL` / `AIOP_LLM_MODEL`）后可让模型起草或修复；
  LLM 输出必须通过完全相同的 Stage 3–6 门禁，失败自动回退模板。
- 生成文件：
  - `{op}.py`：纯 kernel，只有 `build()` + `@T.prim_func`，可独立 import。
  - `test_{op}.py`：自包含 golden + 运行/计时/JSON 输出（本机/远端通用）。

禁止：`T.Kernel(..., threads=)`、多维 grid、动态循环边界、GM→L0 跨级搬运、
reducer 句柄读写、调试插桩留在交付代码。

## 5. 静态门禁规则（节选）

| Rule | 含义 | 修复 |
|---|---|---|
| SYN002 | Python 语法错误 | 修语法 |
| SYN004 | 缺少 `@T.prim_func` | 补 prim_func |
| SYN006 | kernel import torch/numpy/IO | 移除 |
| API001 | 未知 `T.*` API | 查 API 白名单/源码 |
| API002 | `T.Kernel(threads=)` | 移到 `T.SimtVF` |
| HW002 | 容量/预算超限 | 降 tile/rows/stages |
| HW007 | GEMM block 不整除 | 用 gcd 选 block |
| SEM001 | reducer init/finalize 不匹配 | 补齐生命周期 |
| SEM011/12 | guarded 路径缺 guard | 补 `idx<N` / `row<BATCH` |

## 6. 编译门禁

```bash
python -m aiop compile workspace/<task>
```

- L1：`tilelang.lower(target="ascend")`，必须产出 CCE 源。
- L2：`bisheng --npu-arch=dav-3510` 编成 aibin。
- 失败分类：`cann_too_old` / `toolchain_host_headers` / `capacity` /
  `lowering_unsupported` / `device_compile` / `timeout`。
- `cann_too_old` 属于环境问题，不要改代码硬扛。

## 7. 仿真门禁

```bash
python -m aiop simulate workspace/<task>
```

- 后端选择：`local_npu` > `npusim` > `static`；探测结果写入报告。
- `static` 只代表 L1/L2 通过，**不是**数值验证。
- 精度判定：`|a-g| <= atol + rtol*|g|` 的通过率 ≥ required，且 max_abs ≤ 上限。
- 失败按 `knowledge/07` §7 分层定位；不要用仿真时延代表真机性能。

## 8. 上板门禁

```bash
python -m aiop board workspace/<task> --host user@a5-host
```

- 预检：ssh、npu-smi、torch_npu、TileLang、磁盘。
- 部署：`<remote_dir>/<task_id>/`，上传 kernel + test + manifest。
- 采集：latency/p50/p90、带宽、TFLOPS、max_abs、matched_ratio。
- 不可测指标（AIC/AIV 利用率）写 `null`，不填估算值。

## 9. 修复循环

| 失败类型 | 首先尝试 | 禁止 |
|---|---|---|
| 语法/API | 回退模板；核对 API 白名单 | 猜 API |
| 容量 | rows/tile ↓、stages ↓、复用 buffer | `annotate_unlimit_memory` 当修复 |
| 尾部/对齐 | 启用 guard；tile 取整 | 忽略尾部 |
| 精度 | fp32 累加、max-shift、补零 lane 显式写值 | 改 golden 迁就 |
| 死锁/挂死 | stages=1、grid=1 复现 | 反复重试同一配置 |
| 环境 | 切 CANN 9.2 / 配置 torch_npu / 检查 flags | 改 kernel 绕过 |

每次修复必须记录到 `debug_log.md`（attempt、mode、分类、错误、plan）。

## 10. 产物与验收

```text
workspace/<task_id>/
├── DESIGN.md
├── spec.yaml
├── {op}.py
├── test_{op}.py
├── static_report.json
├── compile_result.json + compile_result.cce.c
├── simulation_report.json
├── board_report.json
├── result.json / report.md / manifest.json
├── state.json
├── debug_log.md
└── history/
```

验收：

- 生成通过率 = 静态 + 编译通过 / 总数，目标 ≥ 90%。
- 仿真可用时，逻辑校验必须通过。
- 上板精度在业务阈值内；性能达到 baseline 水平（无 baseline 时记录原始数据）。
- 未验证项必须在 `notes` 中如实披露。

## 11. 命令速查

```bash
python -m aiop env --probe-compile
python -m aiop search "softmax 归约 精度"
python -m aiop generate --spec case.yaml [--simulate/--no-simulate] [--board]
python -m aiop check    workspace/<task>
python -m aiop compile  workspace/<task>
python -m aiop simulate workspace/<task>
python -m aiop board    workspace/<task> --host user@a5-host
python -m aiop eval     --cases aiop/eval/cases.yaml
python -m aiop serve    --host 0.0.0.0 --port 8790
```
