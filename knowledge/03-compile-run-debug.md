# 编译、运行、调试

目标：从源码到可执行 kernel 的全部可复现命令，以及失败时的分层定位方法。
命令均在上游仓根 `<tilelang-repo>` 执行。

## 1. 环境前置

| 项 | 要求 |
|---|---|
| CANN | 含 `bisheng`、支持 CCE 的 `ld.lld`、Ascend runtime 库 |
| `bisheng` | 在 `PATH` 或 `BISHENG_HOME/bin` |
| PyTorch | `torch_npu` 可用，`torch.npu.is_available()` 为 `True` |
| 架构 | 后端默认 `dav-3510`；`ASCEND_NPU_ARCH` 对本目标可选 |

## 2. 构建

```bash
git submodule update --init --recursive
USE_ASCEND=ON USE_CUDA=OFF python -m pip install -v .
```

可编辑开发安装：

```bash
python -m pip install -r requirements-dev.txt
USE_ASCEND=ON USE_CUDA=OFF python -m pip install -e . -v --no-build-isolation
```

已有 build 目录时，用 `CMAKE_ARGS` 传入 `-DUSE_*=...` 覆盖缓存配置。
更多命令见 skill `tilelang-build`。

## 3. 编译入口

```python
# AscendC 代码生成（默认）
kernel = tilelang.compile(func, target="ascend", execution_backend="auto")

# PTO 设备代码生成
kernel = tilelang.compile(func, target="pto")

# JIT 写法
@tilelang.jit(target="ascend")
def f(...): ...
```

| 选择项 | 取值 | 说明 |
|---|---|---|
| `target` | `"ascend"` / `"pto"` | `pto` 走 `codegen_pto.cc`；`target="pto"` 时 `execution_backend` 保持 `auto` |
| `execution_backend` | `"auto"` / `"pto"` | 仅 `target="ascend"` + `execution_backend="pto"` 会被拒绝 |
| `TILELANG_DEFAULT_TARGET` | `ascend` / `pto` | 未显式传 `target` 时的默认值 |
| `TILELANG_EXECUTION_BACKEND` | `pto` | 默认执行后端 |

完全手动同步的 kernel 需要追加：

```python
pass_configs={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False}
```

`T.Stage` 走手动阶段约束但**不需要**关闭自动调度。

## 4. 看生成结果

```python
print(kernel.get_kernel_source())
```

源码级检查（不执行、不需要设备）的三种方式：

| 手段 | 命令 / 开关 | 用途 |
|---|---|---|
| 生成源码 | `tilelang.lower(kernel, target="ascend")` → `kernel_source` | 看最终设备代码 |
| compile-only | `python -I -m tilelang.tools.compile_only --target <target> --output_file out.c input.py` | 只 lower 不编译；`--target` 必须显式，`auto` 被拒绝，默认 `c` |
| IR 逐 pass 追踪 | `TL_LOWER_TRACE=1 python3 my_script.py` | 每个 pass 前后 IR、codegen 源码、HTML 报告、可编辑回注 |

`TL_LOWER_TRACE` 的产物包含：逐 pass 的 `.tir` dump、按 phase 标注的 HTML diff、
最终 codegen 源码；支持改生成源码后重跑注入。旧工具 `TILELANG_PASS_DIFF` 已被它取代。
其他辅助：`docs/tools/analyzer.md`、`docs/tools/pass_diff.md`、`docs/tools/layout_visualization.md`。

## 5. 运行与正确性

```python
x = torch.randn((M, K), device="npu", dtype=torch.bfloat16)
y = kernel(x, w)
torch.npu.synchronize()
torch.testing.assert_close(y, ref, rtol=1e-2, atol=1e-2)
```

示例程序直接运行即做正确性 + 性能：

```bash
python examples/ascend/example_gemm.py
python examples/ascend/example_vecadd.py
python examples/ascend/example_per_token_cast_to_fp8.py --mode simd
python examples/ascend/example_rmsnorm.py --no-bench
```

## 6. 性能与剖析

| 手段 | 开关 / 路径 | 说明 |
|---|---|---|
| 工程内计时 | `do_bench`（见 `references/profiling.md`） | 稳态时延；区分首次 JIT 与稳态 |
| Ascend host 侧 range | `TILELANG_ASCEND_PROFILE=1`（import tilelang 前设置） | 在 Torch profiler 中加 `tilelang::<global_symbol>` range；不启动 profiling 会话 |
| 周期级 | npusim 单算子 cycle 采集 | 量 VF / DMA / Cube / 同步交接 |
| 回归 | `maint/scripts/ascend_perf_regression.py`、`maint/scripts/run_perf_regression_ascend.sh` | 批量 GEMM / 向量 / DeepGEMM / attention |
| 任务代价标注 | `T.Task(latency=..., ii=...)`、`T.SimtVF(..., latency=)` | 把实测代价喂给调度模型 |

结论口径：模型 cycle、求解出的 II、编译耗时、设备实测时延是四种不同量，不能互相替代。

## 7. 测试

```bash
python -m pytest testing/ascend -q                 # 编译器/语言回归
python -m pytest examples/ascend/test_rmsnorm.py -q
```

`testing/ascend` 按"行为归属的最早边界"分目录：

| 目录 | 守护的契约 |
|---|---|
| `analysis` | 源程序合法性检查 |
| `transform` | 单个 IR pass 的前后结构 |
| `auto_schedule` | 调度、存储归属、核放置、同步与复用 |
| `layout` | fractal 几何与数值矩阵路径 |
| `runtime` / `target` | 启动 ABI、当前 stream、target/编译器选择 |
| `language` | 前端、发出的设备代码、数值行为 |

## 8. 失败分层

| 层 | 典型症状 | 先看什么 |
|---|---|---|
| 前端/语义 | `ValueError`/`TypeError`、语义检查拒绝 | `05-limits-and-pitfalls.md`；skill `tilelang-semantic` |
| 调度/容量 | 分配超限、版本数异常 | 生成 allocation 大小、buffer 版本、VF 模式、compile flags |
| 厂商编译 | `bisheng`/`ld.lld` 报错 | 生成源码、`TL_LOWER_TRACE` 的 codegen 产物 |
| 设备启动/运行 | 启动失败、结果错 | 容量与 ABI、`ascendc-api-kernel-debug` 的 `printf`/`asc_dump`/`assert` |

## 9. 缓存

| 变量 | 默认 | 说明 |
|---|---|---|
| `TILELANG_CACHE_DIR` | `~/.tilelang/cache` | kernel 二进制/源码缓存根目录 |
| `TILELANG_DISABLE_CACHE` | `0` | 置真值关闭缓存 |

注意：清理 TileLang cache **不会**重建过期的参考 C++ 扩展；对比实验要同时冻结两个 revision
与已加载的原生扩展。
