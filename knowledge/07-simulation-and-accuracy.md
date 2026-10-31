# 仿真验证、调试与精度判定

面向 Ascend 950（`dav-3510`）的分层验证方法。原则：**能在更低成本层级发现的错误，不要带到
设备上**；每层都有可复现命令与明确产物。

## 1. 五层验证模型

| 层 | 做什么 | 需要设备 | 失败含义 | aiop 命令 |
|---|---|---|---|---|
| L0 语法/API | Python AST、TileLang API 白名单、规格一致性 | 否 | 生成代码不可导入 | `python -m aiop check <task>` |
| L1 Lowering | `tilelang.lower(func, target="ascend")` 出 CCE 源码 | 否 | 前端/layout/调度拒绝 | 同上 |
| L2 设备编译 | `bisheng` 编成 aibin（`dav-3510`） | 否 | 厂商编译器/头文件/模板错误 | `python -m aiop compile <task>` |
| L3 功能仿真 | npusim 运行 kernel，与 golden 比精度 | 否（模拟器） | 数值/同步/越界错误 | `python -m aiop simulate <task>` |
| L4 上板实测 | 远端 A5 真机运行 + 计时 | 是 | 真实性能/精度/稳定性 | `python -m aiop board <task>` |

L1/L2 是生成 Agent 的**强制门禁**；L3 是默认验收门禁；L4 在配置了远端主机后执行。

## 2. 环境事实（本机已验证）

| 项 | 值 | 说明 |
|---|---|---|
| A5 编译所需 CANN | **9.2.0** | 9.1.1 缺少 A5 指令/类型（如 `asc_load_l2_cache_mode`） |
| 编译器 | `/usr/local/Ascend/cann-9.2.0/bin/bisheng` | `--npu-arch=dav-3510` |
| 模拟器 | `/usr/local/Ascend/cann-9.2.0/x86_64-linux/bin/npusim` | `cannsim` 已更名；只接受 `-s Ascend950` / `Ascend950DT` |
| GPU 式运行时 | `torch_npu` | 功能仿真（张量在 `npu` 上）必需 |
| Host 头文件兼容 | `-D__INTEL_LLVM_COMPILER=1 -DSYCL_LANGUAGE_VERSION=1` | 规避 glibc `floatn.h` 在 bisheng clang 15 上的 `__float128` 报错 |

> 本机没有真实 A5 设备（无 `/dev/davinci*`，默认 Python 环境未装 `torch_npu`）。
> 因此 L3 是否可执行由 `aiop simulate` 的探测结果决定，**不会伪造通过**；
> L4 通过配置远端主机执行。

## 3. L1/L2：源码级与设备编译

```python
import tilelang
art = tilelang.lower(func, target="ascend")   # 需要 Target 上下文
src = art.kernel_source                        # CCE 源码，可做静态审计
kernel = tilelang.compile(
    func, target="ascend",
    pass_configs={"tl.device_compile_flags": [
        "-D__INTEL_LLVM_COMPILER=1", "-DSYCL_LANGUAGE_VERSION=1"]},
)
```

- `tilelang.lower` 需要 `Target("ascend")` 上下文，否则 C++ 侧报
  `Target context required`。
- `compile` 会调用 `bisheng` + `ld.lld` 产出 aibin，并写入 TileLang cache。
- 源码审计点：`asc_copy_gm2ub_align` / `asc_copy_ub2gm_align` 的字节数、`asc_vf_call` 的线程数、
  `asc_sync_notify/wait` 是否成对、是否出现标量化的逐元素 GM 访问。

## 4. L3：npusim 功能仿真

```bash
# 单算子运行（runner 由 aiop 生成，输出 JSON）
npusim record -s Ascend950 -o <out_dir> \
  "bash -c 'source <cann>/set_env.sh && <python> <runner.py> --out result.json'"
npusim report -e <out_dir>/npusim_<ts>_<label> -g vf   # 可选：VF IPC / 周期
```

要求：

1. runner 内使用 `torch.randn(..., device="npu")`，kernel 由 `tilelang.compile` JIT，输出与
   CPU golden 比较，写出 `max_abs/mean_abs/matched_ratio/latency_ms` JSON。
2. npusim 只模拟 Ascend 算子执行；不要用 `do_bench(backend="msprof")` 采时延，
   仿真时延只用于相对比较，不能当真实性能。
3. 仿真失败先区分：**编译失败**（L2 已拦）、**启动失败**（ABI/容量/设备初始化）、
   **数值错误**（同步、越界、layout、dtype、mask）。
4. 出现挂死/超时：优先缩短 shape、把 `num_stages` 降为 1、把 grid 降为 1 复现。

## 5. 精度判定标准

逐元素混合容差（与业务口径一致）：

```text
element_ok = |actual - golden| <= atol + rtol * |golden|
pass = matched_ratio >= required_ratio and max_abs_error <= max_abs_limit
```

| dtype | atol | rtol | max_abs_limit | required_ratio |
|---|---|---|---|---|
| float32 | 1e-5 | 1e-5 | 1e-4 | 0.9999 |
| float16 | 1e-3 | 1e-3 | 1e-2 | 0.999 |
| bfloat16 | 1e-2 | 1e-2 | 1e-1 | 0.999 |
| int8/int16/int32 | 0 | 0 | 0 | 1.0（精确） |

规则：

- 整型与索引类输出必须**逐元素精确相等**。
- 归约/softmax 类先比较中间量（如 max、sum）再比较最终值；大面积轻微偏差通常是累加顺序或
  dtype，个别元素爆炸通常是越界/未初始化/同步。
- NaN/Inf 输入按业务定义域处理；不支持的输入应显式拒绝，而不是静默产出 NaN。
- golden 与 kernel 必须使用同一 dtype 语义（bf16 输入先转 fp32 计算再比较）。

## 6. 调试手段

| 手段 | 用法 | 注意 |
|---|---|---|
| `T.print` | 打印 buffer / 标量表达式 / 纯消息 | 改变性能与寄存器压力；A3 的 `T.printf` 为 A5架构暂无（无格式化变参），统一改用 `T.print(obj, msg=...)` |
| `T.device_assert` | 越界/条件断言 | 最终版本移除 |
| `TL_LOWER_TRACE=1` | 逐 pass IR、codegen 源码、HTML diff | 定位 lowering 拒绝点 |
| `npusim report -g vf` | VF IPC、周期分解 | 模型数据，不等于真机时延 |
| `asc_dump` / `clock` | 上板中间结果与时间戳 | 见 `ascendc-api-kernel-debug` |

## 7. 失败分层与修复方向

| 现象 | 层级 | 首选修复 |
|---|---|---|
| `Target context required` | L1 | 在 `Target("ascend")` 上下文里 lower |
| `Ascend device compilation failed` + 缺类型/指令 | L2 | 切换 CANN 9.2；检查 `dav-3510` |
| `floatn.h ... __float128` | L2 | 加 `-D__INTEL_LLVM_COMPILER=1 -DSYCL_LANGUAGE_VERSION=1` |
| UB/L1/L0 容量拒绝 | L1 | 降 tile/stages/rows，复用 buffer |
| `T.Parallel` 向量化拒绝 | L1 | 先 `T.copy` 到 UB；去掉 GM 直接访问/原子/内嵌循环 |
| 启动失败、ABI 错误 | L3 | 检查 grid、线程域、动态 shape、参数顺序 |
| 大面积精度偏差 | L3 | dtype/累加位置、归约顺序、padding lane 未初始化 |
| 个别元素爆炸 | L3 | 越界、同步缺失、tail 未处理、原子未初始化 |
| 挂死/超时 | L3 | 降 stages/grid 复现；检查同步配对 |

## 8. 复现命令

```bash
# 本仓
python -m aiop check     workspace/<task>
python -m aiop compile   workspace/<task>
python -m aiop simulate  workspace/<task>
python -m aiop board     workspace/<task> --host user@a5-host
python -m aiop eval      --cases aiop/eval/cases --out reports/eval.json
```

所有报告写入任务目录，字段与阈值固定，便于回归对比。
