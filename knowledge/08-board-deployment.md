# 上板部署与验证流程（跨服务器 A5 实测）

目标：把仿真通过的算子提交到远端 Ascend 950 服务器，完成**部署 → 运行 → 计时 →
精度对比 → 指标采集 → 报告归档**，失败可复现、可回滚。

## 1. 前置条件

| 项 | 要求 | 校验命令 |
|---|---|---|
| 远端主机 | Ascend 950PR/DT（A5），`dav-3510` | `npu-smi info` |
| CANN | 9.2.0（A5 头文件/指令完整） | `cat $ASCEND_HOME_PATH/version.cfg` |
| Python | 与 kernel 构建版本一致；`torch_npu` 可用 | `python -c "import torch_npu; print(torch.npu.is_available())"` |
| TileLang | 与本地同一 revision 的 A5 工作树/构建产物 | `python -c "import tilelang; print(tilelang.__file__)"` |
| 访问方式 | SSH 公钥登录，无需交互密码 | `ssh -o BatchMode=yes <host> true` |
| 目录 | 远端可写的独立工作目录 | `mkdir -p <remote_dir>` |

**禁止**在共享目录直接覆盖他人文件；每次部署使用 `task_id` 独立子目录。

## 2. 配置

`config/aiop.yaml`：

```yaml
board:
  host: "user@10.0.0.20"      # 为空则跳过上板，报告中标记 skipped
  port: 22
  key: "~/.ssh/id_ed25519"    # 可选；为空用 ssh-agent / 默认密钥
  remote_dir: "/home/user/aiop_runs"
  python: "python3"
  env_script: "source /usr/local/Ascend/cann-9.2.0/set_env.sh"
  tilelang_repo: "/home/user/tilelang"
  timeout_s: 1800
  repeats: 50
  warmup: 10
```

也可用环境变量覆盖：`AIOP_BOARD_HOST`、`AIOP_BOARD_KEY`、`AIOP_BOARD_DIR`。

## 3. 部署包

```text
<task_id>/
├── <op>.py            # 生成的 kernel（可独立 import）
├── run_board.py       # 远端 runner：构造输入、跑 kernel、golden 对比、计时、写 JSON
├── manifest.json      # 任务元数据：spec、plan、git revision、编译 flags
└── requirements.txt   # 远端依赖说明（不自动安装）
```

传输：本地打 tar.gz → `scp` → 远端解包到 `<remote_dir>/<task_id>/`。
不传输 aibin：远端首次运行会按同一源码与 flags 重新 JIT，确保可复现。

## 4. 执行流程

```text
1. 预检：ssh BatchMode 连通、npu-smi 可见、CANN/torch_npu 版本、磁盘余量
2. 上传：tar.gz（kernel + runner + manifest）
3. 远端运行：
   bash -lc "<env_script> && cd <remote_dir>/<task_id> &&
             <python> run_board.py --out result.json --warmup N --repeats M"
4. 拉回：result.json、可选 profiling 摘要、失败日志
5. 判定：精度门禁 + 性能门禁
6. 归档：board_report.json / board_report.md；失败保留现场，不自动删除
```

远端 runner 约定：

- 输入用固定 seed 生成，dtype/shape 与 spec 一致；golden 用 CPU 计算。
- 计时：`torch.npu.synchronize()` 后 `time.perf_counter()` 循环，取中位数与 P90；
  同时记录 min/max，报告样本数。
- 指标：`latency_ms`、`bandwidth_gbps`（逻辑 Bytes / latency）、
  `tflops`（逻辑 FLOPs / latency）、`max_abs_error`、`matched_ratio`。
- 若可用：调用 CANN profiling / `npu-smi` 采集 AIC/AIV 利用率、HBM 带宽；
  不可用时如实标记 `null`，不用估算值填充。

## 5. 验收门禁

| 门禁 | 条件 | 失败处理 |
|---|---|---|
| 精度 | 达到 `knowledge/07` 的混合容差 | 回传失败样本，触发 Agent 重生成 |
| 性能 | 达到任务 `perf_target`；未设目标时记录 baseline | 记录差距，进入性能调优阶段 |
| 稳定性 | 重复运行无 ACL/越界错误 | 保留日志，标记 `BLOCKED_BOARD` |
| 可复现 | manifest 中 revision/flags 完整 | 缺项视为无效报告 |

## 6. 失败模式

| 症状 | 原因 | 处理 |
|---|---|---|
| `ssh: Permission denied` | 未配置公钥 | 配置 key，或置空 host 跳过 |
| `npu-smi: command not found` | 未 source CANN 环境 | 修正 `env_script` |
| `torch.npu.is_available() == False` | 驱动/torch_npu 不匹配 | 阻断，不重试 |
| `ACL_ERROR_DEVICE_MEM_ERROR` | HBM 不足/越界 | 缩小 shape 复现，检查 UB/GM 访问 |
| 设备被占用 | 其他任务在跑 | 排队或换设备，不 kill 他人进程 |
| 时延波动 > 20% | 频率/负载/缓存未固定 | 增加 warmup、固定频率、记录环境 |
| 精度与仿真不一致 | 仿真未覆盖的硬件行为 | 保留两端输入输出，按 dtype/同步分层排查 |

## 7. 安全与清理

- 只用用户自己的账号与目录；不修改远端系统配置。
- 不传输密钥、token、内部数据。
- 任务结束保留 `result.json` 与日志；大文件（profile、dump）按需拉回后远端清理。
- 失败现场至少保留 24 小时，便于复现。

## 8. 报告字段

```json
{
  "task_id": "...",
  "host": "...",
  "device": "Ascend950PR",
  "cann": "9.2.0",
  "kernel_sha256": "...",
  "accuracy": {"max_abs_error": 0.0, "matched_ratio": 1.0, "passed": true},
  "perf": {"latency_ms": 0.0, "bandwidth_gbps": 0.0, "tflops": 0.0,
           "p90_ms": 0.0, "samples": 50},
  "utilization": {"aic": null, "aiv": null, "hbm": null},
  "status": "passed | accuracy_failed | blocked | skipped",
  "notes": []
}
```
