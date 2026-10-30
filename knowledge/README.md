# 知识库索引

本知识库回答「用 TileLang 在昇腾 950 上生成一个算子，需要知道什么」。
skill 负责**流程与约束**，本知识库负责**事实、速查与骨架**，上游源码是最终权威。

## 阅读顺序

| 顺序 | 文档 | 何时读 | 关键产出 |
|---|---|---|---|
| 1 | [04-examples-and-patterns.md](04-examples-and-patterns.md) | 拿到算子需求，先定位骨架 | 模板文件 + 关键 API + 参数化维度 |
| 2 | [02-tilelang-ascend-dsl.md](02-tilelang-ascend-dsl.md) | 落笔写 kernel | 正确的 API 名与参数 |
| 3 | [01-ascend950-platform.md](01-ascend950-platform.md) | 选 dtype、算容量、判断对齐 | 平台可行/不可行 |
| 4 | [06-performance-optimization.md](06-performance-optimization.md) | 定 tiling / 核类型 / 流水 | 预算与优化决策 |
| 5 | [05-limits-and-pitfalls.md](05-limits-and-pitfalls.md) | 生成后自检、编译被拒 | 需要改写的位置 |
| 6 | [03-compile-run-debug.md](03-compile-run-debug.md) | 编译、看 IR、测性能 | 可复现命令与证据 |
| 7 | [07-simulation-and-accuracy.md](07-simulation-and-accuracy.md) | 仿真、精度判定、失败定位 | 门禁结论 |
| 8 | [08-board-deployment.md](08-board-deployment.md) | 远端 A5 实测 | 上板报告 |
| 9 | [09-verified-samples.md](09-verified-samples.md) | 找已验证样例 | 可复用 kernel 路径 |

## 检索协议

1. **先骨架后细节**：任何算子需求先到 `04` 找同类模板，避免从零拼 API。
2. **事实以源码为准**：`02` 是速查，签名与行为以
   `/home/cjl/workspace/tilelang/tilelang/ascend/language/` 为准；平台事实以
   `/home/cjl/workspace/tilelang/.agents/skills/asc-npu-arch/references/npu-arch-facts.json`
   为准。
3. **不确定就查 skill**：`tilelang-ascend` 是编程约束的第一手资料；
   `tilelang-semantic` 说明某个写法为何非法。
4. **不猜硬件**：C API / 指令级能力问题走 `asc-search`（覆盖 `SIMD-API/c_api`），
   不要凭函数名推断。

## 与 skill 的分工

```text
skill（.agent/skills/）        知识库（knowledge/）
├─ aiop-op-pipeline     →      全流程门禁与命令（生成/仿真/上板）
├─ tilelang-ascend      →      02 + 05 的精简版，例外与边界更全
├─ tilelang-semantic    →      05 的语义规则来源
├─ tilelang-layout      →      02 中布局相关条目的来源
├─ tilelang-build       →      03 中构建命令的来源
├─ asc-npu-arch         →      01 的事实来源（含 JSON）
├─ asc-search           →      需要 C API 能力边界时的检索入口
└─ ascendc-api-kernel-debug →  03 中上板调试的接口选型
```

检索入口：`python -m aiop search "<query>"`（BM25，索引 `.index/knowledge.json`）。

## 上游路径映射

skill 与本文档中的相对路径统一按上游仓根解析：

| 写法 | 实际位置 |
|---|---|
| `examples/ascend/...` | `/home/cjl/workspace/tilelang/examples/ascend/...` |
| `testing/ascend/...` | `/home/cjl/workspace/tilelang/testing/ascend/...` |
| `docs/...` | `/home/cjl/workspace/tilelang/docs/...` |
| `tilelang/ascend/language/...` | `/home/cjl/workspace/tilelang/tilelang/ascend/language/...` |
| `${DEVKIT_PATH}` | `/home/cjl/workspace/asc-devkit` |

## 维护规则

- `knowledge/` 只写**结论 + 出处**，不复述上游文档整段内容。
- 每条平台事实必须能指到源码或官方文档路径；无法核实的标「待确认」，不写推测。
- 上游升级后重跑 `scripts/sync_skills.sh`，并复核 `01`/`05` 中与版本相关的条目。
