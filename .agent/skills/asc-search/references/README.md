# asc-search reference 导航

本目录保存 asc-search 两个工作流（建立索引、使用索引）的流程文档与推导契约。运行所需配置、
脚本与测试在各同级目录。

## 目录划分

| 文档 | 内容边界 |
|------|----------|
| [README.md](README.md) | 本 reference 导航和维护规则 |
| [index.md](index.md) | 建立索引：三类头 → 按适用条件筛选的持久索引的构建流程、资产结构与查询 CLI |
| [retrieve.md](retrieve.md) | 使用索引：意图分析、多轮查询、输出格式与检索策略 |
| [index_derivation_contract.md](index_derivation_contract.md) | 推导规则唯一事实源：推导维度、LLM 兜底边界、索引筛选条件 |
| [三类知识标注与检索工具链框架图](asc-search-knowledge-toolchain-9.2.0.svg) | 三类头的标准格式、真实样例、关联关系及工具链消费流程 |

## 维护规则

- `okf-test` 与 `okf-knowledge-user` 的 `.agent/skills/asc-search/` 完整目录始终保持一致，
  包括 README、指令、reference、配置、脚本、测试和图片。任何修改都需同步两分支并提交、推送；
  交付前运行 `git diff okf-test okf-knowledge-user -- .agent/skills/asc-search`，结果必须为空。
- 推导规则只在《index_derivation_contract.md》维护；流程文档只引用，不复述推导规则。
- 构建与前置保障只依赖三类头与本 skill `config/`，不得引用其他 skill 的文件。
- 文档头/include/examples 变化后重跑 `scripts/build_index.py --scope c_api` 并提交索引 diff；
  检索触发的本地重建 diff 只在维护流程确认后提交。
