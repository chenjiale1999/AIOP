---
name: asc-search
description: OKF 用户侧入口：把 Ascend C API 文档的三类头建立为按适用条件筛选的持久索引，并按算子需求检索候选接口文档。用户需要建索引/检查索引新鲜度、询问“用哪个接口/怎么找 API/某接口配什么用/哪个版本可用”时使用；写头、改词表、标注文档请用 asc-knowledge-create。
---

> **aiop 副本说明**：本 skill 复制自 `asc-devkit`（`由 DEVKIT_ROOT 指定的 asc-devkit 仓`）。文中相对路径以该仓为根解析；`${DEVKIT_PATH}` 指该仓根目录。重新同步：`scripts/sync_skills.sh`。

# OKF 索引与检索（asc-search）

本 Skill 覆盖「建立索引 → 使用索引」全流程：索引是三类头（文档 slim 头、include Doxygen、
examples marker）的纯派生数据，查询统一走本 skill 的 `scripts/query.py`。推导规则唯一事实源是
[references/index_derivation_contract.md](references/index_derivation_contract.md)；
建索引细节见 [references/index.md](references/index.md)，检索流程细节见
[references/retrieve.md](references/retrieve.md)。命令均从仓库根目录执行。

通过用户 skill 目录的软链加载时，先将当前 `SKILL.md` 路径解析为真实路径，向上三级取得
对应的 asc-devkit 仓库根目录，并从该目录执行以下命令和读取候选文档。用户当前目录
可以位于其他项目；检索始终使用安装入口指向的 devkit 仓库。

目标架构统一使用 NPU 架构号表述，默认使用 `3510`。

## 建立索引

```bash
python3 .agent/skills/asc-search/scripts/build_index.py --scope c_api [--arch 3510] [--cann-version 9.2.0] [--check]
```

构建按 CANN 版本、架构和 API 层级筛选，只索引同时满足 version 覆盖、archs 命中、层级命中的 API；
校验不过不出索引，同输入两次构建字节一致。`--check` 做文档漂移检测。

## 使用索引

### 步骤 0：索引前置保障

任何检索开始前先运行（用户显式指定架构/版本时透传参数，否则默认使用 3510 架构并自动检测本地 CANN 版本）：

```bash
python3 .agent/skills/asc-search/scripts/ensure_index.py --json [--arch A3] [--cann-version 9.2.0]
```

筛选条件匹配且指纹新鲜 → 零动作；缺失、筛选条件不符或漂移 → 原地重建（本地 diff，不主动提交）。
成功后以输出 JSON（`cann_version/arch_label/categories`）作为本次检索的**文档范围**；
`actions` 含 `rebuilt` 时向用户说明一句。失败时如实报告并停止，不得绕过索引猜测文档。

### 检索循环

1. **意图分析**：抽取阶段/操作/分类/dtype/行为约束/版本产品/具名 API/参数枚举/配套依赖等信号，
   映射到查询维度；至少 1 个可映射信号且无影响候选集的歧义才检索。不明确时先反问，
   同一检索需求累计最多 3 轮，每轮最多 3 个问题并给出选项；澄清期间不执行任何查询。
   达到上限仍不明确时停止反问，说明信息不足，列出已确认条件与待确认项，不猜测条件或推荐接口。
2. **初查与收窄**：主信号查询，结果过宽时加维度（维度间 AND、逗号值内 OR）。
3. **空结果降级链**：精确 `--operation` → `--kw` 同义词 → `--api-class` + `--kw` →
   `--gap` 确认范围外；仍无则如实说明。
4. **配套展开**：`--deps-of`（含 `--reverse`）、`--enum`/`--param` 反查，保留家族变体。
5. **收敛与把关**：意图要素全部有候选或确认范围外、依赖闭合后按固定格式作答；
   脚本只给候选，输出前由你复核相关性与适用条件（筛选条件/condition/deprecated），
   不相关的不展示；按需只读候选文档支撑推荐所需的章节，禁止全量精读。

完整维度映射表、降级链与策略要点见 [references/retrieve.md](references/retrieve.md)；
查询 CLI 全部参数见 [references/index.md](references/index.md)。

### 输出格式（不落盘）

```text
## 候选
- 路径: docs/zh/api/SIMD-API/c_api/xxx/yyy.md 或 include/... | API: asc_yyy | 用途: 一句话
```

未检索到候选时直接说明结果；适用条件和待确认事项随候选说明，不单设未覆盖章节。
返回多个候选且无法确认已覆盖当前需求的全部适用接口时，必须在候选列表后原样注明：
“候选仅供参考，不代表全量接口。”索引查询成功或候选均已核对，不等于候选集完整。

## 边界

- 只帮找文档，不生成算子代码、不替用户写 kernel。
- 当前覆盖范围为 `docs/zh/api/SIMD-API/c_api`；范围外层级（adv_api/tensor_api/basic_api/SIMT/
  AI-CPU 等）如实标注为未覆盖并给出线索。
- 不改写 `docs/` 正文、include、样例或索引构建产物。
