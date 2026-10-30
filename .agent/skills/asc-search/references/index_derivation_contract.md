# 索引推导契约

本契约是 asc-search 的唯一推导规则文档：索引是三类头的纯派生数据，本文定义推导维度、残留
裁决边界与筛选条件规则。文档头如何写入、标注证据规则属于建设侧契约，本 skill 不复制。
当前实例为 c_api 全量（SCOPES categories=None）。

## 1. 三类头（推导输入）

| 头类型 | 载体 | 提供的事实 |
| --- | --- | --- |
| 文档头 | Markdown front matter（`okftype/description/tag/dependencies/version`） | 语义信息：页面种类、摘要、角色/操作/特性标签、关联文档、CANN 生命周期区间 |
| include 头 | 头文件 Doxygen（`@ingroup/@brief/@override_id/@archs/@param`）与声明 | 重载功能摘要与结构事实：逐重载编号、适用架构、参数方向与类型、原型与地址空间 |
| examples 头 | 源文件共同范围首尾的 `//! [id1, id2, ...]` marker（兼容单 ID） | 同一展示范围与多个重载的关联 |

三类头都是只读输入；索引构建不修改 `docs/` 正文、include 或样例。文档头中的 tag 值合法性
以本 skill `config/tag-taxonomy.json`（随包分发的冻结词表）校验。
include 的 `@brief` 是面向开发者的英文重载摘要，当前解析器不将它写入索引；
文档语义仍来自 Markdown 头的 `description/tag/dependencies`，现有索引格式保持不变。

## 2. 推导契约（索引推导期，脚本执行）

下表全部由本 skill 的推导引擎（`scripts/derive/`）从三类头与仓内数据生成，不写入文档头或标签：

| 推导维度 | 来源 | 规则 |
| --- | --- | --- |
| `title` | 文档 H1 | 一级标题 |
| `api_class` | include 路径 / `@ingroup` | `include/c_api/<子目录>` 即 API 分类（vector_compute/reg_compute/cube_compute/cube_datamove/scalar_compute/atomic/cache_ctrl/sync/spr/utils/defs/composite/…） |
| `execution_model` | include 路径 + 原型 mangling | vector_compute→memory_vector；reg_compute→register_vector；scalar_compute→scalar；cube_*→matrix_compute；mangling 形态佐证（`Rv*` 寄存器侧、`PU*` 内存侧） |
| `source/destination_locations` | 原型 mangling 地址空间码 | `G`=GM、`U`=UB、`L1`=L1、`FB`=fixpipe、`CA/CB/CC`=L0A/L0B/L0C；`v`=矢量寄存器（vreg）、`vb`=掩码寄存器（preg）；mangling 解析实现见 `scripts/derive/include_db.py` |
| `signature/override_id/dtype` | include Doxygen + 声明 | `@override_id` 逐重载枚举；dtype 从 mangling 类型码还原 |
| `archs` | include `@archs` + `__NPU_ARCH__` 守卫 | 逐重载架构，文档级取并集 |
| `header` | include 声明位置 | 唯一声明头文件路径 |
| `masked` | mangling `_vb` 后缀 | 任一重载含掩码参数即支持掩码模式 |
| `deprecated` | include `ASC_DEPRECATED(v,date,replacement)` + 文档声明 | 替代目标同步推导 |
| `params`（参数反查） | include 声明 + `@param` | 参数名/类型（含枚举类型形参）→ API 反查视图 |
| `enums/macros 目录` | `include/c_api/defs/*.h` | 枚举值、常量、宏、union、类型别名全量目录 |
| `examples` | 样例成对 marker 的 ID 列表，展开为独立 `override_id` | 记录 `file/start_line/end_line`（行号不含 marker）；每个重载关联同一展示范围；支持 `.asc/.c/.cc/.cpp/.cxx/.h/.hpp` |
| `navigation` | 正文链接 | 仓内相对链接保序去重 |
| `products` | 正文 `<!-- npu="X" -->` 标记 × `config/product-map.yaml` | 产品级支持/不支持矩阵 |
| `feature 视图` | dtype 家族 × roles/operations | 低精度（fp8_*/hifloat8/fp4x2_*）、int4 等特性族 |
| `api_surfaces` | 文档路径 | `SIMD-API/<surface>/` 即层级 |

## 3. LLM 辅助边界（索引构建）

1. 确定性脚本管线承载第 2 节全部主推导。
2. 脚本无法裁决的残留（复杂模板声明、Release Notes 歧义条目、文档与 include 冲突、
   缺口归类）写入 `index/<surface>/unresolved.json`。
3. 单轮 LLM 裁决残留，结果固化为 `derive-overrides/<surface>.json`（落盘、可审查、可回放）。
4. 构建确定性消费 overrides；同输入两次构建字节一致；构建过程不内联调用 LLM。

## 4. 索引筛选条件（CANN 版本 × 架构 × 层级）

- CANN 版本检测链：`--cann-version` 显式参数 > `$ASCEND_HOME_PATH` 安装目录版本 >
  仓 `version.cmake`。
- 构建参数：`--scope`（默认 c_api）、`--arch`（默认 3510，经 `config/product-map.yaml` 映射架构号）。
- 只索引同时满足三条件的 API：version 区间覆盖本地 CANN 版本 ∧ archs 含目标架构 ∧ 属于目标
  层级；被过滤 API 计入 `meta.filtered_by_version/arch` 供核对。
- 检索前置（`scripts/ensure_index.py`，检索流程步骤 0）：按同一检测链得出目标筛选条件，
  committed 索引筛选条件匹配且指纹新鲜（build-meta 哈希 + scope 文档数；`--strict` 全量重建比对）
  才直接使用，否则原地重建后再检索。
