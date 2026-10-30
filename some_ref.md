

# 一. Kernel Agent 概述

## 1. Mindspore AKG

![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026101213253656/52206042/925cbfff3f6941e6af8f64726a2fd718.png)

AKG(Auto Kernel Generator)是Mindspore的一个深度学习编译器，其中的AKG-AGENT是其使用的算子生成工具，目标DSL为Triton-Ascend/SWFT。
AKG主要将算子生成分为两层，设计层输出sketch，生成层输出算子。
sketch是其自定义的特殊最小化DSL，用于表达算子设计意图，标记了循环类型、硬件上下文提示、存储类型等llm_hint，如下所示：

```python
sketch gelu{
	symbols: N;
	tensors: X[N]: f32; Y[N]: f32;
	
	tile_size = 512
	
	@llm_hint("parallel")
	for i in range(0,ceil(N, tile_size)):
		x_tile = alloc([tile_size], llm_hint=["fastest","l1_buffer","input_cache"])
		y_tile = alloc([tile_size], llm_hint=["l1_buffer","init_zero"])
	
		load(X[i:i+tile_size] -> x_tile)
		gelu(x_tile, y_tile)  #coder
		store(y_tile -> Y[i:i+tile_size])
}
```

我说这和tilelang设计理念基本一致；

然后通过三条搜索路线：初始kernel生成、并行子进程探索、迭代优化来对具体算子进行生成。
迭代优化会持久化记忆算子优化路径。

## 2. TIRx-harness

![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026101213253656/52206975/34400f13387f41e7bcdeb9876151f838.png)

TIRx-harness主要是一个agent生成算子的外部框架，没有很深入地探讨如何使用agent来生成算子。
其中输入Task定义了数学公式、接口、workload；Toolset定义了可用的skill和工具。
evolve agent只用于准备环境，benchmark adapter只用于测试和计时。
frontier保存并维护了值得继续探索的候选算子。
可靠性由静态的各类分析和精度判定来保证，具体来说，使用了多个agent通过不同skill在非运行态确认算子的各方面正确性：NumSim分析具体的输入如何产生数值结果；RaceCheck确认算子实际内存访问是否存在读写冲突、界外接入等问题；SyncCheck检查barrier生命周期、generation/phase等同步协议是否正确。benchmark提供运行时的精度正确和各类profiling指标用于继续优化。

## 3. Cake IR

![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026101213253656/52207701/5a9ddbead94649e598b72340b408bd42.png)

有篇论文提出的算子生成流程，自定义了一个中间的Cake IR；我认为其主要关键点在于其将算子生成工作分为了两个循环，内层优化算子，外层将瓶颈、错误诊断额外交给了新agent，用于修改编译器，做到了算子和编译器共同演进。

## 4. KernelSkill

![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026101213253656/52207025/733000f7a36f4e179985e5740e9d4b81.png)

有篇论文的算子多agent联合工作架构，将算子优化和编译问题修复分开，并分别处理为长期记忆和短期记忆，长期记忆主要保存可复用的决策规则和优化方法，短期记忆是当前算子的优化进展。另外代码计划和代码编写被严格分为两份工作。

## 5. auto-gpu-kernel

![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026101213253656/52207216/a1a59a4dd3af4028bbdda590c76f320b.png)

NIVDIA算子生成竞赛的DSA算子#1使用的框架。优化agent每轮规划一个主要改动并进行尝试。另外可以调用几个专项子agent进行深入分析，如Profiler分析阶段耗时和瓶颈；Workload Inspector分析输入特征，batch均衡度，是否稀疏；Research分析下一步优化的尝试方向。

## 6. han-lab-workflow

![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026101213253656/52207227/91950502766b408d80ae2d9c9da0ada2.png)

NIVDIA算子生成竞赛的Fused MoE#1、DSA算子#2使用的框架，先规划计划草稿，再使用Humanize来生成详细计划，然后交给code agent，并使用verifier agent审查正确性与性能证据，同时在算子优化时维护已开发算子优化方向的父子dag图。

# 二. Tilelang Agent 设计

## 1. 现有agent算子生成分析

当前agent生成算子框架基本上总体采用「计划生成、算子生成、审查结果、持久化方案」的自循环构建。论文的相关算子生成框架做得相对比较复杂，而算子生成竞赛的优胜者采用的框架则都是比较常规的自循环工作流。按现在的AI能力来说，有明确划分阶段和每阶段skill、明确任务通常足够agent进行工作。

注意到大多数框架目前还是按已总结完成的kernel开发skill来生成算子，对于快速演进的DSL，这一部分显然也是需要与编译器一起做一个反馈流程并固化已实践过的优化方法的。

可靠性比较完善的是TIRx harness，现有的资料基本是研究阶段/比赛时的临时构建产物，主要还是依赖于直接输入的pytorch参考程序标杆，并通过verifier agent确认。不过按我之前开发的经验来说，AI对可靠性验证一般比较冗余，我个人认为可以按TIRx harness的方法去多阶段检查来保证可靠性。

## 2. Tilelang Agent 设计

![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026101213253656/52242166/73b1e8e7f0454701a9b346155e97c5e2.png)

### 模块和职责

| 模块 | 职责 | 产物 |
|---|---|---|
| TaskRegistry | 固定语义、输入域、ABI、oracle、精度、目标、参考访问及资源限制 | TaskSpec |
| CampaignSupervisor | 管理阶段、预算、设备资源、候选晋升、失败恢复和终止 | 状态事件、frontier、incumbent |
| SkillRegistry 与 SkillAdapter | 选择兼容技能，固定文件内容，将经维护的要求映射为生成契约和检查义务 | SkillBundle、skill.lock、ContextManifest |
| Planner 与 SearchController | 算法家族探索、结构计划、实验假设、参数批次和 shape 特化 | KernelPlan、ExperimentPlan |
| Implementer | 编写 TileLang builder、stage macros、caller loop、wrapper，并有限修复错误 | 固定源码及配置的 Candidate |
| AutoTuneAdapter | 接收 builder、受约束配置批次和探索协议，调用现有 AutoTuner；绑定预算、设备与版本，将优胜配置固定为待验收候选 | TuningRequest、探索 TuningResult、固定 Candidate/Build 引用 |
| CompilerBridge | 复用 TileLang 后端与编译路径，提供 IR、源码、资源、实际调度及诊断 | BuildArtifact、LoweringAudit |
| IndependentEvaluator | 固定输入与 oracle，验证全部输出，执行完整调用的测量及复核 | 四类独立 EvaluationEvidence |
| EvidenceStore 与 Diagnoser | 追加事实、维护候选图、定位瓶颈，将问题送到正确循环 | EvidenceRecord、FailureCluster、EvolutionProposal |
| SkillEvolutionService | 起草、版本化并验证知识与流程变更 | SkillPatch、SkillTrialReport |
| CompilerEvolutionService | 隔离实现 checker、API、lowering、调度和模型提案 | CompilerPatch、RegressionReport |
| CompatibilityGate 与 ReleaseRegistry | 检查版本兼容及组合效果，发布、撤回和回滚可用组合 | ReleaseManifest |

Agent包含Planner、Implementer、Diagnoser，应该也需要checker。

### 当前 Skill 如何驱动生成

SkillAdapter 保留指导文本和出处，并用经审核的映射描述其检查要求。性能建议、用户断言和编译器已确认的语言规则需要分开处理。

| 生成契约 | 必需描述 |
|---|---|
| CallerContract | 输入输出、mutation、shape/stride、动态长度、mask/head 映射、精度和初始化约定 |
| StagePlan | 各阶段读写 region、core/PIPE、SIMD/SIMT/DMA/Cube 选择、guard 和物理访问范围 |
| MemoryPlan | storage/alias、物理尺寸/padding、invariant/carry/scratch、owner、版本及生命周期、容量来源 |
| SchedulePlan | automatic/constrained/manual、caller pipeline、num_stages、offset、阶段约束及同步协议 |
| LoweringAudit | 实际 VF、core/PIPE、版本、allocation、layout、guard、初始化及 drain，与计划的差异 |

对重复 Cube/Vector 循环，可以先分阶段并计算 UB/L1/L0 物理预算，再建立带明确 `num_stages` 上限的自动 `T.Pipelined` 候选。适用 scratch 留给编译器管理版本，carry 保留正确依赖。

`T.macro` 组织阶段，caller loop 表达整个 tile epoch 的流水；UB 数学计算优先考虑 SIMD，guarded irregular gather 考虑 SIMT 或受支持的 indexed DMA，GEMM 遵循所选 L1/L0 路径。

有正确的自动调度基线后，再根据实际顺序或 overlap 问题尝试 `T.Stage`。stage、版本上界、物理 slot 和索引时钟分别描述。

## 3. Autotune 能力边界

Autotune之前的主要搜索内容是tilling大小，并承担了profiling的工作。在当前的z3求解器框架下，我们希望在搜索时能够去做一个较大的求解器适配搜索，而不是只依赖编译器的默认设置。当前deepseek开发的算子中，task latency的显式填写值和默认值有较大差别，我认为这个是能够由autotune自动搜索的参数。

### 1. 基本分工

Autotune 搜索影响程序和调度模型的输入；求解器在这些输入和正确性约束下生成调度；autotune 用实际运行结果选择候选。

```text
问题契约：shape、dtype、layout、输出语义、精度和确定性要求
    ↓
autotune 选择：tiling、启动配置、版本上界、offset、可选调度约束/成本假设
    ↓
kernel 模板生成具体程序
    ↓
编译器分析：任务、依赖、缓冲区所有权、单版本大小、资源限制
    ↓
求解器生成：任务开始时间、模型 II、自动缓冲区版本数
    ↓
编译器生成：实际顺序、stage、缓冲版本访问和同步代码
    ↓
正确性检查 + 真机测量
    ↓
autotune 选择最优候选，并保留调度和资源结果
```

### 2. 直接进行autotune的参数

| 参数 | 具体形式或例子 | 为什么适合搜索 | 必需约束 |
|---|---|---|---|
| Tile 几何 | 模板参数 block_M、block_N、block_K、vector_tile | 改变计算粒度、数据复用、搬运量和片上占用，求解器不会自动替用户重新切块 | 模板处理尾块；满足 GEMM/搬运/布局的形状及对齐要求 |
| 启动核心数量 | T.Kernel(core_num) 中的 core_num | 影响工作分配、负载均衡和启动成本，不属于 kernel 内调度求解变量 | 保证逻辑输出覆盖完整，持久化映射与核心数一致 |
| SIMT 线程域 | T.SimtVF(threads=...) | 同一 SIMT 实现内可改变并行分配和开销 | 仅对已使用 SIMT 的模板开放；合法线程域、跨线程访问及同步由编译器检查 |
| 自动版本数上界 | T.Pipelined(..., num_stages=S) | 控制调度器可用的存储/重叠搜索预算，让求解器适应不同 tiling | S 不等于实际版本数，也不限制手动 stage 编号 |
| 是否允许跨迭代 offset | annotations={"enable_offset": bool} | 改变调度可行空间，可探索更多跨迭代重叠 | 非零 T.Stage 要求 True；它不自动证明控制流和版本索引合法 |
| 缓存策略 | 模板转发到 T.copy/T.dual_copy 的 l2_cache_ctrl | 改变缓存行为和实际搬运性能，调度求解器无法完整预测 | 只使用对应加载/存储路径支持的策略；与 benchmark 缓存清理协议分别记录 |

### 3. 可选的高级调优输入

| 参数 | autotuner 可以做什么 | 原因及限制 |
|---|---|---|
| 手动 T.Stage 方案 | 枚举模板预先支持的 load/compute/store stage 组合 | 用实测弥补成本模型偏差；但它是求解器约束，而非必须手动枚举的默认结果。包裹完整任务，非零 stage 开 offset，不能与 T.Pipelined 的 order/stage 参数混用 |
| 同 PIPE 任务源码顺序变体 | 在已实现、语义合法的模板变体间选择 | 约束模式保留每个 PIPE 的源码顺序；只有已确认独立的任务才能重排，不能跨依赖任意置换 |
| VF latency 标注 | 搜索 T.SimdVF(latency=...) 或 T.SimtVF(latency=...) 的小范围候选 | 改变调度成本模型，可能产生更好的调度；最优标注不是 VF 真实耗时的证明 |
| Task latency/ii 标注 | 搜索 T.Task(latency=..., ii=...) 的受约束候选 | 成本作用于完整任务体；最终必须 latency >= ii > 0，包括另一值来自自动估计的情况 |
| 固定 buffer 版本数 | 通过 T.annotate_buffer_versions 固定特定存储的份数做对照 | 适合验证“模型最优版本数未必真机最优”的假设；固定后该存储的份数不再由求解器选择，默认不展开所有 buffer 的版本组合 |

### 4. SIMD/SIMT 与其他实现变体

算子生成时可以对某些函数提供SIMD/SIMT/其他实现这些单函数的不同版本，autotune可以实现在搜索时选择性使用这些函数的不同版本，来探索底层实现发生改变时的具体性能。
