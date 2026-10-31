#!/usr/bin/env python3
"""Generate the Ascend-exclusive op reference from the dialect source tree.

Every public function/class under ``tilelang/ascend/language`` is parsed and
rendered with name, signature, core behaviour, parameters, return rules, A5
hardware adaptation, performance guidance, example, and caveats. The output is
knowledge/13-ascend-op-reference.md plus a machine manifest.

Usage:
    python scripts/gen_ascend_op_kb.py            # generate
    python scripts/gen_ascend_op_kb.py --check    # verify 100% coverage
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from aiop import ascend_ops  # noqa: E402
from aiop.config import load_config  # noqa: E402

DEFAULT_DOC = PROJECT_ROOT / "knowledge" / "13-ascend-op-reference.md"
DEFAULT_MANIFEST = PROJECT_ROOT / "reports" / "ascend_op_manifest.json"

SECTION_ALIASES = {
    "args": "params",
    "arguments": "params",
    "parameters": "params",
    "parameter": "params",
    "params": "params",
    "returns": "returns",
    "return": "returns",
    "yields": "returns",
    "raises": "raises",
    "raise": "raises",
    "exceptions": "raises",
    "example": "example",
    "examples": "example",
    "usage": "example",
    "note": "notes",
    "notes": "notes",
    "warning": "notes",
    "warnings": "notes",
    "attributes": "attributes",
}

MODULE_TITLE = {
    "allocate.py": "内存分配（UB / L1 / L0A / L0B / L0C / SS）",
    "annotations.py": "buffer 版本与容量注解",
    "copy_op.py": "数据搬运（DMA / MTE）",
    "dma.py": "DMA 辅助（pad 寄存器 / ND→NZ）",
    "frame.py": "AIC / AIV 作用域",
    "gemm_op.py": "Cube GEMM",
    "kernel.py": "核启动与线程域",
    "loop.py": "循环构造",
    "math.py": "标量数学",
    "mode.py": "Cube / 存储模式控制",
    "print.py": "调试打印",
    "random.py": "随机数",
    "reduce_op.py": "归约",
    "schedule_hint.py": "调度提示",
    "sync.py": "PIPE / 跨核同步",
    "tile_schedule.py": "持久化 tile 调度",
    "simd.py": "SIMD MicroAPI",
}

MODULE_DEFAULT_HW = {
    "allocate.py": "片上存储层级：UB（shared.dyn）/ L1（shared.l1）/ L0A / L0B / L0C / SS；由 AutoSchedule 管理版本与复用",
    "annotations.py": "编译期注解：多缓冲版本、手动多缓冲、容量约束解除；不直接产生硬件指令",
    "copy_op.py": "DMA/MTE 搬移路径（GM↔UB、GM↔L1、L1↔L0、L0C↔UB/GM），可带 L2 cache 策略与 NZ 转换",
    "dma.py": "GM→UB 行填充与 ND→NZ 后处理；涉及 PIPE_S pad 寄存器",
    "frame.py": "AIC（Cube）与 AIV（Vector）核作用域；SimtVF 线程域 / SimdVF 寄存器域",
    "gemm_op.py": "Cube/AIC MAD 单元；L1 输入或 L0A/L0B→L0C 直接 MAD；支持 MX block-scaled",
    "kernel.py": "1-D AI core 网格启动；线程域由 SimtVF 在核内声明",
    "loop.py": "通用循环构造，unroll_factor 由 Ascend codegen 转成 #pragma unroll",
    "math.py": "标量数学；在 VF 内优先复用已有向量 intrinsic",
    "mode.py": "Cube MAD 方向、HF32 精度模式、GM 存储模式原子",
    "print.py": "NPU 执行模型下的调试打印：kernel 每核执行一次，UB/L1 为核内私有",
    "random.py": "随机数状态初始化；默认 sequence id 由 SimtVF lane 与 core 索引推导",
    "reduce_op.py": "Vector 归约；SimdVF 内直接对 UB 发射，其余路径走 fragment 往返",
    "schedule_hint.py": "自动调度提示：Stage / Task / PerCoreTask / 冲突假设",
    "sync.py": "AscendC PIPE barrier、SetFlag/WaitFlag、跨核 flag、pipe buffer 锁",
    "tile_schedule.py": "持久化 GEMM tile 遍历（GROUP_M swizzle / N-first snake）",
    "simd.py": "CCE 2048-bit SIMD 向量单元；仅在 T.SimdVF() 内使用",
}

MODULE_DEFAULT_PERF = {
    "allocate.py": "按最小生命周期分配；复用交给 AutoSchedule 多缓冲，避免手工放大版本数",
    "copy_op.py": "优先整块 DMA，保持 32B 对齐；避免碎片化小拷贝与多余 NZ 转换",
    "gemm_op.py": "K 循环首步 clear_accum=True；L1 路径保持 transpose_B=True；仅在实测有收益时启用 HF32",
    "reduce_op.py": "优先 reducer epoch 或 T.reduce_*，不要在标量循环里逐元素累加",
    "sync.py": "优先自动调度；手写 flag 必须成对且覆盖所有分支",
    "mode.py": "set_atomic 只在需要 GM 累加时开启，存储结束后立即 set_atomic_none",
    "simd.py": "优先用一条 SIMD 指令替代标量循环；mask 与 dtype 位宽必须匹配，避免额外 vcvt",
    "tile_schedule.py": "按 core 数选择 GROUP_M 分组；用实测 device time 选择 swizzle 顺序",
    "kernel.py": "grid 不超过 Vector 64 / Cube 32；线程域只写在 SimtVF 内",
}
MODULE_DEFAULT_CAVEAT = {
    "simd.py": "仅在 T.SimdVF() 内调用；mask 位宽与源 dtype 对齐",
    "gemm_op.py": "L0 输入要求尾维从 0 开始且为紧凑 tile；L1 路径需匹配 transpose 约定",
    "sync.py": "手写同步必须整体照抄一个完整示例，禁止混搭半自动/半手动片段",
    "allocate.py": "shape 与 scope 必须满足片上容量与对齐约束",
    "copy_op.py": "跨级搬运必须走 T.copy；禁止 GM→L0 直接搬运",
}
SIMD_SUMMARY = {
    "pand": "谓词按位与（谓词寄存器逐 lane 合并）",
    "pnot": "谓词按位取反",
    "por": "谓词按位或",
    "psel": "按谓词选择两个谓词源",
    "pxor": "谓词按位异或",
    "vabs": "向量逐 lane 绝对值",
    "vadd": "向量逐 lane 加法",
    "vadds": "向量与标量逐 lane 加法",
    "vand": "向量按位与",
    "vexp": "向量逐 lane 指数 exp",
    "vln": "向量逐 lane 自然对数 ln",
    "vmax": "向量逐 lane 最大值",
    "vmaxs": "向量与标量逐 lane 取大",
    "vmin": "向量逐 lane 最小值",
    "vmins": "向量与标量逐 lane 取小",
    "vmul": "向量逐 lane 乘法",
    "vmuls": "向量与标量逐 lane 乘法",
    "vneg": "向量逐 lane 取负",
    "vnot": "向量按位取反",
    "vor": "向量按位或",
    "vrelu": "向量 ReLU（负值截断为 0）",
    "vshl": "向量逐 lane 左移",
    "vshls": "向量按标量左移",
    "vshr": "向量逐 lane 右移",
    "vshrs": "向量按标量右移",
    "vsqrt": "向量逐 lane 平方根",
    "vsub": "向量逐 lane 减法",
    "vxor": "向量按位异或",
}
PARAM_HINTS = {
    "shape": "buffer 形状（静态或符号维度）；需满足对齐与容量约束",
    "dtype": "元素数据类型，支持 float32/bfloat16/float16/int8/int16/int32/uint32 等",
    "scope": "内存 scope；默认值见签名，必须与目标存储层级一致",
    "src": "源 buffer 或区域",
    "dst": "目的 buffer 或区域",
    "src0": "第一源操作数",
    "src1": "第二源操作数",
    "base": "基地址 buffer/指针",
    "index": "逐 lane 索引向量（无符号整型）",
    "mask": "谓词掩码；缺省按 dtype 全 lane 激活",
    "mode": "越界 lane 处理模式，缺省 MODE_ZEROING",
    "part": "LOWER/HIGHER 半区选择",
    "scalar": "标量操作数",
    "value": "写入/累加值",
    "clear_accum": "是否在累加前清零",
    "transpose_A": "是否转置 A",
    "transpose_B": "是否转置 B",
    "buffer": "输入 buffer（UB/共享内存）",
    "out": "归约输出 buffer",
    "dim": "归约/累积维度",
    "clear": "是否先清零输出",
    "batch": "批量归约宽度",
}

PERF_WORDS = (
    "prefer",
    "avoid",
    "faster",
    "throughput",
    "overlap",
    "pipeline",
    "cache",
    "align",
    "tile",
    "double buffer",
    "bandwidth",
    "coalesced",
    "latency",
    "reuse",
    "vectorized",
    "bank",
    "one instruction",
    "single instruction",
)
CAVEAT_WORDS = (
    "must",
    "only",
    "requires",
    "cannot",
    "never",
    "not supported",
    "invalid",
    "raise",
    "reject",
    "deprecated",
    "warning",
    "ensure",
    "禁止",
    "必须",
    "仅",
    "不支持",
)
HW_WORDS = (
    "AIC",
    "AIV",
    "Cube",
    "Vector",
    "UB",
    "L1",
    "L0A",
    "L0B",
    "L0C",
    "PIPE",
    "MTE",
    "SimdVF",
    "SimtVF",
    "cross-core",
    "atomic",
    "dav-3510",
    "dav_3510",
    "Ascend",
    "NPU",
    "CCE",
    "scale factor",
    "fractal",
    "NZ",
)

SIMD_GROUPS = (
    ("分配与辅助", {"alloc_var", "alloc_local", "SimdPair", "pair_get", "make_ubuf_ptr", "mem_bar"}),
    ("访存与索引", {"vld", "vld2", "vsts", "vsstb", "vgather2", "vgatherb", "vscatter"}),
    ("谓词与掩码", set()),
    ("向量计算", set()),
)


@dataclass
class OpDoc:
    info: ascend_ops.OpInfo
    display_signature: str = ""
    summary: str = ""
    params: list[dict[str, str]] = field(default_factory=list)
    returns: str = ""
    hw: str = ""
    perf: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    example: str = ""
    exposed: bool = True


def display_signature(info: ascend_ops.OpInfo) -> str:
    return f"{info.qualname}{info.signature[len(info.name):]}"


def raise_constraints(node: ast.AST) -> list[str]:
    messages = []
    for item in ast.walk(node):
        if not isinstance(item, ast.Raise) or item.exc is None:
            continue
        exc = item.exc
        if isinstance(exc, ast.Call) and exc.args:
            try:
                text = ast.literal_eval(exc.args[0])
            except (ValueError, TypeError):
                text = ast.unparse(exc.args[0])
            if isinstance(text, str):
                messages.append(re.sub(r"\s+", " ", text).strip())
    return list(dict.fromkeys(messages))[:3]


def fill_param_hints(params: list[dict[str, str]]) -> None:
    for param in params:
        if param.get("desc"):
            continue
        name = param["name"].lstrip("*")
        param["desc"] = PARAM_HINTS.get(name, "见签名与源码约束")


def returns_rule(info: ascend_ops.OpInfo, node: ast.AST, section: str) -> str:
    if section:
        return section.strip()
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.returns:
        return f"返回 `{ast.unparse(node.returns)}`"
    inline = re.search(r"\bReturns?[^.]*\.", info.doc or "")
    if inline:
        return inline.group(0).strip()
    if info.kind == "class":
        return "构造并返回该类实例"
    if any(isinstance(item, ast.Return) and item.value is not None for item in ast.walk(node)):
        return "返回 TIR 表达式（句柄/向量值，dtype 由输入与参数决定）"
    return "无显式返回值（intrinsic / 副作用）"


def _dedent(lines: list[str]) -> str:
    body = [line.rstrip() for line in lines]
    while body and not body[0].strip():
        body.pop(0)
    while body and not body[-1].strip():
        body.pop()
    indents = [len(line) - len(line.lstrip()) for line in body if line.strip()]
    cut = min(indents) if indents else 0
    return "\n".join(line[cut:] if len(line) >= cut else line for line in body)


def split_sections(doc: str) -> tuple[str, dict[str, str]]:
    lines = doc.splitlines()
    headers: list[tuple[int, str]] = []
    for idx, raw in enumerate(lines):
        stripped = raw.strip().rstrip(":").strip()
        key = SECTION_ALIASES.get(stripped.lower())
        if not key:
            continue
        nxt = lines[idx + 1].strip() if idx + 1 < len(lines) else ""
        if raw.strip().endswith(":") or not raw.strip() or re.fullmatch(r"-{3,}", nxt or "x"):
            headers.append((idx, key))
    if not headers:
        return _dedent(lines), {}
    summary = _dedent(lines[: headers[0][0]])
    sections: dict[str, str] = {}
    for pos, (idx, key) in enumerate(headers):
        end = headers[pos + 1][0] if pos + 1 < len(headers) else len(lines)
        body = lines[idx + 1 : end]
        if body and re.fullmatch(r"\s*-{3,}\s*", body[0]):
            body = body[1:]
        text = _dedent(body)
        if text:
            sections[key] = (sections.get(key, "") + "\n" + text).strip()
    return summary, sections


def parse_params(block: str) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for raw in block.splitlines():
        if not raw.strip():
            continue
        match = re.match(r"^\s*([\w\*]+(?:\s*,\s*[\w\*]+)*)\s*(?:\(([^)]*)\))?\s*:\s*(.*)$", raw)
        if match:
            type_ = (match.group(2) or "").strip()
            desc = match.group(3).strip()
            # NumPy style is "name : type" followed by an indented description.
            if not type_ and re.search(r"\s:\s", raw):
                type_, desc = desc, ""
            current = {"name": match.group(1).strip(), "type": type_, "desc": desc}
            entries.append(current)
        elif current is not None:
            current["desc"] = (current["desc"] + " " + raw.strip()).strip()
        else:
            entries.append({"name": "", "type": "", "desc": raw.strip()})
    return entries


def ast_params(node: ast.AST) -> list[dict[str, str]]:
    if isinstance(node, ast.ClassDef):
        init = next(
            (item for item in node.body if isinstance(item, ast.FunctionDef) and item.name == "__init__"),
            None,
        )
        if init is None:
            return []
        node = init
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return []
    args = node.args
    positional = list(args.posonlyargs) + list(args.args)
    defaults = [""] * (len(positional) - len(args.defaults)) + [
        ast.unparse(d) for d in args.defaults
    ]
    entries = []
    for arg, default in zip(positional, defaults):
        if arg.arg in ("self", "cls"):
            continue
        entries.append(
            {
                "name": arg.arg,
                "type": ast.unparse(arg.annotation) if arg.annotation else "",
                "default": default,
                "desc": "",
            }
        )
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        entries.append(
            {
                "name": arg.arg,
                "type": ast.unparse(arg.annotation) if arg.annotation else "",
                "default": ast.unparse(default) if default else "",
                "desc": "",
            }
        )
    if args.vararg:
        entries.append({"name": "*" + args.vararg.arg, "type": "", "default": "", "desc": ""})
    if args.kwarg:
        entries.append({"name": "**" + args.kwarg.arg, "type": "", "default": "", "desc": ""})
    return entries


def merge_params(node: ast.AST, doc_params: list[dict[str, str]]) -> list[dict[str, str]]:
    merged = ast_params(node)
    by_name = {entry["name"]: entry for entry in merged}
    doc_map: dict[str, dict[str, str]] = {}
    for entry in doc_params:
        for name in entry["name"].split(","):
            doc_map[name.strip()] = entry
    for name, entry in doc_map.items():
        if name in by_name:
            by_name[name]["desc"] = entry["desc"]
            if not by_name[name]["type"]:
                by_name[name]["type"] = entry["type"]
        elif name:
            merged.append({"name": name, "type": entry["type"], "default": "", "desc": entry["desc"]})
    return merged


def sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?;])\s+|(?<=[。；])", text)
    return [part.strip() for part in parts if len(part.strip()) > 12]


def pick(text: str, words: tuple[str, ...], limit: int) -> list[str]:
    picked = []
    for sentence in sentences(text):
        lowered = sentence.lower()
        if any(word in lowered for word in words) and sentence not in picked:
            picked.append(sentence)
    return picked[:limit]


def hardware_tags(text: str) -> list[str]:
    tags = []
    mapping = [
        ("AIC", "AIC/Cube"),
        ("Cube", "Cube"),
        ("AIV", "AIV/Vector"),
        ("SimdVF", "SimdVF"),
        ("SimtVF", "SimtVF"),
        ("UB", "UB"),
        ("L1", "L1/CBuf"),
        ("L0A", "L0A"),
        ("L0B", "L0B"),
        ("L0C", "L0C"),
        ("PIPE", "PIPE"),
        ("MTE", "MTE/DMA"),
        ("cross-core", "跨核同步"),
        ("atomic", "原子"),
        ("scale factor", "MX ScaleFactor"),
        ("fractal", "fractal 布局"),
        ("dav-3510", "dav-3510"),
        ("dav_3510", "dav-3510"),
    ]
    lowered = text.lower()
    for needle, label in mapping:
        if needle.lower() in lowered and label not in tags:
            tags.append(label)
    return tags


def doc_example(block: str) -> str:
    if not block:
        return ""
    fence = re.search(r"```(?:python|py)?\s*\n(.*?)```", block, re.S)
    if fence:
        return _dedent(fence.group(1).splitlines())
    lines = [line.strip() for line in block.splitlines() if line.strip().startswith(">>>")]
    if lines:
        return "\n".join(line[4:].strip() for line in lines)
    return ""


def synth_example(info: ascend_ops.OpInfo, params: list[dict[str, str]]) -> str:
    required = [p["name"] for p in params if not p.get("default") and p.get("name") and not p["name"].startswith(("*", "**"))]
    args = ", ".join(f"<{name}>" for name in required) if required else ""
    if info.namespace == "T.simd":
        return f"with T.SimdVF():\n    {info.qualname}({args})"
    return f"{info.qualname}({args})"


def docstring_chain(node: ast.AST, class_map: dict[str, ast.ClassDef], depth: int = 0) -> list[str]:
    texts = [ast.get_docstring(node) or ""]
    if isinstance(node, ast.ClassDef) and depth < 3:
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id in class_map:
                texts.extend(docstring_chain(class_map[base.id], class_map, depth + 1))
    return texts


def params_node(node: ast.AST, class_map: dict[str, ast.ClassDef], depth: int = 0) -> ast.AST:
    if isinstance(node, ast.ClassDef):
        init = next(
            (item for item in node.body if isinstance(item, ast.FunctionDef) and item.name == "__init__"),
            None,
        )
        if init is not None:
            return init
        if depth < 3:
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id in class_map:
                    return params_node(class_map[base.id], class_map, depth + 1)
    return node


def build_doc(info: ascend_ops.OpInfo, node: ast.AST, exposed: bool, class_map: dict[str, ast.ClassDef]) -> OpDoc:
    summary, sections = split_sections(info.doc)
    entry = OpDoc(info=info, display_signature=display_signature(info), exposed=exposed)
    flat_summary = re.sub(r"\s+", " ", summary).strip()
    if len(flat_summary) > 320:
        flat_summary = flat_summary[:317].rstrip() + "..."
    entry.summary = (
        flat_summary
        or SIMD_SUMMARY.get(info.name)
        or f"CCE SIMD intrinsic `{info.name}`（源码未提供 docstring；参数与 dtype 约束以签名为准）"
    )
    pnode = params_node(node, class_map)
    doc_params = parse_params(sections.get("params", ""))
    if not doc_params:
        for text in docstring_chain(node, class_map)[1:]:
            _, inherited_sections = split_sections(text)
            doc_params = parse_params(inherited_sections.get("params", ""))
            if doc_params:
                break
    entry.params = merge_params(pnode, doc_params)
    fill_param_hints(entry.params)
    entry.returns = returns_rule(info, node, sections.get("returns", ""))
    tags = hardware_tags(info.doc)
    module_hw = MODULE_DEFAULT_HW.get(info.module, "A5 通用面；以 lowering 结果为准")
    entry.hw = "、".join(tags) if tags else ""
    entry.hw = (entry.hw + "。" + module_hw) if entry.hw else module_hw
    hw_sentences = pick(info.doc, tuple(HW_WORDS) + ("硬件",), 1)
    if hw_sentences:
        entry.hw += "；" + hw_sentences[0]
    entry.perf = pick(info.doc, PERF_WORDS, 2) or [MODULE_DEFAULT_PERF.get(info.module, "以实测为准；先保证正确性再调优")]
    caveats = []
    if sections.get("raises"):
        caveats.append(sections["raises"].splitlines()[0].strip())
    caveats.extend(raise_constraints(node))
    caveats.extend(pick(info.doc, CAVEAT_WORDS, 2))
    if not exposed:
        caveats.append("内部接口：不出现在 T.__all__，不应从生成代码直接调用")
    if not caveats:
        caveats.append(MODULE_DEFAULT_CAVEAT.get(info.module, "无额外约束；以源码与静态检查为准"))
    entry.caveats = list(dict.fromkeys(c for c in caveats if c))[:4]
    entry.example = doc_example(sections.get("example", "")) or synth_example(info, entry.params)
    return entry


def simd_group(name: str) -> str:
    for title, members in SIMD_GROUPS:
        if name in members:
            return title
    if name.startswith("p"):
        return "谓词与掩码"
    if name.startswith("v"):
        return "向量计算"
    return "其他"


def render_entry(entry: OpDoc) -> list[str]:
    info = entry.info
    lines = [f"#### `{info.qualname}`", ""]
    status = "" if entry.exposed or info.namespace == "T.simd" else "（内部接口）"
    lines.append(f"- **签名**：`{entry.display_signature}`{status}")
    lines.append(f"- **核心功能**：{entry.summary or '见源码 docstring'}")
    if entry.params:
        lines.append("- **入参**：")
        for param in entry.params:
            parts = [f"`{param['name']}`"]
            meta = []
            if param.get("type"):
                meta.append(param["type"])
            if param.get("default"):
                meta.append(f"默认 `{param['default']}`")
            if meta:
                parts.append("(" + ", ".join(meta) + ")")
            desc = param.get("desc", "")
            lines.append(f"  - {' '.join(parts)}" + (f"：{desc}" if desc else ""))
    else:
        lines.append("- **入参**：无显式入参")
    lines.append(f"- **返回值**：{entry.returns}")
    lines.append(f"- **A5 硬件适配**：{entry.hw}")
    lines.append("- **性能建议**：")
    lines.extend(f"  - {item}" for item in entry.perf)
    lines.append("- **示例**：")
    lines.append("")
    lines.append("```python")
    lines.extend(entry.example.splitlines())
    lines.append("```")
    lines.append("")
    if entry.caveats:
        lines.append("- **注意事项**：")
        lines.extend(f"  - {item}" for item in entry.caveats)
    else:
        lines.append("- **注意事项**：—")
    lines.append("")
    return lines


def build_markdown(scope: ascend_ops.OpScope, entries: list[OpDoc]) -> str:
    by_module: dict[str, list[OpDoc]] = {}
    for entry in entries:
        if entry.info.namespace == "T":
            by_module.setdefault(entry.info.module, []).append(entry)
    simd_entries = [entry for entry in entries if entry.info.namespace == "T.simd"]

    lines = [
        "# Ascend 专属 OP 参考（A5 / Ascend 后端）",
        "",
        "> 本文件由 `scripts/gen_ascend_op_kb.py` 从源码自动生成，请勿手工编辑。",
        f"> 解析目录：`{scope.language_dir}`",
        f"> 覆盖：目录内 {len(entries)} 个公开定义（`T.*` {len(entries) - len(simd_entries)} 个，`T.simd.*` {len(simd_entries)} 个）",
        f"> 合法调用面：`tilelang.ascend.language.__all__` {len(scope.declared)} 项（含再导出的通用面）",
        "",
        "## 0. 使用规则",
        "",
        "1. 只允许 `import tilelang.ascend.language as T`（或 `from tilelang.ascend import language as T`）。",
        "2. `T.<op>` 必须是 `tilelang.ascend.language` 的声明面成员；`T.simd.<op>` 必须是 "
        "`tilelang/ascend/language/simd.py` 中定义的算子。",
        "3. 禁止 `T.tile.*`、`T.ascend.*`、`T.printf`、`alloc_L1/L0A/L0B/L0C/ub` 等 A3 拼写。",
        "4. 非法调用由静态检查拦截：`API001`（未知）、`API003`（A3 别名）、`API004`（越界命名空间）；"
        "reducer 误用由 `SEM018` 拦截。",
        "5. 本文件是生成算子代码时的第一手 OP 参考；签名与行为以源码为准，本文随源码重新生成。",
        "",
        "## 1. 覆盖总览",
        "",
        "| 模块 | 标题 | 算子数 |",
        "|---|---|---:|",
    ]
    simd_modules = {entry.info.module for entry in simd_entries}
    for module in sorted(by_module):
        lines.append(f"| `{module}` | {MODULE_TITLE.get(module, module)} | {len(by_module[module])} |")
    for module in sorted(simd_modules):
        lines.append(f"| `{module}` | {MODULE_TITLE.get(module, module)} | {len(simd_entries)} |")
    lines.append(f"| **合计** | — | **{len(entries)}** |")
    lines.append("")

    lines.append("## 2. `T.*` 顶层方言算子")
    lines.append("")
    for module in sorted(by_module):
        lines.append(f"### 2.{list(sorted(by_module)).index(module) + 1} `{module}` — {MODULE_TITLE.get(module, module)}")
        lines.append("")
        for entry in by_module[module]:
            lines.extend(render_entry(entry))

    lines.append("## 3. `T.simd.*` SIMD MicroAPI")
    lines.append("")
    grouped: dict[str, list[OpDoc]] = {}
    for entry in simd_entries:
        grouped.setdefault(simd_group(entry.info.name), []).append(entry)
    for index, title in enumerate(("分配与辅助", "谓词与掩码", "访存与索引", "向量计算", "其他"), 1):
        group = grouped.get(title)
        if not group:
            continue
        lines.append(f"### 3.{index} {title}")
        lines.append("")
        for entry in group:
            lines.extend(render_entry(entry))

    lines.append("## 4. 覆盖校验")
    lines.append("")
    lines.append("```bash")
    lines.append("cd <aiop-repo-root>")
    lines.append("python scripts/gen_ascend_op_kb.py --check")
    lines.append("```")
    lines.append("")
    lines.append("校验会逐个核对：目录内每个公开定义都有对应条目，且签名与源码一致。")
    lines.append("")
    return "\n".join(lines)


def collect(scope: ascend_ops.OpScope) -> list[OpDoc]:
    entries = []
    nodes: dict[str, ast.AST] = {}
    class_map: dict[str, ast.ClassDef] = {}
    for path in sorted(scope.language_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
                nodes[node.name] = node
                if isinstance(node, ast.ClassDef):
                    class_map[node.name] = node
    for name in sorted(scope.local):
        info = scope.local[name]
        exposed = info.namespace == "T.simd" or name in scope.declared
        entries.append(build_doc(info, nodes[name], exposed, class_map))
    entries.sort(key=lambda e: (e.info.namespace != "T", e.info.module, e.info.lineno))
    return entries


def manifest(scope: ascend_ops.OpScope, entries: list[OpDoc]) -> dict[str, object]:
    return {
        "language_dir": str(scope.language_dir),
        "scope_source": scope.source,
        "declared_count": len(scope.declared),
        "local_count": len(scope.local),
        "simd_count": len(scope.simd),
        "ops": [
            {
                "name": entry.info.name,
                "qualname": entry.info.qualname,
                "module": entry.info.module,
                "kind": entry.info.kind,
                "signature": entry.display_signature,
                "exposed": entry.exposed,
                "summary": entry.summary,
                "returns": entry.returns,
                "hardware": entry.hw,
                "perf": entry.perf,
                "caveats": entry.caveats,
                "example": entry.example,
            }
            for entry in entries
        ],
    }


REQUIRED_SECTIONS = ("核心功能", "入参", "返回值", "A5 硬件适配", "性能建议", "示例", "注意事项")


def check(doc_path: Path, scope: ascend_ops.OpScope) -> int:
    if not doc_path.is_file():
        print(f"missing document: {doc_path}")
        return 1
    text = doc_path.read_text(encoding="utf-8")
    parts = re.split(r"^#### `([^`]+)`\s*$", text, flags=re.M)
    blocks = {parts[i]: parts[i + 1] for i in range(1, len(parts) - 1, 2)}
    missing = [info.qualname for info in scope.local.values() if info.qualname not in blocks]
    problems = []
    incomplete = []
    for name, info in scope.local.items():
        body = blocks.get(info.qualname, "")
        if f"`{display_signature(info)}`" not in body:
            problems.append(info.qualname)
        absent = [label for label in REQUIRED_SECTIONS if f"**{label}**" not in body]
        empty_summary = re.search(r"\*\*核心功能\*\*：\s*\n", body) is not None
        no_example = "```python" not in body
        if absent or empty_summary or no_example:
            incomplete.append(f"{info.qualname}({','.join(absent) or ('example' if no_example else 'summary')})")
    print(f"directory-defined ops: {len(scope.local)}")
    print(f"missing entries: {len(missing)}")
    if missing:
        print("  " + ", ".join(sorted(missing)))
    print(f"signature mismatches: {len(problems)}")
    if problems:
        print("  " + ", ".join(sorted(problems)))
    print(f"incomplete entries: {len(incomplete)}")
    if incomplete:
        print("  " + ", ".join(sorted(incomplete)[:20]))
    return 0 if not missing and not problems and not incomplete else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", default=str(DEFAULT_DOC))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--tilelang-repo", default="")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config()
    if args.tilelang_repo:
        cfg.paths.tilelang_repo = args.tilelang_repo
    scope = ascend_ops.load_scope(cfg)
    doc_path = Path(args.doc)

    if args.check:
        return check(doc_path, scope)

    entries = collect(scope)
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(build_markdown(scope, entries), encoding="utf-8")
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest(scope, entries), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {doc_path} ({len(entries)} ops)")
    print(f"wrote {manifest_path}")
    return check(doc_path, scope)


if __name__ == "__main__":
    raise SystemExit(main())
