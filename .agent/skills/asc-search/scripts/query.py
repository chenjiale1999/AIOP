#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""OKF 索引查询 CLI。

索引实例按 surface 隔离（默认 c_api），构建时已按 CANN 版本 × 架构筛选条件。
过滤查询各维度 AND、逗号值内 OR；独立查询（--api/--deps-of/--enum/--gap）返回结构化卡片。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path, PurePosixPath
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from build_index import EMPTY_GAPS, INDEX_DIR, SCOPES, normalize_dtype  # noqa: E402

FILTER_KEYS = ("stage", "operation", "category", "api_class", "dtype", "kw", "convert", "feature", "version", "product")


def load(index_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    index = json.loads((index_dir / "index.json").read_text(encoding="utf-8"))
    gaps_path = index_dir / "gaps.json"
    gaps = json.loads(gaps_path.read_text(encoding="utf-8")) if gaps_path.is_file() else dict(EMPTY_GAPS)
    return index, gaps


def csv(value: str | None) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()] if value else []


def resolve_api(index: dict[str, Any], name: str) -> str | None:
    """API 名 / 文档全路径 / 文件名 → 索引文档路径；未命中返回 None（由调用方空结果处理）。"""
    if name in index["documents"]:
        return name
    view = index["views"]["by_api"]
    if name in view:
        candidates = view[name]
        # 家族 canonical 页（<family>/<family>.md）优先于废弃/变体成员作 owner
        for path in candidates:
            if PurePosixPath(path).stem == name and PurePosixPath(path).parent.name == name:
                return path
        return candidates[0]
    for path in index["documents"]:
        if Path(path).stem == name or Path(path).name == f"{name}.md":
            return path
    return None


def why_tags(index: dict[str, Any], path: str, dims: dict[str, list[str]]) -> list[str]:
    reasons = []
    for stage in dims.get("stage", []):
        if path in index["views"]["by_stage"].get(stage, []):
            reasons.append(f"stage:{stage}")
    for op in dims.get("operation", []):
        if path in index["views"]["by_operation"].get(op, []):
            reasons.append(f"operation:{op}")
    for cat in dims.get("category", []):
        if path in index["views"]["by_category"].get(cat, []):
            reasons.append(f"category:{cat}")
    for cls in dims.get("api_class", []):
        if path in index["views"]["by_api_class"].get(cls, []):
            reasons.append(f"api_class:{cls}")
    for dtype in dims.get("dtype", []):
        if path in index["views"]["by_dtype"].get(dtype, []):
            reasons.append(f"dtype:{dtype}")
    for pair in dims.get("pair", []):
        if path in index["views"]["by_dtype_pair"].get(pair, []):
            reasons.append(f"convert:{pair}")
    for feature in dims.get("feature", []):
        if path in index["views"]["by_feature"].get(feature, []):
            reasons.append(f"feature:{feature}")
    for version in dims.get("version", []):
        if path in index["views"]["by_version"].get(version, []):
            reasons.append(f"version:{version}")
    for product in dims.get("product", []):
        if path in index["views"]["by_product"].get(product, []):
            reasons.append(f"product:{product}")
    for kw in dims.get("kw", []):
        doc = index["documents"][path]
        haystack = f"{doc['title']} {doc['description']}".lower()
        if kw.lower() in haystack:
            reasons.append(f"kw:{kw}")
    return reasons


def filter_query(index: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    dims: dict[str, list[str]] = {
        "stage": [s for s in csv(args.stage) if s] if args.stage else [],
        "operation": csv(args.operation),
        "category": csv(args.category),
        "api_class": csv(args.api_class),
        "dtype": [normalize_dtype(d) for d in csv(args.dtype)],
        "kw": csv(args.kw),
        "pair": [],
        "feature": csv(args.feature),
        "version": [args.version.strip()] if args.version and args.version.strip() else [],
        "product": csv(args.product),
    }
    if args.convert:
        src = normalize_dtype(args.convert[0])
        dims["pair"] = [f"{src}->{normalize_dtype(dst)}" for dst in csv(args.convert[1])]
    if not any(dims[key] for key in dims):
        raise ValueError(
            "no filter given; use --stage/--operation/--category/--api-class/--dtype/--kw/--convert/--feature/--version/--product"
        )
    # 维度值不合法时不报错：视作空结果，交由 Agent 走降级链并用同义词重试
    candidates: set[str] | None = None
    for key, view_name in (
        ("stage", "by_stage"),
        ("operation", "by_operation"),
        ("category", "by_category"),
        ("api_class", "by_api_class"),
        ("dtype", "by_dtype"),
        ("pair", "by_dtype_pair"),
        ("feature", "by_feature"),
        ("version", "by_version"),
        ("product", "by_product"),
    ):
        values = dims[key]
        if not values:
            continue
        matched: set[str] = set()
        for value in values:
            matched.update(index["views"][view_name].get(value, []))
        candidates = matched if candidates is None else candidates & matched
    if dims["kw"]:
        matched = {
            path
            for path in (candidates if candidates is not None else index["documents"])
            if any(
                kw.lower() in f"{index['documents'][path]['title']} {index['documents'][path]['description']}".lower()
                for kw in dims["kw"]
            )
        }
        candidates = matched
    return [{"path": path, "why": why_tags(index, path, dims)} for path in sorted(candidates or set())]


def api_card(index: dict[str, Any], name: str) -> list[dict[str, Any]]:
    owner = resolve_api(index, name)
    if owner is None:
        return []
    doc = index["documents"][owner]
    results = [{"path": owner, "why": ["api:owner"]}]
    for member in index["views"]["by_api"].get(doc["family"], []):
        if member != owner:
            results.append({"path": member, "why": ["family-member"]})
    if doc["nav_page"] != owner and doc["nav_page"] in index["documents"]:
        results.append({"path": doc["nav_page"], "why": ["category-page"]})
    return results


def deps_query(index: dict[str, Any], target: str, kind: str | None, reverse: bool) -> list[dict[str, Any]]:
    path = resolve_api(index, target)
    if path is None:
        return []
    doc = index["documents"][path]
    results: list[dict[str, Any]] = []
    if reverse:
        # 反向按解析到的家族聚合：族名/成员名/全路径都能查到全族入边
        family_paths = set(index["views"]["by_api"].get(doc["family"], []))
        family_paths.add(path)
        for edge in index["graph"]["edges"]:
            if edge["to"] in family_paths and (not kind or edge["kind"] == kind):
                results.append(
                    {
                        "path": edge["from"],
                        "why": [f"reverse:{edge['kind']}"],
                        "condition": edge["condition"],
                        "out_of_scope": edge["from"] not in index["documents"],
                    }
                )
    else:
        for dep in doc["dependencies"]:
            if kind and dep["kind"] != kind:
                continue
            results.append(
                {
                    "path": dep["path"],
                    "why": [f"dep:{dep['kind']}"],
                    "condition": dep["condition"],
                    "out_of_scope": dep["path"] not in index["documents"],
                }
            )
    return results


def param_query(index: dict[str, Any], token: str) -> list[dict[str, Any]]:
    """参数名或枚举类型反查：--param round_mode / --param asc_cast_round_mode。"""
    results = []
    for path, doc in sorted(index["documents"].items()):
        hits = [
            p
            for sig in doc.get("signatures", [])
            for p in sig.get("params", [])
            if p.get("name") == token or p.get("type") == token
        ]
        if hits:
            param = hits[0]
            reasons = [f"param:{param.get('name', '')}"]
            if str(param.get("type", "")).startswith("asc_"):
                reasons.append(f"type:{param['type']}")
            results.append({"path": path, "why": reasons})
    if not results and token in index.get("defs", {}).get("enums", {}):
        enum = index["defs"]["enums"][token]
        results.append(
            {
                "path": enum["header"],
                "why": [f"enum:{token}"],
                "enum": {"values": enum["values"][:20], "header": enum["header"], "total_values": len(enum["values"])},
            }
        )
    return results


def enum_query(index: dict[str, Any], token: str) -> list[dict[str, Any]]:
    """枚举名或枚举值反查：--enum asc_cast_round_mode / --enum ASC_CAST_RN。"""
    enums = index.get("defs", {}).get("enums", {})
    constants = {c["name"]: c for c in index.get("defs", {}).get("constants", [])}
    results = []
    if token in enums:
        users = index["views"].get("by_enum", {}).get(token, [])
        enum = enums[token]
        results.append(
            {
                "path": f"include/{enum['header']}",
                "why": [f"enum:{token}"],
                "enum": {"values": enum["values"], "used_by": users},
            }
        )
    value_owner = None
    for name, enum in enums.items():
        if token in enum["values"]:
            value_owner = name
            break
    if value_owner is None and token in constants:
        value_owner = next(
            (e for e, info in enums.items() if any(c["name"] == token for c in constants.values() if True) and False),
            None,
        )
        # 常量名 ASC_CAST_RN → 归属枚举（defs 常量与枚举同名族）
        stem = token[4:] if token.startswith("ASC_") else token
        for name in enums:
            if name.lower().endswith(stem.split("_")[0].lower()):
                value_owner = name
                break
    if value_owner:
        users = index["views"].get("by_enum", {}).get(value_owner, [])
        results.append(
            {
                "path": f"include/{enums[value_owner]['header']}",
                "why": [f"enum-value:{token}", f"enum:{value_owner}"],
                "enum": {"value": token, "used_by": users},
            }
        )
    if not results:
        raise ValueError(f"unknown enum or value: {token}")
    return results


def gap_query(gaps: dict[str, Any], keyword: str) -> list[dict[str, Any]]:
    keywords = [k.strip().lower() for k in keyword.split(",") if k.strip()]
    if not keywords:
        raise ValueError("--gap requires a non-empty keyword")

    def hit(haystack: str) -> bool:
        return any(kw in haystack.lower() for kw in keywords)

    results = []
    for gap in gaps["scope_out_gaps"]:
        haystack = " ".join([gap["id"], gap["conclusion"], *gap["keywords"]])
        if hit(haystack):
            results.append({"gap_id": gap["id"], "conclusion": gap["conclusion"], "hints": gap["hints"]})
    for gap in gaps["relation_gaps"]:
        haystack = f"{gap['from']} {gap['note']}"
        if hit(haystack):
            results.append(
                {
                    "gap_id": f"relation:{Path(gap['from']).stem}",
                    "conclusion": gap["note"],
                    "hints": [{"path": gap["from"], "note": "范围内部文档"}],
                }
            )
    return results


def run_query(args: argparse.Namespace, index: dict[str, Any], gaps: dict[str, Any]) -> list[dict[str, Any]]:
    if args.gap is not None:
        return gap_query(gaps, args.gap)
    if args.api:
        return api_card(index, args.api)
    if args.deps_of:
        return deps_query(index, args.deps_of, args.kind, args.reverse)
    if args.param:
        return param_query(index, args.param)
    if args.enum:
        return enum_query(index, args.enum)
    return filter_query(index, args)


def render(index: dict[str, Any], results: list[dict[str, Any]]) -> str:
    if not results:
        return "no match"
    lines = []
    for row in results:
        if "gap_id" in row:
            hints = "; ".join(h["path"] for h in row["hints"])
            lines.append(f"{row['gap_id']}\n  {row['conclusion']}\n  hints: {hints}")
            continue
        if "enum" in row:
            enum = row["enum"]
            detail = f"值: {', '.join(enum['values'][:12])}" if "value" not in enum else f"枚举值 {enum['value']}"
            users = f"；使用方 {len(enum['used_by'])} 篇" if enum.get("used_by") else ""
            lines.append(f"{row['path']} | {','.join(row['why'])}\n  {detail}{users}")
            continue
        doc = index["documents"].get(row["path"])
        name = doc["title"] if doc else Path(row["path"]).stem
        desc = (doc["description"] if doc else "")[:60]
        flags = " [范围外]" if row.get("out_of_scope") else ""
        condition = f"（{row['condition']}）" if row.get("condition") else ""
        lines.append(f"{row['path']}{flags} | {name} | {','.join(row['why'])}{condition}\n  {desc}")
    return "\n".join(lines)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--surface",
        default="c_api",
        choices=sorted(SCOPES),
        help="API surface 索引实例（默认 c_api）；索引构建时已按 CANN×架构筛选条件",
    )
    p.add_argument("--index-dir", type=Path, default=None, help="显式索引目录（默认 index/<surface>）")
    p.add_argument("--json", action="store_true", help="机器可读输出")
    p.add_argument("--stage", help="逗号列表: in,out,compute")
    p.add_argument("--operation", help="逗号列表（如 copy,divide,gather）")
    p.add_argument("--category", help="逗号列表（如 vector_compute,vector_datamove）")
    p.add_argument("--api-class", dest="api_class", help="API 分类=include 子目录（如 vector_datamove,reg_compute）")
    p.add_argument("--dtype", help="逗号列表（如 half,fp8_e4m3fn）")
    p.add_argument("--kw", help="title+description 子串")
    p.add_argument("--convert", nargs=2, metavar=("SRC", "DST"), help="类型转换对，DST 可为逗号列表")
    p.add_argument("--feature", help="特性（如 low_precision,low_precision_matrix,int4）")
    p.add_argument("--version", help="按 version 区间精确匹配（如 '>=9.1.0'；含 >= 按 shell 规则加引号）")
    p.add_argument("--product", help="产品 npu 标记（如 950,A3,910b）")
    p.add_argument("--api", help="家族卡：主页+同族成员+分类页")
    p.add_argument("--deps-of", dest="deps_of", help="依赖图遍历（API 名或路径）")
    p.add_argument("--kind", choices=["requires", "configures", "constrained_by", "related", "replaced_by"])
    p.add_argument("--reverse", action="store_true", help="配合 --deps-of：反向边")
    p.add_argument("--param", help="参数名或枚举类型反查（如 round_mode,asc_cast_round_mode）")
    p.add_argument("--enum", help="枚举名或枚举值反查（如 asc_cast_round_mode,ASC_CAST_RN）")
    p.add_argument("--gap", help="缺口登记检索")
    return p


def main() -> int:
    args = parser().parse_args()
    try:
        index, gaps = load(args.index_dir if args.index_dir is not None else INDEX_DIR / args.surface)
        results = run_query(args, index, gaps)
        if args.gap is not None and not results and not gaps.get("scope_out_gaps"):
            print("[]" if args.json else "无登记缺口：原范围外缺口已全部纳入索引范围")
            return 0
        print(json.dumps(results, ensure_ascii=False, indent=1) if args.json else render(index, results))
        return 0
    except (ValueError, KeyError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
