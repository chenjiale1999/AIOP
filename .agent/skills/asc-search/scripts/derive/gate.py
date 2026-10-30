# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""索引筛选条件：CANN 版本检测、架构映射与版本区间判定。

检测链（references/index_derivation_contract.md 第 4 节）：--cann-version 显式参数 > $ASCEND_HOME_PATH
安装目录版本 > 仓 version.cmake。架构标签经 product-map.yaml 映射架构号。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

VERSION_RANGE_RE = re.compile(r"^>=(?P<floor>\d+\.\d+(?:\.\d+)?)(?:,<=(?P<ceil>\d+\.\d+(?:\.\d+)?))?$")


def load_product_map(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "archs" not in payload:
        raise ValueError(f"invalid product map: {path}")
    return payload


def arch_label_to_number(label: str, product_map: dict[str, Any]) -> str:
    if label in product_map["archs"]:
        return label
    for arch, info in product_map["archs"].items():
        if label in info.get("labels", []):
            return str(arch)
    raise ValueError(f"unknown arch label: {label!r}; known architecture numbers: {list(product_map['archs'])}")


def npu_to_arch(marker: str, product_map: dict[str, Any]) -> str | None:
    for arch, info in product_map["archs"].items():
        if marker in info.get("npu_markers", []):
            return str(arch)
    return None


def detect_cann_version(explicit: str | None, repo_root: Path) -> str:
    if explicit:
        return explicit
    home = os.environ.get("ASCEND_HOME_PATH", "")
    if home:
        match = re.search(r"cann-(\d+\.\d+(?:\.\d+)?)", home)
        if match:
            return match.group(1)
    cmake = repo_root / "version.cmake"
    if cmake.is_file():
        match = re.search(r"VERSION\s+\"(\d+\.\d+(?:\.\d+)?)\"", cmake.read_text(encoding="utf-8"))
        if match:
            return match.group(1)
    raise ValueError("cannot detect local CANN version; pass --cann-version explicitly")


def version_covers(version: str, cann_version: str) -> bool:
    """version 形如 >=9.0.0 或 >=9.0.0,<=9.2.0；判定是否覆盖给定 CANN 版本。"""
    match = VERSION_RANGE_RE.fullmatch(version)
    if not match:
        return False
    floor, ceil = match.group("floor"), match.group("ceil")
    target = tuple(int(x) for x in cann_version.split("."))
    low = tuple(int(x) for x in floor.split("."))
    if target < low:
        return False
    if ceil:
        high = tuple(int(x) for x in ceil.split("."))
        return target <= high
    return True


def version_sort_key(version: str) -> tuple[int, ...]:
    floor = version.split(",", 1)[0].removeprefix(">=")
    return tuple(int(x) for x in floor.split("."))
