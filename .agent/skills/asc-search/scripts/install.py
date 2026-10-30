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

"""Install asc-search discovery links while keeping its devkit source in place."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

AGENTS = ("codex", "claude", "opencode")
SKILL_ROOT = Path(__file__).resolve().parents[1]


def target_directories(repo_root: Path, scope: str) -> dict[str, Path]:
    if scope == "project":
        return {
            "codex": repo_root / ".agents/skills",
            "claude": repo_root / ".claude/skills",
            "opencode": repo_root / ".opencode/skills",
        }
    user_home = Path.home()
    config_home = Path(os.environ.get("XDG_CONFIG_HOME") or user_home / ".config")
    if not config_home.is_absolute():
        config_home = user_home / ".config"
    return {
        "codex": user_home / ".agents/skills",
        "claude": user_home / ".claude/skills",
        "opencode": config_home / "opencode/skills",
    }


def install(source: Path, targets: dict[str, Path], dry_run: bool = False) -> None:
    source = source.resolve(strict=True)
    if not (source / "SKILL.md").is_file():
        raise ValueError(f"Skill manifest not found: {source / 'SKILL.md'}")
    planned = []
    for agent, directory in targets.items():
        target = directory / source.name
        if target.exists() or target.is_symlink():
            if target.resolve() == source:
                planned.append((agent, target, False))
                continue
            raise ValueError(f"Existing skill left unchanged: {target}")
        # Check every destination before creating any links or directories.
        ancestor = directory
        while not ancestor.exists() and not ancestor.is_symlink():
            ancestor = ancestor.parent
        if not ancestor.is_dir():
            raise ValueError(f"Skill directory is not accessible: {ancestor}")
        planned.append((agent, target, True))
    for agent, target, create in planned:
        if create and not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source, target_is_directory=True)
        status = "would install" if create and dry_run else "installed" if create else "already installed"
        print(f"{agent}: {status}: {target} -> {source}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=(*AGENTS, "all"), default="all")
    parser.add_argument("--scope", choices=("user", "project"), default="user")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    repo_root = SKILL_ROOT.parents[2]
    if not (repo_root / "docs/zh/api/SIMD-API/c_api").is_dir() or not (repo_root / "include/c_api").is_dir():
        print("error: Run the installer from a complete asc-devkit checkout.", file=sys.stderr)
        return 1
    targets = target_directories(repo_root, args.scope)
    if args.agent != "all":
        targets = {args.agent: targets[args.agent]}
    try:
        install(SKILL_ROOT, targets, args.dry_run)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Devkit checkout: {repo_root}")
    print("Restart or reload your agent to discover asc-search. Keep this checkout in place.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
