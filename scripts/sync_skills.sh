#!/usr/bin/env bash
# 从上游仓库同步 aiop 收编的 skill，并在 SKILL.md 顶部标注路径基准。
# 用法：
#   bash scripts/sync_skills.sh                 # 同步全部
#   bash scripts/sync_skills.sh tilelang-ascend # 只同步指定 skill
set -euo pipefail

AIO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TILELANG_ROOT="${TILELANG_ROOT:-${AIOP_TILELANG_REPO:-$AIO_ROOT/../tilelang}}"
DEVKIT_ROOT="${DEVKIT_ROOT:-$AIO_ROOT/../asc-devkit}"
DEST="$AIO_ROOT/.agent/skills"

ASC_SKILLS=(asc-npu-arch asc-search ascendc-api-kernel-debug)
TL_SKILLS=(tilelang-ascend tilelang-layout tilelang-semantic tilelang-build)

ASC_NOTE="> **aiop 副本说明**：本 skill 复制自 \`asc-devkit\`（\`${DEVKIT_ROOT}\`）。文中相对路径以该仓为根解析；\`\${DEVKIT_PATH}\` 指该仓根目录。重新同步：\`scripts/sync_skills.sh\`。"
TL_NOTE="> **aiop 副本说明**：本 skill 复制自 \`tilelang\`（\`${TILELANG_ROOT}\`）。文中 \`../../..\` 相对路径以该仓为根解析，\`examples/\`、\`testing/\`、\`docs/\` 均指该仓目录。重新同步：\`scripts/sync_skills.sh\`。"

insert_note() {
  python3 - "$1" "$2" <<'PY'
import sys
import pathlib

path = pathlib.Path(sys.argv[1])
note = sys.argv[2]
text = path.read_text(encoding="utf-8")
if note in text:
    sys.exit(0)
lines = text.splitlines()
if not lines or lines[0].strip() != "---":
    sys.exit(f"missing YAML front matter: {path}")
end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
lines[end + 1:end + 1] = ["", note]
path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

sync_one() {
  local src_root="$1" name="$2" note="$3"
  if [[ ! -d "$src_root/$name" ]]; then
    echo "warn: $src_root/$name 不存在，跳过" >&2
    return 0
  fi
  rm -rf "$DEST/$name"
  mkdir -p "$DEST/$name"
  rsync -a --exclude '__pycache__' --exclude '.pytest_cache' "$src_root/$name/" "$DEST/$name/"
  insert_note "$DEST/$name/SKILL.md" "$note"
  echo "synced: $name <- $src_root"
}

want() {
  [[ $# -eq 0 ]] && return 0
  local name="$1"; shift
  local x
  for x in "$@"; do [[ "$x" == "$name" ]] && return 0; done
  return 1
}

mkdir -p "$DEST"

for name in "${ASC_SKILLS[@]}"; do
  want "$name" "$@" && sync_one "$DEVKIT_ROOT/.agent/skills" "$name" "$ASC_NOTE"
done

for name in "${TL_SKILLS[@]}"; do
  want "$name" "$@" && sync_one "$TILELANG_ROOT/.agents/skills" "$name" "$TL_NOTE"
done

# 统一入口软链（与上游仓约定一致）
for d in .claude .codex .opencode; do
  mkdir -p "$AIO_ROOT/$d"
  ln -sfn ../.agent/skills "$AIO_ROOT/$d/skills"
done

echo "done. skills in $DEST"
