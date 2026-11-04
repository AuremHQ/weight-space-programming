#!/usr/bin/env bash
#
# Weight-Space Programming — skill installer
#
# Installs SKILL.md from the public repo into your AI agent's skills folder.
#
# This script only downloads ONE markdown file. It never runs downloaded
# code, never asks for sudo, never touches your models, and never sends
# anything anywhere. If you do not trust it, read it first — it is short.
#
# Usage:
#   bash install.sh                  Install to the default folder
#   bash install.sh --dir <path>     Install to a folder you choose
#   bash install.sh --help           Show this help
#
# One-line install (fetches this script from the repo and runs it):
#   curl -fsSL https://raw.githubusercontent.com/AuremHQ/weight-space-programming/main/install.sh | bash
#
# Prefer to read it first? Download it, read it, then run it:
#   curl -fsSL https://raw.githubusercontent.com/AuremHQ/weight-space-programming/main/install.sh -o install.sh
#   bash install.sh
#
# To uninstall, just delete the folder it printed at the end.
#
set -euo pipefail

REPO_RAW="https://raw.githubusercontent.com/AuremHQ/weight-space-programming/main"
SKILL_FILE="SKILL.md"
SKILL_URL="$REPO_RAW/$SKILL_FILE"

usage() {
    echo "Usage: bash install.sh [--dir <path>]"
    echo "  --dir <path>   Install folder (default: \$HOME/.claude/skills/weight-space-programming)"
    echo "  --help         Show this help"
}

TARGET_DIR=""
while [ $# -gt 0 ]; do
    case "$1" in
        --dir)
            if [ $# -lt 2 ]; then
                echo "Error: --dir needs a path." >&2
                usage >&2
                exit 1
            fi
            TARGET_DIR="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [ -z "$TARGET_DIR" ]; then
    if [ -z "${HOME:-}" ]; then
        echo "Error: HOME is not set. Pass a folder explicitly: bash install.sh --dir <path>" >&2
        exit 1
    fi
    TARGET_DIR="$HOME/.claude/skills/weight-space-programming"
fi

# Pick a downloader. curl preferred, wget as fallback.
if command -v curl >/dev/null 2>&1; then
    download() { curl -fsSL "$1" -o "$2"; }
elif command -v wget >/dev/null 2>&1; then
    download() { wget -qO "$2" "$1"; }
else
    echo "Error: neither curl nor wget was found. Install one of them and run this again." >&2
    exit 1
fi

mkdir -p "$TARGET_DIR"

# Back up an existing install instead of silently overwriting it.
if [ -f "$TARGET_DIR/$SKILL_FILE" ]; then
    cp "$TARGET_DIR/$SKILL_FILE" "$TARGET_DIR/$SKILL_FILE.bak"
    echo "Backed up your existing skill to: $TARGET_DIR/$SKILL_FILE.bak"
fi

# Download to a temp file first, so a failed download never leaves a
# half-written skill behind. Then sanity-check it before moving it in.
TMP_FILE="$(mktemp)"
trap 'rm -f "$TMP_FILE"' EXIT

echo "Downloading $SKILL_URL ..."
if ! download "$SKILL_URL" "$TMP_FILE"; then
    echo "Error: download failed. Check your internet connection and try again." >&2
    exit 1
fi

if [ ! -s "$TMP_FILE" ]; then
    echo "Error: downloaded file is empty. Nothing was installed." >&2
    exit 1
fi

if ! grep -q "Weight-Space Programming" "$TMP_FILE"; then
    echo "Error: downloaded file does not look like the skill. Nothing was installed." >&2
    exit 1
fi

mv "$TMP_FILE" "$TARGET_DIR/$SKILL_FILE"
trap - EXIT

echo ""
echo "Installed: $TARGET_DIR/$SKILL_FILE"
echo ""
echo "Next step: ask your AI agent to follow that skill file on your own"
echo "small open-weight model. The skill refuses closed API models, models"
echo "over 3B parameters, and any edit over 1% of weights — by design."
echo ""
echo "To uninstall, delete the folder: rm -rf \"$TARGET_DIR\""
