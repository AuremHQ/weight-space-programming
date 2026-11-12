#!/usr/bin/env bash
#
# Weight-Space Programming — easy installer
#
# Installs the skill AND the kit, sets up the `wsp` command, and runs
# the kit's own demo automatically — so you see a result in seconds,
# with nothing to configure.
#
# What it does, plainly:
#   1. Downloads the kit code into ~/.wsp-kit (git clone, or a
#      tarball if git is not installed), at the ref in --ref /
#      WSP_REF (default: main).
#   2. Installs SKILL.md from that SAME checkout into your AI
#      agent's skills folder (backs up an existing copy first),
#      so the skill and the kit can never differ.
#   3. Sets up the `wsp` command in ~/.local/bin (no sudo, and it
#      never changes your PATH — it tells you if ~/.local/bin is
#      not on your PATH yet).
#   4. Runs the kit's own demo (numpy-only, no network, no model
#      download, nothing leaves your machine) and shows the result.
#
# After install, fix your own model with one command:
#   wsp auto
# It asks 4 simple questions (which model, what goes wrong, what
# goes right, what should not change), then runs the pipeline by
# itself and shows the result. Nothing runs on a model you did not
# point it at, and no edit ever overwrites your original model on disk.
#
# Usage:
#   bash install.sh                  Install + run the demo
#   bash install.sh --dir <path>     Skill folder you choose
#   bash install.sh --no-demo        Install, skip the demo
#   bash install.sh --ref <tag>      Install from a tag or branch
#   bash install.sh --dry-run        Print the ref and exit, install nothing
#   bash install.sh --help           Show this help
#
# The ref comes from --ref, else the WSP_REF environment variable,
# else "main".
#
# One-line install (fetches this script from the repo and runs it):
#   curl -fsSL https://raw.githubusercontent.com/AuremHQ/weight-space-programming/main/install.sh | bash
#
# Prefer to read it first? Download it, read it, then run it:
#   curl -fsSL https://raw.githubusercontent.com/AuremHQ/weight-space-programming/main/install.sh -o install.sh
#   bash install.sh
#
# To uninstall, delete ~/.wsp-kit, ~/.local/bin/wsp, and the skill
# folder it printed at the end.
#

set -euo pipefail

REPO="AuremHQ/weight-space-programming"
REPO_GIT="https://github.com/$REPO.git"
SKILL_FILE="SKILL.md"
KIT_DIR="${WSP_KIT_DIR:-$HOME/.wsp-kit}"
BIN_DIR="$HOME/.local/bin"

# Release tag goes here once v0.3.0 is tagged: change the default
# below from "main" to the tag (e.g. "v0.3.0").
WSP_REF="${WSP_REF:-main}"

usage() {
    echo "Usage: bash install.sh [--dir <path>] [--no-demo] [--ref <tag>] [--dry-run]"
    echo "  --dir <path>   Skill folder (default: \$HOME/.claude/skills/weight-space-programming)"
    echo "  --no-demo      Install, skip the automatic demo"
    echo "  --ref <tag>    Tag or branch to install (default: \$WSP_REF or main)"
    echo "  --dry-run      Print the ref and exit, install nothing"
    echo "  --help         Show this help"
}

TARGET_DIR=""
NO_DEMO=0
DRY_RUN=0
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
        --ref)
            if [ $# -lt 2 ]; then
                echo "Error: --ref needs a tag or branch." >&2
                usage >&2
                exit 1
            fi
            WSP_REF="$2"
            shift 2
            ;;
        --no-demo)
            NO_DEMO=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
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

# --dry-run: print what would be installed and exit before any
# download, clone, or file write (used by the test suite and CI).
if [ "$DRY_RUN" = 1 ]; then
    echo "ref: $WSP_REF"
    echo "repo: $REPO_GIT"
    echo "skill dir: $TARGET_DIR"
    echo "kit dir: $KIT_DIR"
    exit 0
fi

# Pick a downloader. curl preferred, wget as fallback. Only used
# for the tarball path below (no git installed).
if command -v curl >/dev/null 2>&1; then
    download() { curl -fsSL "$1" -o "$2"; }
elif command -v wget >/dev/null 2>&1; then
    download() { wget -qO "$2" "$1"; }
else
    echo "Error: neither curl nor wget was found. Install one of them and run this again." >&2
    exit 1
fi

# --- Kit code: clone the repo (or a tarball when git is missing). ---
# The skill comes from THIS checkout (below), so the skill and the
# kit can never differ.
echo ""
echo "Getting the kit code ($WSP_REF) into $KIT_DIR ..."
if [ -d "$KIT_DIR/.git" ]; then
    if ! git -C "$KIT_DIR" fetch -q origin 2>/dev/null \
        || ! git -C "$KIT_DIR" checkout -q "$WSP_REF" 2>/dev/null; then
        echo "Note: could not update the kit code to $WSP_REF (kept the existing copy)."
    fi
elif command -v git >/dev/null 2>&1; then
    if ! git clone --depth 1 --branch "$WSP_REF" -q "$REPO_GIT" "$KIT_DIR"; then
        echo "Error: could not download the kit code." >&2
        exit 1
    fi
else
    # Tarball: try the tag ref first, then the branch ref.
    TMP_TGZ="$(mktemp).tgz"
    trap 'rm -f "$TMP_TGZ"' EXIT
    if download "https://codeload.github.com/$REPO/tar.gz/refs/tags/$WSP_REF" "$TMP_TGZ" \
        || download "https://codeload.github.com/$REPO/tar.gz/refs/heads/$WSP_REF" "$TMP_TGZ"; then
        mkdir -p "$KIT_DIR"
        if ! tar xzf "$TMP_TGZ" --strip-components=1 -C "$KIT_DIR"; then
            echo "Error: could not unpack the kit code." >&2
            exit 1
        fi
    else
        echo "Error: could not download the kit code." >&2
        exit 1
    fi
    trap - EXIT
fi

# Print the resolved commit so the install is traceable.
if [ -d "$KIT_DIR/.git" ]; then
    echo "Installed kit commit: $(git -C "$KIT_DIR" rev-parse HEAD)"
else
    echo "Installed kit ref: $WSP_REF (tarball; commit not resolved)"
fi

# --- Skill: install SKILL.md from the SAME checkout as the kit. ---
mkdir -p "$TARGET_DIR"

# Back up an existing install instead of silently overwriting it.
if [ -f "$TARGET_DIR/$SKILL_FILE" ]; then
    cp "$TARGET_DIR/$SKILL_FILE" "$TARGET_DIR/$SKILL_FILE.bak"
    echo "Backed up your existing skill to: $TARGET_DIR/$SKILL_FILE.bak"
fi

# Sanity-check it before moving it in.
if [ ! -s "$KIT_DIR/$SKILL_FILE" ]; then
    echo "Error: $KIT_DIR/$SKILL_FILE is missing. Nothing was installed." >&2
    exit 1
fi

if ! grep -q "Weight-Space Programming" "$KIT_DIR/$SKILL_FILE"; then
    echo "Error: $KIT_DIR/$SKILL_FILE does not look like the skill. Nothing was installed." >&2
    exit 1
fi

cp "$KIT_DIR/$SKILL_FILE" "$TARGET_DIR/$SKILL_FILE"

echo ""
echo "Installed: $TARGET_DIR/$SKILL_FILE"

# --- `wsp` command: a small launcher in ~/.local/bin (no sudo). ---
mkdir -p "$BIN_DIR"
printf '#!/bin/sh\n# wsp-kit launcher — installed by install.sh.\nexec env PYTHONPATH="%s/kit${PYTHONPATH:+:$PYTHONPATH}" python3 -m wsp.cli "$@"\n' "$KIT_DIR" > "$BIN_DIR/wsp"
chmod +x "$BIN_DIR/wsp"
echo "Installed: $BIN_DIR/wsp"

case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *)
        echo ""
        echo "Note: $BIN_DIR is not on your PATH yet. Add it with:"
        echo "  export PATH=\"$BIN_DIR:\$PATH\""
        echo "Then run: wsp demo"
        ;;
esac

# --- Demo: run it automatically so you see a result right away. ---
if [ "$NO_DEMO" = 0 ]; then
    echo ""
    echo "Running the demo automatically (no model download, nothing leaves your machine) ..."
    if ! command -v python3 >/dev/null 2>&1; then
        echo "Demo skipped: python3 was not found. Install Python 3.9+, then run: wsp demo"
    elif ! python3 -c "import numpy" 2>/dev/null; then
        echo "Demo skipped: numpy was not found. Install it (pip install numpy), then run: wsp demo"
    elif ! "$BIN_DIR/wsp" demo --out-dir "$KIT_DIR/demo-out"; then
        echo "Demo did not pass — the kit is installed, but the test did not pass. See the output above."
    fi
fi

echo ""
echo "Next step: fix your own model. One command, 4 simple questions:"
echo "  wsp auto"
echo ""
echo "It asks which model, what goes wrong, what goes right, and what"
echo "should not change — then runs the pipeline by itself and shows"
echo "the result. The kit never runs on a model you did not point it at."
echo ""
echo "To uninstall, delete: $KIT_DIR, $BIN_DIR/wsp, and $TARGET_DIR"
