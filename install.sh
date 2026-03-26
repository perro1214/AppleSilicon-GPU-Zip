#!/usr/bin/env bash
# install.sh — APLZ インストーラ
#
# Usage:
#   ./install.sh              ~/.local/bin にインストール (デフォルト)
#   ./install.sh /usr/local   /usr/local/bin にインストール (sudo 必要)
#   ./install.sh uninstall    アンインストール
#
# 実行内容:
#   1. -O3 -flto による最適化ビルド
#   2. gpu_zip, aplz, compression.metal を PREFIX/bin にインストール
#   3. PATH 設定の案内

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

SRC="${SCRIPT_DIR}/main.mm"
SHADER="${SCRIPT_DIR}/compression.metal"
WRAPPER="${SCRIPT_DIR}/aplz"

DEFAULT_PREFIX="$HOME/.local"

# ─── アンインストール ────────────────────────────────────────────────────────
if [[ "${1:-}" == "uninstall" ]]; then
    echo "==> Uninstalling APLZ..."
    for prefix in "$HOME/.local" "/usr/local"; do
        local_bin="${prefix}/bin"
        for f in gpu_zip aplz compression.metal; do
            if [[ -f "${local_bin}/${f}" ]]; then
                echo "    Removing: ${local_bin}/${f}"
                rm -f "${local_bin}/${f}" 2>/dev/null || sudo rm -f "${local_bin}/${f}"
            fi
        done
    done
    echo "Done. Run './setup_finder.sh uninstall' to remove Finder actions."
    exit 0
fi

PREFIX="${1:-$DEFAULT_PREFIX}"
BIN_DIR="${PREFIX}/bin"

echo "============================================================"
echo " APLZ Installer"
echo " Install to: ${BIN_DIR}"
echo "============================================================"
echo ""

# ─── 依存チェック ────────────────────────────────────────────────────────────
if ! command -v xcrun &>/dev/null; then
    echo "Error: Xcode Command Line Tools not found." >&2
    echo "Install with: xcode-select --install" >&2
    exit 1
fi

# ─── Metal シェーダコンパイル ────────────────────────────────────────────────
echo "==> Compiling Metal shader (optimized)..."
xcrun -sdk macosx metal \
    -std=metal3.0 \
    -O2 \
    -Wall \
    -Wno-unused-variable \
    -c "$SHADER" \
    -o "${SCRIPT_DIR}/compression.air"

xcrun -sdk macosx metallib \
    "${SCRIPT_DIR}/compression.air" \
    -o "${SCRIPT_DIR}/compression.metallib"

echo "    -> compression.metallib"

# ─── ホストコード最適化ビルド ────────────────────────────────────────────────
SDK=$(xcrun --sdk macosx --show-sdk-path)
echo "==> Building gpu_zip with -O3 -flto (SDK: ${SDK})..."

clang++ \
    -std=c++17 \
    -O3 \
    -flto \
    -DNDEBUG \
    -Wall \
    -Wextra \
    -Wno-unused-parameter \
    -isysroot "$SDK" \
    -framework Metal \
    -framework Foundation \
    -framework QuartzCore \
    -o "${SCRIPT_DIR}/gpu_zip" \
    "$SRC"

echo "    -> gpu_zip (optimized)"

# ─── インストール ────────────────────────────────────────────────────────────
echo "==> Installing to ${BIN_DIR}..."
mkdir -p "$BIN_DIR"

NEED_SUDO=false
if [[ ! -w "$BIN_DIR" ]]; then
    NEED_SUDO=true
    echo "    (requires sudo for ${BIN_DIR})"
fi

do_install() {
    local src="$1" dst="$2" mode="${3:-755}"
    if $NEED_SUDO; then
        sudo install -m "$mode" "$src" "$dst"
    else
        install -m "$mode" "$src" "$dst"
    fi
}

do_install "${SCRIPT_DIR}/gpu_zip" "${BIN_DIR}/gpu_zip" 755
do_install "$WRAPPER"              "${BIN_DIR}/aplz"    755
do_install "$SHADER"               "${BIN_DIR}/compression.metal" 644

echo "    gpu_zip           -> ${BIN_DIR}/gpu_zip"
echo "    aplz              -> ${BIN_DIR}/aplz"
echo "    compression.metal -> ${BIN_DIR}/compression.metal"

# ─── PATH チェック ───────────────────────────────────────────────────────────
echo ""
if echo "$PATH" | tr ':' '\n' | grep -qx "$BIN_DIR"; then
    echo "==> ${BIN_DIR} is already in PATH."
else
    echo "==> Add to PATH (add to ~/.zshrc):"
    echo ""
    echo "    export PATH=\"${BIN_DIR}:\$PATH\""
    echo ""
fi

# ─── 完了 ────────────────────────────────────────────────────────────────────
echo "============================================================"
echo " Installation complete!"
echo ""
echo " Quick start:"
echo "   aplz compress myfile.txt          → myfile.txt.aplz"
echo "   aplz extract  myfile.txt.aplz     → myfile.txt"
echo "   aplz compress mydir/              → mydir.tar.aplz"
echo "   aplz extract  mydir.tar.aplz      → mydir/"
echo "   aplz info     myfile.txt.aplz"
echo ""
echo " Finder integration (optional):"
echo "   ./setup_finder.sh"
echo "============================================================"
