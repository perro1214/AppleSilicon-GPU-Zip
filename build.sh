#!/usr/bin/env bash
# build.sh — GPU_ZIP Phase 1 & 2 build script
# Usage: ./build.sh [clean]
set -euo pipefail

TARGET="gpu_zip"
SRC="main.mm"
SHADER="compression.metal"
AIR="${SHADER%.metal}.air"
METALLIB="${SHADER%.metal}.metallib"

# ── Optional clean ────────────────────────────────────────────────────────────
if [[ "${1:-}" == "clean" ]]; then
    echo "==> Cleaning build artifacts"
    rm -f "${TARGET}" "${AIR}" "${METALLIB}"
    exit 0
fi

# ── Metal shader: compile (.metal → .air → .metallib) ────────────────────────
# .metallib is pre-compiled for fastest load time at runtime,
# but main.mm also supports source-level compilation as fallback.
echo "==> Compiling Metal shader: ${SHADER}"
xcrun -sdk macosx metal \
    -std=metal3.0 \
    -O2 \
    -Wall \
    -Wno-unused-variable \
    -c "${SHADER}" \
    -o "${AIR}"

xcrun -sdk macosx metallib \
    "${AIR}" \
    -o "${METALLIB}"

echo "    -> ${METALLIB}"

# ── Host code: Objective-C++ ──────────────────────────────────────────────────
SDK=$(xcrun --sdk macosx --show-sdk-path)
echo "==> Compiling host: ${SRC}  (SDK: ${SDK})"
clang++ \
    -std=c++17 \
    -O2 \
    -Wall \
    -Wextra \
    -Wno-unused-parameter \
    -isysroot "${SDK}" \
    -framework Metal \
    -framework Foundation \
    -framework QuartzCore \
    -o "${TARGET}" \
    "${SRC}"

echo "    -> ${TARGET}"
echo ""

# ── Quick smoke test instructions ─────────────────────────────────────────────
echo "============================================================"
echo " Build complete!  Run with:"
echo ""
echo "   # Generate a random test file (10 MB)"
echo "   dd if=/dev/urandom of=test_random.bin bs=1M count=10 2>/dev/null"
echo ""
echo "   # Compressible data (repeating pattern)"
echo "   python3 -c \\"
echo "     \"import sys; sys.stdout.buffer.write((b'hello world ' * 100000)[:10*1024*1024])\" \\"
echo "     > test_text.bin"
echo ""
echo "   # Run"
echo "   ./${TARGET} test_random.bin out.bin ${SHADER}"
echo "   ./${TARGET} test_text.bin  out.bin ${SHADER}"
echo "============================================================"
