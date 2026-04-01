#!/usr/bin/env bash
# bench_compress.sh — APLZ / gzip / ZIP / Zstandard 圧縮ベンチマーク
#
# Usage:
#   ./bench_compress.sh              デフォルト 1GB テスト
#   ./bench_compress.sh 512          512MB テスト
#   ./bench_compress.sh 2048         2GB テスト
#   ./bench_compress.sh <file>       既存ファイルでテスト

set -euo pipefail

# ─── 設定 ──────────────────────────────────────────────────────────────────────
APLZ_DIR="${APLZ_DIR:-$(cd "$(dirname "$0")/../project/GPU_ZIP" 2>/dev/null && pwd || echo "$HOME/project/GPU_ZIP")}"
GPU_ZIP="${APLZ_DIR}/gpu_zip"
SHADER="${APLZ_DIR}/compression.metal"
TMPDIR_BENCH="/tmp/aplz_bench_$$"

# aplz が見つからない場合はインストール済みを使う
if [[ ! -x "$GPU_ZIP" ]]; then
    GPU_ZIP="$(which gpu_zip 2>/dev/null || echo "")"
    SHADER="$(dirname "$GPU_ZIP")/compression.metal"
fi
if [[ ! -x "$GPU_ZIP" ]]; then
    echo "Error: gpu_zip not found. Set APLZ_DIR or install APLZ." >&2
    exit 1
fi

# ─── クリーンアップ ────────────────────────────────────────────────────────────
cleanup() {
    rm -rf "$TMPDIR_BENCH"
}
trap cleanup EXIT INT TERM

mkdir -p "$TMPDIR_BENCH"

# purge 用に sudo 認証を事前取得 (失敗時はキャッシュフラッシュなしで続行)
CAN_PURGE=false
if command -v purge &>/dev/null; then
    echo "==> Requesting sudo for cache flush (purge)..."
    if sudo -v 2>/dev/null; then
        CAN_PURGE=true
        echo "    purge enabled"
    else
        echo "    sudo not available — running without cache flush"
    fi
fi

# ─── ツール検出 ────────────────────────────────────────────────────────────────
check_tool() {
    if command -v "$1" &>/dev/null; then
        echo "  $1: $(command -v "$1")"
        return 0
    else
        echo "  $1: not found (skipped)"
        return 1
    fi
}

HAS_GZIP=false; HAS_ZIP=false; HAS_ZSTD=false
echo "==> Tools:"
echo "  aplz: $GPU_ZIP"
check_tool gzip && HAS_GZIP=true
check_tool zip && HAS_ZIP=true
check_tool zstd && HAS_ZSTD=true
echo ""

# ─── テストデータ生成 ──────────────────────────────────────────────────────────
generate_test_data() {
    local size_mb="$1"
    local text_file="$TMPDIR_BENCH/test_text.bin"
    local rand_file="$TMPDIR_BENCH/test_rand.bin"

    echo "==> Generating test data (${size_mb} MB each)..."

    python3 -c "
import sys
block = b'The quick brown fox jumps over the lazy dog. ' * 1000
total = ${size_mb} * 1024 * 1024
written = 0
while written < total:
    chunk = block[:min(len(block), total - written)]
    sys.stdout.buffer.write(chunk)
    written += len(chunk)
" > "$text_file"

    dd if=/dev/urandom of="$rand_file" bs=1048576 count="$size_mb" 2>/dev/null

    echo "    Text: $(ls -lh "$text_file" | awk '{print $5}')"
    echo "    Random: $(ls -lh "$rand_file" | awk '{print $5}')"
    echo ""
}

# ─── ベンチマーク関数 ──────────────────────────────────────────────────────────
# 引数: tool_name, compress_cmd, decompress_cmd, compressed_file, decompressed_file
# 出力: time (秒), サイズ
flush_cache() {
    if $CAN_PURGE; then
        sudo purge 2>/dev/null || true
    fi
}

bench_compress() {
    local name="$1"
    local input="$2"
    local output="$3"
    shift 3
    local cmd=("$@")

    local input_size
    input_size=$(stat -f%z "$input" 2>/dev/null || stat -c%s "$input")

    flush_cache

    local t0 t1 elapsed
    t0=$(python3 -c "import time; print(time.monotonic())")
    "${cmd[@]}" >/dev/null 2>&1
    t1=$(python3 -c "import time; print(time.monotonic())")
    elapsed=$(python3 -c "print(f'{$t1 - $t0:.3f}')")

    local output_size
    if [[ -f "$output" ]]; then
        output_size=$(stat -f%z "$output" 2>/dev/null || stat -c%s "$output")
    else
        output_size=0
    fi

    local ratio speed_mb
    ratio=$(python3 -c "print(f'{100.0 * $output_size / $input_size:.1f}')")
    speed_mb=$(python3 -c "print(f'{$input_size / 1048576 / ($t1 - $t0):.1f}')")

    printf "  %-12s %8s  %5s%%  %8s MB/s  %6ss\n" \
        "$name" \
        "$(numfmt_size "$output_size")" \
        "$ratio" \
        "$speed_mb" \
        "$elapsed"
}

bench_decompress() {
    local name="$1"
    local input="$2"
    local original_size="$3"
    shift 3
    local cmd=("$@")

    flush_cache

    local t0 t1 elapsed
    t0=$(python3 -c "import time; print(time.monotonic())")
    "${cmd[@]}" >/dev/null 2>&1
    t1=$(python3 -c "import time; print(time.monotonic())")
    elapsed=$(python3 -c "print(f'{$t1 - $t0:.3f}')")

    local speed_mb
    speed_mb=$(python3 -c "print(f'{$original_size / 1048576 / ($t1 - $t0):.1f}')")

    printf "  %-12s %8s MB/s  %6ss\n" "$name" "$speed_mb" "$elapsed"
}

numfmt_size() {
    local bytes="$1"
    python3 -c "
b = $bytes
if b >= 1073741824: print(f'{b/1073741824:.2f} GB')
elif b >= 1048576: print(f'{b/1048576:.1f} MB')
elif b >= 1024: print(f'{b/1024:.1f} KB')
else: print(f'{b} B')
"
}

# ─── メイン ────────────────────────────────────────────────────────────────────
run_benchmark() {
    local input_file="$1"
    local label="$2"
    local input_size
    input_size=$(stat -f%z "$input_file" 2>/dev/null || stat -c%s "$input_file")
    local input_human
    input_human=$(numfmt_size "$input_size")

    echo "━━━ $label ($input_human) ━━━"
    echo ""

    # --- 圧縮 ---
    echo "  [圧縮]"
    printf "  %-12s %8s  %5s  %8s       %6s\n" "Tool" "Output" "Ratio" "Speed" "Time"
    printf "  %-12s %8s  %5s  %8s       %6s\n" "────" "──────" "─────" "─────" "────"

    bench_compress "APLZ" "$input_file" "$TMPDIR_BENCH/out.aplz" \
        "$GPU_ZIP" -c "$input_file" "$TMPDIR_BENCH/out.aplz" "$SHADER"

    if $HAS_GZIP; then
        cp "$input_file" "$TMPDIR_BENCH/out_gz.bin"
        bench_compress "gzip" "$input_file" "$TMPDIR_BENCH/out_gz.bin.gz" \
            gzip -f "$TMPDIR_BENCH/out_gz.bin"
    fi

    if $HAS_ZIP; then
        bench_compress "ZIP" "$input_file" "$TMPDIR_BENCH/out.zip" \
            zip -j -q "$TMPDIR_BENCH/out.zip" "$input_file"
    fi

    if $HAS_ZSTD; then
        bench_compress "Zstandard" "$input_file" "$TMPDIR_BENCH/out.zst" \
            zstd -f "$input_file" -o "$TMPDIR_BENCH/out.zst"
    fi

    echo ""

    # --- 解凍 ---
    echo "  [解凍]"
    printf "  %-12s %8s       %6s\n" "Tool" "Speed" "Time"
    printf "  %-12s %8s       %6s\n" "────" "─────" "────"

    bench_decompress "APLZ" "$TMPDIR_BENCH/out.aplz" "$input_size" \
        "$GPU_ZIP" -d "$TMPDIR_BENCH/out.aplz" "$TMPDIR_BENCH/dec_aplz.bin" "$SHADER"

    if $HAS_GZIP && [[ -f "$TMPDIR_BENCH/out_gz.bin.gz" ]]; then
        bench_decompress "gzip" "$TMPDIR_BENCH/out_gz.bin.gz" "$input_size" \
            gzip -d -k -f "$TMPDIR_BENCH/out_gz.bin.gz"
    fi

    if $HAS_ZIP && [[ -f "$TMPDIR_BENCH/out.zip" ]]; then
        bench_decompress "ZIP" "$TMPDIR_BENCH/out.zip" "$input_size" \
            unzip -o -q -d "$TMPDIR_BENCH" "$TMPDIR_BENCH/out.zip"
    fi

    if $HAS_ZSTD && [[ -f "$TMPDIR_BENCH/out.zst" ]]; then
        bench_decompress "Zstandard" "$TMPDIR_BENCH/out.zst" "$input_size" \
            zstd -d -f "$TMPDIR_BENCH/out.zst" -o "$TMPDIR_BENCH/dec_zst.bin"
    fi

    echo ""

    # クリーンアップ (次のテスト用)
    rm -f "$TMPDIR_BENCH"/out.* "$TMPDIR_BENCH"/out_gz.* "$TMPDIR_BENCH"/dec_*
}

# ─── エントリポイント ──────────────────────────────────────────────────────────
ARG="${1:-1024}"

if [[ -f "$ARG" ]]; then
    # 既存ファイルでベンチマーク
    echo "============================================================"
    echo " APLZ Compression Benchmark"
    echo " File: $ARG"
    echo "============================================================"
    echo ""
    run_benchmark "$ARG" "$(basename "$ARG")"
else
    # サイズ指定でテストデータ生成
    SIZE_MB="$ARG"
    echo "============================================================"
    echo " APLZ Compression Benchmark (${SIZE_MB} MB)"
    echo "============================================================"
    echo ""
    generate_test_data "$SIZE_MB"

    run_benchmark "$TMPDIR_BENCH/test_text.bin" "Text (repeating pattern)"
    run_benchmark "$TMPDIR_BENCH/test_rand.bin" "Random (incompressible)"
fi

echo "============================================================"
echo " Benchmark complete."
echo "============================================================"
