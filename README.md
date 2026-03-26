# APLZ — Apple Parallel LZ

Apple Silicon GPU で動作する高速ロスレス圧縮/解凍エンジン。
Metal Compute Shader (MSL 3.0) + tANS (Asymmetric Numeral Systems) を組み合わせた GPU パイプラインにより、圧縮・解凍の **完全 Round-trip** を実現する。

## ハイライト

- **Zero-copy I/O**: Apple Silicon の Unified Memory Architecture (UMA) を活用。`mmap` + `newBufferWithBytesNoCopy` により CPU↔GPU 間のデータコピーを完全に排除
- **並列 LZ77**: 256 スレッドが 64 KB チャンクを協調処理。2-way ハッシュ (atomic_fetch_min) で高速マッチ探索
- **256 Interleaved tANS**: 各スレッドが独立した ANS 状態を保持し、256 本のビットストリームを並列にエンコード/デコード
- **並列 LZ77 デコード**: 非重複マッチは 256 スレッドで並列コピー、重複マッチ (dist < len) はシリアルフォールバック。Barrier 削減最適化付き
- **Perfect Round-trip**: 圧縮→解凍でバイトレベルの完全一致を保証

## アーキテクチャ

### 圧縮パイプライン (`-c`)

```
mmap(input)
    |  zero-copy MTLBuffer (Apple Silicon UMA)
    v
+--------------------------------------------------+
|  GPU Pass 1: compress_chunk                      |
|  Phase A-1  Hash table init                      |
|  Phase A-2  3-gram hash (atomic_fetch_min x 2)   |
|  Phase A-3  LZ77 match search -> sparse[]        |
|  Phase B    Greedy overlap resolution -> compact[]|
|  1 threadgroup (256 threads) = 1 chunk (64 KB)   |
+--------------------------------------------------+
    | compact tokens + count
    v
+--------------------------------------------------+
|  CPU: tANS table construction                    |
|  1. Histogram (512 symbols)                      |
|  2. Normalize (sum = L = 1024)                   |
|  3. Duda's spread table (step = 643)             |
|  4. Encode table                                 |
+--------------------------------------------------+
    | SymInfo[512] + enc_table[1024]
    v
+--------------------------------------------------+
|  GPU Pass 2: tans_encode                         |
|  256 interleaved ANS streams                     |
|  + sentinel bit + SIMD prefix sum                |
+--------------------------------------------------+
    | 256 x bitstreams
    v
  .aplz file
```

### 解凍パイプライン (`-d`)

```
read .aplz
    |
    v
+--------------------------------------------------+
|  CPU: Decode table construction                  |
|  SymInfo[512] -> spread table -> DecodeEntry[L]  |
+--------------------------------------------------+
    | dec_table[1024]
    v
+--------------------------------------------------+
|  GPU Pass 1: tans_decode                         |
|  256 streams: reverse bitstream playback         |
|  sentinel detection via clz()                    |
|  -> interleaved token array                      |
+--------------------------------------------------+
    | LzToken[]
    v
+--------------------------------------------------+
|  GPU Pass 2: lz77_decode                         |
|  Hybrid serial/parallel expansion                |
|  - Literals: Phase 1 (all threads)               |
|  - Matches: Phase 2 (cooperative copy)           |
|  Barrier reduction via max_match_written tracking|
+--------------------------------------------------+
    v
  restored original data
```

## ファイルフォーマット (.aplz)

| Offset | Size | Content |
|---|---|---|
| 0 | 24 B | `FileHeader` (magic "APLZ", version, original_size, ...) |
| 24 | 4 B | `n_streams` = 256 |
| 28 | 4 B | `ans_log_l` = 10 |
| 32 | 2048 B | `SymInfo[512]` (freq + cum_freq) |
| 2080 | 8xN B | `chunk_offsets[N]` (seek table) |
| ... | variable | Chunk data (per chunk: token_cnt + stream_sizes[256] + streams) |

## ビルド

macOS + Xcode Command Line Tools が必要。

```bash
./build.sh
```

ビルド成果物:
- `gpu_zip` -- 実行バイナリ
- `compression.metallib` -- プリコンパイル済みシェーダ (ランタイムコンパイルも可)

クリーン:
```bash
./build.sh clean
```

## 使い方

```bash
# 圧縮
./gpu_zip -c <input> <output.aplz> compression.metal

# 解凍
./gpu_zip -d <input.aplz> <output> compression.metal
```

例:

```bash
# テスト用データ生成
python3 -c "import sys; sys.stdout.buffer.write((b'hello world ' * 100000)[:1200000])" > test_text.bin
dd if=/dev/urandom of=test_random.bin bs=1M count=10 2>/dev/null

# 圧縮 + 解凍 + 検証
./gpu_zip -c test_text.bin out.aplz compression.metal
./gpu_zip -d out.aplz decoded.bin compression.metal
cmp test_text.bin decoded.bin  # Perfect Match
```

## パフォーマンス (Apple M4)

### 圧縮

| Data | Input | Output | Ratio | Throughput |
|---|---|---|---|---|
| Text (repeating pattern) | 1.14 MB | 36 KB | 3.0% | 46 MB/s |
| Random binary | 10 MB | 10.9 MB | 104% | 78 MB/s |

### 解凍

| Data | Speed | Notes |
|---|---|---|
| Text (1.14 MB) | 27.9 MB/s | Parallel LZ77 decode (256 threads) |
| Random (10 MB) | 329 MB/s | Mostly literals, minimal match processing |

> Random data is incompressible so output slightly exceeds input (tANS/distance field overhead).

## ソースファイル構成

| File | Role |
|---|---|
| `APLZ.h` | Shared header (structs & constants for CPU and GPU) |
| `compression.metal` | GPU kernels (MSL 3.0): compress_chunk, tans_encode, tans_decode, lz77_decode |
| `main.mm` | Host driver (Objective-C++): compress/decompress pipelines |
| `build.sh` | Build script |

## 技術詳細

- **LZ77**: 3-gram hash, 2-way hash table (atomic_fetch_min), max match 255, min match 3, distance up to 65534
- **tANS**: 512 symbols (256 literals + 256 match lengths), L=1024, Duda's fast spread
- **ANS Encoding**: Forward encode with renormalization, sentinel bit for bitstream end detection
- **ANS Decoding**: Reverse playback from final state, sentinel via `clz()`, O(1) decode table lookup
- **LZ77 Parallel Decode**: Hybrid approach -- serial fallback for token count > 4096, parallel cooperative copy with barrier reduction for dense match streams

## 既知の制約

- **BS_CAP = 512 B**: 1 stream max output is 512 bytes. Highly incompressible data may truncate bits.
- **Global tANS table**: Single table for entire file. Per-chunk adaptive tables could improve compression ratio.
- **Chunk size fixed at 64 KB**: Tunable but requires rebuild.

## License

MIT License. See [LICENSE](LICENSE) for details.
