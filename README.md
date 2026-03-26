# APLZ — Apple Parallel LZ

Apple Silicon GPU で動作する高速ロスレス圧縮/解凍エンジン。
Metal Compute Shader (MSL 3.0) + tANS (Asymmetric Numeral Systems) を組み合わせた GPU パイプラインにより、圧縮・解凍の **完全 Round-trip** を実現する。

## ハイライト

- **Zero-copy I/O**: Apple Silicon の Unified Memory Architecture (UMA) を活用。`mmap` + `newBufferWithBytesNoCopy` により CPU↔GPU 間のデータコピーを完全に排除
- **並列 LZ77**: 256 スレッドが 64 KB チャンクを協調処理。2-way ハッシュ (atomic_fetch_min) で高速マッチ探索
- **256 Interleaved tANS**: 各スレッドが独立した ANS 状態を保持し、256 本のビットストリームを並列にエンコード/デコード
- **並列 LZ77 デコード**: 非重複マッチは 256 スレッドで並列コピー、重複マッチ (dist < len) はシリアルフォールバック。Barrier 削減最適化付き
- **非同期ダブルバッファリング**: GCD (`dispatch_semaphore` / `dispatch_group`) によるバッチパイプライン。GPU 実行と CPU I/O を完全にオーバーラップ
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
|  GPU Pass 2: tans_encode (double-buffered batch) |
|  256 interleaved ANS streams                     |
|  + sentinel bit + SIMD prefix sum                |
|  dispatch_semaphore: slot exclusion               |
|  addCompletedHandler -> serial write queue        |
+--------------------------------------------------+
    | 256 x bitstreams (async write overlap)
    v
  .aplz file
```

### 解凍パイプライン (`-d`)

```
read .aplz header + tANS tables
    |
    v
+--------------------------------------------------+
|  CPU: Decode table construction                  |
|  SymInfo[512] -> spread table -> DecodeEntry[L]  |
+--------------------------------------------------+
    | dec_table[1024]
    v
+--------------------------------------------------+
|  Double-buffered batch pipeline                  |
|  CPU: read batch N+1  |  GPU: decode batch N     |
|  (dispatch_semaphore slot exclusion)             |
|                                                  |
|  Per batch:                                      |
|    GPU Pass 1: tans_decode                       |
|    256 streams: reverse bitstream playback       |
|    GPU Pass 2: lz77_decode                       |
|    Hybrid serial/parallel expansion              |
|    -> buf_out[chunk_offset] (direct write)       |
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

## インストール

macOS + Xcode Command Line Tools が必要。

```bash
# ~/.local/bin にインストール (-O3 -flto 最適化ビルド)
./install.sh

# /usr/local/bin にインストール (sudo 必要)
./install.sh /usr/local

# アンインストール
./install.sh uninstall
```

### Finder 統合 (オプション)

```bash
# 右クリック →「APLZ で圧縮」「APLZ で解凍」を追加
./setup_finder.sh

# 削除
./setup_finder.sh uninstall
```

## 使い方

### aplz コマンド (推奨)

```bash
# ファイル圧縮
aplz compress myfile.txt              → myfile.txt.aplz

# ディレクトリ圧縮 (自動 tar)
aplz compress myproject/              → myproject.tar.aplz

# 解凍
aplz extract myfile.txt.aplz          → myfile.txt
aplz extract myproject.tar.aplz       → myproject/

# ファイル情報
aplz info myfile.txt.aplz
```

### gpu_zip (低レベル API)

```bash
./gpu_zip -c <input> <output.aplz> compression.metal
./gpu_zip -d <input.aplz> <output> compression.metal
```

### 開発用ビルド

```bash
./build.sh        # -O2 ビルド
./build.sh clean   # 成果物削除
```

## パフォーマンス (Apple M4)

### 圧縮

| Data | Input | Output | Ratio | Throughput |
|---|---|---|---|---|
| Text (repeating pattern) | 1.14 MB | 36 KB | 3.0% | 36 MB/s |
| Random binary | 10 MB | 10.9 MB | 104% | 84 MB/s |
| **Text (large)** | **100 MB** | **2.9 MB** | **2.8%** | **157 MB/s** |

### 解凍

| Data | Speed | Notes |
|---|---|---|
| Text (1.14 MB) | 280 MB/s | 1 batch, pipeline overhead minimal |
| Random (10 MB) | 140 MB/s | 5 batches, mostly literals |
| **Text (100 MB)** | **4.4 GB/s** | **50 batches, full pipeline overlap** |

> Random data is incompressible so output slightly exceeds input (tANS/distance field overhead).
> Large file performance demonstrates the benefit of the async double-buffered pipeline.

## ソースファイル構成

| File | Role |
|---|---|
| `APLZ.h` | Shared header (structs & constants for CPU and GPU) |
| `compression.metal` | GPU kernels (MSL 3.0): compress_chunk, tans_encode, tans_decode, lz77_decode |
| `main.mm` | Host driver (Objective-C++): async double-buffered compress/decompress pipelines |
| `aplz` | User-facing CLI wrapper (Bash): file/directory support, tar integration |
| `install.sh` | Installer: optimized build (-O3 -flto) + system-wide deployment |
| `setup_finder.sh` | macOS Finder Quick Action generator (right-click compress/extract) |
| `build.sh` | Development build script |

## 技術詳細

- **LZ77**: 3-gram hash, 2-way hash table (atomic_fetch_min), max match 255, min match 3, distance up to 65534
- **tANS**: 512 symbols (256 literals + 256 match lengths), L=1024, Duda's fast spread
- **ANS Encoding**: Forward encode with renormalization, sentinel bit for bitstream end detection
- **ANS Decoding**: Reverse playback from final state, sentinel via `clz()`, O(1) decode table lookup
- **LZ77 Parallel Decode**: Hybrid approach -- serial fallback for token count > 4096, parallel cooperative copy with barrier reduction for dense match streams
- **Async Double Buffering**: GCD-based batch pipeline (32 chunks/batch, 2 slots). `dispatch_semaphore` for slot exclusion, `addCompletedHandler` + serial `dispatch_queue` for ordered async writes

## 既知の制約

- **BS_CAP = 512 B**: 1 stream max output is 512 bytes. Highly incompressible data may truncate bits.
- **Global tANS table**: Single table for entire file. Per-chunk adaptive tables could improve compression ratio.
- **Chunk size fixed at 64 KB**: Tunable but requires rebuild.

## License

MIT License. See [LICENSE](LICENSE) for details.
