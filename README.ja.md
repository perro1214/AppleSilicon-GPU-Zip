# APLZ — Apple Parallel LZ

[English](README.md) | **日本語** | [中文](README.zh.md) | [Español](README.es.md) | [Français](README.fr.md) | [Deutsch](README.de.md)

Apple Silicon GPU で動作する高速ロスレス圧縮/解凍エンジン。Metal Compute Shader (MSL 3.0) + tANS (Asymmetric Numeral Systems) を組み合わせた GPU パイプラインにより、圧縮・解凍の **完全 Round-trip** を実現する。

## ハイライト

- **Zero-copy I/O**: Apple Silicon の Unified Memory Architecture (UMA) を活用。`mmap` + `newBufferWithBytesNoCopy` により CPU↔GPU 間のデータコピーを完全に排除
- **並列 LZ77**: 256 スレッドが 64 KB チャンクを協調処理。2-way ハッシュ (atomic_fetch_min) で高速マッチ探索
- **Per-chunk 256 Interleaved tANS**: チャンクごとに最適な tANS テーブルを動的構築。各スレッドが独立した ANS 状態を保持し、256 本のビットストリームを並列にエンコード/デコード
- **並列 LZ77 デコード**: 非重複マッチは 256 スレッドで並列コピー、重複マッチ (dist < len) はシリアルフォールバック。Barrier 削減最適化付き
- **非同期ダブルバッファリング**: GCD (`dispatch_semaphore` / `dispatch_group`) によるバッチパイプライン。GPU 実行と CPU I/O を完全にオーバーラップ
- **O(1) ストリーミング・メモリ**: 入力サイズに依存しない固定メモリ使用量。メガバッチ (32MB) 単位で処理し、5GB+ のファイルも安定動作
- **Perfect Round-trip**: 圧縮→解凍でバイトレベルの完全一致を保証

## アーキテクチャ

### 圧縮パイプライン (`-c`)

```
mmap(input)  <- zero-copy MTLBuffer (Apple Silicon UMA)
    |
    v
╔══════════════════════════════════════════════════════╗
║  Phase A: GPU Pass 1 x mega-batch (32MB streaming)   ║
║  ┌────────────────────────────────────────────────┐  ║
║  │ compress_chunk (512 chunks/mega-batch)         │  ║
║  │   A-1  Hash table init                         │  ║
║  │   A-2  3-gram hash (atomic_fetch_min x 2)      │  ║
║  │   A-3  LZ77 match search -> sparse[]           │  ║
║  │   B    Greedy overlap resolution -> compact[]  │  ║
║  │   1 threadgroup (256 threads) = 1 chunk (64KB) │  ║
║  └────────────────────────────────────────────────┘  ║
║  -> per-chunk histogram -> reuse buffers -> next 32MB║
╚══════════════════════════════════════════════════════╝
    | チャンクごとのヒストグラム (512 シンボル)
    v
+------------------------------------------------------+
|  Phase B: CPU per-chunk tANS テーブル構築              |
|  Per chunk: Normalize -> Duda's spread -> Encode table|
+------------------------------------------------------+
    | SymInfo[512] + enc_table[1024] per chunk
    v
╔══════════════════════════════════════════════════════╗
║  Phase C: Pass2 tANS encode (double-buffered batch)  ║
║  for each mega-batch (32MB):                         ║
║    ┌──────────────────────────────────────────────┐  ║
║    │ Pass2: tans_encode (double-buffered batch)   │  ║
║    │ 256 interleaved ANS streams per chunk        │  ║
║    │ dispatch_semaphore -> serial write queue     │  ║
║    └──────────────────────────────────────────────┘  ║
╚══════════════════════════════════════════════════════╝
    | 256 x bitstreams (async write overlap)
    v
  .aplz file
```

### 解凍パイプライン (`-d`)

```
read .aplz header + chunk_offsets (seek table)
    |
    v
╔══════════════════════════════════════════════════════╗
║  Mega-batch streaming pipeline (32MB buf_out)        ║
║  for each mega-batch:                                ║
║    ┌──────────────────────────────────────────────┐  ║
║    │ Double-buffered batch pipeline               │  ║
║    │                                              │  ║
║    │ CPU (batch N+1):                             │  ║
║    │   read per-chunk compact_freq + streams      │  ║
║    │   build DecodeEntry[L] per chunk             │  ║
║    │                                              │  ║
║    │ GPU (batch N):                               │  ║
║    │   tans_decode -> lz77_decode -> buf_out      │  ║
║    └──────────────────────────────────────────────┘  ║
║    -> fwrite(buf_out) -> reuse buffers               ║
╚══════════════════════════════════════════════════════╝
    v
  復元データ
```

## ファイルフォーマット (.aplz)

| Offset | Size | Content |
|---|---|---|
| 0 | 24 B | `FileHeader` (magic "APLZ", version=5, original_size, chunk_size, num_chunks) |
| 24 | 4 B | `n_streams` = 256 |
| 28 | 4 B | `ans_log_l` = 10 |
| 32 | 8xN B | `chunk_offsets[N]` (seek table) |
| ... | variable | Chunk data (N チャンク、v5 形式) |

Per-chunk データ構造 (v5 形式):
- `chunk_kind` (1 B): 0 = encoded (tANS), 1 = raw (フォールバック)

**encoded (chunk_kind = 0) の場合:**
- `chunk_n_streams` (2 B): このチャンクのアクティブな ANS ストリーム数
- `token_cnt` (4 B): dense token 数
- `compact_freq`: `n_nonzero` (2 B) + `[sym_id, freq]` pairs (4 B each) — 非ゼロシンボルのみ
- `stream_sizes[chunk_n_streams]` (2 B each): 各 ANS ストリームのバイトサイズ
- stream data: ANS ビットストリーム (v5 **compact distance coding** 使用)
  - 短距離 (≤255): 8-bit payload + 1-bit flag
  - 長距離 (>255): 16-bit payload + 1-bit flag

**raw (chunk_kind = 1) の場合:**
- `raw_size` (4 B): 非圧縮チャンクサイズ
- raw bytes: 元データ (エンコード出力が入力を超える場合のフォールバック)

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
# 右クリック ->「APLZ で圧縮」「APLZ で解凍」を追加
./setup_finder.sh

# 削除
./setup_finder.sh uninstall
```

Google Drive やクラウドストレージ上のファイルにも対応。サービス実行時は自動的に `/tmp` 経由で処理することで、macOS サンドボックス制限を回避する。

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

| データ | 入力 | 出力 | 圧縮率 | スループット |
|---|---|---|---|---|
| テキスト (繰り返しパターン) | 1 MB | 31 KB | 3.0% | 50 MB/s |
| テキスト (大規模) | 100 MB | 3.0 MB | 3.0% | 251 MB/s |
| ランダムバイナリ | 10 MB | 10.6 MB | 106% | 121 MB/s |

### 解凍

| データ | 速度 | 備考 |
|---|---|---|
| テキスト (1 MB) | 246 MB/s | 1 mega-batch、パイプラインオーバーヘッド最小 |
| テキスト (100 MB) | 1.4 GB/s | 4 mega-batches、フルパイプラインオーバーラップ |
| ランダム (10 MB) | 246 MB/s | 1 mega-batch、ほぼリテラル |

> ランダムデータは非圧縮のため、出力が入力をわずかに超える (tANS/distance フィールドのオーバーヘッド)。
> メガバッチストリーミングにより、入力サイズに関係なくメモリ使用量は一定 (~512 MB)。

## ソースファイル構成

| ファイル | 役割 |
|---|---|
| `APLZ.h` | 共有ヘッダ (CPU/GPU 共通の構造体・定数) |
| `compression.metal` | GPU カーネル (MSL 3.0): compress_chunk, tans_encode, tans_decode, lz77_decode |
| `main.mm` | ホストドライバ (Objective-C++): O(1) ストリーミング・メガバッチ + 非同期ダブルバッファパイプライン |
| `aplz` | ユーザー向け CLI ラッパー (Bash): ファイル/ディレクトリ対応、tar 統合 |
| `install.sh` | インストーラ: 最適化ビルド (-O3 -flto) + システムワイドデプロイ |
| `setup_finder.sh` | macOS Finder Quick Action 生成 (右クリック圧縮/解凍) |
| `build.sh` | 開発用ビルドスクリプト |

## 技術詳細

- **LZ77**: 3-gram hash, 2-way hash table (atomic_fetch_min), max match 255, min match 3, distance up to 65534
- **Per-chunk tANS (v3)**: 各 64KB チャンクごとに最適な tANS テーブルを動的構築。512 symbols (256 literals + 256 match lengths), L=1024, Duda's fast spread。Threadgroup cooperative loading で per-chunk テーブルを高速ロード
- **ANS Encoding**: Forward encode with renormalization, sentinel bit for bitstream end detection
- **ANS Decoding**: Reverse playback from final state, sentinel via `clz()`, O(1) decode table lookup
- **Compact Frequency Serialization**: Per-chunk 頻度テーブルを非ゼロエントリのみ保存 (`n_nonzero + [sym_id, freq]` pairs) でファイルサイズオーバーヘッドを最小化
- **LZ77 Parallel Decode**: Hybrid approach — serial fallback for token count > 4096, parallel cooperative copy with barrier reduction for dense match streams
- **Async Double Buffering**: GCD-based batch pipeline (32 chunks/batch, 2 slots). `dispatch_semaphore` for slot exclusion, `addCompletedHandler` + serial `dispatch_queue` for ordered async writes
- **O(1) Streaming Memory**: Mega-batch architecture (512 chunks = 32MB per batch). Buffers are reused across mega-batches. Fixed ~512 MB memory regardless of input size.

## 既知の制約

- **BS_CAP = 512 B**: 1 ストリームの最大出力は 512 バイト。高度に非圧縮なデータではビットが切り捨てられる可能性がある。
- **チャンクサイズ固定 64 KB**: 変更可能だが再ビルドが必要。

## License

MIT License. See [LICENSE](LICENSE) for details.
