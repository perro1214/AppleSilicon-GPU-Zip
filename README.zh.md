# APLZ — Apple Parallel LZ

[English](README.md) | [日本語](README.ja.md) | **中文** | [Español](README.es.md) | [Français](README.fr.md) | [Deutsch](README.de.md)

基于 Apple Silicon GPU 的高速无损压缩/解压引擎。结合 Metal Compute Shader (MSL 3.0) 与 tANS (Asymmetric Numeral Systems) 的全 GPU 加速流水线，实现**完美的往返压缩 (Round-trip)**。

## 亮点

- **Zero-copy I/O**: 充分利用 Apple Silicon 的统一内存架构 (UMA)。`mmap` + `newBufferWithBytesNoCopy` 完全消除 CPU↔GPU 数据拷贝
- **并行 LZ77**: 256 个线程协同处理每个 64 KB 块。2-way 哈希 (atomic_fetch_min) 实现快速匹配搜索
- **Per-chunk 256 交织 tANS**: 为每个块动态构建最优 tANS 表。每个线程维护独立的 ANS 状态，并行编码/解码 256 个比特流
- **并行 LZ77 解码**: 非重叠匹配由 256 个线程并行复制；重叠匹配 (dist < len) 使用串行回退。含 Barrier 削减优化
- **异步双缓冲**: 基于 GCD (`dispatch_semaphore` / `dispatch_group`) 的批处理流水线。GPU 执行与 CPU I/O 完全重叠
- **O(1) 流式内存**: 固定内存用量，与输入大小无关。以 mega-batch (32 MB) 为单位处理，5 GB+ 文件稳定运行
- **完美往返**: 压缩→解压后保证字节级完全一致

## 架构

### 压缩流水线 (`-c`)

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
    | 每块直方图 (512 符号)
    v
+------------------------------------------------------+
|  Phase B: CPU per-chunk tANS 表构建                   |
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

### 解压流水线 (`-d`)

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
  恢复原始数据
```

## 文件格式 (.aplz)

| 偏移 | 大小 | 内容 |
|---|---|---|
| 0 | 24 B | `FileHeader` (magic "APLZ", version=5, original_size, chunk_size, num_chunks) |
| 24 | 4 B | `n_streams` = 256 |
| 28 | 4 B | `ans_log_l` = 10 |
| 32 | 8xN B | `chunk_offsets[N]` (seek table) |
| ... | 可变 | Chunk data (N 个块，v5 编码) |

Per-chunk 数据布局 (v5 编码):
- `chunk_kind` (1 B): 0 = encoded (tANS), 1 = raw (回退)

**编码块 (chunk_kind = 0) 的情况:**
- `chunk_n_streams` (2 B): 此块的活跃 ANS 流数
- `token_cnt` (4 B): dense token 数量
- `compact_freq`: `n_nonzero` (2 B) + `[sym_id, freq]` 对 (各 4 B) — 仅非零符号
- `stream_sizes[chunk_n_streams]` (各 2 B): 每个 ANS 流的字节大小
- stream data: ANS 比特流 (采用 v5 **紧凑距离编码**)
  - 短距离 (≤255): 8-bit payload + 1-bit flag
  - 长距离 (>255): 16-bit payload + 1-bit flag

**原始块 (chunk_kind = 1) 的情况:**
- `raw_size` (4 B): 未压缩块大小
- raw bytes: 原始数据 (当编码输出超过输入时的回退)

## 安装

需要 macOS + Xcode Command Line Tools。

```bash
# 安装到 ~/.local/bin (使用 -O3 -flto 优化构建)
./install.sh

# 安装到 /usr/local/bin (需要 sudo)
./install.sh /usr/local

# 卸载
./install.sh uninstall
```

### Finder 集成 (可选)

```bash
# 添加右键菜单 -> "Compress with APLZ" / "Extract with APLZ"
./setup_finder.sh

# 移除
./setup_finder.sh uninstall
```

支持 Google Drive 和云存储文件。服务运行时自动通过 `/tmp` 处理，绕过 macOS 沙盒限制。

## 使用方法

### aplz 命令 (推荐)

```bash
# 压缩文件
aplz compress myfile.txt              → myfile.txt.aplz

# 压缩目录 (自动 tar)
aplz compress myproject/              → myproject.tar.aplz

# 解压
aplz extract myfile.txt.aplz          → myfile.txt
aplz extract myproject.tar.aplz       → myproject/

# 文件信息
aplz info myfile.txt.aplz
```

### gpu_zip (底层 API)

```bash
./gpu_zip -c <input> <output.aplz> compression.metal
./gpu_zip -d <input.aplz> <output> compression.metal
```

### 开发构建

```bash
./build.sh        # -O2 构建
./build.sh clean   # 清理产物
```

## 性能 (Apple M4)

### 压缩

| 数据 | 输入 | 输出 | 压缩率 | 吞吐量 |
|---|---|---|---|---|
| 文本 (重复模式) | 1 MB | 31 KB | 3.0% | 50 MB/s |
| 文本 (大规模) | 100 MB | 3.0 MB | 3.0% | 251 MB/s |
| 随机二进制 | 10 MB | 10.6 MB | 106% | 121 MB/s |

### 解压

| 数据 | 速度 | 备注 |
|---|---|---|
| 文本 (1 MB) | 246 MB/s | 1 mega-batch，流水线开销最小 |
| 文本 (100 MB) | 1.4 GB/s | 4 mega-batches，完整流水线重叠 |
| 随机 (10 MB) | 246 MB/s | 1 mega-batch，大部分为字面量 |

> 随机数据不可压缩，因此输出略大于输入 (tANS/distance 字段开销)。
> 得益于 mega-batch 流式处理，内存用量恒定 (~512 MB)，与输入大小无关。

## 源文件结构

| 文件 | 作用 |
|---|---|
| `APLZ.h` | 共享头文件 (CPU/GPU 通用结构体与常量) |
| `compression.metal` | GPU 内核 (MSL 3.0): compress_chunk, tans_encode, tans_decode, lz77_decode |
| `main.mm` | 主机驱动 (Objective-C++): O(1) 流式 mega-batch + 异步双缓冲流水线 |
| `aplz` | 用户 CLI 封装 (Bash): 文件/目录支持, tar 集成 |
| `install.sh` | 安装器: 优化构建 (-O3 -flto) + 系统级部署 |
| `setup_finder.sh` | macOS Finder Quick Action 生成器 (右键压缩/解压) |
| `build.sh` | 开发构建脚本 |

## 技术细节

- **LZ77**: 3-gram hash, 2-way hash table (atomic_fetch_min), max match 255, min match 3, distance up to 65534
- **Per-chunk tANS (v3)**: 为每个 64KB 块动态构建最优 tANS 表。512 symbols (256 literals + 256 match lengths), L=1024, Duda's fast spread。Threadgroup cooperative loading 实现 GPU 上 per-chunk 表的快速加载
- **ANS Encoding**: Forward encode with renormalization, sentinel bit for bitstream end detection
- **ANS Decoding**: Reverse playback from final state, sentinel via `clz()`, O(1) decode table lookup
- **Compact Frequency Serialization**: 每块频率表仅存储非零条目 (`n_nonzero + [sym_id, freq]` pairs)，最小化文件大小开销
- **LZ77 Parallel Decode**: Hybrid approach — serial fallback for token count > 4096, parallel cooperative copy with barrier reduction for dense match streams
- **Async Double Buffering**: GCD-based batch pipeline (32 chunks/batch, 2 slots). `dispatch_semaphore` for slot exclusion, `addCompletedHandler` + serial `dispatch_queue` for ordered async writes
- **O(1) Streaming Memory**: Mega-batch architecture (512 chunks = 32MB per batch). Buffers are reused across mega-batches. Fixed ~512 MB memory regardless of input size.

## 已知限制

- **BS_CAP = 512 B**: 单个流最大输出 512 字节。高度不可压缩的数据可能会截断比特。
- **块大小固定为 64 KB**: 可调整但需要重新构建。

## License

MIT License. See [LICENSE](LICENSE) for details.
