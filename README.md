# APLZ — Apple Parallel LZ

**English** | [日本語](README.ja.md) | [中文](README.zh.md) | [Español](README.es.md) | [Français](README.fr.md) | [Deutsch](README.de.md)

A high-speed lossless compression/decompression engine running on Apple Silicon GPU. Combines Metal Compute Shaders (MSL 3.0) with tANS (Asymmetric Numeral Systems) in a fully GPU-accelerated pipeline, achieving **perfect round-trip** compression.

## Highlights

- **Zero-copy I/O**: Leverages Apple Silicon's Unified Memory Architecture (UMA). `mmap` + `newBufferWithBytesNoCopy` eliminates all CPU↔GPU data copies
- **Parallel LZ77**: 256 threads cooperatively process each 64 KB chunk. 2-way hash (atomic_fetch_min) for fast match search
- **Per-chunk 256 Interleaved tANS**: Dynamically builds optimal tANS tables per chunk. Each thread maintains an independent ANS state, encoding/decoding 256 bitstreams in parallel
- **Parallel LZ77 Decode**: Non-overlapping matches are copied in parallel by 256 threads; overlapping matches (dist < len) use serial fallback. Includes barrier reduction optimization
- **Async Double Buffering**: GCD-based batch pipeline (`dispatch_semaphore` / `dispatch_group`). Fully overlaps GPU execution with CPU I/O
- **O(1) Streaming Memory**: Fixed memory usage independent of input size. Processes in mega-batches (32 MB) for stable operation on 5 GB+ files
- **Perfect Round-trip**: Guarantees byte-level exact match after compress → decompress

## Architecture

### Compression Pipeline (`-c`)

```
mmap(input)  ← zero-copy MTLBuffer (Apple Silicon UMA)
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
    | per-chunk histogram (512 symbols each)
    v
+------------------------------------------------------+
|  Phase B: CPU per-chunk tANS table construction      |
|  Per chunk: Normalize → Duda's spread → Encode table |
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

### Decompression Pipeline (`-d`)

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
  restored original data
```

## File Format (.aplz)

| Offset | Size | Content |
|---|---|---|
| 0 | 24 B | `FileHeader` (magic "APLZ", version=5, original_size, chunk_size, num_chunks) |
| 24 | 4 B | `n_streams` = 256 |
| 28 | 4 B | `ans_log_l` = 10 |
| 32 | 8×N B | `chunk_offsets[N]` (seek table) |
| ... | variable | Chunk data (N chunks with v5 encoding) |

Per-chunk data layout (v5 encoding):
- `chunk_kind` (1 B): 0 = encoded (tANS), 1 = raw (fallback)

**If encoded (chunk_kind = 0):**
- `chunk_n_streams` (2 B): number of active ANS streams for this chunk
- `token_cnt` (4 B): number of dense tokens
- `compact_freq`: `n_nonzero` (2 B) + `[sym_id, freq]` pairs (4 B each) — non-zero symbols only
- `stream_sizes[chunk_n_streams]` (2 B each): byte size of each ANS stream
- stream data: concatenated ANS bitstreams with **v5 compact distance coding**
  - Short distance (≤255): 8-bit payload + 1-bit flag
  - Long distance (>255): 16-bit payload + 1-bit flag

**If raw (chunk_kind = 1):**
- `raw_size` (4 B): uncompressed chunk size
- raw bytes: original data (fallback when encoded output exceeds input)

## Installation

Requires macOS + Xcode Command Line Tools.

```bash
# Install to ~/.local/bin (optimized build with -O3 -flto)
./install.sh

# Install to /usr/local/bin (requires sudo)
./install.sh /usr/local

# Uninstall
./install.sh uninstall
```

### Finder Integration (Optional)

```bash
# Add right-click → "Compress with APLZ" / "Extract with APLZ"
./setup_finder.sh

# Remove
./setup_finder.sh uninstall
```

Supports Google Drive and cloud storage files. The service automatically processes files via `/tmp` to work around macOS sandbox restrictions.

## Usage

### aplz command (recommended)

```bash
# Compress a file
aplz compress myfile.txt              → myfile.txt.aplz

# Compress a directory (auto tar)
aplz compress myproject/              → myproject.tar.aplz

# Extract
aplz extract myfile.txt.aplz          → myfile.txt
aplz extract myproject.tar.aplz       → myproject/

# File info
aplz info myfile.txt.aplz
```

### gpu_zip (low-level API)

```bash
./gpu_zip -c <input> <output.aplz> compression.metal
./gpu_zip -d <input.aplz> <output> compression.metal
```

### Development Build

```bash
./build.sh        # -O2 build
./build.sh clean   # clean artifacts
```

## Performance (Apple M4)

### Compression

| Data | Input | Output | Ratio | Throughput |
|---|---|---|---|---|
| Text (repeating pattern) | 1 MB | 31 KB | 3.0% | 50 MB/s |
| Text (large) | 100 MB | 3.0 MB | 3.0% | 251 MB/s |
| Random binary | 10 MB | 10.6 MB | 106% | 121 MB/s |

### Decompression

| Data | Speed | Notes |
|---|---|---|
| Text (1 MB) | 246 MB/s | 1 mega-batch, minimal pipeline overhead |
| Text (100 MB) | 1.4 GB/s | 4 mega-batches, full pipeline overlap |
| Random (10 MB) | 246 MB/s | 1 mega-batch, mostly literals |

> Random data is incompressible so output slightly exceeds input (tANS/distance field overhead).
> Memory usage is constant (~512 MB) regardless of input size thanks to mega-batch streaming.

## Source Files

| File | Role |
|---|---|
| `APLZ.h` | Shared header (structs & constants for CPU and GPU) |
| `compression.metal` | GPU kernels (MSL 3.0): compress_chunk, tans_encode, tans_decode, lz77_decode |
| `main.mm` | Host driver (Objective-C++): O(1) streaming mega-batch + async double-buffered pipelines |
| `aplz` | User-facing CLI wrapper (Bash): file/directory support, tar integration |
| `install.sh` | Installer: optimized build (-O3 -flto) + system-wide deployment |
| `setup_finder.sh` | macOS Finder Quick Action generator (right-click compress/extract) |
| `build.sh` | Development build script |

## Technical Details

- **LZ77**: 3-gram hash, 2-way hash table (atomic_fetch_min), max match 255, min match 3, distance up to 65534
- **Per-chunk tANS (v3)**: Dynamically builds optimal tANS tables per 64 KB chunk. 512 symbols (256 literals + 256 match lengths), L=1024, Duda's fast spread. Threadgroup cooperative loading for fast per-chunk table access on GPU
- **ANS Encoding**: Forward encode with renormalization, sentinel bit for bitstream end detection
- **ANS Decoding**: Reverse playback from final state, sentinel via `clz()`, O(1) decode table lookup
- **Compact Frequency Serialization**: Stores only non-zero entries per chunk (`n_nonzero + [sym_id, freq]` pairs), minimizing file size overhead
- **LZ77 Parallel Decode**: Hybrid approach — serial fallback for token count > 4096, parallel cooperative copy with barrier reduction for dense match streams
- **Async Double Buffering**: GCD-based batch pipeline (32 chunks/batch, 2 slots). `dispatch_semaphore` for slot exclusion, `addCompletedHandler` + serial `dispatch_queue` for ordered async writes
- **O(1) Streaming Memory**: Mega-batch architecture (512 chunks = 32MB per batch). Buffers are reused across mega-batches. Fixed ~512 MB memory regardless of input size.

## Known Limitations

- **BS_CAP = 512 B**: 1 stream max output is 512 bytes. Highly incompressible data may truncate bits.
- **Chunk size fixed at 64 KB**: Tunable but requires rebuild.

## License

MIT License. See [LICENSE](LICENSE) for details.
