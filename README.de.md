# APLZ — Apple Parallel LZ

[English](README.md) | [日本語](README.ja.md) | [中文](README.zh.md) | [Español](README.es.md) | [Français](README.fr.md) | **Deutsch**

Hochgeschwindigkeits-Engine fuer verlustfreie Kompression/Dekompression auf Apple Silicon GPU. Kombiniert Metal Compute Shaders (MSL 3.0) mit tANS (Asymmetric Numeral Systems) in einer vollstaendig GPU-beschleunigten Pipeline und erreicht **perfekte Round-trip-Kompression**.

## Highlights

- **Zero-copy I/O**: Nutzt die Unified Memory Architecture (UMA) von Apple Silicon. `mmap` + `newBufferWithBytesNoCopy` eliminiert alle CPU↔GPU-Datenkopien
- **Paralleles LZ77**: 256 Threads verarbeiten kooperativ jeden 64-KB-Block. 2-Wege-Hash (atomic_fetch_min) fuer schnelle Uebereinstimmungssuche
- **Per-chunk 256 verschraenktes tANS**: Baut dynamisch optimale tANS-Tabellen pro Block auf. Jeder Thread haelt einen unabhaengigen ANS-Zustand und kodiert/dekodiert 256 Bitstreams parallel
- **Parallele LZ77-Dekodierung**: Nicht ueberlappende Uebereinstimmungen werden parallel von 256 Threads kopiert; ueberlappende Uebereinstimmungen (dist < len) nutzen seriellen Rueckfall. Mit Barrier-Reduktionsoptimierung
- **Asynchrones Doppelpuffern**: GCD-basierte Batch-Pipeline (`dispatch_semaphore` / `dispatch_group`). Ueberlappt GPU-Ausfuehrung vollstaendig mit CPU-E/A
- **O(1) Streaming-Speicher**: Fester Speicherverbrauch unabhaengig von der Eingabegroesse. Verarbeitet in Mega-Batches (32 MB) fuer stabilen Betrieb bei 5 GB+ Dateien
- **Perfekter Round-trip**: Garantiert Byte-genaue Uebereinstimmung nach Kompression → Dekompression

## Architektur

### Kompressionspipeline (`-c`)

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
    | Histogramm pro Block (512 Symbole)
    v
+------------------------------------------------------+
|  Phase B: CPU per-chunk tANS table construction      |
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

### Dekompressionspipeline (`-d`)

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
  wiederhergestellte Originaldaten
```

## Dateiformat (.aplz)

| Offset | Groesse | Inhalt |
|---|---|---|
| 0 | 24 B | `FileHeader` (magic "APLZ", version=3, original_size, chunk_size, num_chunks) |
| 24 | 4 B | `n_streams` = 256 |
| 28 | 4 B | `ans_log_l` = 10 |
| 32 | 8xN B | `chunk_offsets[N]` (seek table) |
| ... | variabel | Chunk data (per chunk: token_cnt + compact_freq + stream_sizes[256] + streams) |

Per-chunk Datenstruktur:

- `token_cnt` (4 B): Anzahl der dichten Tokens
- `compact_freq`: `n_nonzero` (2 B) + `[sym_id, freq]` Paare (je 4 B) — nur Nicht-Null-Symbole
- `stream_sizes[256]` (512 B): Bytegroesse jedes ANS-Streams
- stream data: 256 verkettete Bitstreams

## Installation

Erfordert macOS + Xcode Command Line Tools.

```bash
# In ~/.local/bin installieren (optimierter Build mit -O3 -flto)
./install.sh

# In /usr/local/bin installieren (erfordert sudo)
./install.sh /usr/local

# Deinstallieren
./install.sh uninstall
```

### Finder-Integration (optional)

```bash
# Rechtsklick -> "Compress with APLZ" / "Extract with APLZ" hinzufuegen
./setup_finder.sh

# Entfernen
./setup_finder.sh uninstall
```

Unterstuetzt Google Drive und Cloud-Speicher-Dateien. Der Dienst verarbeitet Dateien automatisch ueber `/tmp`, um die macOS-Sandbox-Einschraenkungen zu umgehen.

## Verwendung

### aplz-Befehl (empfohlen)

```bash
# Datei komprimieren
aplz compress myfile.txt              → myfile.txt.aplz

# Verzeichnis komprimieren (automatisches tar)
aplz compress myproject/              → myproject.tar.aplz

# Extrahieren
aplz extract myfile.txt.aplz          → myfile.txt
aplz extract myproject.tar.aplz       → myproject/

# Dateiinformationen
aplz info myfile.txt.aplz
```

### gpu_zip (Low-Level-API)

```bash
./gpu_zip -c <input> <output.aplz> compression.metal
./gpu_zip -d <input.aplz> <output> compression.metal
```

### Entwicklungs-Build

```bash
./build.sh        # -O2 Build
./build.sh clean   # Artefakte bereinigen
```

## Leistung (Apple M4)

### Kompression

| Daten | Eingabe | Ausgabe | Ratio | Durchsatz |
|---|---|---|---|---|
| Text (wiederholendes Muster) | 1 MB | 31 KB | 3.0% | 50 MB/s |
| Text (gross) | 100 MB | 3.0 MB | 3.0% | 251 MB/s |
| Zufaellige Binaerdaten | 10 MB | 10.6 MB | 106% | 121 MB/s |

### Dekompression

| Daten | Geschwindigkeit | Anmerkungen |
|---|---|---|
| Text (1 MB) | 246 MB/s | 1 Mega-Batch, minimaler Pipeline-Overhead |
| Text (100 MB) | 1.4 GB/s | 4 Mega-Batches, vollstaendige Pipeline-Ueberlappung |
| Zufaellig (10 MB) | 246 MB/s | 1 Mega-Batch, ueberwiegend Literale |

> Zufaellige Daten sind inkompressibel, daher uebersteigt die Ausgabe leicht die Eingabe (tANS/distance-Feld-Overhead).
> Der Speicherverbrauch ist dank Mega-Batch-Streaming konstant (~512 MB) unabhaengig von der Eingabegroesse.

## Quelldateien

| Datei | Rolle |
|---|---|
| `APLZ.h` | Gemeinsamer Header (Structs und Konstanten fuer CPU und GPU) |
| `compression.metal` | GPU-Kernel (MSL 3.0): compress_chunk, tans_encode, tans_decode, lz77_decode |
| `main.mm` | Host-Treiber (Objective-C++): O(1)-Streaming-Mega-Batch + asynchrone Doppelpuffer-Pipelines |
| `aplz` | Benutzer-CLI-Wrapper (Bash): Datei-/Verzeichnisunterstuetzung, tar-Integration |
| `install.sh` | Installer: optimierter Build (-O3 -flto) + systemweite Bereitstellung |
| `setup_finder.sh` | macOS Finder Quick Action Generator (Rechtsklick-Kompression/-Extraktion) |
| `build.sh` | Entwicklungs-Build-Skript |

## Technische Details

- **LZ77**: 3-gram hash, 2-way hash table (atomic_fetch_min), max match 255, min match 3, distance up to 65534
- **Per-chunk tANS (v3)**: Baut dynamisch optimale tANS-Tabellen pro 64-KB-Block auf. 512 symbols (256 literals + 256 match lengths), L=1024, Duda's fast spread. Threadgroup cooperative loading fuer schnellen per-chunk Tabellenzugriff auf der GPU
- **ANS Encoding**: Forward encode with renormalization, sentinel bit for bitstream end detection
- **ANS Decoding**: Reverse playback from final state, sentinel via `clz()`, O(1) decode table lookup
- **Compact Frequency Serialization**: Speichert nur Nicht-Null-Eintraege pro Block (`n_nonzero + [sym_id, freq]` Paare), minimiert den Dateigroessen-Overhead
- **LZ77 Parallel Decode**: Hybrid approach — serial fallback for token count > 4096, parallel cooperative copy with barrier reduction for dense match streams
- **Async Double Buffering**: GCD-based batch pipeline (32 chunks/batch, 2 slots). `dispatch_semaphore` for slot exclusion, `addCompletedHandler` + serial `dispatch_queue` for ordered async writes
- **O(1) Streaming Memory**: Mega-batch architecture (512 chunks = 32MB per batch). Buffers are reused across mega-batches. Fixed ~512 MB memory regardless of input size.

## Bekannte Einschraenkungen

- **BS_CAP = 512 B**: Maximale Ausgabe eines Streams betraegt 512 Bytes. Hochgradig inkompressible Daten koennen Bits abschneiden.
- **Blockgroesse auf 64 KB festgelegt**: Anpassbar, erfordert aber Neukompilierung.

## License

MIT License. See [LICENSE](LICENSE) for details.
