# APLZ — Apple Parallel LZ

[English](README.md) | [日本語](README.ja.md) | [中文](README.zh.md) | **Español** | [Français](README.fr.md) | [Deutsch](README.de.md)

Motor de compresión/descompresión sin pérdida de alta velocidad que funciona en GPU Apple Silicon. Combina Metal Compute Shaders (MSL 3.0) con tANS (Asymmetric Numeral Systems) en un pipeline completamente acelerado por GPU, logrando una **compresión Round-trip perfecta**.

## Características

- **Zero-copy I/O**: Aprovecha la Unified Memory Architecture (UMA) de Apple Silicon. `mmap` + `newBufferWithBytesNoCopy` elimina todas las copias de datos CPU↔GPU
- **LZ77 paralelo**: 256 hilos procesan cooperativamente cada bloque de 64 KB. Hash de 2 vías (atomic_fetch_min) para búsqueda rápida de coincidencias
- **tANS intercalado por bloque (256 flujos)**: Construye dinámicamente tablas tANS óptimas por bloque. Cada hilo mantiene un estado ANS independiente, codificando/decodificando 256 flujos de bits en paralelo
- **Decodificación LZ77 paralela**: Las coincidencias sin superposición se copian en paralelo con 256 hilos; las coincidencias superpuestas (dist < len) usan respaldo serial. Incluye optimización de reducción de barreras
- **Doble buffer asíncrono**: Pipeline por lotes basado en GCD (`dispatch_semaphore` / `dispatch_group`). Superpone completamente la ejecución GPU con la E/S de CPU
- **Memoria O(1) en streaming**: Uso de memoria fijo independiente del tamaño de entrada. Procesa en mega-batches (32 MB) para operación estable con archivos de 5 GB+
- **Round-trip perfecto**: Garantiza coincidencia exacta a nivel de bytes tras comprimir → descomprimir

## Arquitectura

### Pipeline de compresión (`-c`)

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
    | histograma por bloque (512 simbolos)
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

### Pipeline de descompresión (`-d`)

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
  datos originales restaurados
```

## Formato de archivo (.aplz)

| Offset | Tamaño | Contenido |
|---|---|---|
| 0 | 24 B | `FileHeader` (magic "APLZ", version=5, original_size, chunk_size, num_chunks) |
| 24 | 4 B | `n_streams` = 256 |
| 28 | 4 B | `ans_log_l` = 10 |
| 32 | 8xN B | `chunk_offsets[N]` (seek table) |
| ... | variable | Chunk data (N bloques, codificación v5) |

Estructura de datos por bloque (codificación v5):
- `chunk_kind` (1 B): 0 = encoded (tANS), 1 = raw (fallback)

**Bloque codificado (chunk_kind = 0):**
- `chunk_n_streams` (2 B): número de flujos ANS activos para este bloque
- `token_cnt` (4 B): número de tokens densos
- `compact_freq`: `n_nonzero` (2 B) + `[sym_id, freq]` pares (4 B cada) — solo símbolos no nulos
- `stream_sizes[chunk_n_streams]` (2 B cada): tamaño en bytes de cada flujo ANS
- stream data: flujos de bits ANS concatenados con **codificación de distancia compacta v5**
  - Distancia corta (≤255): payload de 8 bits + 1-bit flag
  - Distancia larga (>255): payload de 16 bits + 1-bit flag

**Bloque sin comprimir (chunk_kind = 1):**
- `raw_size` (4 B): tamaño de bloque sin comprimir
- raw bytes: datos originales (fallback cuando la salida codificada excede la entrada)

## Instalación

Requiere macOS + Xcode Command Line Tools.

```bash
# Instalar en ~/.local/bin (compilación optimizada con -O3 -flto)
./install.sh

# Instalar en /usr/local/bin (requiere sudo)
./install.sh /usr/local

# Desinstalar
./install.sh uninstall
```

### Integración con Finder (opcional)

```bash
# Agregar clic derecho -> "Compress with APLZ" / "Extract with APLZ"
./setup_finder.sh

# Eliminar
./setup_finder.sh uninstall
```

Compatible con Google Drive y archivos de almacenamiento en la nube. El servicio procesa automáticamente los archivos vía `/tmp` para eludir las restricciones del sandbox de macOS.

## Uso

### Comando aplz (recomendado)

```bash
# Comprimir un archivo
aplz compress myfile.txt              → myfile.txt.aplz

# Comprimir un directorio (tar automático)
aplz compress myproject/              → myproject.tar.aplz

# Extraer
aplz extract myfile.txt.aplz          → myfile.txt
aplz extract myproject.tar.aplz       → myproject/

# Información del archivo
aplz info myfile.txt.aplz
```

### gpu_zip (API de bajo nivel)

```bash
./gpu_zip -c <input> <output.aplz> compression.metal
./gpu_zip -d <input.aplz> <output> compression.metal
```

### Compilación de desarrollo

```bash
./build.sh        # compilación -O2
./build.sh clean   # limpiar artefactos
```

## Rendimiento (Apple M4)

### Compresión

| Datos | Entrada | Salida | Ratio | Rendimiento |
|---|---|---|---|---|
| Texto (patrón repetitivo) | 1 MB | 31 KB | 3.0% | 50 MB/s |
| Texto (grande) | 100 MB | 3.0 MB | 3.0% | 251 MB/s |
| Binario aleatorio | 10 MB | 10.6 MB | 106% | 121 MB/s |

### Descompresión

| Datos | Velocidad | Notas |
|---|---|---|
| Texto (1 MB) | 246 MB/s | 1 mega-batch, overhead mínimo del pipeline |
| Texto (100 MB) | 1.4 GB/s | 4 mega-batches, superposición completa del pipeline |
| Aleatorio (10 MB) | 246 MB/s | 1 mega-batch, mayormente literales |

> Los datos aleatorios son incompresibles, por lo que la salida supera ligeramente la entrada (overhead de campos tANS/distance).
> El uso de memoria es constante (~512 MB) independientemente del tamaño de entrada gracias al streaming por mega-batch.

## Archivos fuente

| Archivo | Función |
|---|---|
| `APLZ.h` | Header compartido (structs y constantes para CPU y GPU) |
| `compression.metal` | Kernels GPU (MSL 3.0): compress_chunk, tans_encode, tans_decode, lz77_decode |
| `main.mm` | Driver del host (Objective-C++): mega-batch O(1) + pipelines asíncronos con doble buffer |
| `aplz` | CLI wrapper para usuario (Bash): soporte de archivos/directorios, integración tar |
| `install.sh` | Instalador: compilación optimizada (-O3 -flto) + despliegue a nivel de sistema |
| `setup_finder.sh` | Generador de Quick Actions para macOS Finder (compresión/extracción con clic derecho) |
| `build.sh` | Script de compilación para desarrollo |

## Detalles técnicos

- **LZ77**: 3-gram hash, 2-way hash table (atomic_fetch_min), max match 255, min match 3, distance up to 65534
- **Per-chunk tANS (v3)**: Construye dinámicamente tablas tANS óptimas por bloque de 64 KB. 512 symbols (256 literals + 256 match lengths), L=1024, Duda's fast spread. Threadgroup cooperative loading para acceso rápido a tablas per-chunk en GPU
- **ANS Encoding**: Forward encode with renormalization, sentinel bit for bitstream end detection
- **ANS Decoding**: Reverse playback from final state, sentinel via `clz()`, O(1) decode table lookup
- **Compact Frequency Serialization**: Almacena solo entradas no nulas por bloque (`n_nonzero + [sym_id, freq]` pairs), minimizando el overhead del tamaño del archivo
- **LZ77 Parallel Decode**: Hybrid approach — serial fallback for token count > 4096, parallel cooperative copy with barrier reduction for dense match streams
- **Async Double Buffering**: GCD-based batch pipeline (32 chunks/batch, 2 slots). `dispatch_semaphore` for slot exclusion, `addCompletedHandler` + serial `dispatch_queue` for ordered async writes
- **O(1) Streaming Memory**: Mega-batch architecture (512 chunks = 32MB per batch). Buffers are reused across mega-batches. Fixed ~512 MB memory regardless of input size.

## Limitaciones conocidas

- **BS_CAP = 512 B**: La salida máxima de 1 flujo es 512 bytes. Los datos altamente incompresibles pueden truncar bits.
- **Tamaño de bloque fijo en 64 KB**: Configurable pero requiere recompilación.

## License

MIT License. See [LICENSE](LICENSE) for details.
