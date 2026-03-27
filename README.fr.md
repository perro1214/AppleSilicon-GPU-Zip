# APLZ — Apple Parallel LZ

[English](README.md) | [日本語](README.ja.md) | [中文](README.zh.md) | [Español](README.es.md) | **Français** | [Deutsch](README.de.md)

Moteur de compression/décompression sans perte à haute vitesse fonctionnant sur GPU Apple Silicon. Combine les Metal Compute Shaders (MSL 3.0) avec tANS (Asymmetric Numeral Systems) dans un pipeline entièrement accéléré par GPU, réalisant une **compression Round-trip parfaite**.

## Points forts

- **Zero-copy I/O** : Exploite l'Unified Memory Architecture (UMA) d'Apple Silicon. `mmap` + `newBufferWithBytesNoCopy` élimine toutes les copies de données CPU↔GPU
- **LZ77 parallèle** : 256 threads traitent coopérativement chaque bloc de 64 Ko. Hash à 2 voies (atomic_fetch_min) pour une recherche rapide de correspondances
- **tANS entrelacé par bloc (256 flux)** : Construit dynamiquement des tables tANS optimales par bloc. Chaque thread maintient un état ANS indépendant, encodant/décodant 256 flux de bits en parallèle
- **Décodage LZ77 parallèle** : Les correspondances sans chevauchement sont copiées en parallèle par 256 threads ; les correspondances chevauchantes (dist < len) utilisent un repli séquentiel. Optimisation par réduction de barrières incluse
- **Double tampon asynchrone** : Pipeline par lots basé sur GCD (`dispatch_semaphore` / `dispatch_group`). Recouvre complètement l'exécution GPU avec les E/S CPU
- **Mémoire O(1) en streaming** : Utilisation mémoire fixe indépendante de la taille d'entrée. Traite en mega-batches (32 Mo) pour un fonctionnement stable sur des fichiers de 5 Go+
- **Round-trip parfait** : Garantit une correspondance exacte au niveau octet après compression → décompression

## Architecture

### Pipeline de compression (`-c`)

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
    | histogramme par bloc (512 symboles)
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

### Pipeline de décompression (`-d`)

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
  donnees originales restaurees
```

## Format de fichier (.aplz)

| Offset | Taille | Contenu |
|---|---|---|
| 0 | 24 o | `FileHeader` (magic "APLZ", version=3, original_size, chunk_size, num_chunks) |
| 24 | 4 o | `n_streams` = 256 |
| 28 | 4 o | `ans_log_l` = 10 |
| 32 | 8xN o | `chunk_offsets[N]` (seek table) |
| ... | variable | Chunk data (per chunk: token_cnt + compact_freq + stream_sizes[256] + streams) |

Structure des donnees par bloc :

- `token_cnt` (4 o) : nombre de tokens denses
- `compact_freq` : `n_nonzero` (2 o) + `[sym_id, freq]` pairs (4 o each) — symboles non nuls uniquement
- `stream_sizes[256]` (512 o) : taille en octets de chaque flux ANS
- stream data : 256 flux de bits concatenes

## Installation

Necessite macOS + Xcode Command Line Tools.

```bash
# Installer dans ~/.local/bin (compilation optimisee avec -O3 -flto)
./install.sh

# Installer dans /usr/local/bin (necessite sudo)
./install.sh /usr/local

# Desinstaller
./install.sh uninstall
```

### Integration Finder (optionnelle)

```bash
# Ajouter clic droit -> "Compress with APLZ" / "Extract with APLZ"
./setup_finder.sh

# Supprimer
./setup_finder.sh uninstall
```

Compatible avec Google Drive et les fichiers de stockage cloud. Le service traite automatiquement les fichiers via `/tmp` pour contourner les restrictions du sandbox macOS.

## Utilisation

### Commande aplz (recommandee)

```bash
# Compresser un fichier
aplz compress myfile.txt              → myfile.txt.aplz

# Compresser un repertoire (tar automatique)
aplz compress myproject/              → myproject.tar.aplz

# Extraire
aplz extract myfile.txt.aplz          → myfile.txt
aplz extract myproject.tar.aplz       → myproject/

# Informations sur le fichier
aplz info myfile.txt.aplz
```

### gpu_zip (API bas niveau)

```bash
./gpu_zip -c <input> <output.aplz> compression.metal
./gpu_zip -d <input.aplz> <output> compression.metal
```

### Compilation de developpement

```bash
./build.sh        # compilation -O2
./build.sh clean   # nettoyer les artefacts
```

## Performances (Apple M4)

### Compression

| Donnees | Entree | Sortie | Ratio | Debit |
|---|---|---|---|---|
| Texte (motif repetitif) | 1 Mo | 31 Ko | 3.0% | 50 Mo/s |
| Texte (volumineux) | 100 Mo | 3.0 Mo | 3.0% | 251 Mo/s |
| Binaire aleatoire | 10 Mo | 10.6 Mo | 106% | 121 Mo/s |

### Decompression

| Donnees | Vitesse | Notes |
|---|---|---|
| Texte (1 Mo) | 246 Mo/s | 1 mega-batch, overhead minimal du pipeline |
| Texte (100 Mo) | 1.4 Go/s | 4 mega-batches, recouvrement complet du pipeline |
| Aleatoire (10 Mo) | 246 Mo/s | 1 mega-batch, principalement des literaux |

> Les donnees aleatoires sont incompressibles, donc la sortie depasse legerement l'entree (overhead des champs tANS/distance).
> L'utilisation memoire est constante (~512 Mo) quelle que soit la taille d'entree grace au streaming par mega-batch.

## Fichiers sources

| Fichier | Role |
|---|---|
| `APLZ.h` | Header partage (structs et constantes pour CPU et GPU) |
| `compression.metal` | Kernels GPU (MSL 3.0) : compress_chunk, tans_encode, tans_decode, lz77_decode |
| `main.mm` | Pilote hote (Objective-C++) : mega-batch O(1) + pipelines asynchrones a double tampon |
| `aplz` | CLI wrapper utilisateur (Bash) : support fichiers/repertoires, integration tar |
| `install.sh` | Installateur : compilation optimisee (-O3 -flto) + deploiement systeme |
| `setup_finder.sh` | Generateur de Quick Actions macOS Finder (compression/extraction par clic droit) |
| `build.sh` | Script de compilation de developpement |

## Details techniques

- **LZ77** : 3-gram hash, 2-way hash table (atomic_fetch_min), max match 255, min match 3, distance up to 65534
- **Per-chunk tANS (v3)** : Construit dynamiquement des tables tANS optimales par bloc de 64 Ko. 512 symbols (256 literals + 256 match lengths), L=1024, Duda's fast spread. Threadgroup cooperative loading pour un acces rapide aux tables per-chunk sur GPU
- **ANS Encoding** : Forward encode with renormalization, sentinel bit for bitstream end detection
- **ANS Decoding** : Reverse playback from final state, sentinel via `clz()`, O(1) decode table lookup
- **Compact Frequency Serialization** : Stocke uniquement les entrees non nulles par bloc (`n_nonzero + [sym_id, freq]` pairs), minimisant l'overhead de taille du fichier
- **LZ77 Parallel Decode** : Hybrid approach — serial fallback for token count > 4096, parallel cooperative copy with barrier reduction for dense match streams
- **Async Double Buffering** : GCD-based batch pipeline (32 chunks/batch, 2 slots). `dispatch_semaphore` for slot exclusion, `addCompletedHandler` + serial `dispatch_queue` for ordered async writes
- **O(1) Streaming Memory** : Mega-batch architecture (512 chunks = 32MB per batch). Buffers are reused across mega-batches. Fixed ~512 MB memory regardless of input size.

## Limitations connues

- **BS_CAP = 512 o** : La sortie maximale d'un flux est de 512 octets. Les donnees hautement incompressibles peuvent tronquer des bits.
- **Taille de bloc fixee a 64 Ko** : Configurable mais necessite une recompilation.

## License

MIT License. See [LICENSE](LICENSE) for details.
