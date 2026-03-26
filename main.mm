// main.mm — APLZ Host Driver (圧縮 + 解凍)
//
// 圧縮パイプライン (-c):
//   mmap(input) → ゼロコピー MTLBuffer
//   → GPU Pass 1: compress_chunk  [LZ77 + greedy overlap resolution]
//   → CPU       : histogram → normalize → Duda's spread/encode tables (tANS)
//   → GPU Pass 2: tans_encode      [256 interleaved ANS → bitstreams]
//   → CPU       : write output file (FileHeader + tANS tables + bitstreams)
//
// 解凍パイプライン (-d):
//   read .aplz → SymInfo[512] 復元 → CPU: decode table 構築
//   → GPU Pass 1: tans_decode  [256 interleaved ANS 逆再生 → tokens]
//   → GPU Pass 2: lz77_decode  [tokens → 元データ展開]
//   → write output file

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <QuartzCore/CABase.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <vector>
#include <algorithm>

#include "APLZ.h"

// ─── ローカルエイリアス ────────────────────────────────────────────────────────
static const uint32_t CHUNK_SIZE  = APLZ_CHUNK_SIZE;
static const uint32_t TG_SIZE     = APLZ_TG_SIZE;
static const uint32_t N_STREAMS   = APLZ_N_STREAMS;
static const uint32_t ANS_LOG_L   = APLZ_ANS_LOG_L;
static const uint32_t ANS_L       = APLZ_ANS_L;
static const uint32_t N_SYMBOLS   = APLZ_N_SYMBOLS;
static const uint32_t BS_CAP      = APLZ_BS_CAP;

// ─── ヘルパー ──────────────────────────────────────────────────────────────────
[[noreturn]] static void die(const char* msg) { perror(msg); exit(EXIT_FAILURE); }

static id<MTLLibrary> compile_shader(id<MTLDevice> dev, const char* path) {
    NSError* err = nil;
    NSString* src = [NSString stringWithContentsOfFile:@(path)
                                              encoding:NSUTF8StringEncoding
                                                 error:&err];
    if (!src) {
        fprintf(stderr, "[GPU_ZIP] Cannot read '%s': %s\n",
                path, err.localizedDescription.UTF8String);
        exit(EXIT_FAILURE);
    }
    MTLCompileOptions* opt = [MTLCompileOptions new];
    opt.languageVersion = MTLLanguageVersion3_0;
    opt.mathMode        = MTLMathModeFast;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opt error:&err];
    if (!lib) {
        fprintf(stderr, "[GPU_ZIP] Shader error:\n%s\n",
                err.localizedDescription.UTF8String);
        exit(EXIT_FAILURE);
    }
    return lib;
}

// ─── ヒストグラム集計 (CPU) ────────────────────────────────────────────────────
static void compute_histogram(const LzToken* compact, const uint32_t* cnt,
                              uint32_t num_chunks, uint32_t* freq) {
    for (uint32_t c = 0; c < num_chunks; ++c) {
        const LzToken* base = compact + (uint64_t)c * CHUNK_SIZE;
        for (uint32_t i = 0; i < cnt[c]; ++i) {
            if (base[i].is_match) freq[256 + base[i].val]++;
            else                  freq[base[i].val]++;
        }
    }
}

// ─── tANS: ヒストグラム正規化 → SymInfo[512] ─────────────────────────────────
static void normalize_histogram(const uint32_t* raw, SymInfo* si) {
    uint64_t total = 0;
    for (uint32_t i = 0; i < N_SYMBOLS; i++) total += raw[i];

    if (total == 0) {
        memset(si, 0, N_SYMBOLS * sizeof(SymInfo));
        si[0].freq = (uint16_t)ANS_L;
        for (uint32_t i = 0; i < N_SYMBOLS; i++) si[i].cum_freq = (i == 0) ? 0 : si[0].freq;
        return;
    }

    uint32_t nf[N_SYMBOLS];
    uint32_t assigned = 0;
    for (uint32_t i = 0; i < N_SYMBOLS; i++) {
        if (raw[i] == 0) { nf[i] = 0; continue; }
        nf[i] = std::max(1u, (uint32_t)((uint64_t)raw[i] * ANS_L / total));
        assigned += nf[i];
    }

    uint32_t best = 0;
    for (uint32_t i = 1; i < N_SYMBOLS; i++)
        if (raw[i] > raw[best]) best = i;

    int32_t diff = (int32_t)ANS_L - (int32_t)assigned;
    if ((int32_t)nf[best] + diff >= 1) {
        nf[best] += diff;
    } else {
        while (assigned != ANS_L) {
            for (uint32_t i = 0; i < N_SYMBOLS && assigned != ANS_L; i++) {
                if (assigned < ANS_L && nf[i] > 0)  { nf[i]++; assigned++; }
                if (assigned > ANS_L && nf[i] > 1)  { nf[i]--; assigned--; }
            }
        }
    }

    uint16_t cum = 0;
    for (uint32_t i = 0; i < N_SYMBOLS; i++) {
        si[i].freq     = (uint16_t)nf[i];
        si[i].cum_freq = cum;
        cum += (uint16_t)nf[i];
    }
}

// ─── tANS: Duda's fast spread table ─────────────────────────────────────────
static void build_spread_table(const SymInfo* si, uint16_t* spread) {
    const uint32_t step = (ANS_L >> 1) + (ANS_L >> 3) + 3;
    const uint32_t mask = ANS_L - 1;
    uint32_t pos = 0;
    for (uint32_t s = 0; s < N_SYMBOLS; s++) {
        for (uint32_t j = 0; j < si[s].freq; j++) {
            spread[pos] = (uint16_t)s;
            pos = (pos + step) & mask;
        }
    }
}

// ─── tANS: エンコードテーブル構築 ─────────────────────────────────────────────
static void build_encode_table(const SymInfo* si, const uint16_t* spread,
                               uint16_t* enc_table) {
    uint32_t rank[N_SYMBOLS] = {};
    for (uint32_t p = 0; p < ANS_L; p++) {
        uint32_t s = spread[p];
        uint32_t j = rank[s]++;
        enc_table[si[s].cum_freq + j] = (uint16_t)(ANS_L + p);
    }
}

// ─── tANS: デコードテーブル構築 ──────────────────────────────────────────────
//  spread table から各状態 x ∈ [L, 2L-1] の逆引きテーブルを構築する。
//
//  状態 x のデコード:
//    1. idx = x - L → spread[idx] でシンボル s を得る
//    2. s の spread 内 rank を j とすると、reduced state x_s = freq[s] + j
//    3. x_s ∈ [freq, 2*freq-1] から読み取りビット数 nb を計算:
//       nb = ANS_LOG_L - floor(log2(x_s))
//    4. new_base = x_s << nb  (∈ [L, 2L-1])
//    5. デコーダは nb ビット読み取り → new_state = new_base + bits
static void build_decode_table(const SymInfo* si, const uint16_t* spread,
                               DecodeEntry* dec) {
    uint32_t rank[N_SYMBOLS] = {};
    for (uint32_t i = 0; i < ANS_L; i++) {
        uint32_t s = spread[i];
        uint32_t j = rank[s]++;
        uint32_t x_s = si[s].freq + j;  // reduced state ∈ [freq, 2*freq-1]

        // floor(log2(x_s)): 最上位ビット位置
        uint32_t k = 0, tmp = x_s;
        while (tmp > 1) { tmp >>= 1; k++; }

        uint32_t nb = ANS_LOG_L - k;
        uint32_t new_base = x_s << nb;  // ∈ [L, 2L-1]

        dec[i].symbol   = (uint16_t)s;
        dec[i].num_bits = (uint16_t)nb;
        dec[i].new_base = (uint16_t)new_base;
        dec[i]._pad     = 0;
    }
}

// ─── 圧縮ファイル書き出し ──────────────────────────────────────────────────────
// フォーマット:
//   FileHeader (24 B) | n_streams (4 B) | ans_log_l (4 B)
//   SymInfo[512] (2048 B) | chunk_offsets[num_chunks] (8 B 各)
//   Per chunk: token_cnt (4 B) | uint16_t stream_sizes[256] | stream_data...
static uint64_t write_compressed(
    const char*       out_path,
    const SymInfo*    sym_info,
    const uint8_t*    bs_out,
    const uint32_t*   bs_sizes,
    const uint32_t*   token_counts,
    uint32_t          num_chunks,
    uint64_t          original_size)
{
    FILE* f = fopen(out_path, "wb");
    if (!f) { perror("fopen"); return 0; }

    FileHeader hdr;
    memcpy(hdr.magic, "APLZ", 4);
    hdr.version       = 2;
    hdr.original_size = original_size;
    hdr.chunk_size    = CHUNK_SIZE;
    hdr.num_chunks    = num_chunks;
    fwrite(&hdr, sizeof(hdr), 1, f);

    uint32_t ns = N_STREAMS, al = ANS_LOG_L;
    fwrite(&ns, 4, 1, f);
    fwrite(&al, 4, 1, f);
    fwrite(sym_info, sizeof(SymInfo), N_SYMBOLS, f);

    long seek_tbl_pos = ftell(f);
    std::vector<uint64_t> offsets(num_chunks, 0);
    fwrite(offsets.data(), 8, num_chunks, f);

    for (uint32_t c = 0; c < num_chunks; c++) {
        offsets[c] = (uint64_t)ftell(f);

        // token count (デコーダ用)
        fwrite(&token_counts[c], 4, 1, f);

        std::vector<uint16_t> ssz(N_STREAMS);
        for (uint32_t s = 0; s < N_STREAMS; s++)
            ssz[s] = (uint16_t)bs_sizes[(uint64_t)c * N_STREAMS + s];
        fwrite(ssz.data(), 2, N_STREAMS, f);

        for (uint32_t s = 0; s < N_STREAMS; s++) {
            const uint8_t* sdata = bs_out + ((uint64_t)c * N_STREAMS + s) * BS_CAP;
            fwrite(sdata, 1, ssz[s], f);
        }
    }

    uint64_t end_pos = (uint64_t)ftell(f);

    fseek(f, seek_tbl_pos, SEEK_SET);
    fwrite(offsets.data(), 8, num_chunks, f);
    fclose(f);

    return end_pos;
}

// ═══════════════════════════════════════════════════════════════════════════════
// 圧縮モード (-c)
// ═══════════════════════════════════════════════════════════════════════════════
static int compress(const char* in_path, const char* out_path, const char* shader_path) {
    // ── ゼロコピー I/O ──────────────────────────────────────────────────────
    int fd = open(in_path, O_RDONLY);
    if (fd < 0) die("open");
    struct stat st;
    if (fstat(fd, &st) < 0) die("fstat");
    const size_t file_size = (size_t)st.st_size;
    if (file_size == 0) {
        fprintf(stderr, "[GPU_ZIP] Empty input.\n"); close(fd); return EXIT_FAILURE;
    }
    void* mapped = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) die("mmap");
    madvise(mapped, file_size, MADV_SEQUENTIAL);

    printf("[GPU_ZIP] Input   : %s  (%zu bytes, %.2f MB)\n",
           in_path, file_size, file_size / (1024.0 * 1024.0));

    // ── Metal ────────────────────────────────────────────────────────────────
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) { fprintf(stderr, "[GPU_ZIP] Metal unavailable.\n"); return EXIT_FAILURE; }
    printf("[GPU_ZIP] Device  : %s\n", dev.name.UTF8String);
    id<MTLCommandQueue> cq = [dev newCommandQueue];

    id<MTLBuffer> buf_in = [dev newBufferWithBytesNoCopy:mapped
                                                  length:file_size
                                                 options:MTLResourceStorageModeShared
                                             deallocator:nil];
    if (!buf_in) { fprintf(stderr, "[GPU_ZIP] NoCopy failed.\n"); return EXIT_FAILURE; }

    const uint32_t num_chunks = (uint32_t)((file_size + CHUNK_SIZE - 1) / CHUNK_SIZE);
    printf("[GPU_ZIP] Chunks  : %u\n", num_chunks);

    const size_t sz_tok = (size_t)num_chunks * CHUNK_SIZE * sizeof(LzToken);
    const size_t sz_cnt = (size_t)num_chunks * sizeof(uint32_t);

    id<MTLBuffer> buf_sparse  = [dev newBufferWithLength:sz_tok options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_compact = [dev newBufferWithLength:sz_tok options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_cnt     = [dev newBufferWithLength:sz_cnt options:MTLResourceStorageModeShared];

    id<MTLLibrary> lib = compile_shader(dev, shader_path);

    CFTimeInterval t_total_start = CACurrentMediaTime();

    // ── GPU Pass 1: LZ77 ────────────────────────────────────────────────────
    {
        id<MTLFunction> fn = [lib newFunctionWithName:@"compress_chunk"];
        NSError* err = nil;
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) {
            fprintf(stderr, "[GPU_ZIP] PSO1: %s\n", err.localizedDescription.UTF8String);
            return EXIT_FAILURE;
        }

        id<MTLCommandBuffer> cb = [cq commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:buf_in      offset:0 atIndex:0];
        [enc setBuffer:buf_sparse  offset:0 atIndex:1];
        [enc setBuffer:buf_compact offset:0 atIndex:2];
        [enc setBuffer:buf_cnt     offset:0 atIndex:3];
        uint32_t total = (uint32_t)file_size;
        [enc setBytes:&total length:4 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(num_chunks, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(TG_SIZE, 1, 1)];
        [enc endEncoding];

        CFTimeInterval t0 = CACurrentMediaTime();
        [cb commit]; [cb waitUntilCompleted];
        CFTimeInterval t1 = CACurrentMediaTime();

        if (cb.error) {
            fprintf(stderr, "[GPU_ZIP] Pass 1 error: %s\n",
                    cb.error.localizedDescription.UTF8String);
            return EXIT_FAILURE;
        }
        printf("[GPU_ZIP] Pass 1  : %.2f ms  (LZ77 + overlap resolution)\n", (t1-t0)*1000.0);
    }

    // ── CPU: ヒストグラム + tANS テーブル ────────────────────────────────────
    const LzToken*  compact_ptr = (const LzToken*)buf_compact.contents;
    const uint32_t* cnt_ptr     = (const uint32_t*)buf_cnt.contents;

    uint64_t total_tokens = 0, total_matches = 0;
    for (uint32_t c = 0; c < num_chunks; c++) total_tokens += cnt_ptr[c];
    for (uint32_t c = 0; c < num_chunks; c++) {
        const LzToken* base = compact_ptr + (uint64_t)c * CHUNK_SIZE;
        for (uint32_t i = 0; i < cnt_ptr[c]; i++)
            if (base[i].is_match) total_matches++;
    }
    printf("[GPU_ZIP] Tokens  : %llu  (%llu matches, %.1f%%)\n",
           (unsigned long long)total_tokens,
           (unsigned long long)total_matches,
           100.0 * total_matches / std::max(total_tokens, (uint64_t)1));

    uint32_t raw_freq[N_SYMBOLS] = {};
    compute_histogram(compact_ptr, cnt_ptr, num_chunks, raw_freq);

    SymInfo sym_info[N_SYMBOLS];
    normalize_histogram(raw_freq, sym_info);

    uint16_t spread[ANS_L], enc_table[ANS_L];
    build_spread_table(sym_info, spread);
    build_encode_table(sym_info, spread, enc_table);

    // ── Pass 2 用バッファ ────────────────────────────────────────────────────
    const size_t sz_bs    = (size_t)num_chunks * N_STREAMS * BS_CAP;
    const size_t sz_bsz   = (size_t)num_chunks * N_STREAMS * sizeof(uint32_t);
    const size_t sz_ccomp = (size_t)num_chunks * sizeof(uint32_t);

    id<MTLBuffer> buf_sym   = [dev newBufferWithLength:N_SYMBOLS * sizeof(SymInfo)
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_enc   = [dev newBufferWithLength:ANS_L * sizeof(uint16_t)
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_bs    = [dev newBufferWithLength:sz_bs
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_bsz   = [dev newBufferWithLength:sz_bsz
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_ccomp = [dev newBufferWithLength:sz_ccomp
                                               options:MTLResourceStorageModeShared];

    if (!buf_sym || !buf_enc || !buf_bs || !buf_bsz || !buf_ccomp) {
        fprintf(stderr, "[GPU_ZIP] Buffer alloc failed (Pass 2).\n"); return EXIT_FAILURE;
    }
    memcpy(buf_sym.contents, sym_info, N_SYMBOLS * sizeof(SymInfo));
    memcpy(buf_enc.contents, enc_table, ANS_L * sizeof(uint16_t));
    memset(buf_bs.contents, 0, sz_bs);

    printf("[GPU_ZIP] Buffers : bs=%.1f MB, bsz=%.0f KB\n",
           sz_bs / (1024.0*1024.0), sz_bsz / 1024.0);

    // ── GPU Pass 2: tANS encode ─────────────────────────────────────────────
    {
        id<MTLFunction> fn = [lib newFunctionWithName:@"tans_encode"];
        if (!fn) {
            fprintf(stderr, "[GPU_ZIP] 'tans_encode' not found.\n"); return EXIT_FAILURE;
        }
        NSError* err = nil;
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) {
            fprintf(stderr, "[GPU_ZIP] PSO2: %s\n", err.localizedDescription.UTF8String);
            return EXIT_FAILURE;
        }

        id<MTLCommandBuffer> cb = [cq commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:buf_compact offset:0 atIndex:0];
        [enc setBuffer:buf_cnt     offset:0 atIndex:1];
        [enc setBuffer:buf_sym     offset:0 atIndex:2];
        [enc setBuffer:buf_enc     offset:0 atIndex:3];
        [enc setBuffer:buf_bs      offset:0 atIndex:4];
        [enc setBuffer:buf_bsz     offset:0 atIndex:5];
        [enc setBuffer:buf_ccomp   offset:0 atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(num_chunks, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(TG_SIZE, 1, 1)];
        [enc endEncoding];

        CFTimeInterval t0 = CACurrentMediaTime();
        [cb commit]; [cb waitUntilCompleted];
        CFTimeInterval t1 = CACurrentMediaTime();

        if (cb.error) {
            fprintf(stderr, "[GPU_ZIP] Pass 2 error: %s\n",
                    cb.error.localizedDescription.UTF8String);
            return EXIT_FAILURE;
        }
        printf("[GPU_ZIP] Pass 2  : %.2f ms  (tANS encoding)\n", (t1-t0)*1000.0);
    }

    CFTimeInterval t_total_end = CACurrentMediaTime();
    double total_ms  = (t_total_end - t_total_start) * 1000.0;
    double throughput = (file_size / (1024.0 * 1024.0)) / (t_total_end - t_total_start);

    // ── 書き出し ─────────────────────────────────────────────────────────────
    const uint8_t*  bs_ptr  = (const uint8_t*)buf_bs.contents;
    const uint32_t* bsz_ptr = (const uint32_t*)buf_bsz.contents;

    uint64_t comp_size = write_compressed(out_path, sym_info, bs_ptr, bsz_ptr,
                                          cnt_ptr, num_chunks, (uint64_t)file_size);
    if (comp_size == 0) {
        fprintf(stderr, "[GPU_ZIP] Write failed.\n"); return EXIT_FAILURE;
    }

    printf("[GPU_ZIP] Output  : %s\n", out_path);
    printf("[GPU_ZIP] Size    : %zu → %llu bytes  (%.1f%%)\n",
           file_size, (unsigned long long)comp_size,
           100.0 * (double)comp_size / (double)file_size);
    printf("[GPU_ZIP] Total   : %.2f ms  (%.1f MB/s)\n", total_ms, throughput);

    munmap(mapped, file_size);
    close(fd);
    return EXIT_SUCCESS;
}

// ═══════════════════════════════════════════════════════════════════════════════
// 解凍モード (-d)
// ═══════════════════════════════════════════════════════════════════════════════
static int decompress(const char* in_path, const char* out_path, const char* shader_path) {
    // ── .aplz ファイル読み込み ───────────────────────────────────────────────
    FILE* f = fopen(in_path, "rb");
    if (!f) { perror("fopen"); return EXIT_FAILURE; }

    FileHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) {
        fprintf(stderr, "[GPU_ZIP] Failed to read header.\n"); fclose(f); return EXIT_FAILURE;
    }
    if (memcmp(hdr.magic, "APLZ", 4) != 0) {
        fprintf(stderr, "[GPU_ZIP] Not an APLZ file.\n"); fclose(f); return EXIT_FAILURE;
    }

    uint32_t ns, al;
    fread(&ns, 4, 1, f);
    fread(&al, 4, 1, f);

    SymInfo sym_info[N_SYMBOLS];
    fread(sym_info, sizeof(SymInfo), N_SYMBOLS, f);

    const uint32_t num_chunks = hdr.num_chunks;
    const uint64_t original_size = hdr.original_size;

    std::vector<uint64_t> offsets(num_chunks);
    fread(offsets.data(), 8, num_chunks, f);

    printf("[GPU_ZIP] Decompress: %s\n", in_path);
    printf("[GPU_ZIP] Original : %llu bytes (%.2f MB)\n",
           (unsigned long long)original_size, original_size / (1024.0 * 1024.0));
    printf("[GPU_ZIP] Chunks   : %u\n", num_chunks);

    // ── CPU: tANS テーブル再構築 ────────────────────────────────────────────
    // SymInfo[512] からエンコーダと同じ spread table を再構築し、
    // そこからデコードテーブルを生成する。
    uint16_t spread[ANS_L];
    build_spread_table(sym_info, spread);

    DecodeEntry dec_table[ANS_L];
    build_decode_table(sym_info, spread, dec_table);

    // ── Metal セットアップ ──────────────────────────────────────────────────
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) { fprintf(stderr, "[GPU_ZIP] Metal unavailable.\n"); fclose(f); return EXIT_FAILURE; }
    printf("[GPU_ZIP] Device   : %s\n", dev.name.UTF8String);
    id<MTLCommandQueue> cq = [dev newCommandQueue];
    id<MTLLibrary> lib = compile_shader(dev, shader_path);

    // ── GPU バッファ確保 ─────────────────────────────────────────────────────
    const size_t sz_bs   = (size_t)num_chunks * N_STREAMS * BS_CAP;
    const size_t sz_bsz  = (size_t)num_chunks * N_STREAMS * sizeof(uint32_t);
    const size_t sz_tcnt = (size_t)num_chunks * sizeof(uint32_t);
    const size_t sz_tok  = (size_t)num_chunks * CHUNK_SIZE * sizeof(LzToken);
    const size_t sz_out  = (size_t)original_size;

    id<MTLBuffer> buf_bs   = [dev newBufferWithLength:sz_bs   options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_bsz  = [dev newBufferWithLength:sz_bsz  options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_tcnt = [dev newBufferWithLength:sz_tcnt options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_dec  = [dev newBufferWithLength:ANS_L * sizeof(DecodeEntry)
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_tok  = [dev newBufferWithLength:sz_tok  options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_out  = [dev newBufferWithLength:sz_out  options:MTLResourceStorageModeShared];

    if (!buf_bs || !buf_bsz || !buf_tcnt || !buf_dec || !buf_tok || !buf_out) {
        fprintf(stderr, "[GPU_ZIP] Buffer alloc failed (decompress).\n"); fclose(f); return EXIT_FAILURE;
    }

    memcpy(buf_dec.contents, dec_table, ANS_L * sizeof(DecodeEntry));

    // ── チャンクデータをファイルから GPU バッファに読み込む ───────────────
    uint8_t*  bs_data  = (uint8_t*)buf_bs.contents;
    uint32_t* bsz_data = (uint32_t*)buf_bsz.contents;
    uint32_t* tcnt_data = (uint32_t*)buf_tcnt.contents;
    memset(bs_data, 0, sz_bs);

    for (uint32_t c = 0; c < num_chunks; c++) {
        fseek(f, (long)offsets[c], SEEK_SET);

        // token count
        fread(&tcnt_data[c], 4, 1, f);

        // stream sizes (uint16_t → uint32_t に拡張)
        uint16_t ssz[N_STREAMS];
        fread(ssz, 2, N_STREAMS, f);

        for (uint32_t s = 0; s < N_STREAMS; s++) {
            bsz_data[(uint64_t)c * N_STREAMS + s] = (uint32_t)ssz[s];
            fread(bs_data + ((uint64_t)c * N_STREAMS + s) * BS_CAP, 1, ssz[s], f);
        }
    }
    fclose(f);

    CFTimeInterval t_total_start = CACurrentMediaTime();

    // ── GPU Pass 1: tANS デコード ───────────────────────────────────────────
    {
        id<MTLFunction> fn = [lib newFunctionWithName:@"tans_decode"];
        if (!fn) {
            fprintf(stderr, "[GPU_ZIP] 'tans_decode' not found.\n"); return EXIT_FAILURE;
        }
        NSError* err = nil;
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) {
            fprintf(stderr, "[GPU_ZIP] PSO decode1: %s\n", err.localizedDescription.UTF8String);
            return EXIT_FAILURE;
        }

        id<MTLCommandBuffer> cb = [cq commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:buf_bs   offset:0 atIndex:0];
        [enc setBuffer:buf_bsz  offset:0 atIndex:1];
        [enc setBuffer:buf_dec  offset:0 atIndex:2];
        [enc setBuffer:buf_tok  offset:0 atIndex:3];
        [enc setBuffer:buf_tcnt offset:0 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(num_chunks, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(TG_SIZE, 1, 1)];
        [enc endEncoding];

        CFTimeInterval t0 = CACurrentMediaTime();
        [cb commit]; [cb waitUntilCompleted];
        CFTimeInterval t1 = CACurrentMediaTime();

        if (cb.error) {
            fprintf(stderr, "[GPU_ZIP] tANS decode error: %s\n",
                    cb.error.localizedDescription.UTF8String);
            return EXIT_FAILURE;
        }
        printf("[GPU_ZIP] Decode 1 : %.2f ms  (tANS decode)\n", (t1-t0)*1000.0);
    }

    // ── GPU Pass 2: LZ77 展開 ───────────────────────────────────────────────
    {
        id<MTLFunction> fn = [lib newFunctionWithName:@"lz77_decode"];
        if (!fn) {
            fprintf(stderr, "[GPU_ZIP] 'lz77_decode' not found.\n"); return EXIT_FAILURE;
        }
        NSError* err = nil;
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) {
            fprintf(stderr, "[GPU_ZIP] PSO decode2: %s\n", err.localizedDescription.UTF8String);
            return EXIT_FAILURE;
        }

        id<MTLCommandBuffer> cb = [cq commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:buf_tok  offset:0 atIndex:0];
        [enc setBuffer:buf_tcnt offset:0 atIndex:1];
        [enc setBuffer:buf_out  offset:0 atIndex:2];
        uint32_t total = (uint32_t)original_size;
        [enc setBytes:&total length:4 atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(num_chunks, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(TG_SIZE, 1, 1)];
        [enc endEncoding];

        CFTimeInterval t0 = CACurrentMediaTime();
        [cb commit]; [cb waitUntilCompleted];
        CFTimeInterval t1 = CACurrentMediaTime();

        if (cb.error) {
            fprintf(stderr, "[GPU_ZIP] LZ77 decode error: %s\n",
                    cb.error.localizedDescription.UTF8String);
            return EXIT_FAILURE;
        }
        printf("[GPU_ZIP] Decode 2 : %.2f ms  (LZ77 expand)\n", (t1-t0)*1000.0);
    }

    CFTimeInterval t_total_end = CACurrentMediaTime();
    double total_ms = (t_total_end - t_total_start) * 1000.0;
    double throughput = (original_size / (1024.0 * 1024.0)) / (t_total_end - t_total_start);

    // ── 出力書き出し ─────────────────────────────────────────────────────────
    FILE* fout = fopen(out_path, "wb");
    if (!fout) { perror("fopen"); return EXIT_FAILURE; }
    fwrite(buf_out.contents, 1, (size_t)original_size, fout);
    fclose(fout);

    printf("[GPU_ZIP] Output   : %s (%llu bytes)\n", out_path, (unsigned long long)original_size);
    printf("[GPU_ZIP] Total    : %.2f ms  (%.1f MB/s)\n", total_ms, throughput);

    return EXIT_SUCCESS;
}

// ─── main ──────────────────────────────────────────────────────────────────────
int main(int argc, const char* argv[]) {
    @autoreleasepool {

    if (argc < 5) {
        fprintf(stderr, "Usage:\n");
        fprintf(stderr, "  %s -c <input> <output.aplz> <compression.metal>\n", argv[0]);
        fprintf(stderr, "  %s -d <input.aplz> <output> <compression.metal>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char* mode = argv[1];
    const char* path1 = argv[2];
    const char* path2 = argv[3];
    const char* shader = argv[4];

    if (strcmp(mode, "-c") == 0) {
        return compress(path1, path2, shader);
    } else if (strcmp(mode, "-d") == 0) {
        return decompress(path1, path2, shader);
    } else {
        fprintf(stderr, "Unknown mode '%s'. Use -c (compress) or -d (decompress).\n", mode);
        return EXIT_FAILURE;
    }

    } // @autoreleasepool
}
