// main.mm — APLZ Host Driver (圧縮 + 解凍) v3
//
// Per-chunk tANS テーブル + O(1) ストリーミング・メガバッチ・パイプライン
//
//   圧縮: メガバッチ単位で Pass1 → CPU per-chunk テーブル構築 → Pass2
//          Pass1 は 1 回のみ (v2 の 2 回実行を解消)
//   解凍: メガバッチ単位で読み込み → CPU per-chunk デコードテーブル構築 → GPU decode

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
#include <string>
#include <algorithm>
#include <dispatch/dispatch.h>
#include <limits>

#include "APLZ.h"

// ─── ローカルエイリアス ────────────────────────────────────────────────────────
static const uint32_t CHUNK_SIZE  = APLZ_CHUNK_SIZE;
static const uint32_t TG_SIZE     = APLZ_TG_SIZE;
static const uint32_t N_STREAMS   = APLZ_N_STREAMS;
static const uint32_t ANS_LOG_L   = APLZ_ANS_LOG_L;
static const uint32_t ANS_L       = APLZ_ANS_L;
static const uint32_t N_SYMBOLS   = APLZ_N_SYMBOLS;
static const uint32_t BS_CAP      = APLZ_BS_CAP;

// ─── パイプライン定数 ────────────────────────────────────────────────────────
static const uint32_t MEGA_BATCH_CHUNKS = 512;
static const uint32_t BATCH_CHUNKS      = 32;
static const uint32_t N_BUFS            = 2;

// ─── ヘルパー ──────────────────────────────────────────────────────────────────
[[noreturn]] static void die(const char* msg) { perror(msg); exit(EXIT_FAILURE); }

static id<MTLLibrary> compile_shader(id<MTLDevice> dev, const char* path) {
    NSError* err = nil;
    NSString* src = [NSString stringWithContentsOfFile:@(path)
                                              encoding:NSUTF8StringEncoding
                                                 error:&err];
    if (!src) {
        fprintf(stderr, "[APLZ] Cannot read '%s': %s\n",
                path, err.localizedDescription.UTF8String);
        exit(EXIT_FAILURE);
    }
    MTLCompileOptions* opt = [MTLCompileOptions new];
    opt.languageVersion = MTLLanguageVersion3_0;
    opt.mathMode        = MTLMathModeFast;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opt error:&err];
    if (!lib) {
        fprintf(stderr, "[APLZ] Shader error:\n%s\n",
                err.localizedDescription.UTF8String);
        exit(EXIT_FAILURE);
    }
    return lib;
}

// ─── Per-chunk ヒストグラム → SymInfo 正規化 ─────────────────────────────────
static uint32_t compute_chunk_histogram(const LzToken* compact, uint32_t cnt,
                                        uint32_t* freq) {
    memset(freq, 0, N_SYMBOLS * sizeof(uint32_t));
    uint32_t matches = 0;
    for (uint32_t i = 0; i < cnt; ++i) {
        if (compact[i].is_match) {
            freq[256 + compact[i].val]++;
            matches++;
        } else {
            freq[compact[i].val]++;
        }
    }
    return matches;
}

static void normalize_histogram(const uint32_t* raw, SymInfo* si) {
    uint64_t total = 0;
    for (uint32_t i = 0; i < N_SYMBOLS; i++) total += raw[i];

    if (total == 0) {
        memset(si, 0, N_SYMBOLS * sizeof(SymInfo));
        si[0].freq = (uint16_t)ANS_L;
        for (uint32_t i = 0; i < N_SYMBOLS; i++)
            si[i].cum_freq = (i == 0) ? 0 : si[0].freq;
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

static void build_encode_table(const SymInfo* si, const uint16_t* spread,
                               uint16_t* enc_table) {
    uint32_t rank[N_SYMBOLS] = {};
    for (uint32_t p = 0; p < ANS_L; p++) {
        uint32_t s = spread[p];
        uint32_t j = rank[s]++;
        enc_table[si[s].cum_freq + j] = (uint16_t)(ANS_L + p);
    }
}

static void build_decode_table(const SymInfo* si, const uint16_t* spread,
                               DecodeEntry* dec) {
    uint32_t rank[N_SYMBOLS] = {};
    for (uint32_t i = 0; i < ANS_L; i++) {
        uint32_t s = spread[i];
        uint32_t j = rank[s]++;
        uint32_t x_s = si[s].freq + j;
        uint32_t k = 0, tmp = x_s;
        while (tmp > 1) { tmp >>= 1; k++; }
        uint32_t nb = ANS_LOG_L - k;
        uint32_t new_base = x_s << nb;
        dec[i].symbol   = (uint16_t)s;
        dec[i].num_bits = (uint16_t)nb;
        dec[i].new_base = (uint16_t)new_base;
        dec[i]._pad     = 0;
    }
}

// ─── コンパクト頻度テーブル シリアライズ/デシリアライズ ─────────────────────────
// format: uint16_t n_nonzero, [uint16_t symbol_id, uint16_t freq] × n_nonzero
static void write_compact_freq(FILE* f, const SymInfo* si) {
    uint16_t n_nz = 0;
    for (uint32_t i = 0; i < N_SYMBOLS; i++)
        if (si[i].freq > 0) n_nz++;
    fwrite(&n_nz, 2, 1, f);
    for (uint32_t i = 0; i < N_SYMBOLS; i++) {
        if (si[i].freq > 0) {
            uint16_t sym = (uint16_t)i;
            fwrite(&sym, 2, 1, f);
            fwrite(&si[i].freq, 2, 1, f);
        }
    }
}

static void read_compact_freq(FILE* f, SymInfo* si) {
    memset(si, 0, N_SYMBOLS * sizeof(SymInfo));
    uint16_t n_nz;
    fread(&n_nz, 2, 1, f);
    for (uint16_t i = 0; i < n_nz; i++) {
        uint16_t sym, freq;
        fread(&sym, 2, 1, f);
        fread(&freq, 2, 1, f);
        si[sym].freq = freq;
    }
    // cum_freq を再計算
    uint16_t cum = 0;
    for (uint32_t i = 0; i < N_SYMBOLS; i++) {
        si[i].cum_freq = cum;
        cum += si[i].freq;
    }
}

// ─── GPU Pass1 dispatch ヘルパー ──────────────────────────────────────────────
static void dispatch_pass1(id<MTLCommandQueue> cq,
                           id<MTLComputePipelineState> pso,
                           id<MTLBuffer> buf_in,
                           id<MTLBuffer> buf_sparse,
                           id<MTLBuffer> buf_compact,
                           id<MTLBuffer> buf_cnt,
                           uint32_t mc_start,
                           uint32_t mc_count,
                           size_t   file_size) {
    size_t in_offset = (size_t)mc_start * CHUNK_SIZE;
    uint32_t mb_bytes = (uint32_t)std::min(
        (uint64_t)mc_count * CHUNK_SIZE,
        (uint64_t)file_size - (uint64_t)in_offset);

    id<MTLCommandBuffer> cb = [cq commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:buf_in      offset:in_offset atIndex:0];
    [enc setBuffer:buf_sparse  offset:0         atIndex:1];
    [enc setBuffer:buf_compact offset:0         atIndex:2];
    [enc setBuffer:buf_cnt     offset:0         atIndex:3];
    [enc setBytes:&mb_bytes length:4 atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(mc_count, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(TG_SIZE, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    if (cb.error) {
        fprintf(stderr, "[APLZ] Pass 1 error: %s\n",
                cb.error.localizedDescription.UTF8String);
        exit(EXIT_FAILURE);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 圧縮モード (-c) v3 — Per-chunk tANS
// ═══════════════════════════════════════════════════════════════════════════════
static int compress(const char* in_path, const char* out_path, const char* shader_path) {
    int fd = open(in_path, O_RDONLY);
    if (fd < 0) die("open");
    struct stat st;
    if (fstat(fd, &st) < 0) die("fstat");
    const size_t file_size = (size_t)st.st_size;
    if (file_size == 0) {
        close(fd);
        // 空ファイル: ヘッダのみ書き出して正常終了
        FILE* fout = fopen(out_path, "wb");
        if (!fout) { perror("fopen"); return EXIT_FAILURE; }
        FileHeader hdr;
        memcpy(hdr.magic, "APLZ", 4);
        hdr.version       = APLZ_MAGIC_V3;
        hdr.original_size = 0;
        hdr.chunk_size    = CHUNK_SIZE;
        hdr.num_chunks    = 0;
        fwrite(&hdr, sizeof(hdr), 1, fout);
        uint32_t ns_val = N_STREAMS, al_val = ANS_LOG_L;
        fwrite(&ns_val, 4, 1, fout);
        fwrite(&al_val, 4, 1, fout);
        fclose(fout);
        printf("[APLZ] Empty file → %s (header only)\n", out_path);
        return EXIT_SUCCESS;
    }

    void* mapped = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) die("mmap");
    madvise(mapped, file_size, MADV_SEQUENTIAL);

    printf("[APLZ] Input    : %s  (%zu bytes, %.2f MB)\n",
           in_path, file_size, file_size / (1024.0 * 1024.0));

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) { fprintf(stderr, "[APLZ] Metal unavailable.\n"); return EXIT_FAILURE; }
    printf("[APLZ] Device   : %s\n", dev.name.UTF8String);
    id<MTLCommandQueue> cq = [dev newCommandQueue];

    id<MTLBuffer> buf_in = [dev newBufferWithBytesNoCopy:mapped
                                                  length:file_size
                                                 options:MTLResourceStorageModeShared
                                             deallocator:nil];
    if (!buf_in) { fprintf(stderr, "[APLZ] NoCopy failed.\n"); return EXIT_FAILURE; }

    const uint32_t num_chunks = (uint32_t)((file_size + CHUNK_SIZE - 1) / CHUNK_SIZE);
    printf("[APLZ] Chunks   : %u\n", num_chunks);

    const uint32_t mb_chunks = std::min((uint32_t)MEGA_BATCH_CHUNKS, num_chunks);
    const size_t sz_tok_mb = (size_t)mb_chunks * CHUNK_SIZE * sizeof(LzToken);
    const size_t sz_cnt_mb = (size_t)mb_chunks * sizeof(uint32_t);

    id<MTLBuffer> buf_sparse  = [dev newBufferWithLength:sz_tok_mb options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_compact = [dev newBufferWithLength:sz_tok_mb options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_cnt     = [dev newBufferWithLength:sz_cnt_mb options:MTLResourceStorageModeShared];

    if (!buf_sparse || !buf_compact || !buf_cnt) {
        fprintf(stderr, "[APLZ] Buffer alloc failed.\n"); return EXIT_FAILURE;
    }

    id<MTLLibrary> lib = compile_shader(dev, shader_path);
    id<MTLFunction> fn1 = [lib newFunctionWithName:@"compress_chunk"];
    NSError* err = nil;
    id<MTLComputePipelineState> pso1 = [dev newComputePipelineStateWithFunction:fn1 error:&err];
    if (!pso1) {
        fprintf(stderr, "[APLZ] PSO1: %s\n", err.localizedDescription.UTF8String);
        return EXIT_FAILURE;
    }

    id<MTLFunction> fn2 = [lib newFunctionWithName:@"tans_encode"];
    if (!fn2) { fprintf(stderr, "[APLZ] 'tans_encode' not found.\n"); return EXIT_FAILURE; }
    NSError* pso2_err = nil;
    id<MTLComputePipelineState> pso2 = [dev newComputePipelineStateWithFunction:fn2 error:&pso2_err];
    if (!pso2) {
        fprintf(stderr, "[APLZ] PSO2: %s\n", pso2_err.localizedDescription.UTF8String);
        return EXIT_FAILURE;
    }

    // Per-batch テーブルバッファ (ダブルバッファ)
    // sym_arr: SymInfo[BATCH_CHUNKS][N_SYMBOLS], enc_arr: uint16_t[BATCH_CHUNKS][ANS_L]
    id<MTLBuffer> slot_sym[N_BUFS], slot_enc[N_BUFS];
    id<MTLBuffer> slot_bs[N_BUFS], slot_bsz[N_BUFS];
    for (uint32_t s = 0; s < N_BUFS; s++) {
        slot_sym[s]   = [dev newBufferWithLength:(size_t)BATCH_CHUNKS * N_SYMBOLS * sizeof(SymInfo)
                                          options:MTLResourceStorageModeShared];
        slot_enc[s]   = [dev newBufferWithLength:(size_t)BATCH_CHUNKS * ANS_L * sizeof(uint16_t)
                                          options:MTLResourceStorageModeShared];
        slot_bs[s]    = [dev newBufferWithLength:(size_t)BATCH_CHUNKS * N_STREAMS * BS_CAP
                                         options:MTLResourceStorageModeShared];
        slot_bsz[s]   = [dev newBufferWithLength:(size_t)BATCH_CHUNKS * N_STREAMS * sizeof(uint32_t)
                                          options:MTLResourceStorageModeShared];
    }

    CFTimeInterval t_total_start = CACurrentMediaTime();

    // ── ファイルヘッダ (一時ファイルに書き込み、完了後にrename) ─────────────
    std::string tmp_path = std::string(out_path) + ".tmp";
    FILE* fout = fopen(tmp_path.c_str(), "wb");
    if (!fout) { perror("fopen"); return EXIT_FAILURE; }

    FileHeader hdr;
    memcpy(hdr.magic, "APLZ", 4);
    hdr.version       = APLZ_MAGIC_V3;
    hdr.original_size = (uint64_t)file_size;
    hdr.chunk_size    = CHUNK_SIZE;
    hdr.num_chunks    = num_chunks;
    fwrite(&hdr, sizeof(hdr), 1, fout);

    uint32_t ns_val = N_STREAMS, al_val = ANS_LOG_L;
    fwrite(&ns_val, 4, 1, fout);
    fwrite(&al_val, 4, 1, fout);
    // v3: グローバル SymInfo は書き出さない

    long seek_tbl_pos = ftell(fout);
    std::vector<uint64_t> offsets(num_chunks, 0);
    fwrite(offsets.data(), 8, num_chunks, fout);

    // パイプライン同期
    dispatch_semaphore_t buf_sem = dispatch_semaphore_create(N_BUFS);
    dispatch_queue_t write_q = dispatch_queue_create("aplz.write", DISPATCH_QUEUE_SERIAL);
    dispatch_group_t done_grp = dispatch_group_create();
    uint64_t* offs_ptr = offsets.data();

    uint32_t n_mega = (num_chunks + mb_chunks - 1) / mb_chunks;
    uint64_t total_tokens = 0, total_matches = 0;

    printf("[APLZ] MegaBatch: %u mega-batches x %u chunks (%.1f MB/batch)\n",
           n_mega, mb_chunks, (double)mb_chunks * CHUNK_SIZE / (1024.0 * 1024.0));

    CFTimeInterval t_pipe_start = CACurrentMediaTime();

    for (uint32_t mb = 0; mb < n_mega; mb++) {
        uint32_t mc_start = mb * mb_chunks;
        uint32_t mc_count = std::min(mb_chunks, num_chunks - mc_start);

        // ── GPU Pass1 ────────────────────────────────────────────────────
        dispatch_pass1(cq, pso1, buf_in, buf_sparse, buf_compact, buf_cnt,
                        mc_start, mc_count, file_size);

        const LzToken*  compact_ptr = (const LzToken*)buf_compact.contents;
        const uint32_t* cnt_ptr     = (const uint32_t*)buf_cnt.contents;

        // カウント収集
        for (uint32_t c = 0; c < mc_count; c++) {
            total_tokens += cnt_ptr[c];
        }

        // buf_cnt のローカルコピー
        std::vector<uint32_t> mb_cnt(cnt_ptr, cnt_ptr + mc_count);

        // ── Per-chunk テーブル構築 + Pass2 バッチパイプライン ─────────────
        // per-chunk の SymInfo を構築して配列に格納
        // 全メガバッチ分の SymInfo を保持 (ファイル書き出しに必要)
        std::vector<SymInfo> mb_sym_infos(mc_count * N_SYMBOLS);

        uint32_t raw_freq[N_SYMBOLS];
        uint16_t spread[ANS_L];

        for (uint32_t c = 0; c < mc_count; c++) {
            SymInfo* si = &mb_sym_infos[c * N_SYMBOLS];
            total_matches += compute_chunk_histogram(
                compact_ptr + (uint64_t)c * CHUNK_SIZE, cnt_ptr[c], raw_freq);
            normalize_histogram(raw_freq, si);
        }

        // Pass2 ダブルバッファ バッチループ
        uint32_t mb_n_batches = (mc_count + BATCH_CHUNKS - 1) / BATCH_CHUNKS;
        for (uint32_t b = 0; b < mb_n_batches; b++) {
            dispatch_semaphore_wait(buf_sem, DISPATCH_TIME_FOREVER);

            uint32_t sl = b % N_BUFS;
            uint32_t lc_start = b * BATCH_CHUNKS;
            uint32_t lc_count = std::min(BATCH_CHUNKS, mc_count - lc_start);
            uint32_t gc_start = mc_start + lc_start;

            // Per-chunk テーブルをスロットバッファにコピー
            SymInfo*  sym_dst = (SymInfo*)slot_sym[sl].contents;
            uint16_t* enc_dst = (uint16_t*)slot_enc[sl].contents;

            for (uint32_t lc = 0; lc < lc_count; lc++) {
                SymInfo* si = &mb_sym_infos[(lc_start + lc) * N_SYMBOLS];
                memcpy(sym_dst + (size_t)lc * N_SYMBOLS, si, N_SYMBOLS * sizeof(SymInfo));

                build_spread_table(si, spread);
                uint16_t* enc = enc_dst + (size_t)lc * ANS_L;
                build_encode_table(si, spread, enc);
            }

            memset(slot_bs[sl].contents, 0, (size_t)lc_count * N_STREAMS * BS_CAP);

            id<MTLCommandBuffer> cb = [cq commandBuffer];
            id<MTLComputeCommandEncoder> enc2 = [cb computeCommandEncoder];
            [enc2 setComputePipelineState:pso2];
            [enc2 setBuffer:buf_compact    offset:(size_t)lc_start * CHUNK_SIZE * sizeof(LzToken) atIndex:0];
            [enc2 setBuffer:buf_cnt        offset:(size_t)lc_start * sizeof(uint32_t)             atIndex:1];
            [enc2 setBuffer:slot_sym[sl]   offset:0 atIndex:2];
            [enc2 setBuffer:slot_enc[sl]   offset:0 atIndex:3];
            [enc2 setBuffer:slot_bs[sl]    offset:0 atIndex:4];
            [enc2 setBuffer:slot_bsz[sl]   offset:0 atIndex:5];
            [enc2 dispatchThreadgroups:MTLSizeMake(lc_count, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(TG_SIZE, 1, 1)];
            [enc2 endEncoding];

            // キャプチャ
            id<MTLBuffer> cap_bs  = slot_bs[sl];
            id<MTLBuffer> cap_bsz = slot_bsz[sl];
            uint32_t cap_gc_start = gc_start;
            uint32_t cap_lc_count = lc_count;
            std::vector<uint32_t> cap_cnt(mb_cnt.begin() + lc_start,
                                          mb_cnt.begin() + lc_start + lc_count);
            // per-chunk SymInfo のキャプチャ (ファイル書き出し用)
            std::vector<SymInfo> cap_sym(mb_sym_infos.begin() + lc_start * N_SYMBOLS,
                                         mb_sym_infos.begin() + (lc_start + lc_count) * N_SYMBOLS);

            dispatch_group_enter(done_grp);
            [cb addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull) {
                dispatch_async(write_q, ^{
                    const uint8_t*  bs  = (const uint8_t*)cap_bs.contents;
                    const uint32_t* bsz = (const uint32_t*)cap_bsz.contents;

                    for (uint32_t lc = 0; lc < cap_lc_count; lc++) {
                        uint32_t gc = cap_gc_start + lc;
                        offs_ptr[gc] = (uint64_t)ftell(fout);

                        uint32_t tc = cap_cnt[lc];
                        fwrite(&tc, 4, 1, fout);

                        // per-chunk 頻度テーブル (コンパクト形式)
                        const SymInfo* si = &cap_sym[lc * N_SYMBOLS];
                        write_compact_freq(fout, si);

                        // stream sizes + data (coalesced payload write)
                        uint16_t ssz[256];
                        size_t payload_size = 0;
                        for (uint32_t s = 0; s < 256; s++) {
                            ssz[s] = (uint16_t)bsz[(size_t)lc * 256 + s];
                            payload_size += ssz[s];
                        }
                        fwrite(ssz, 2, 256, fout);

                        std::vector<uint8_t> payload;
                        payload.reserve(payload_size);
                        for (uint32_t s = 0; s < 256; s++) {
                            const uint8_t* sdata = bs + ((size_t)lc * 256 + s) * BS_CAP;
                            payload.insert(payload.end(), sdata, sdata + ssz[s]);
                        }
                        fwrite(payload.data(), 1, payload.size(), fout);
                    }

                    dispatch_semaphore_signal(buf_sem);
                    dispatch_group_leave(done_grp);
                });
            }];
            [cb commit];
        }

        // メガバッチ完了待ち
        dispatch_group_wait(done_grp, DISPATCH_TIME_FOREVER);
    }

    CFTimeInterval t_pipe_end = CACurrentMediaTime();

    printf("[APLZ] Tokens   : %llu  (%llu matches, %.1f%%)\n",
           (unsigned long long)total_tokens,
           (unsigned long long)total_matches,
           100.0 * total_matches / std::max(total_tokens, (uint64_t)1));
    printf("[APLZ] Pipeline : %.2f ms  (Pass1 + per-chunk tANS + Pass2, streaming)\n",
           (t_pipe_end - t_pipe_start) * 1000.0);

    // ── シークテーブル書き戻し ──────────────────────────────────────────────
    fseek(fout, seek_tbl_pos, SEEK_SET);
    fwrite(offsets.data(), 8, num_chunks, fout);
    fseek(fout, 0, SEEK_END);
    uint64_t comp_size = (uint64_t)ftell(fout);
    fclose(fout);

    // 一時ファイルを最終パスにアトミック移動
    if (rename(tmp_path.c_str(), out_path) != 0) {
        perror("rename");
        unlink(tmp_path.c_str());
        return EXIT_FAILURE;
    }

    // sparse バッファ解放
    buf_sparse = nil;

    CFTimeInterval t_total_end = CACurrentMediaTime();
    double total_ms  = (t_total_end - t_total_start) * 1000.0;
    double throughput = (file_size / (1024.0 * 1024.0)) / (t_total_end - t_total_start);

    printf("[APLZ] Output   : %s\n", out_path);
    printf("[APLZ] Size     : %zu → %llu bytes  (%.1f%%)\n",
           file_size, (unsigned long long)comp_size,
           100.0 * (double)comp_size / (double)file_size);
    printf("[APLZ] Total    : %.2f ms  (%.1f MB/s)\n", total_ms, throughput);

    munmap(mapped, file_size);
    close(fd);
    return EXIT_SUCCESS;
}

// ═══════════════════════════════════════════════════════════════════════════════
// 解凍モード (-d) v3 — Per-chunk tANS
// ═══════════════════════════════════════════════════════════════════════════════
static int decompress(const char* in_path, const char* out_path, const char* shader_path) {
    FILE* f = fopen(in_path, "rb");
    if (!f) { perror("fopen"); return EXIT_FAILURE; }

    FileHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) {
        fprintf(stderr, "[APLZ] Failed to read header.\n"); fclose(f); return EXIT_FAILURE;
    }
    if (memcmp(hdr.magic, "APLZ", 4) != 0) {
        fprintf(stderr, "[APLZ] Not an APLZ file.\n"); fclose(f); return EXIT_FAILURE;
    }

    uint32_t ns, al;
    fread(&ns, 4, 1, f);
    fread(&al, 4, 1, f);

    // v3: グローバル SymInfo はない。v2 互換のためバージョンチェック。
    if (hdr.version == 2) {
        // v2: グローバルテーブルを読み飛ばす
        fseek(f, N_SYMBOLS * sizeof(SymInfo), SEEK_CUR);
    }

    const uint32_t num_chunks = hdr.num_chunks;
    const uint64_t original_size = hdr.original_size;

    // 空ファイル: 0バイトの出力ファイルを作成して正常終了
    if (num_chunks == 0 && original_size == 0) {
        fclose(f);
        FILE* fout = fopen(out_path, "wb");
        if (!fout) { perror("fopen"); return EXIT_FAILURE; }
        fclose(fout);
        printf("[APLZ] Empty archive → %s (0 bytes)\n", out_path);
        return EXIT_SUCCESS;
    }

    std::vector<uint64_t> chunk_offsets(num_chunks);
    fread(chunk_offsets.data(), 8, num_chunks, f);

    printf("[APLZ] Decompress: %s (v%u)\n", in_path, hdr.version);
    printf("[APLZ] Original  : %llu bytes (%.2f MB)\n",
           (unsigned long long)original_size, original_size / (1024.0 * 1024.0));
    printf("[APLZ] Chunks    : %u\n", num_chunks);

    // Metal セットアップ
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) { fprintf(stderr, "[APLZ] Metal unavailable.\n"); fclose(f); return EXIT_FAILURE; }
    printf("[APLZ] Device    : %s\n", dev.name.UTF8String);
    id<MTLCommandQueue> cq = [dev newCommandQueue];
    id<MTLLibrary> lib = compile_shader(dev, shader_path);

    NSError* err = nil;
    id<MTLFunction> fn_tans = [lib newFunctionWithName:@"tans_decode"];
    id<MTLFunction> fn_lz77 = [lib newFunctionWithName:@"lz77_decode"];
    if (!fn_tans || !fn_lz77) {
        fprintf(stderr, "[APLZ] Decode kernel not found.\n"); return EXIT_FAILURE;
    }
    id<MTLComputePipelineState> pso_tans = [dev newComputePipelineStateWithFunction:fn_tans error:&err];
    if (!pso_tans) {
        fprintf(stderr, "[APLZ] PSO tans_decode: %s\n", err.localizedDescription.UTF8String);
        return EXIT_FAILURE;
    }
    id<MTLComputePipelineState> pso_lz77 = [dev newComputePipelineStateWithFunction:fn_lz77 error:&err];
    if (!pso_lz77) {
        fprintf(stderr, "[APLZ] PSO lz77_decode: %s\n", err.localizedDescription.UTF8String);
        return EXIT_FAILURE;
    }

    // 出力バッファ (メガバッチサイズ)
    const uint32_t mb_chunks = std::min((uint32_t)MEGA_BATCH_CHUNKS, num_chunks);
    const size_t mb_out_size = (size_t)mb_chunks * CHUNK_SIZE;
    id<MTLBuffer> buf_out = [dev newBufferWithLength:mb_out_size
                                             options:MTLResourceStorageModeShared];
    if (!buf_out) {
        fprintf(stderr, "[APLZ] Output buffer alloc failed.\n"); return EXIT_FAILURE;
    }

    // ダブルバッファ スロット
    id<MTLBuffer> slot_bs[N_BUFS], slot_bsz[N_BUFS], slot_tcnt[N_BUFS], slot_tok[N_BUFS];
    id<MTLBuffer> slot_dec[N_BUFS], slot_anserr[N_BUFS], slot_lzerr[N_BUFS];
    for (uint32_t s = 0; s < N_BUFS; s++) {
        slot_bs[s]   = [dev newBufferWithLength:(size_t)BATCH_CHUNKS * N_STREAMS * BS_CAP
                                        options:MTLResourceStorageModeShared];
        slot_bsz[s]  = [dev newBufferWithLength:(size_t)BATCH_CHUNKS * N_STREAMS * sizeof(uint32_t)
                                         options:MTLResourceStorageModeShared];
        slot_tcnt[s] = [dev newBufferWithLength:(size_t)BATCH_CHUNKS * sizeof(uint32_t)
                                         options:MTLResourceStorageModeShared];
        slot_tok[s]  = [dev newBufferWithLength:(size_t)BATCH_CHUNKS * CHUNK_SIZE * sizeof(LzToken)
                                         options:MTLResourceStorageModeShared];
        slot_dec[s]  = [dev newBufferWithLength:(size_t)BATCH_CHUNKS * ANS_L * sizeof(DecodeEntry)
                                         options:MTLResourceStorageModeShared];
        slot_anserr[s] = [dev newBufferWithLength:(size_t)BATCH_CHUNKS * sizeof(uint32_t)
                                           options:MTLResourceStorageModeShared];
        slot_lzerr[s] = [dev newBufferWithLength:(size_t)BATCH_CHUNKS * sizeof(uint32_t)
                                          options:MTLResourceStorageModeShared];
    }

    FILE* fout = fopen(out_path, "wb");
    if (!fout) { perror("fopen"); return EXIT_FAILURE; }

    dispatch_semaphore_t buf_sem = dispatch_semaphore_create(N_BUFS);
    dispatch_group_t done_grp = dispatch_group_create();
    __block bool decode_failed = false;

    uint32_t n_mega = (num_chunks + mb_chunks - 1) / mb_chunks;
    uint32_t total_batches = (num_chunks + BATCH_CHUNKS - 1) / BATCH_CHUNKS;
    printf("[APLZ] Pipeline  : %u batches x %u chunks/batch (double-buffered, %u mega-batches)\n",
           total_batches, BATCH_CHUNKS, n_mega);

    CFTimeInterval t_total_start = CACurrentMediaTime();

    SymInfo chunk_si[N_SYMBOLS];
    uint16_t spread[ANS_L];

    for (uint32_t mb = 0; mb < n_mega; mb++) {
        uint32_t mc_start = mb * mb_chunks;
        uint32_t mc_count = std::min(mb_chunks, num_chunks - mc_start);

        uint32_t mb_n_batches = (mc_count + BATCH_CHUNKS - 1) / BATCH_CHUNKS;

        for (uint32_t b = 0; b < mb_n_batches; b++) {
            if (decode_failed) break;
            dispatch_semaphore_wait(buf_sem, DISPATCH_TIME_FOREVER);

            uint32_t sl = b % N_BUFS;
            uint32_t lc_start = b * BATCH_CHUNKS;
            uint32_t lc_count = std::min(BATCH_CHUNKS, mc_count - lc_start);
            uint32_t gc_start = mc_start + lc_start;

            uint8_t*  bs_data   = (uint8_t*)slot_bs[sl].contents;
            uint32_t* bsz_data  = (uint32_t*)slot_bsz[sl].contents;
            uint32_t* tcnt_data = (uint32_t*)slot_tcnt[sl].contents;
            uint32_t* anserr_data = (uint32_t*)slot_anserr[sl].contents;
            uint32_t* lzerr_data = (uint32_t*)slot_lzerr[sl].contents;
            DecodeEntry* dec_data = (DecodeEntry*)slot_dec[sl].contents;
            memset(bsz_data, 0, (size_t)lc_count * N_STREAMS * sizeof(uint32_t));
            memset(tcnt_data, 0, (size_t)lc_count * sizeof(uint32_t));
            memset(anserr_data, 0, (size_t)lc_count * sizeof(uint32_t));
            memset(lzerr_data, 0, (size_t)lc_count * sizeof(uint32_t));

            for (uint32_t lc = 0; lc < lc_count; lc++) {
                uint32_t gc = gc_start + lc;
                fseek(f, (long)chunk_offsets[gc], SEEK_SET);
                fread(&tcnt_data[lc], 4, 1, f);

                // per-chunk 頻度テーブル読み込み + デコードテーブル構築
                if (hdr.version >= 3) {
                    read_compact_freq(f, chunk_si);
                } else {
                    // v2 互換: グローバルテーブルは既に読み飛ばし済み
                    // (v2ファイルはこのコードパスに来ないはず)
                }

                build_spread_table(chunk_si, spread);
                build_decode_table(chunk_si, spread, dec_data + (size_t)lc * ANS_L);

                uint16_t ssz[256];
                fread(ssz, 2, 256, f);
                size_t stream_total = 0;
                for (uint32_t s = 0; s < 256; s++) stream_total += ssz[s];
                std::vector<uint8_t> stream_buf(stream_total);
                fread(stream_buf.data(), 1, stream_total, f);
                const uint8_t* src = stream_buf.data();
                for (uint32_t s = 0; s < 256; s++) {
                    bsz_data[(size_t)lc * 256 + s] = (uint32_t)ssz[s];
                    memcpy(bs_data + ((size_t)lc * 256 + s) * BS_CAP, src, ssz[s]);
                    src += ssz[s];
                }
            }

            uint32_t local_out_offset = lc_start * CHUNK_SIZE;
            uint32_t batch_bytes = (uint32_t)std::min(
                (uint64_t)lc_count * CHUNK_SIZE,
                original_size - (uint64_t)gc_start * CHUNK_SIZE);

            id<MTLCommandBuffer> cb = [cq commandBuffer];

            {
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso_tans];
                [enc setBuffer:slot_bs[sl]   offset:0 atIndex:0];
                [enc setBuffer:slot_bsz[sl]  offset:0 atIndex:1];
                [enc setBuffer:slot_dec[sl]  offset:0 atIndex:2];
                [enc setBuffer:slot_tok[sl]  offset:0 atIndex:3];
                [enc setBuffer:slot_tcnt[sl] offset:0 atIndex:4];
                [enc setBuffer:slot_anserr[sl] offset:0 atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(lc_count, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(TG_SIZE, 1, 1)];
                [enc endEncoding];
            }

            {
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso_lz77];
                [enc setBuffer:slot_tok[sl]  offset:0 atIndex:0];
                [enc setBuffer:slot_tcnt[sl] offset:0 atIndex:1];
                [enc setBuffer:buf_out       offset:(size_t)local_out_offset atIndex:2];
                [enc setBuffer:slot_lzerr[sl] offset:0 atIndex:3];
                [enc setBytes:&batch_bytes   length:4 atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(lc_count, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(TG_SIZE, 1, 1)];
                [enc endEncoding];
            }

            dispatch_group_enter(done_grp);
            id<MTLBuffer> cap_anserr = slot_anserr[sl];
            id<MTLBuffer> cap_lzerr = slot_lzerr[sl];
            [cb addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull completed_cb) {
                if (completed_cb.error) {
                    fprintf(stderr, "[APLZ] Decode command buffer failed: %s\n",
                            completed_cb.error.localizedDescription.UTF8String);
                    dispatch_semaphore_signal(buf_sem);
                    dispatch_group_leave(done_grp);
                    return;
                }
                const uint32_t* ans_err = (const uint32_t*)cap_anserr.contents;
                const uint32_t* lz_err = (const uint32_t*)cap_lzerr.contents;
                for (uint32_t i = 0; i < lc_count; i++) {
                    if (ans_err[i] != 0u) {
                        fprintf(stderr, "[APLZ] Invalid ANS stream at chunk %u.\n", gc_start + i);
                        decode_failed = true;
                        dispatch_semaphore_signal(buf_sem);
                        dispatch_group_leave(done_grp);
                        return;
                    }
                    if (lz_err[i] != 0u) {
                        fprintf(stderr, "[APLZ] Invalid LZ token stream at chunk %u.\n", gc_start + i);
                        decode_failed = true;
                        dispatch_semaphore_signal(buf_sem);
                        dispatch_group_leave(done_grp);
                        return;
                    }
                }
                dispatch_semaphore_signal(buf_sem);
                dispatch_group_leave(done_grp);
            }];
            [cb commit];
        }

        dispatch_group_wait(done_grp, DISPATCH_TIME_FOREVER);
        if (decode_failed) {
            fclose(f);
            fclose(fout);
            unlink(out_path);
            return EXIT_FAILURE;
        }

        size_t mb_bytes = std::min(
            (size_t)mc_count * CHUNK_SIZE,
            (size_t)(original_size - (uint64_t)mc_start * CHUNK_SIZE));
        fwrite(buf_out.contents, 1, mb_bytes, fout);
    }

    fclose(f);
    fclose(fout);

    CFTimeInterval t_total_end = CACurrentMediaTime();
    double total_ms = (t_total_end - t_total_start) * 1000.0;
    double throughput = (original_size / (1024.0 * 1024.0)) / (t_total_end - t_total_start);

    printf("[APLZ] Output    : %s (%llu bytes)\n", out_path, (unsigned long long)original_size);
    printf("[APLZ] Total     : %.2f ms  (%.1f MB/s)\n", total_ms, throughput);

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
