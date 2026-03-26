// main.mm — APLZ Host Driver (圧縮 + 解凍)
//
// ストリーミング・メガバッチ・パイプライン:
//   入力サイズに依存しない O(1) メモリ使用量で動作する。
//   MEGA_BATCH_CHUNKS (512) チャンク = 32MB 単位で処理し、
//   バッファを再利用する。
//
//   圧縮: Phase A: Pass1 メガバッチループ (ヒストグラム収集)
//          Phase B: CPU tANS テーブル構築
//          Phase C: Pass1 再実行 + Pass2 ダブルバッファ バッチパイプライン
//   解凍: メガバッチループ (読み込み → GPU decode → fwrite)

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
#include <dispatch/dispatch.h>

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
static const uint32_t MEGA_BATCH_CHUNKS = 512;   // メガバッチ: 512 * 64KB = 32MB
static const uint32_t BATCH_CHUNKS      = 32;    // Pass2 バッチあたりのチャンク数
static const uint32_t N_BUFS            = 2;     // ダブルバッファ

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

// ─── ヒストグラム集計 (CPU) ──────────────────────────────────────────────────
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

// ─── GPU Pass1 をメガバッチ分 dispatch するヘルパー ────────────────────────────
// buf_in の offset をずらし、mc_count チャンク分を処理する。
// buf_sparse/buf_compact/buf_cnt は先頭から再利用される。
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
// 圧縮モード (-c)
//
// ストリーミング・パイプライン:
//   Phase A: GPU Pass1 をメガバッチ (32MB) 単位で実行し、ヒストグラムを累積。
//            バッファは MEGA_BATCH_CHUNKS 分のみ確保。
//   Phase B: CPU histogram + tANS テーブル構築
//   Phase C: GPU Pass1 を再実行 → Pass2 ダブルバッファ バッチパイプライン。
//            sparse バッファは解放済み。compact + cnt のみ使用。
// ═══════════════════════════════════════════════════════════════════════════════
static int compress(const char* in_path, const char* out_path, const char* shader_path) {
    // ── ゼロコピー I/O ──────────────────────────────────────────────────────
    int fd = open(in_path, O_RDONLY);
    if (fd < 0) die("open");
    struct stat st;
    if (fstat(fd, &st) < 0) die("fstat");
    const size_t file_size = (size_t)st.st_size;
    if (file_size == 0) {
        fprintf(stderr, "[APLZ] Empty input.\n"); close(fd); return EXIT_FAILURE;
    }

    // mmap: 仮想アドレス空間のみ確保。物理メモリは OS がページ単位で管理。
    void* mapped = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) die("mmap");
    madvise(mapped, file_size, MADV_SEQUENTIAL);

    printf("[APLZ] Input    : %s  (%zu bytes, %.2f MB)\n",
           in_path, file_size, file_size / (1024.0 * 1024.0));

    // ── Metal セットアップ ──────────────────────────────────────────────────
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) { fprintf(stderr, "[APLZ] Metal unavailable.\n"); return EXIT_FAILURE; }
    printf("[APLZ] Device   : %s\n", dev.name.UTF8String);
    id<MTLCommandQueue> cq = [dev newCommandQueue];

    // mmap 全体を NoCopy バッファとして保持 (仮想メモリ、物理はページフォールトで遅延確保)
    id<MTLBuffer> buf_in = [dev newBufferWithBytesNoCopy:mapped
                                                  length:file_size
                                                 options:MTLResourceStorageModeShared
                                             deallocator:nil];
    if (!buf_in) { fprintf(stderr, "[APLZ] NoCopy failed.\n"); return EXIT_FAILURE; }

    const uint32_t num_chunks = (uint32_t)((file_size + CHUNK_SIZE - 1) / CHUNK_SIZE);
    printf("[APLZ] Chunks   : %u\n", num_chunks);

    // ── メガバッチサイズのバッファ確保 (O(1) メモリ) ─────────────────────────
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

    CFTimeInterval t_total_start = CACurrentMediaTime();

    // ═══ Phase A: GPU Pass1 メガバッチループ (ヒストグラム収集) ══════════════
    uint32_t n_mega = (num_chunks + mb_chunks - 1) / mb_chunks;
    uint32_t raw_freq[N_SYMBOLS] = {};
    uint64_t total_tokens = 0, total_matches = 0;

    printf("[APLZ] MegaBatch: %u mega-batches x %u chunks (%.1f MB/batch)\n",
           n_mega, mb_chunks, (double)mb_chunks * CHUNK_SIZE / (1024.0 * 1024.0));

    CFTimeInterval t1_start = CACurrentMediaTime();

    for (uint32_t mb = 0; mb < n_mega; mb++) {
        uint32_t mc_start = mb * mb_chunks;
        uint32_t mc_count = std::min(mb_chunks, num_chunks - mc_start);

        dispatch_pass1(cq, pso1, buf_in, buf_sparse, buf_compact, buf_cnt,
                        mc_start, mc_count, file_size);

        // ヒストグラム累積
        const LzToken*  compact_ptr = (const LzToken*)buf_compact.contents;
        const uint32_t* cnt_ptr     = (const uint32_t*)buf_cnt.contents;

        compute_histogram(compact_ptr, cnt_ptr, mc_count, raw_freq);

        for (uint32_t c = 0; c < mc_count; c++) {
            total_tokens += cnt_ptr[c];
            const LzToken* base = compact_ptr + (uint64_t)c * CHUNK_SIZE;
            for (uint32_t i = 0; i < cnt_ptr[c]; i++)
                if (base[i].is_match) total_matches++;
        }
    }

    CFTimeInterval t1_end = CACurrentMediaTime();
    printf("[APLZ] Pass 1a  : %.2f ms  (LZ77 + histogram, streaming)\n", (t1_end-t1_start)*1000.0);

    printf("[APLZ] Tokens   : %llu  (%llu matches, %.1f%%)\n",
           (unsigned long long)total_tokens,
           (unsigned long long)total_matches,
           100.0 * total_matches / std::max(total_tokens, (uint64_t)1));

    // ═══ Phase B: CPU tANS テーブル構築 ════════════════════════════════════
    SymInfo sym_info[N_SYMBOLS];
    normalize_histogram(raw_freq, sym_info);

    uint16_t spread[ANS_L], enc_table_cpu[ANS_L];
    build_spread_table(sym_info, spread);
    build_encode_table(sym_info, spread, enc_table_cpu);

    // sparse バッファ解放 (Phase C では不要… と思いきや Pass1 再実行に必要)
    // → Phase C でも Pass1 を走らせるので sparse は保持する

    // ═══ Phase C: Pass1 再実行 + Pass2 ダブルバッファ パイプライン ═══════════
    // 共有読み取り専用バッファ
    id<MTLBuffer> buf_sym = [dev newBufferWithLength:N_SYMBOLS * sizeof(SymInfo)
                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_enc = [dev newBufferWithLength:ANS_L * sizeof(uint16_t)
                                             options:MTLResourceStorageModeShared];
    memcpy(buf_sym.contents, sym_info, N_SYMBOLS * sizeof(SymInfo));
    memcpy(buf_enc.contents, enc_table_cpu, ANS_L * sizeof(uint16_t));

    // スロットごとのビットストリーム出力バッファ (ダブルバッファ)
    id<MTLBuffer> slot_bs[N_BUFS], slot_bsz[N_BUFS], slot_ccomp[N_BUFS];
    for (uint32_t s = 0; s < N_BUFS; s++) {
        slot_bs[s]    = [dev newBufferWithLength:(size_t)BATCH_CHUNKS * N_STREAMS * BS_CAP
                                         options:MTLResourceStorageModeShared];
        slot_bsz[s]   = [dev newBufferWithLength:(size_t)BATCH_CHUNKS * N_STREAMS * sizeof(uint32_t)
                                          options:MTLResourceStorageModeShared];
        slot_ccomp[s]  = [dev newBufferWithLength:(size_t)BATCH_CHUNKS * sizeof(uint32_t)
                                           options:MTLResourceStorageModeShared];
    }

    id<MTLFunction> fn2 = [lib newFunctionWithName:@"tans_encode"];
    if (!fn2) { fprintf(stderr, "[APLZ] 'tans_encode' not found.\n"); return EXIT_FAILURE; }
    NSError* pso2_err = nil;
    id<MTLComputePipelineState> pso2 = [dev newComputePipelineStateWithFunction:fn2 error:&pso2_err];
    if (!pso2) {
        fprintf(stderr, "[APLZ] PSO2: %s\n", pso2_err.localizedDescription.UTF8String);
        return EXIT_FAILURE;
    }

    // ── ファイルヘッダ + シークテーブルプレースホルダ書き出し ────────────────
    FILE* fout = fopen(out_path, "wb");
    if (!fout) { perror("fopen"); return EXIT_FAILURE; }

    FileHeader hdr;
    memcpy(hdr.magic, "APLZ", 4);
    hdr.version       = 2;
    hdr.original_size = (uint64_t)file_size;
    hdr.chunk_size    = CHUNK_SIZE;
    hdr.num_chunks    = num_chunks;
    fwrite(&hdr, sizeof(hdr), 1, fout);

    uint32_t ns_val = N_STREAMS, al_val = ANS_LOG_L;
    fwrite(&ns_val, 4, 1, fout);
    fwrite(&al_val, 4, 1, fout);
    fwrite(sym_info, sizeof(SymInfo), N_SYMBOLS, fout);

    long seek_tbl_pos = ftell(fout);
    std::vector<uint64_t> offsets(num_chunks, 0);
    fwrite(offsets.data(), 8, num_chunks, fout);

    // ── パイプライン同期プリミティブ ─────────────────────────────────────────
    dispatch_semaphore_t buf_sem = dispatch_semaphore_create(N_BUFS);
    dispatch_queue_t write_q = dispatch_queue_create("aplz.write", DISPATCH_QUEUE_SERIAL);
    dispatch_group_t done_grp = dispatch_group_create();

    uint64_t* offs_ptr = offsets.data();

    CFTimeInterval t2_start = CACurrentMediaTime();

    // メガバッチループ: Pass1 再実行 → Pass2 ダブルバッファ
    for (uint32_t mb = 0; mb < n_mega; mb++) {
        uint32_t mc_start = mb * mb_chunks;
        uint32_t mc_count = std::min(mb_chunks, num_chunks - mc_start);

        // GPU Pass1 再実行 (このメガバッチ分のトークンを buf_compact に再生成)
        dispatch_pass1(cq, pso1, buf_in, buf_sparse, buf_compact, buf_cnt,
                        mc_start, mc_count, file_size);

        // buf_cnt の内容をローカルにコピー (Pass2 非同期ループ中に上書きされないよう)
        std::vector<uint32_t> mb_cnt(mc_count);
        memcpy(mb_cnt.data(), buf_cnt.contents, mc_count * sizeof(uint32_t));

        // Pass2 ダブルバッファ バッチパイプライン (メガバッチ内)
        uint32_t mb_n_batches = (mc_count + BATCH_CHUNKS - 1) / BATCH_CHUNKS;
        for (uint32_t b = 0; b < mb_n_batches; b++) {
            dispatch_semaphore_wait(buf_sem, DISPATCH_TIME_FOREVER);

            uint32_t sl = b % N_BUFS;
            uint32_t lc_start = b * BATCH_CHUNKS;            // メガバッチ内ローカル
            uint32_t lc_count = std::min(BATCH_CHUNKS, mc_count - lc_start);
            uint32_t gc_start = mc_start + lc_start;          // グローバルチャンクID

            memset(slot_bs[sl].contents, 0, (size_t)lc_count * N_STREAMS * BS_CAP);

            id<MTLCommandBuffer> cb = [cq commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso2];
            [enc setBuffer:buf_compact   offset:(size_t)lc_start * CHUNK_SIZE * sizeof(LzToken) atIndex:0];
            [enc setBuffer:buf_cnt       offset:(size_t)lc_start * sizeof(uint32_t)             atIndex:1];
            [enc setBuffer:buf_sym       offset:0 atIndex:2];
            [enc setBuffer:buf_enc       offset:0 atIndex:3];
            [enc setBuffer:slot_bs[sl]   offset:0 atIndex:4];
            [enc setBuffer:slot_bsz[sl]  offset:0 atIndex:5];
            [enc setBuffer:slot_ccomp[sl] offset:0 atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake(lc_count, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(TG_SIZE, 1, 1)];
            [enc endEncoding];

            id<MTLBuffer> cap_bs  = slot_bs[sl];
            id<MTLBuffer> cap_bsz = slot_bsz[sl];
            uint32_t cap_gc_start = gc_start;
            uint32_t cap_lc_count = lc_count;
            // mb_cnt のデータをキャプチャ用にコピー
            std::vector<uint32_t> cap_cnt(mb_cnt.begin() + lc_start,
                                          mb_cnt.begin() + lc_start + lc_count);

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

                        uint16_t ssz[256];
                        for (uint32_t s = 0; s < 256; s++)
                            ssz[s] = (uint16_t)bsz[(size_t)lc * 256 + s];
                        fwrite(ssz, 2, 256, fout);

                        for (uint32_t s = 0; s < 256; s++) {
                            const uint8_t* sdata = bs + ((size_t)lc * 256 + s) * BS_CAP;
                            fwrite(sdata, 1, ssz[s], fout);
                        }
                    }

                    dispatch_semaphore_signal(buf_sem);
                    dispatch_group_leave(done_grp);
                });
            }];
            [cb commit];
        }

        // メガバッチ内の全バッチ完了を待機してから次のメガバッチへ
        // (次の Pass1 が buf_compact/buf_cnt を上書きするため)
        dispatch_group_wait(done_grp, DISPATCH_TIME_FOREVER);
    }

    CFTimeInterval t2_end = CACurrentMediaTime();
    printf("[APLZ] Pass 1b+2: %.2f ms  (LZ77 re-run + tANS encode, streaming)\n",
           (t2_end-t2_start)*1000.0);

    // ── シークテーブル書き戻し ──────────────────────────────────────────────
    fseek(fout, seek_tbl_pos, SEEK_SET);
    fwrite(offsets.data(), 8, num_chunks, fout);
    fseek(fout, 0, SEEK_END);
    uint64_t comp_size = (uint64_t)ftell(fout);
    fclose(fout);

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
// 解凍モード (-d)
//
// ストリーミング・パイプライン:
//   メガバッチ (32MB) 単位で buf_out を確保し、GPU デコード後に fwrite。
//   各メガバッチ内は既存のダブルバッファ パイプラインで処理する。
// ═══════════════════════════════════════════════════════════════════════════════
static int decompress(const char* in_path, const char* out_path, const char* shader_path) {
    // ── .aplz ファイルヘッダ読み込み ────────────────────────────────────────
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

    SymInfo sym_info[N_SYMBOLS];
    fread(sym_info, sizeof(SymInfo), N_SYMBOLS, f);

    const uint32_t num_chunks = hdr.num_chunks;
    const uint64_t original_size = hdr.original_size;

    std::vector<uint64_t> chunk_offsets(num_chunks);
    fread(chunk_offsets.data(), 8, num_chunks, f);

    printf("[APLZ] Decompress: %s\n", in_path);
    printf("[APLZ] Original  : %llu bytes (%.2f MB)\n",
           (unsigned long long)original_size, original_size / (1024.0 * 1024.0));
    printf("[APLZ] Chunks    : %u\n", num_chunks);

    // ── CPU: tANS デコードテーブル構築 ──────────────────────────────────────
    uint16_t spread[ANS_L];
    build_spread_table(sym_info, spread);

    DecodeEntry dec_table_cpu[ANS_L];
    build_decode_table(sym_info, spread, dec_table_cpu);

    // ── Metal セットアップ ──────────────────────────────────────────────────
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

    // ── 共有バッファ (全バッチ共通) ──────────────────────────────────────────
    id<MTLBuffer> buf_dec = [dev newBufferWithLength:ANS_L * sizeof(DecodeEntry)
                                             options:MTLResourceStorageModeShared];
    memcpy(buf_dec.contents, dec_table_cpu, ANS_L * sizeof(DecodeEntry));

    // ── 出力バッファ: メガバッチサイズのみ確保 (O(1) メモリ) ────────────────
    const uint32_t mb_chunks = std::min((uint32_t)MEGA_BATCH_CHUNKS, num_chunks);
    const size_t mb_out_size = (size_t)mb_chunks * CHUNK_SIZE;
    id<MTLBuffer> buf_out = [dev newBufferWithLength:mb_out_size
                                             options:MTLResourceStorageModeShared];
    if (!buf_out) {
        fprintf(stderr, "[APLZ] Output buffer alloc failed.\n"); return EXIT_FAILURE;
    }

    // ── ダブルバッファ: スロットごとの作業領域 ──────────────────────────────
    id<MTLBuffer> slot_bs[N_BUFS], slot_bsz[N_BUFS], slot_tcnt[N_BUFS], slot_tok[N_BUFS];
    for (uint32_t s = 0; s < N_BUFS; s++) {
        slot_bs[s]   = [dev newBufferWithLength:(size_t)BATCH_CHUNKS * N_STREAMS * BS_CAP
                                        options:MTLResourceStorageModeShared];
        slot_bsz[s]  = [dev newBufferWithLength:(size_t)BATCH_CHUNKS * N_STREAMS * sizeof(uint32_t)
                                         options:MTLResourceStorageModeShared];
        slot_tcnt[s] = [dev newBufferWithLength:(size_t)BATCH_CHUNKS * sizeof(uint32_t)
                                         options:MTLResourceStorageModeShared];
        slot_tok[s]  = [dev newBufferWithLength:(size_t)BATCH_CHUNKS * CHUNK_SIZE * sizeof(LzToken)
                                         options:MTLResourceStorageModeShared];
    }

    // ── 出力ファイル ────────────────────────────────────────────────────────
    FILE* fout = fopen(out_path, "wb");
    if (!fout) { perror("fopen"); return EXIT_FAILURE; }

    // ── パイプライン同期 ────────────────────────────────────────────────────
    dispatch_semaphore_t buf_sem = dispatch_semaphore_create(N_BUFS);
    dispatch_group_t done_grp = dispatch_group_create();

    uint32_t n_mega = (num_chunks + mb_chunks - 1) / mb_chunks;
    uint32_t total_batches = (num_chunks + BATCH_CHUNKS - 1) / BATCH_CHUNKS;
    printf("[APLZ] Pipeline  : %u batches x %u chunks/batch (double-buffered, %u mega-batches)\n",
           total_batches, BATCH_CHUNKS, n_mega);

    CFTimeInterval t_total_start = CACurrentMediaTime();

    for (uint32_t mb = 0; mb < n_mega; mb++) {
        uint32_t mc_start = mb * mb_chunks;
        uint32_t mc_count = std::min(mb_chunks, num_chunks - mc_start);

        uint32_t mb_n_batches = (mc_count + BATCH_CHUNKS - 1) / BATCH_CHUNKS;

        for (uint32_t b = 0; b < mb_n_batches; b++) {
            dispatch_semaphore_wait(buf_sem, DISPATCH_TIME_FOREVER);

            uint32_t sl = b % N_BUFS;
            uint32_t lc_start = b * BATCH_CHUNKS;
            uint32_t lc_count = std::min(BATCH_CHUNKS, mc_count - lc_start);
            uint32_t gc_start = mc_start + lc_start;

            // ── CPU: ファイルからバッチデータ読み込み ─────────────────────
            uint8_t*  bs_data   = (uint8_t*)slot_bs[sl].contents;
            uint32_t* bsz_data  = (uint32_t*)slot_bsz[sl].contents;
            uint32_t* tcnt_data = (uint32_t*)slot_tcnt[sl].contents;
            memset(bs_data, 0, (size_t)BATCH_CHUNKS * N_STREAMS * BS_CAP);

            for (uint32_t lc = 0; lc < lc_count; lc++) {
                uint32_t gc = gc_start + lc;
                fseek(f, (long)chunk_offsets[gc], SEEK_SET);
                fread(&tcnt_data[lc], 4, 1, f);

                uint16_t ssz[256];
                fread(ssz, 2, 256, f);
                for (uint32_t s = 0; s < 256; s++) {
                    bsz_data[(size_t)lc * 256 + s] = (uint32_t)ssz[s];
                    fread(bs_data + ((size_t)lc * 256 + s) * BS_CAP, 1, ssz[s], f);
                }
            }

            // ── GPU: tANS デコード + LZ77 展開 ──────────────────────────
            // buf_out のオフセットはメガバッチ内ローカル
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
                [enc setBuffer:buf_dec       offset:0 atIndex:2];
                [enc setBuffer:slot_tok[sl]  offset:0 atIndex:3];
                [enc setBuffer:slot_tcnt[sl] offset:0 atIndex:4];
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
                [enc setBytes:&batch_bytes   length:4 atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(lc_count, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(TG_SIZE, 1, 1)];
                [enc endEncoding];
            }

            dispatch_group_enter(done_grp);
            [cb addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull) {
                dispatch_semaphore_signal(buf_sem);
                dispatch_group_leave(done_grp);
            }];
            [cb commit];
        }

        // メガバッチ内の全バッチ完了を待機
        dispatch_group_wait(done_grp, DISPATCH_TIME_FOREVER);

        // buf_out の有効部分をファイルに書き出し
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
