// compression.metal — APLZ GPU Kernels: LZ77 + Per-Chunk tANS Encode/Decode
//
// カーネル構成:
//   compress_chunk  — LZ77 マッチング + greedy overlap resolution
//   tans_encode     — 256 インターリーブ ANS ストリーム並列エンコード (per-chunk table)
//   tans_decode     — 256 インターリーブ ANS ストリーム並列デコード (per-chunk table)
//   lz77_decode     — LZ77 展開 (ハイブリッド並列/シリアル)

#include <metal_stdlib>
using namespace metal;

// ─── LZ77 定数 ─────────────────────────────────────────────────────────────────
constant uint CHUNK_SIZE = 65536u;
constant uint HASH_BITS  = 11u;
constant uint HASH_SIZE  = 1u << HASH_BITS;
constant uint MAX_MATCH  = 255u;
constant uint MIN_MATCH  = 3u;
constant uint SERIAL_FALLBACK_TOKEN_THRESHOLD = 8192u;

// ─── tANS 定数 ─────────────────────────────────────────────────────────────────
constant uint ANS_LOG_L  = 10u;
constant uint ANS_L      = 1u << ANS_LOG_L;
constant uint N_SYMBOLS  = 512u;
constant uint BS_CAP     = 512u;

// ─── 構造体 ──────────────────────────────────────────────────────────────────────
struct LzToken {
    uint8_t  is_match;
    uint8_t  _pad0;
    uint16_t val;
    uint16_t dist;
    uint16_t _pad1;
};

struct SymInfo {
    uint16_t freq;
    uint16_t cum_freq;
};

struct DecodeEntry {
    uint16_t symbol;
    uint16_t num_bits;
    uint16_t new_base;
    uint16_t _pad;
};

// ─── 3-gram 乗算ハッシュ ───────────────────────────────────────────────────────
inline uint h3(uint8_t a, uint8_t b, uint8_t c) {
    return ((uint(a) << 16) | (uint(b) << 8) | uint(c)) * 2654435761u
           >> (32u - HASH_BITS);
}

// ════════════════════════════════════════════════════════════════════════════════
// Kernel 1: compress_chunk — LZ77 (変更なし)
// ════════════════════════════════════════════════════════════════════════════════
kernel void compress_chunk(
    device const uint8_t*  in_data     [[ buffer(0) ]],
    device       LzToken*  out_sparse  [[ buffer(1) ]],
    device       LzToken*  out_compact [[ buffer(2) ]],
    device       uint32_t* compact_cnt [[ buffer(3) ]],
    constant     uint32_t& total_bytes [[ buffer(4) ]],

    uint tid     [[ thread_index_in_threadgroup ]],
    uint tg_size [[ threads_per_threadgroup ]],
    uint gid     [[ threadgroup_position_in_grid ]]
) {
    threadgroup atomic_uint ht_a[HASH_SIZE];
    threadgroup atomic_uint ht_b[HASH_SIZE];

    const uint base = gid * CHUNK_SIZE;
    if (base >= total_bytes) return;
    const uint clen = min(CHUNK_SIZE, total_bytes - base);

    device LzToken* sparse  = out_sparse  + (uint64_t)gid * CHUNK_SIZE;
    device LzToken* compact = out_compact + (uint64_t)gid * CHUNK_SIZE;

    for (uint i = tid; i < HASH_SIZE; i += tg_size) {
        atomic_store_explicit(&ht_a[i], 0xFFFFFFFFu, memory_order_relaxed);
        atomic_store_explicit(&ht_b[i], 0xFFFFFFFFu, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i + 2u < clen; i += tg_size) {
        uint slot  = h3(in_data[base+i], in_data[base+i+1], in_data[base+i+2]);
        uint old_a = atomic_fetch_min_explicit(&ht_a[slot], i, memory_order_relaxed);
        if (old_a != 0xFFFFFFFFu && old_a > i)
            atomic_fetch_min_explicit(&ht_b[slot], old_a, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < clen; i += tg_size) {
        LzToken t;
        t.is_match = 0; t._pad0 = 0; t._pad1 = 0;
        t.val = in_data[base + i]; t.dist = 0;

        if (i + 2u < clen) {
            uint slot = h3(in_data[base+i], in_data[base+i+1], in_data[base+i+2]);
            uint ca = atomic_load_explicit(&ht_a[slot], memory_order_relaxed);
            uint cb = atomic_load_explicit(&ht_b[slot], memory_order_relaxed);

            uint best = 0xFFFFFFFFu;
            for (uint k = 0u; k < 2u; ++k) {
                uint c = (k == 0u) ? ca : cb;
                if (c != 0xFFFFFFFFu && c < i && (i - c) <= 65534u)
                    best = (best == 0xFFFFFFFFu) ? c : max(best, c);
            }
            if (best != 0xFFFFFFFFu) {
                uint maxl = min(MAX_MATCH, clen - i), mlen = 0u;
                while (mlen < maxl &&
                       in_data[base+best+mlen] == in_data[base+i+mlen]) ++mlen;
                if (mlen >= MIN_MATCH) {
                    t.is_match = 1;
                    t.val      = uint16_t(mlen);
                    t.dist     = uint16_t(i - best);
                }
            }
        }
        sparse[i] = t;
    }
    threadgroup_barrier(mem_flags::mem_device);

    // ── コンパクション並列化 (parallel prefix-sum) ─────────────────────────
    // アルゴリズム:
    //   Step 1 (tid==0): sparse を1回走査し、各スレッドの担当開始位置 tg_starts[t] を決定。
    //                    担当区間は compact トークン数が均等になるよう割り当てる。
    //   Step 2 (全スレッド): 各スレッドが [tg_starts[tid], tg_starts[tid+1]) を走査し、
    //                         compact トークンを sparse バッファの専用スロットに書く。
    //                         (sparse はこの時点で読み終わっており安全に再利用可能)
    //   Step 3 (tid==0): exclusive prefix-sum で書き込みオフセットを計算。
    //   Step 4 (全スレッド): sparse 一時領域 -> compact 最終バッファへコピー。
    //                         sparse と compact は別バッファなので競合なし。
    threadgroup uint tg_starts[256];    // 各 tid の sparse 走査開始インデックス
    threadgroup uint tg_local_cnt[256]; // 各 tid が生成するトークン数
    threadgroup uint tg_offsets[257];   // exclusive prefix-sum (size tg_size+1)

    // Step 1
    if (tid == 0u) {
        uint total_tok = 0u;
        uint i = 0u;
        while (i < clen) {
            total_tok++;
            i += sparse[i].is_match ? uint(sparse[i].val) : 1u;
        }
        uint toks_per = (total_tok + tg_size - 1u) / tg_size;

        uint tok_count = 0u;
        uint cur_tid = 0u;
        tg_starts[0u] = 0u;
        i = 0u;
        while (i < clen) {
            tok_count++;
            uint step = sparse[i].is_match ? uint(sparse[i].val) : 1u;
            i += step;
            if (tok_count % toks_per == 0u && cur_tid + 1u < tg_size) {
                cur_tid++;
                tg_starts[cur_tid] = i;
            }
        }
        for (uint t = cur_tid + 1u; t < tg_size; t++)
            tg_starts[t] = clen;
        compact_cnt[gid] = total_tok;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: 担当区間をコンパクション。
    // 一時書き込み先: compact[tid * max_per_thread + k]。
    // sparse と compact は別バッファなので sparse の読み取りに干渉しない。
    // tg_offsets[tid] <= tid * max_per_thread が常に成立するため、
    // Step 4 の前方コピーでスレッド間のデータ競合も起きない（後述）。
    uint my_start = tg_starts[tid];
    uint my_end   = (tid + 1u < tg_size) ? tg_starts[tid + 1u] : clen;
    const uint max_per_thread = (CHUNK_SIZE + tg_size - 1u) / tg_size;
    device LzToken* my_tmp = compact + tid * max_per_thread;

    uint local_cnt = 0u;
    {
        uint i = my_start;
        while (i < my_end && local_cnt < max_per_thread) {
            LzToken tok = sparse[i];
            my_tmp[local_cnt++] = tok;
            i += tok.is_match ? uint(tok.val) : 1u;
        }
    }
    tg_local_cnt[tid] = local_cnt;
    threadgroup_barrier(mem_flags::mem_device);

    // Step 3: exclusive prefix-sum
    if (tid == 0u) {
        tg_offsets[0u] = 0u;
        for (uint t = 0u; t < tg_size; t++)
            tg_offsets[t + 1u] = tg_offsets[t] + tg_local_cnt[t];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 4: tid==0 が compact 一時領域を前から順に最終位置へ詰める。
    // 複数スレッドが同バッファを同時に読み書きすると競合するため tid==0 でシリアル実施。
    if (tid == 0u) {
        uint total_tok = tg_offsets[tg_size];
        uint dst = 0u;
        for (uint t = 0u; t < tg_size && dst < total_tok; t++) {
            uint src = t * max_per_thread;
            uint cnt = tg_local_cnt[t];
            for (uint k = 0u; k < cnt; k++)
                compact[dst++] = compact[src + k];
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════════
// Kernel 2: tans_encode — Per-chunk tANS テーブルで並列エンコード
//
// buffer(2): sym_arr   — SymInfo[N_SYMBOLS] の配列 (チャンクごと)
// buffer(3): enc_arr   — uint16_t[ANS_L] の配列 (チャンクごと)
// 各スレッドグループが chunk_id で自分のテーブルを threadgroup メモリにロードする。
// ════════════════════════════════════════════════════════════════════════════════
kernel void tans_encode(
    device const LzToken*   tokens     [[ buffer(0) ]],
    device const uint32_t*  token_cnt  [[ buffer(1) ]],
    device const SymInfo*   sym_arr    [[ buffer(2) ]],
    device const uint16_t*  enc_arr    [[ buffer(3) ]],
    device       uint8_t*   bs_out     [[ buffer(4) ]],
    device       uint32_t*  bs_sizes   [[ buffer(5) ]],

    uint tid     [[ thread_index_in_threadgroup ]],
    uint tg_size [[ threads_per_threadgroup ]],
    uint gid     [[ threadgroup_position_in_grid ]]
) {
    // ── Per-chunk テーブルを threadgroup メモリに協調ロード ────────────────
    threadgroup SymInfo   tg_sym[N_SYMBOLS];  // 512 * 4 = 2048 bytes
    threadgroup uint16_t  tg_enc[ANS_L];      // 1024 * 2 = 2048 bytes

    device const SymInfo*  my_sym = sym_arr + (uint64_t)gid * N_SYMBOLS;
    device const uint16_t* my_enc = enc_arr + (uint64_t)gid * ANS_L;

    for (uint i = tid; i < N_SYMBOLS; i += tg_size)
        tg_sym[i] = my_sym[i];
    for (uint i = tid; i < ANS_L; i += tg_size)
        tg_enc[i] = my_enc[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── エンコード ────────────────────────────────────────────────────────
    const uint n_tok = token_cnt[gid];
    device const LzToken* ct = tokens + (uint64_t)gid * CHUNK_SIZE;
    uint state   = ANS_L;
    uint bit_buf = 0u;
    uint bit_cnt = 0u;

    device uint8_t* my_out = bs_out + ((uint64_t)gid * tg_size + tid) * BS_CAP;
    uint bp = 0u;

    for (uint i = tid; i < n_tok; i += tg_size) {
        LzToken tok = ct[i];
        uint sym = tok.is_match ? (256u + uint(tok.val)) : uint(tok.val);
        uint f   = uint(tg_sym[sym].freq);
        uint cum = uint(tg_sym[sym].cum_freq);
        if (f == 0u) continue;

        uint nb = 0u, tmp = state;
        while (tmp >= 2u * f) { tmp >>= 1u; nb++; }

        if (nb > 0u) {
            bit_buf |= ((state & ((1u << nb) - 1u)) << bit_cnt);
            bit_cnt += nb;
            while (bit_cnt >= 8u && bp < BS_CAP - 4u) {
                my_out[bp++] = uint8_t(bit_buf & 0xFFu);
                bit_buf >>= 8u; bit_cnt -= 8u;
            }
        }

        state = uint(tg_enc[cum + (state >> nb) - f]);

        if (tok.is_match) {
            bit_buf |= (uint(tok.dist) << bit_cnt);
            bit_cnt += 16u;
            while (bit_cnt >= 8u && bp < BS_CAP - 4u) {
                my_out[bp++] = uint8_t(bit_buf & 0xFFu);
                bit_buf >>= 8u; bit_cnt -= 8u;
            }
        }
    }

    // Sentinel + flush
    bit_buf |= (1u << bit_cnt);
    bit_cnt += 1u;

    while (bit_cnt > 0u && bp < BS_CAP - 2u) {
        my_out[bp++] = uint8_t(bit_buf & 0xFFu);
        bit_buf >>= 8u;
        bit_cnt = bit_cnt >= 8u ? bit_cnt - 8u : 0u;
    }

    if (bp + 2u <= BS_CAP) {
        my_out[bp++] = uint8_t(state & 0xFFu);
        my_out[bp++] = uint8_t((state >> 8u) & 0xFFu);
    }

    bs_sizes[gid * tg_size + tid] = bp;
}

// ════════════════════════════════════════════════════════════════════════════════
// Kernel 3: tans_decode — Per-chunk tANS テーブルで並列デコード
//
// buffer(2): dec_arr — DecodeEntry[ANS_L] の配列 (チャンクごと)
// 各スレッドグループが chunk_id で自分のデコードテーブルを threadgroup にロード。
// ════════════════════════════════════════════════════════════════════════════════
kernel void tans_decode(
    device const uint8_t*      bs_in       [[ buffer(0) ]],
    device const uint32_t*     bs_sizes    [[ buffer(1) ]],
    device const DecodeEntry*  dec_arr     [[ buffer(2) ]],
    device       LzToken*      out_tokens  [[ buffer(3) ]],
    device const uint32_t*     token_cnt   [[ buffer(4) ]],
    device       atomic_uint*  ans_err     [[ buffer(5) ]],

    uint tid     [[ thread_index_in_threadgroup ]],
    uint tg_size [[ threads_per_threadgroup ]],
    uint gid     [[ threadgroup_position_in_grid ]]
) {
    // ── Per-chunk デコードテーブルを threadgroup メモリにロード ────────────
    threadgroup DecodeEntry tg_dec[ANS_L];  // 1024 * 8 = 8192 bytes

    device const DecodeEntry* my_dec = dec_arr + (uint64_t)gid * ANS_L;
    for (uint i = tid; i < ANS_L; i += tg_size)
        tg_dec[i] = my_dec[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u)
        atomic_store_explicit(&ans_err[gid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── デコード ──────────────────────────────────────────────────────────
    const uint n_tok = token_cnt[gid];
    uint my_count = (tid < n_tok) ? ((n_tok - 1u - tid) / tg_size + 1u) : 0u;
    if (my_count == 0u) return;

    uint stream_idx = gid * tg_size + tid;
    uint bp = bs_sizes[stream_idx];
    device const uint8_t* my_in = bs_in + (uint64_t)stream_idx * BS_CAP;
    if (bp < 2u || bp > BS_CAP) {
        atomic_store_explicit(&ans_err[gid], 1u, memory_order_relaxed);
        return;
    }

    uint state = uint(my_in[bp - 2u]) | (uint(my_in[bp - 1u]) << 8u);
    uint data_bytes = bp - 2u;
    if (state < ANS_L || state >= 2u * ANS_L) {
        atomic_store_explicit(&ans_err[gid], 1u, memory_order_relaxed);
        return;
    }

    int bit_pos = -1;
    if (data_bytes > 0u) {
        uint last_byte = uint(my_in[data_bytes - 1u]);
        uint sentinel_pos = (last_byte > 0u) ? (31u - clz(last_byte)) : 0u;
        bit_pos = int((data_bytes - 1u) * 8u + sentinel_pos) - 1;
    }

    device LzToken* chunk_out = out_tokens + (uint64_t)gid * CHUNK_SIZE;

    for (int t = int(my_count) - 1; t >= 0; t--) {
        if (state < ANS_L || state >= 2u * ANS_L) {
            atomic_store_explicit(&ans_err[gid], 1u, memory_order_relaxed);
            return;
        }
        uint idx = state - ANS_L;
        DecodeEntry e = tg_dec[idx];

        LzToken tok;
        tok._pad0 = 0; tok._pad1 = 0;

        if (e.symbol >= 256u) {
            tok.is_match = 1;
            tok.val = uint16_t(e.symbol - 256u);
            // 距離フィールド 16 ビットをバッチ取得。
            // エンコード: bit_buf |= (dist << bit_cnt) → dist LSB が小ビット位置に書かれる。
            // デコード逆走: bit_pos は最後に書かれた dist MSB 位置を指す。
            // [bit_pos-15 .. bit_pos] の 16 ビットを取り出すと raw16 = dist そのまま (反転不要)。
            if (bit_pos < 15) {
                atomic_store_explicit(&ans_err[gid], 1u, memory_order_relaxed);
                return;
            }
            uint lo16 = uint(bit_pos) - 15u;    // dist の LSB 側ビット位置
            uint byte_lo = lo16 >> 3u;
            uint bit_lo  = lo16 & 7u;
            // 最大 3 バイトのウィンドウで 16 ビットを取り出す
            uint w = uint(my_in[byte_lo]);
            if (byte_lo + 1u < data_bytes) w |= uint(my_in[byte_lo + 1u]) << 8u;
            if (byte_lo + 2u < data_bytes) w |= uint(my_in[byte_lo + 2u]) << 16u;
            uint raw16 = (w >> bit_lo) & 0xFFFFu;
            tok.dist = uint16_t(raw16);
            bit_pos -= 16;
        } else {
            tok.is_match = 0;
            tok.val = uint16_t(e.symbol);
            tok.dist = 0;
        }

        uint nb = uint(e.num_bits);
        uint bits = 0u;
        if (nb > 0u) {
            // ANS num_bits は最大 ANS_LOG_L = 10 ビット。
            // エンコード: bit_buf |= (state_lower_bits << bit_cnt) → LSB が小ビット位置
            // [bit_pos-nb+1 .. bit_pos] を取り出すと raw_nb = state_lower_bits そのまま (反転不要)
            if (bit_pos < int(nb) - 1) {
                atomic_store_explicit(&ans_err[gid], 1u, memory_order_relaxed);
                return;
            }
            uint lo_nb = uint(bit_pos) - nb + 1u;
            uint byte_nb = lo_nb >> 3u;
            uint bit_nb  = lo_nb & 7u;
            uint w2 = uint(my_in[byte_nb]);
            if (byte_nb + 1u < data_bytes) w2 |= uint(my_in[byte_nb + 1u]) << 8u;
            if (byte_nb + 2u < data_bytes) w2 |= uint(my_in[byte_nb + 2u]) << 16u;
            bits = (w2 >> bit_nb) & ((1u << nb) - 1u);
            bit_pos -= int(nb);
        }
        state = uint(e.new_base) + bits;

        chunk_out[tid + tg_size * uint(t)] = tok;
    }

    if (state != ANS_L || bit_pos != -1)
        atomic_store_explicit(&ans_err[gid], 1u, memory_order_relaxed);
}

// ════════════════════════════════════════════════════════════════════════════════
// Kernel 4: lz77_decode — LZ77 展開 (変更なし)
// ════════════════════════════════════════════════════════════════════════════════
kernel void lz77_decode(
    device const LzToken*   tokens      [[ buffer(0) ]],
    device const uint32_t*  token_cnt   [[ buffer(1) ]],
    device       uint8_t*   out_data    [[ buffer(2) ]],
    device       atomic_uint* lz_err    [[ buffer(3) ]],
    constant     uint32_t&  total_bytes [[ buffer(4) ]],

    uint tid     [[ thread_index_in_threadgroup ]],
    uint tg_size [[ threads_per_threadgroup ]],
    uint gid     [[ threadgroup_position_in_grid ]]
) {
    if (tid == 0u)
        atomic_store_explicit(&lz_err[gid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint base = gid * CHUNK_SIZE;
    if (base >= total_bytes) return;
    const uint clen = min(CHUNK_SIZE, total_bytes - base);
    const uint n_tok = token_cnt[gid];

    device const LzToken* ct = tokens + (uint64_t)gid * CHUNK_SIZE;
    device uint8_t* out = out_data + base;

    threadgroup uint tg_serial_mode;
    threadgroup uint tg_invalid;

    tg_serial_mode = 0u;
    tg_invalid = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        uint pos = 0u;
        uint match_cnt = 0u;
        for (uint i = 0u; i < n_tok && pos < clen; i++) {
            if (ct[i].is_match) {
                uint len = uint(ct[i].val), dist = uint(ct[i].dist);
                if (len == 0u || dist == 0u || dist > pos || pos + len > clen) {
                    tg_invalid = 1u;
                    break;
                }
                match_cnt++;
                pos += len;
            } else {
                pos += 1u;
            }
        }
        if (tg_invalid == 0u && pos != clen)
            tg_invalid = 1u;
        if (tg_invalid != 0u) {
            atomic_store_explicit(&lz_err[gid], 1u, memory_order_relaxed);
        } else if (n_tok > SERIAL_FALLBACK_TOKEN_THRESHOLD &&
                   match_cnt * 16u <= n_tok) {
            tg_serial_mode = 1u;
            pos = 0u;
            for (uint i = 0u; i < n_tok && pos < clen; i++) {
                if (ct[i].is_match) {
                    uint len = uint(ct[i].val), dist = uint(ct[i].dist);
                    for (uint j = 0u; j < len && pos < clen; j++) {
                        out[pos] = out[pos - dist];
                        pos++;
                    }
                } else {
                    out[pos++] = uint8_t(ct[i].val);
                }
            }
            if (pos != clen) {
                tg_invalid = 1u;
                atomic_store_explicit(&lz_err[gid], 1u, memory_order_relaxed);
            }
        } else {
            pos = 0u;
            for (uint i = 0u; i < n_tok; i++) {
                if (!ct[i].is_match)
                    out[pos] = uint8_t(ct[i].val);
                pos += ct[i].is_match ? uint(ct[i].val) : 1u;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_device);

    if (tg_invalid != 0u || tg_serial_mode != 0u)
        return;

    {
        uint pos = 0u;
        uint max_mw = 0u;
        for (uint i = 0u; i < n_tok; i++) {
            if (!ct[i].is_match) {
                pos += 1u;
                continue;
            }

            uint off  = pos;
            uint len  = uint(ct[i].val);
            uint dist = uint(ct[i].dist);
            if (len == 0u || dist == 0u || dist > off || off + len > clen) {
                atomic_store_explicit(&lz_err[gid], 1u, memory_order_relaxed);
                return;
            }

            if (off - dist < max_mw) {
                threadgroup_barrier(mem_flags::mem_device);
            }

            if (dist >= len) {
                for (uint j = tid; j < len; j += tg_size)
                    out[off + j] = out[off - dist + j];
            } else {
                if (tid == 0u) {
                    for (uint j = 0u; j < len; j++)
                        out[off + j] = out[off - dist + j];
                }
            }

            max_mw = max(max_mw, off + len);
            pos += len;
        }
        if (tid == 0u) {
            if (pos != clen)
                atomic_store_explicit(&lz_err[gid], 1u, memory_order_relaxed);
        }
    }
}
