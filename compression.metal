// compression.metal — APLZ GPU Kernels: LZ77 + tANS Encode/Decode
//
// カーネル構成:
//   compress_chunk  — LZ77 マッチング + greedy overlap resolution
//   tans_encode     — 256 インターリーブ ANS ストリーム並列エンコード
//   tans_decode     — 256 インターリーブ ANS ストリーム並列デコード (逆再生)
//   lz77_decode     — LZ77 展開 (ハイブリッド並列/シリアル)
//
// threadgroup メモリ使用量 (compress_chunk):
//   ht_a[2048] × 4B + ht_b[2048] × 4B = 16 KB × 2 = 32 KB (上限 32 KB ちょうど)
//   ※ HASH_BITS=12 (4096 slots) にすると sg_sums 分でオーバーするため 11 を使用

#include <metal_stdlib>
using namespace metal;

// ─── LZ77 定数 ─────────────────────────────────────────────────────────────────
constant uint CHUNK_SIZE = 65536u;          // 1 チャンク = 64 KB
constant uint HASH_BITS  = 11u;             // ハッシュテーブルビット幅
constant uint HASH_SIZE  = 1u << HASH_BITS; // 2048 スロット
constant uint MAX_MATCH  = 255u;            // 最大一致長 (val フィールドが uint8 相当)
constant uint MIN_MATCH  = 3u;              // 最小一致長 (3-gram ハッシュと対応)
constant uint SIMD_W     = 32u;             // SIMD グループ幅 (Apple GPU = 32)

// ─── tANS 定数 ─────────────────────────────────────────────────────────────────
constant uint ANS_LOG_L  = 10u;             // log2(L)
constant uint ANS_L      = 1u << ANS_LOG_L; // L = 1024 (ANS 状態空間の下限)
constant uint BS_CAP     = 512u;            // 1 ストリームあたりの最大出力バイト数

// ─── 構造体 ──────────────────────────────────────────────────────────────────────
// LzToken: LZ77 トークン (sparse / compact バッファ共通)
//   is_match=0: リテラル, val=バイト値, dist 未使用
//   is_match=1: バックリファレンス, val=一致長(≥3), dist=後方距離(1-based, ≤65534)
struct LzToken {
    uint8_t  is_match;
    uint8_t  _pad0;
    uint16_t val;   // リテラル: バイト値;  マッチ: 一致長 (≥3)
    uint16_t dist;  // マッチ: 距離 (1-based); リテラル: 0
    uint16_t _pad1;
};

// SymInfo: tANS シンボル情報 (CPU が構築して GPU Pass 2 に渡す)
//   freq の総和 = ANS_L、cum_freq は prefix sum
struct SymInfo {
    uint16_t freq;      // 正規化済み頻度 (sum = ANS_L)
    uint16_t cum_freq;  // 累積頻度
};

// ─── 3-gram 乗算ハッシュ ───────────────────────────────────────────────────────
// Knuth 乗算ハッシュ (定数 2654435761 = Fibonacci hashing) で 3 バイトを
// HASH_BITS=11 ビットのスロット番号に写像する。
inline uint h3(uint8_t a, uint8_t b, uint8_t c) {
    return ((uint(a) << 16) | (uint(b) << 8) | uint(c)) * 2654435761u
           >> (32u - HASH_BITS);
}

// ════════════════════════════════════════════════════════════════════════════════
// Kernel 1: compress_chunk — LZ77 マッチング + greedy overlap 解決
//
// 1 threadgroup = 1 chunk (= 64 KB の入力)
// 256 スレッドが 4 フェーズで協調動作する:
//
//  [Phase A-1] ハッシュテーブル初期化
//    全スレッドが ht_a / ht_b を 0xFFFFFFFF (空) で埋める。
//    threadgroup_barrier で全スレッドの書き込み完了を同期。
//
//  [Phase A-2] ハッシュテーブル構築
//    各スレッドが担当バイト位置 i の 3-gram をハッシュし、
//    atomic_fetch_min で ht_a[slot] に最小(最古)インデックスを書き込む。
//    既存の ht_a より大きい値が来た場合は ht_b にも候補として記録する。
//    → 1 スロットに最大 2 候補 (最古 2 つ) を保持する 2-way ハッシュ。
//    注意: atomic_fetch_max を使うと「最後の」インデックスが残り、
//          c < i の距離チェックが常に失敗してマッチ率 0% になる。
//
//  [Phase A-3] LZ77 マッチ探索 → sparse[]
//    各スレッドが担当バイト位置 i について:
//    - ht_a / ht_b から 2 候補を取得し、距離 ≤ 65534 の制約でフィルタ
//      ※ 65534 = uint16_t dist フィールドの実最大値 (65535 は予約なし)
//    - 最大一致長を線形スキャンで計測
//    - MIN_MATCH(3) 以上なら is_match=1 のトークンを sparse[i] に書き込む
//    最後に mem_device バリアで device memory (sparse[]) の書き込みを確定。
//
//  [Phase B] Greedy overlap 解決 + compaction (tid == 0 のみ)
//    Phase A-3 の sparse[] は各バイト位置独立にマッチを見つけるため、
//    隣接マッチが重複する場合がある。tid==0 がシリアルスキャンし、
//    マッチ長 val バイト分を一度に進んで重複を除去する。
//    結果を compact[] に詰め、有効トークン数を compact_cnt[gid] に記録。
//
//  buffer(0): in_data      device const uint8_t[]
//  buffer(1): out_sparse   device LzToken[]   (1 スロット/バイト)
//  buffer(2): out_compact  device LzToken[]   (dense output)
//  buffer(3): compact_cnt  device uint32_t[]  (有効トークン数/チャンク)
//  buffer(4): total_bytes  constant uint32_t
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
    // 2-way ハッシュテーブル: ht_a に最小(最古)、ht_b に次点インデックスを保持
    threadgroup atomic_uint ht_a[HASH_SIZE];
    threadgroup atomic_uint ht_b[HASH_SIZE];

    const uint base = gid * CHUNK_SIZE;
    if (base >= total_bytes) return;
    const uint clen = min(CHUNK_SIZE, total_bytes - base);

    device LzToken* sparse  = out_sparse  + (uint64_t)gid * CHUNK_SIZE;
    device LzToken* compact = out_compact + (uint64_t)gid * CHUNK_SIZE;

    // ── Phase A-1: ハッシュテーブル初期化 ────────────────────────────────────
    // 0xFFFFFFFF を「空」のセンティネルとして使用する。
    for (uint i = tid; i < HASH_SIZE; i += tg_size) {
        atomic_store_explicit(&ht_a[i], 0xFFFFFFFFu, memory_order_relaxed);
        atomic_store_explicit(&ht_b[i], 0xFFFFFFFFu, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase A-2: ハッシュテーブル構築 ──────────────────────────────────────
    // atomic_fetch_min で最古(最小インデックス)エントリを ht_a に保持する。
    // ht_a に上書きされた古い値は ht_b に追い出し、2 候補体制を維持する。
    for (uint i = tid; i + 2u < clen; i += tg_size) {
        uint slot  = h3(in_data[base+i], in_data[base+i+1], in_data[base+i+2]);
        uint old_a = atomic_fetch_min_explicit(&ht_a[slot], i, memory_order_relaxed);
        if (old_a != 0xFFFFFFFFu && old_a > i)
            atomic_fetch_min_explicit(&ht_b[slot], old_a, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase A-3: LZ77 マッチ探索 ───────────────────────────────────────────
    // 各スレッドが自担当バイト位置 i について独立にマッチ候補を評価する。
    // best = 最も近い(後方距離が最小)有効候補を選択する。
    for (uint i = tid; i < clen; i += tg_size) {
        LzToken t;
        t.is_match = 0; t._pad0 = 0; t._pad1 = 0;
        t.val = in_data[base + i]; t.dist = 0;

        if (i + 2u < clen) {
            uint slot = h3(in_data[base+i], in_data[base+i+1], in_data[base+i+2]);
            uint ca = atomic_load_explicit(&ht_a[slot], memory_order_relaxed);
            uint cb = atomic_load_explicit(&ht_b[slot], memory_order_relaxed);

            // 2 候補のうち距離 ≤ 65534 (uint16_t dist の実最大値) を満たす最近候補を選ぶ
            uint best = 0xFFFFFFFFu;
            for (uint k = 0u; k < 2u; ++k) {
                uint c = (k == 0u) ? ca : cb;
                if (c != 0xFFFFFFFFu && c < i && (i - c) <= 65534u)
                    best = (best == 0xFFFFFFFFu) ? c : max(best, c);  // max = より近い位置
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
    // sparse[] は device memory に書き込まれているため mem_device バリアが必要
    threadgroup_barrier(mem_flags::mem_device);

    // ── Phase B: Greedy overlap 解決 + compaction (tid 0 のみ) ───────────────
    // sparse[] はバイト位置ごとの独立マッチなのでマッチが重複し得る。
    // 例: pos=10 に len=5 のマッチがあるとき pos=11 のマッチは無効化する。
    // tid==0 がシリアルスキャンしてマッチ長分だけ i を進めることで解決する。
    // 最終的に compact[] は重複のない dense token stream になる。
    if (tid == 0u) {
        uint j = 0u, i = 0u;
        while (i < clen) {
            compact[j++] = sparse[i];
            i += sparse[i].is_match ? uint(sparse[i].val) : 1u;
        }
        compact_cnt[gid] = j;
    }
}

// ════════════════════════════════════════════════════════════════════════════════
// Kernel 2: tans_encode — 並列 tANS ビットストリーム生成
//
// 256 インターリーブストリーム: スレッド tid がトークン tid, tid+256, tid+512, ...
// を担当し、独立した ANS 状態を保持してビットストリームを出力する。
//
// tANS エンコード手順 (1 シンボル):
//   1. Renormalize: state ∈ [ANS_L, 2*ANS_L-1] を維持するため、
//      state >> nb が [f, 2f-1] に入るまで下位 nb ビットをビットバッファに出力。
//   2. 状態遷移: state = enc_table[cum[s] + (state >> nb) - f]
//      enc_table は CPU 側で build_encode_table() が構築したもの。
//      新状態は [ANS_L, 2*ANS_L-1] の範囲が保証される。
//   3. マッチトークン: ANS エンコード後に 16-bit 距離をそのまま出力
//      (距離はエントロピー符号化せず生ビットで送る)。
//
// ストリーム末尾処理:
//   - 残ったビットバッファをフラッシュ
//   - 最終 ANS 状態を 16 bit (LE) でストリーム末尾に追記
//     (デコーダがここから逆順に復元を開始する)
//
// SIMD prefix sum によるチャンク合計計算:
//   - simd_prefix_exclusive_sum(my_bytes) でグループ内のバイトオフセットを算出
//   - simd_sum(my_bytes) でグループ合計を sg_sums[] に記録
//   - tid==0 が全グループ合計を集約して chunk_comp[gid] に書き込む
//
//  buffer(0): tokens      compact token array (compress_chunk 出力)
//  buffer(1): token_cnt   チャンクごとの dense token 数
//  buffer(2): sym_info    SymInfo[512] (CPU 構築, 正規化頻度)
//  buffer(3): enc_table   uint16_t[ANS_L] (CPU 構築, 状態遷移テーブル)
//  buffer(4): bs_out      ストリーム出力バッファ (BS_CAP bytes/stream)
//  buffer(5): bs_sizes    ストリームごとの出力バイト数
//  buffer(6): chunk_comp  チャンクごとの圧縮後合計バイト数
// ════════════════════════════════════════════════════════════════════════════════
kernel void tans_encode(
    device const LzToken*   tokens     [[ buffer(0) ]],
    device const uint32_t*  token_cnt  [[ buffer(1) ]],
    constant     SymInfo*   sym_info   [[ buffer(2) ]],
    constant     uint16_t*  enc_table  [[ buffer(3) ]],
    device       uint8_t*   bs_out     [[ buffer(4) ]],
    device       uint32_t*  bs_sizes   [[ buffer(5) ]],
    device       uint32_t*  chunk_comp [[ buffer(6) ]],

    uint tid     [[ thread_index_in_threadgroup ]],
    uint tg_size [[ threads_per_threadgroup ]],
    uint sg_id   [[ simdgroup_index_in_threadgroup ]],
    uint sg_lid  [[ thread_index_in_simdgroup ]],
    uint gid     [[ threadgroup_position_in_grid ]]
) {
    const uint n_tok = token_cnt[gid];
    device const LzToken* ct = tokens + (uint64_t)gid * CHUNK_SIZE;
    const uint n_sg = tg_size / SIMD_W;  // SIMD グループ数 (256/32 = 8)
    threadgroup uint sg_sums[8];          // 各 SIMD グループの合計バイト数

    // ── スレッドごとの ANS 状態とビットバッファ ─────────────────────────────
    uint state   = ANS_L;   // 初期状態 = L (= 1024)
    uint bit_buf = 0u;      // 出力ビットバッファ (最大 32 bit 使用)
    uint bit_cnt = 0u;      // bit_buf 内の有効ビット数

    // my_out: このスレッドの出力領域 (BS_CAP bytes)
    device uint8_t* my_out = bs_out + ((uint64_t)gid * tg_size + tid) * BS_CAP;
    uint bp = 0u;           // バイト書き込み位置

    // ── エンコードループ ────────────────────────────────────────────────────
    // スレッド tid はトークン配列を 256 ストライドで処理する (インターリーブ)。
    for (uint i = tid; i < n_tok; i += tg_size) {
        LzToken tok = ct[i];
        // シンボル番号: リテラル = 0–255, マッチ長 = 256+(val-1) … 256+254
        uint sym = tok.is_match ? (256u + uint(tok.val)) : uint(tok.val);
        uint f   = uint(sym_info[sym].freq);    // 正規化頻度
        uint cum = uint(sym_info[sym].cum_freq); // 累積頻度
        if (f == 0u) continue;  // 未使用シンボルはスキップ

        // Renormalize: state >> nb が [f, 2f-1] に入るまで下位ビットを出力
        uint nb = 0u, tmp = state;
        while (tmp >= 2u * f) { tmp >>= 1u; nb++; }

        if (nb > 0u) {
            // 下位 nb ビットをビットバッファに追加
            bit_buf |= ((state & ((1u << nb) - 1u)) << bit_cnt);
            bit_cnt += nb;
            // 8 ビット以上溜まったらバイト単位で出力
            while (bit_cnt >= 8u && bp < BS_CAP - 4u) {
                my_out[bp++] = uint8_t(bit_buf & 0xFFu);
                bit_buf >>= 8u; bit_cnt -= 8u;
            }
        }

        // 状態遷移: enc_table でインデックス計算
        // (state >> nb) ∈ [f, 2f-1] → 添字 = cum + (state>>nb) - f ∈ [cum, cum+f-1]
        state = uint(enc_table[cum + (state >> nb) - f]);

        // マッチ: 16-bit 距離を生ビットで出力 (ANS エンコードなし)
        if (tok.is_match) {
            bit_buf |= (uint(tok.dist) << bit_cnt);
            bit_cnt += 16u;
            while (bit_cnt >= 8u && bp < BS_CAP - 4u) {
                my_out[bp++] = uint8_t(bit_buf & 0xFFu);
                bit_buf >>= 8u; bit_cnt -= 8u;
            }
        }
    }

    // ── Sentinel bit + 残留ビットのフラッシュ ────────────────────────────
    // Sentinel 1-bit をデータ末尾に追加する。デコーダは最終データバイトの
    // 最上位セットビットを sentinel として検出し、有効ビット範囲を特定する。
    // sentinel より上のビットは 0 パディングとなるため一意に識別可能。
    bit_buf |= (1u << bit_cnt);
    bit_cnt += 1u;

    while (bit_cnt > 0u && bp < BS_CAP - 2u) {
        my_out[bp++] = uint8_t(bit_buf & 0xFFu);
        bit_buf >>= 8u;
        bit_cnt = bit_cnt >= 8u ? bit_cnt - 8u : 0u;
    }

    // 最終 ANS 状態を 16-bit LE でストリーム末尾に書き込む。
    // デコーダはここから状態を読み取り、逆順デコードを開始する。
    if (bp + 2u <= BS_CAP) {
        my_out[bp++] = uint8_t(state & 0xFFu);
        my_out[bp++] = uint8_t((state >> 8u) & 0xFFu);
    }

    // ── SIMD prefix sum によるバイト数集計 ────────────────────────────────
    // simd_prefix_exclusive_sum: SIMD グループ内での排他的前方和
    //   → 将来の packed output 最適化でストリームをメモリ上に連続配置する際に使用
    // simd_sum: SIMD グループ全体の合計
    uint my_bytes = bp;
    uint sg_off   = simd_prefix_exclusive_sum(my_bytes);  // intra-SG オフセット
    uint sg_total = simd_sum(my_bytes);                   // SG 合計バイト数

    bs_sizes[gid * tg_size + tid] = my_bytes;

    // 各 SIMD グループの先頭スレッド (sg_lid==0) が sg_sums に合計を書き込む
    if (sg_lid == 0u) sg_sums[sg_id] = sg_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // tid==0 が全 SIMD グループの合計を集約 → チャンク圧縮バイト数
    if (tid == 0u) {
        uint total = 0u;
        for (uint s = 0u; s < n_sg; ++s) total += sg_sums[s];
        chunk_comp[gid] = total;
    }

    (void)sg_off;  // 現在は未使用 (packed output 実装時に活用予定)
}

// ─── tANS デコードテーブルエントリ ──────────────────────────────────────────────
// CPU 側で build_decode_table() が構築し、tans_decode に渡す。
// 状態 x ∈ [L, 2L-1] に対し dec_table[x - L] で O(1) ルックアップ。
struct DecodeEntry {
    uint16_t symbol;     // デコードされたシンボル (0-511)
    uint16_t num_bits;   // 次状態を復元するために読むビット数
    uint16_t new_base;   // new_state = new_base + readBits(num_bits)  ∈ [L, 2L-1]
    uint16_t _pad;
};

// ════════════════════════════════════════════════════════════════════════════════
// Kernel 3: tans_decode — 並列 tANS ビットストリームデコード (逆再生)
//
// エンコーダの逆操作。各スレッドが自分のビットストリームを末尾→先頭方向に
// 読み出し、ANS 状態遷移を逆転させてトークン列を復元する。
//
// デコード手順 (1 トークン、逆順):
//   1. 現在の状態 x ∈ [L, 2L-1] から dec_table[x - L] を参照
//      → シンボル s、読み取りビット数 nb、次状態ベース new_base
//   2. s ≥ 256 (マッチ) なら、ビットストリームから 16-bit 距離を逆方向に読む
//   3. nb ビットを逆方向に読み、new_state = new_base + bits で前の状態を復元
//   4. 復元されたトークンをインターリーブ位置 (tid + 256*t) に書き込む
//
// ストリームレイアウト:
//   [data bytes (sentinel 含む)] [final_state 16-bit LE]
//   デコーダは final_state を読み取り、最終データバイトの最上位セットビットを
//   sentinel として検出、そこから逆方向にビットを読む。
//
//  buffer(0): bs_in      ストリーム入力データ (BS_CAP bytes/stream)
//  buffer(1): bs_sizes   ストリームごとのバイト数 (state 2B + sentinel + data)
//  buffer(2): dec_table  DecodeEntry[ANS_L] (CPU 構築)
//  buffer(3): out_tokens 出力トークン配列 (compact 形式)
//  buffer(4): token_cnt  チャンクごとの dense token 数
// ════════════════════════════════════════════════════════════════════════════════
kernel void tans_decode(
    device const uint8_t*      bs_in       [[ buffer(0) ]],
    device const uint32_t*     bs_sizes    [[ buffer(1) ]],
    constant     DecodeEntry*  dec_table   [[ buffer(2) ]],
    device       LzToken*      out_tokens  [[ buffer(3) ]],
    device const uint32_t*     token_cnt   [[ buffer(4) ]],

    uint tid     [[ thread_index_in_threadgroup ]],
    uint tg_size [[ threads_per_threadgroup ]],
    uint gid     [[ threadgroup_position_in_grid ]]
) {
    const uint n_tok = token_cnt[gid];

    // このスレッドがデコードするトークン数
    // エンコーダはトークン tid, tid+256, tid+512, ... を処理したので同数を復元する
    uint my_count = (tid < n_tok) ? ((n_tok - 1u - tid) / tg_size + 1u) : 0u;
    if (my_count == 0u) return;

    uint stream_idx = gid * tg_size + tid;
    uint bp = bs_sizes[stream_idx];       // このストリームの総バイト数
    device const uint8_t* my_in = bs_in + (uint64_t)stream_idx * BS_CAP;

    // ── 最終 ANS 状態を読み取る (ストリーム末尾 2 バイト LE) ────────────
    uint state = uint(my_in[bp - 2u]) | (uint(my_in[bp - 1u]) << 8u);
    uint data_bytes = bp - 2u;

    // ── Sentinel 検出: 最終データバイトの最上位セットビット ──────────────
    // sentinel より上のビットは 0 パディングなので、最上位セットビット = sentinel
    int bit_pos = -1;
    if (data_bytes > 0u) {
        uint last_byte = uint(my_in[data_bytes - 1u]);
        // clz は 32-bit uint の先頭ゼロ数を返す。byte 値 [1,255] → 正しい位置を算出
        uint sentinel_pos = (last_byte > 0u) ? (31u - clz(last_byte)) : 0u;
        bit_pos = int((data_bytes - 1u) * 8u + sentinel_pos) - 1;
    }

    // ── 逆方向デコードループ ────────────────────────────────────────────
    // 最後にエンコードされたトークンから先頭へ向かって復元し、
    // 直接インターリーブ位置に書き込む (ローカル配列不要)
    device LzToken* chunk_out = out_tokens + (uint64_t)gid * CHUNK_SIZE;

    for (int t = int(my_count) - 1; t >= 0; t--) {
        // デコードテーブル参照
        uint idx = state - ANS_L;
        DecodeEntry e = dec_table[idx];

        LzToken tok;
        tok._pad0 = 0; tok._pad1 = 0;

        if (e.symbol >= 256u) {
            // マッチトークン: まず 16-bit 距離を逆方向に読む
            // (エンコーダが ANS bits の後に距離を書いたため、逆では距離が先)
            tok.is_match = 1;
            tok.val = uint16_t(e.symbol - 256u);
            uint dist = 0u;
            for (int b = 0; b < 16; b++) {
                uint bpi = uint(bit_pos);
                dist = (dist << 1u) | ((uint(my_in[bpi >> 3u]) >> (bpi & 7u)) & 1u);
                bit_pos--;
            }
            tok.dist = uint16_t(dist);
        } else {
            // リテラルトークン
            tok.is_match = 0;
            tok.val = uint16_t(e.symbol);
            tok.dist = 0;
        }

        // ANS 状態復元ビットを逆方向に読む
        uint nb = uint(e.num_bits);
        uint bits = 0u;
        for (uint b = 0u; b < nb; b++) {
            uint bpi = uint(bit_pos);
            bits = (bits << 1u) | ((uint(my_in[bpi >> 3u]) >> (bpi & 7u)) & 1u);
            bit_pos--;
        }
        state = uint(e.new_base) + bits;

        // インターリーブ位置に直接書き込み
        chunk_out[tid + tg_size * uint(t)] = tok;
    }
}

// ════════════════════════════════════════════════════════════════════════════════
// Kernel 4: lz77_decode — LZ77 トークン列から元データを並列展開
//
// 2つの実行パスを持つハイブリッド実装:
//
// [Serial path] n_tok > MAX_PAR_TOKENS (4096)
//   トークン数が多い = ほぼリテラル → バリアオーバーヘッドが勝るためシリアル実行
//
// [Parallel path] n_tok ≤ MAX_PAR_TOKENS
//   全 256 スレッドが協調して展開する:
//   Phase 1: 全リテラルを並列書き込み (他トークンの出力に依存しない)
//   Phase 2: マッチを1トークンずつ処理、256 スレッドが協調コピー
//     - dist ≥ len (非重複): ソースが全て過去データ → 全スレッドで並列 memcpy
//     - dist < len (重複/RLE): tid==0 がシリアル展開 (len ≤ 255 で高速)
//     Barrier 削減: max_match_written 追跡で依存なしマッチのバリアをスキップ
//
//  buffer(0): tokens      compact token 配列 (tans_decode 出力)
//  buffer(1): token_cnt   チャンクごとの token 数
//  buffer(2): out_data    展開先バッファ (元ファイルサイズ)
//  buffer(3): total_bytes 元ファイルの総バイト数
// ════════════════════════════════════════════════════════════════════════════════
constant uint MAX_PAR_TOKENS = 4096u;

kernel void lz77_decode(
    device const LzToken*   tokens      [[ buffer(0) ]],
    device const uint32_t*  token_cnt   [[ buffer(1) ]],
    device       uint8_t*   out_data    [[ buffer(2) ]],
    constant     uint32_t&  total_bytes [[ buffer(3) ]],

    uint tid     [[ thread_index_in_threadgroup ]],
    uint tg_size [[ threads_per_threadgroup ]],
    uint gid     [[ threadgroup_position_in_grid ]]
) {
    const uint base = gid * CHUNK_SIZE;
    if (base >= total_bytes) return;
    const uint clen = min(CHUNK_SIZE, total_bytes - base);
    const uint n_tok = token_cnt[gid];

    device const LzToken* ct = tokens + (uint64_t)gid * CHUNK_SIZE;
    device uint8_t* out = out_data + base;

    // ── Serial path: リテラル中心のチャンク (バリアオーバーヘッド回避) ────
    if (n_tok > MAX_PAR_TOKENS) {
        if (tid == 0u) {
            uint pos = 0u;
            for (uint i = 0u; i < n_tok && pos < clen; i++) {
                if (ct[i].is_match) {
                    uint len = uint(ct[i].val), dist = uint(ct[i].dist);
                    for (uint j = 0u; j < len && pos < clen; j++) {
                        out[pos] = out[pos - dist]; pos++;
                    }
                } else {
                    out[pos++] = uint8_t(ct[i].val);
                }
            }
        }
        return;
    }

    // ── Parallel path: 全 256 スレッドが協調展開 ─────────────────────────

    // Phase 1: 全リテラルを書き込む (マッチの出力に依存しない)
    // 全スレッドが冗長にオフセットを計算し、tid==0 がリテラルを書き込む。
    // リテラル数は少ない (高圧縮時 ~5%) ため tid==0 のみで十分。
    {
        uint pos = 0u;
        for (uint i = 0u; i < n_tok; i++) {
            if (!ct[i].is_match) {
                if (tid == 0u && pos < clen)
                    out[pos] = uint8_t(ct[i].val);
                pos += 1u;
            } else {
                pos += uint(ct[i].val);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_device);

    // Phase 2: マッチを順番に処理 (全スレッドが協調コピー)
    //
    // バリア削減最適化: max_match_written でマッチの最大書き込み位置を追跡し、
    // 後続マッチのソース範囲がその手前なら（依存なし）バリアをスキップする。
    // リテラルは Phase 1 で全て書き込み済みのためマッチ間の依存のみ考慮。
    {
        uint pos = 0u;
        uint max_mw = 0u;  // max position written by matches in Phase 2
        for (uint i = 0u; i < n_tok; i++) {
            if (!ct[i].is_match) {
                pos += 1u;
                continue;   // 全スレッドが同じ条件で skip → バリア整合性 OK
            }

            uint off  = pos;
            uint len  = uint(ct[i].val);
            uint dist = uint(ct[i].dist);

            // ソース [off-dist, off-dist+len) が Phase 2 で書き込んだ領域と重なるか
            if (off - dist < max_mw) {
                threadgroup_barrier(mem_flags::mem_device);
            }

            if (dist >= len) {
                // Non-overlapping: ソース [off-dist, off-dist+len) は全て過去データ
                // → 256 スレッドで並列コピー
                for (uint j = tid; j < len; j += tg_size) {
                    if (off + j < clen)
                        out[off + j] = out[off + j - dist];
                }
            } else {
                // Overlapping (dist < len): RLE 的パターン
                // len ≤ MAX_MATCH(255) のため tid==0 のシリアルコピーで十分高速
                if (tid == 0u) {
                    for (uint j = 0u; j < len && off + j < clen; j++)
                        out[off + j] = out[off + j - dist];
                }
            }

            pos += len;
            max_mw = pos;
        }
    }
}
