// APLZ.h — APLZ (Apple Parallel LZ) 共通ヘッダ
//
// CPU (main.mm) と GPU (compression.metal) の両方で使用する
// 構造体・定数の定義をここに集約する。
//
// ファイルフォーマット概要:
//   FileHeader (24 B)
//   n_streams  (4 B)   = 256 (最大ストリーム数)
//   ans_log_l  (4 B)   = 10
//   chunk_offsets[num_chunks] (8 B 各, seek table)
//   --- チャンクデータ (num_chunks 個) ---
//   v4/v5 per chunk:
//     chunk_kind (1 B): 0 = encoded, 1 = raw
//     if encoded:
//       chunk_n_streams (2 B)
//       token_cnt       (4 B)
//       compact_freq: n_nonzero(2B) + [sym_id(2B), freq(2B)]...
//       uint16_t stream_sizes[chunk_n_streams]
//       stream_0_data | ... | stream_(chunk_n_streams-1)_data
//     if raw:
//       raw_size (4 B)
//       raw bytes
//   v5 changes the encoded chunk bitstream to use compact distance coding:
//     short distance (<=255): 8-bit payload + 1-bit flag
//     long distance  (>255): 16-bit payload + 1-bit flag

#pragma once

#include <stdint.h>

// ─── パイプライン定数 ──────────────────────────────────────────────────────────
//  これらの値は compression.metal 内の constant 宣言と一致させること。

static const uint32_t APLZ_MAGIC_V2  = 2u;    // 旧バージョン (グローバルテーブル)
static const uint32_t APLZ_MAGIC_V3  = 3u;    // per-chunk tANS テーブル
static const uint32_t APLZ_MAGIC_V4  = 4u;    // raw chunk fallback + per-chunk stream count
static const uint32_t APLZ_MAGIC_V5  = 5u;    // v4 + compact distance coding
static const uint32_t APLZ_CHUNK_SIZE = 65536u;   // LZ77 チャンク単位 (64 KB)
static const uint32_t APLZ_TG_SIZE   = 256u;      // Metal threadgroup size
static const uint32_t APLZ_N_STREAMS = 256u;      // 並列 ANS ストリーム数 (= TG_SIZE)
static const uint32_t APLZ_ANS_LOG_L = 10u;       // tANS テーブルサイズ: L = 2^10 = 1024
static const uint32_t APLZ_ANS_L     = 1u << APLZ_ANS_LOG_L;
static const uint32_t APLZ_N_SYMBOLS = 512u;      // 0-255: リテラル, 256-511: マッチ長
static const uint32_t APLZ_BS_CAP    = 512u;      // 1ストリームあたりの最大出力バイト数
static const uint8_t  APLZ_CHUNK_ENCODED = 0u;
static const uint8_t  APLZ_CHUNK_RAW     = 1u;

// ─── LZ トークン ───────────────────────────────────────────────────────────────
//  sparse バッファ (1 スロット/入力バイト) および
//  compact バッファ (overlap 解決済み dense ストリーム) 共通レイアウト。
//  compression.metal の struct LzToken と完全一致すること。
#pragma pack(push, 1)
struct LzToken {
    uint8_t  is_match;  // 0 = リテラル, 1 = バックリファレンス
    uint8_t  _pad0;
    uint16_t val;       // リテラル: バイト値 (0-255)
                        // マッチ  : 一致長  (MIN_MATCH=3 以上)
    uint16_t dist;      // マッチ  : 後方距離 (1-based, ≤ 65534)
                        // リテラル: 0
    uint16_t _pad1;
};
#pragma pack(pop)
static_assert(sizeof(LzToken) == 8, "LzToken size mismatch");

// ─── シンボル情報 (tANS テーブル) ─────────────────────────────────────────────
//  CPU 側で正規化ヒストグラムから構築し、GPU Pass 2 に転送する。
//  compression.metal の struct SymInfo と完全一致すること。
struct SymInfo {
    uint16_t freq;      // 正規化済み頻度 (全シンボルの和 = APLZ_ANS_L)
    uint16_t cum_freq;  // 累積頻度 (prefix sum of freq)
};
static_assert(sizeof(SymInfo) == 4, "SymInfo size mismatch");

// ─── ファイルヘッダ ────────────────────────────────────────────────────────────
//  圧縮ファイル先頭 24 バイトに書き込まれる。
#pragma pack(push, 1)
struct FileHeader {
    uint8_t  magic[4];       // "APLZ" (マジックバイト)
    uint32_t version;        // 5 (APLZ_MAGIC_V5)
    uint64_t original_size;  // 元ファイルサイズ (バイト)
    uint32_t chunk_size;     // チャンクサイズ = APLZ_CHUNK_SIZE
    uint32_t num_chunks;     // チャンク数 = ceil(original_size / chunk_size)
};
#pragma pack(pop)
static_assert(sizeof(FileHeader) == 24, "FileHeader size mismatch");

// ─── tANS デコードテーブルエントリ ──────────────────────────────────────────────
//  状態 x ∈ [L, 2L-1] に対し、dec_table[x - L] から O(1) で:
//    symbol   : デコードされたシンボル (0-511)
//    num_bits : 次状態を復元するために読むビット数
//    new_base : new_state = new_base + readBits(num_bits)  ∈ [L, 2L-1]
//  compression.metal の struct DecodeEntry と完全一致すること。
struct DecodeEntry {
    uint16_t symbol;     // デコードされたシンボル (0-511)
    uint16_t num_bits;   // 次の状態を得るために読むビット数
    uint16_t new_base;   // new_state = new_base + readBits(num_bits)
    uint16_t _pad;
};
static_assert(sizeof(DecodeEntry) == 8, "DecodeEntry size mismatch");
