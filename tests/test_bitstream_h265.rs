//! `shiguredo_mp4::bitstream::h265` の決定的テスト
//!
//! 手動構築した Annex B / length-prefixed のバイト列と SPS のビット列に対して
//! パーサーの受理・拒否条件を固定する。実エンコーダー出力による fixture テストは
//! `tests/testdata/h265-vps-sps-pps-annexb.bin` を用いた別テストで補う。

use shiguredo_mp4::{
    Decode, Encode, ErrorKind,
    bitstream::h265::{
        H265ConstantFrameRate, H265NalUnitType, H265SampleEntryConfig, LengthSize, build_hev1_box,
        build_hev1_box_from_annexb, build_hvc1_box, build_hvc1_box_from_annexb, collect_nal_units,
        parse_annexb_nal_units, parse_length_prefixed_nal_units, parse_sps,
    },
    boxes::{Hev1Box, Hvc1Box, VisualSampleEntryFields},
};

/// SPS の RBSP を組み立てる MSB-first ビットライター
#[derive(Debug, Clone, Default)]
struct BitWriter {
    bytes: Vec<u8>,
    bit_pos: u8,
}

impl BitWriter {
    fn new() -> Self {
        Self::default()
    }

    fn push_bits(&mut self, value: u64, n: u32) {
        for i in (0..n).rev() {
            let bit = ((value >> i) & 1) as u8;
            if self.bit_pos == 0 {
                self.bytes.push(0);
            }
            let last = self.bytes.len() - 1;
            self.bytes[last] |= bit << (7 - self.bit_pos);
            self.bit_pos = (self.bit_pos + 1) % 8;
        }
    }

    fn push_bit(&mut self, bit: u8) {
        self.push_bits(u64::from(bit), 1);
    }

    /// 符号なし Exp-Golomb (`ue(v)`) を書き込む
    fn push_ue(&mut self, value: u32) {
        let code_num = value + 1;
        let mut zeros = 0;
        let mut v = code_num;
        while v > 1 {
            v >>= 1;
            zeros += 1;
        }
        self.push_bits(0, zeros);
        self.push_bits(u64::from(code_num), zeros + 1);
    }

    fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

/// `seq_parameter_set_rbsp` の構築パラメタ
///
/// バリデーションはせず、渡した値をそのままビット位置に詰める。
/// sub-layer の present flag は両方とも同じ値で書く (パーサーの読み飛ばしを
/// 固定ケースで検証するため)
#[derive(Debug, Clone, Copy)]
struct SpsParams {
    general_profile_space: u8,
    general_tier_flag: bool,
    general_profile_idc: u8,
    general_profile_compatibility_flags: u32,
    general_constraint_indicator_flags: u64,
    general_level_idc: u8,
    sps_max_sub_layers_minus1: u8,
    sps_temporal_id_nesting_flag: bool,
    sub_layer_profile_present_flag: bool,
    sub_layer_level_present_flag: bool,
    chroma_format_idc: u8,
    separate_colour_plane_flag: bool,
    pic_width_in_luma_samples: u32,
    pic_height_in_luma_samples: u32,
    conformance_window_flag: bool,
    conf_win_left_offset: u32,
    conf_win_right_offset: u32,
    conf_win_top_offset: u32,
    conf_win_bottom_offset: u32,
    bit_depth_luma_minus8: u8,
    bit_depth_chroma_minus8: u8,
}

impl SpsParams {
    /// 有効値で全フィールドを初期化した最小構成 (Main / 320x240)
    fn valid() -> Self {
        Self {
            general_profile_space: 0,
            general_tier_flag: false,
            general_profile_idc: 1,
            general_profile_compatibility_flags: 0,
            general_constraint_indicator_flags: 0,
            general_level_idc: 90,
            sps_max_sub_layers_minus1: 0,
            sps_temporal_id_nesting_flag: true,
            sub_layer_profile_present_flag: false,
            sub_layer_level_present_flag: false,
            chroma_format_idc: 1,
            separate_colour_plane_flag: false,
            pic_width_in_luma_samples: 320,
            pic_height_in_luma_samples: 240,
            conformance_window_flag: false,
            conf_win_left_offset: 0,
            conf_win_right_offset: 0,
            conf_win_top_offset: 0,
            conf_win_bottom_offset: 0,
            bit_depth_luma_minus8: 0,
            bit_depth_chroma_minus8: 0,
        }
    }
}

/// NAL ヘッダー (type 33 / layer 0 / TemporalId 0) 付きの SPS EBSP を組み立てる
///
/// `bit_depth_chroma_minus8` までで打ち切るため、VUI を含む SPS は
/// [`build_sps_with_tail`] を使う。
fn build_sps(p: &SpsParams) -> Vec<u8> {
    let mut w = BitWriter::new();
    // NAL ヘッダー: forbidden_zero_bit = 0 / nal_unit_type = 33 / nuh_layer_id = 0 /
    // nuh_temporal_id_plus1 = 1 (TemporalId = 0)
    w.push_bits(0x42, 8);
    w.push_bits(0x01, 8);
    push_sps_body(&mut w, p);
    w.into_bytes()
}

/// NAL ヘッダーを除く `seq_parameter_set_data` を `bit_depth_chroma_minus8` まで書き込む
fn push_sps_body(w: &mut BitWriter, p: &SpsParams) {
    w.push_bits(0, 4); // sps_video_parameter_set_id
    w.push_bits(u64::from(p.sps_max_sub_layers_minus1), 3);
    w.push_bit(u8::from(p.sps_temporal_id_nesting_flag));
    // profile_tier_level(1, sps_max_sub_layers_minus1) (7.3.3)
    w.push_bits(u64::from(p.general_profile_space), 2);
    w.push_bit(u8::from(p.general_tier_flag));
    w.push_bits(u64::from(p.general_profile_idc), 5);
    w.push_bits(u64::from(p.general_profile_compatibility_flags), 32);
    w.push_bits(p.general_constraint_indicator_flags & 0xFFFF_FFFF_FFFF, 48);
    w.push_bits(u64::from(p.general_level_idc), 8);
    for _ in 0..p.sps_max_sub_layers_minus1 {
        w.push_bit(u8::from(p.sub_layer_profile_present_flag));
        w.push_bit(u8::from(p.sub_layer_level_present_flag));
    }
    if p.sps_max_sub_layers_minus1 > 0 {
        for _ in p.sps_max_sub_layers_minus1..8 {
            w.push_bits(0, 2); // reserved_zero_2bits
        }
    }
    for _ in 0..p.sps_max_sub_layers_minus1 {
        if p.sub_layer_profile_present_flag {
            // sub-layer profile (2 + 1 + 5 + 32 + 48 = 88 bits) を 0 で埋める。
            // push_bits は u64 シフトのため 64 ビットを超えるとパニックするので分割する
            w.push_bits(0, 64);
            w.push_bits(0, 24);
        }
        if p.sub_layer_level_present_flag {
            w.push_bits(0, 8); // sub_layer_level_idc
        }
    }
    w.push_ue(0); // sps_seq_parameter_set_id
    w.push_ue(u32::from(p.chroma_format_idc));
    if p.chroma_format_idc == 3 {
        w.push_bit(u8::from(p.separate_colour_plane_flag));
    }
    w.push_ue(p.pic_width_in_luma_samples);
    w.push_ue(p.pic_height_in_luma_samples);
    w.push_bit(u8::from(p.conformance_window_flag));
    if p.conformance_window_flag {
        w.push_ue(p.conf_win_left_offset);
        w.push_ue(p.conf_win_right_offset);
        w.push_ue(p.conf_win_top_offset);
        w.push_ue(p.conf_win_bottom_offset);
    }
    w.push_ue(u32::from(p.bit_depth_luma_minus8));
    w.push_ue(u32::from(p.bit_depth_chroma_minus8));
}

/// SPS の `bit_depth_chroma_minus8` より後ろ (VUI まで) の構築パラメタ
///
/// VUI に到達するまでに読み飛ばす構文を切り替えて、パーサーのビット位置が
/// ずれないことを確かめるために使う。
#[derive(Debug, Clone, Copy)]
struct SpsTailParams {
    /// `sps_sub_layer_ordering_info_present_flag`
    sub_layer_ordering_info_present: bool,
    /// `scaling_list_enabled_flag` と `sps_scaling_list_data_present_flag` を共に立てる
    scaling_list: bool,
    /// `pcm_enabled_flag`
    pcm: bool,
    /// `st_ref_pic_set` の構成
    st_ref_pic_sets: StRefPicSets,
    /// `num_long_term_ref_pics_sps` (0 なら `long_term_ref_pics_present_flag = 0`)
    long_term_ref_pics: u32,
    /// VUI (`None` なら `vui_parameters_present_flag = 0`)
    vui: Option<VuiParams>,
}

impl SpsTailParams {
    /// 読み飛ばし構文を最小にして VUI だけを載せる構成
    fn with_vui(vui: VuiParams) -> Self {
        Self {
            sub_layer_ordering_info_present: false,
            scaling_list: false,
            pcm: false,
            st_ref_pic_sets: StRefPicSets::None,
            long_term_ref_pics: 0,
            vui: Some(vui),
        }
    }
}

/// テストで書き込む `st_ref_pic_set` の構成
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StRefPicSets {
    /// `num_short_term_ref_pic_sets = 0`
    None,
    /// 明示的な集合 1 個 (負 2 個 / 正 1 個)
    Single,
    /// 明示的な集合 1 個 + `inter_ref_pic_set_prediction_flag == 1` の集合 2 個
    Predicted,
}

/// VUI (ITU-T H.265 E.2.1) の構築パラメタ
#[derive(Debug, Clone, Copy)]
struct VuiParams {
    /// `vui_timing_info_present_flag == 1` のときの
    /// (`vui_num_units_in_tick`, `vui_time_scale`)
    timing: Option<(u32, u32)>,
    /// `vui_hrd_parameters_present_flag` (`timing` が `Some` のときだけ意味を持つ)
    hrd: bool,
    /// `sub_pic_hrd_params_present_flag` (`hrd` が真のときだけ意味を持つ)
    sub_pic_hrd: bool,
    /// `bitstream_restriction_flag == 1` のときの `min_spatial_segmentation_idc`
    min_spatial_segmentation_idc: Option<u32>,
}

impl VuiParams {
    /// timing も bitstream restriction も持たない VUI
    fn empty() -> Self {
        Self {
            timing: None,
            hrd: false,
            sub_pic_hrd: false,
            min_spatial_segmentation_idc: None,
        }
    }
}

/// テストで書き込む `log2_max_pic_order_cnt_lsb_minus4`
///
/// `lt_ref_pic_poc_lsb_sps` の幅 (`+ 4` ビット) を決める
const TEST_LOG2_MAX_PIC_ORDER_CNT_LSB_MINUS4: u32 = 4;

/// 末尾 (VUI まで) を含む SPS を組み立てて EBSP 化する
///
/// 末尾には 32 ビットの timing 値など任意のバイト列が載るため、
/// `00 00 03` が現れても壊れないよう本文だけを EBSP 化して NAL ヘッダーに繋ぐ。
fn build_sps_with_tail(p: &SpsParams, t: &SpsTailParams) -> Vec<u8> {
    let mut w = BitWriter::new();
    push_sps_body(&mut w, p);
    push_sps_tail(&mut w, p, t);

    // NAL ヘッダー: nal_unit_type = 33 / nuh_layer_id = 0 / TemporalId = 0
    let mut nal = vec![0x42, 0x01];
    nal.extend_from_slice(&to_ebsp(&w.into_bytes()));
    nal
}

/// `bit_depth_chroma_minus8` の続きから VUI までを書き込む (7.3.2.2.1)
fn push_sps_tail(w: &mut BitWriter, p: &SpsParams, t: &SpsTailParams) {
    w.push_ue(TEST_LOG2_MAX_PIC_ORDER_CNT_LSB_MINUS4);
    w.push_bit(u8::from(t.sub_layer_ordering_info_present));
    // present なら 0..=sps_max_sub_layers_minus1、非 present なら最上位の 1 個だけ
    let ordering_info_count = if t.sub_layer_ordering_info_present {
        u32::from(p.sps_max_sub_layers_minus1) + 1
    } else {
        1
    };
    for _ in 0..ordering_info_count {
        w.push_ue(1); // sps_max_dec_pic_buffering_minus1
        w.push_ue(0); // sps_max_num_reorder_pics
        w.push_ue(0); // sps_max_latency_increase_plus1
    }

    w.push_ue(0); // log2_min_luma_coding_block_size_minus3
    w.push_ue(3); // log2_diff_max_min_luma_coding_block_size
    w.push_ue(0); // log2_min_luma_transform_block_size_minus2
    w.push_ue(3); // log2_diff_max_min_luma_transform_block_size
    w.push_ue(0); // max_transform_hierarchy_depth_inter
    w.push_ue(0); // max_transform_hierarchy_depth_intra

    w.push_bit(u8::from(t.scaling_list));
    if t.scaling_list {
        w.push_bit(1); // sps_scaling_list_data_present_flag
        push_scaling_list_data(w);
    }

    w.push_bit(0); // amp_enabled_flag
    w.push_bit(1); // sample_adaptive_offset_enabled_flag

    w.push_bit(u8::from(t.pcm));
    if t.pcm {
        w.push_bits(7, 4); // pcm_sample_bit_depth_luma_minus1
        w.push_bits(7, 4); // pcm_sample_bit_depth_chroma_minus1
        w.push_ue(0); // log2_min_pcm_luma_coding_block_size_minus3
        w.push_ue(2); // log2_diff_max_min_pcm_luma_coding_block_size
        w.push_bit(1); // pcm_loop_filter_disabled_flag
    }

    push_st_ref_pic_sets(w, t.st_ref_pic_sets);

    w.push_bit(u8::from(t.long_term_ref_pics > 0));
    if t.long_term_ref_pics > 0 {
        w.push_ue(t.long_term_ref_pics);
        for _ in 0..t.long_term_ref_pics {
            // lt_ref_pic_poc_lsb_sps は u(log2_max_pic_order_cnt_lsb_minus4 + 4)。
            // 値は読み飛ばされるため、ゼロ連続を避けて全ビット 1 にする
            let bits = TEST_LOG2_MAX_PIC_ORDER_CNT_LSB_MINUS4 + 4;
            w.push_bits((1u64 << bits) - 1, bits);
            w.push_bit(1); // used_by_curr_pic_lt_sps_flag
        }
    }

    w.push_bit(1); // sps_temporal_mvp_enabled_flag
    w.push_bit(0); // strong_intra_smoothing_enabled_flag

    match &t.vui {
        None => w.push_bit(0), // vui_parameters_present_flag
        Some(vui) => {
            w.push_bit(1);
            push_vui_parameters(w, p, vui);
        }
    }
}

/// `scaling_list_data` (7.3.4) を書き込む
///
/// 全ての行列で `scaling_list_pred_mode_flag = 1` とし、係数を `se(v) = 0`
/// (1 ビットの `1`) で埋めて、係数読み飛ばしの個数計算を検証する。
fn push_scaling_list_data(w: &mut BitWriter) {
    for size_id in 0..4u32 {
        let mut matrix_id = 0u32;
        while matrix_id < 6 {
            w.push_bit(1); // scaling_list_pred_mode_flag
            let coef_num = core::cmp::min(64u32, 1u32 << (4 + (size_id << 1)));
            if size_id > 1 {
                w.push_ue(0); // scaling_list_dc_coef_minus8 (se(v) = 0)
            }
            for _ in 0..coef_num {
                w.push_ue(0); // scaling_list_delta_coef (se(v) = 0)
            }
            matrix_id += if size_id == 3 { 3 } else { 1 };
        }
    }
}

/// `num_short_term_ref_pic_sets` と `st_ref_pic_set` 群 (7.3.7) を書き込む
///
/// [`StRefPicSets::Predicted`] は 7.4.8 の導出を通さないと後続の集合の
/// フラグ個数が決まらない構成にしてある。1 個目は負 2 個 (`DeltaPocS0` が
/// -1 / -2) と正 1 個 (`DeltaPocS1` が +1) で `NumDeltaPocs = 3`。
/// 2 個目は `deltaRps = -1` で `use_delta_flag[0] = 0` とするため、
/// 導出結果は `DeltaPocS0` が -1 / -3 の 2 個・`DeltaPocS1` が 0 個となり
/// `NumDeltaPocs = 2` に減る。よって 3 個目のフラグは 3 組になる。
fn push_st_ref_pic_sets(w: &mut BitWriter, kind: StRefPicSets) {
    match kind {
        StRefPicSets::None => w.push_ue(0),
        StRefPicSets::Single => {
            w.push_ue(1);
            push_explicit_st_ref_pic_set(w);
        }
        StRefPicSets::Predicted => {
            w.push_ue(3);
            push_explicit_st_ref_pic_set(w);

            // 2 個目: 参照は 1 個目 (NumDeltaPocs = 3) なのでフラグは 4 組
            w.push_bit(1); // inter_ref_pic_set_prediction_flag
            w.push_bit(1); // delta_rps_sign (負)
            w.push_ue(0); // abs_delta_rps_minus1 → deltaRps = -1
            w.push_bit(0); // used_by_curr_pic_flag[0]
            w.push_bit(0); // use_delta_flag[0] (この 1 個が導出から落ちる)
            w.push_bit(1); // used_by_curr_pic_flag[1]
            w.push_bit(1); // used_by_curr_pic_flag[2]
            w.push_bit(1); // used_by_curr_pic_flag[3]

            // 3 個目: 参照は 2 個目 (NumDeltaPocs = 2) なのでフラグは 3 組
            w.push_bit(1); // inter_ref_pic_set_prediction_flag
            w.push_bit(1); // delta_rps_sign (負)
            w.push_ue(0); // abs_delta_rps_minus1 → deltaRps = -1
            w.push_bit(1); // used_by_curr_pic_flag[0]
            w.push_bit(1); // used_by_curr_pic_flag[1]
            w.push_bit(1); // used_by_curr_pic_flag[2]
        }
    }
}

/// `inter_ref_pic_set_prediction_flag == 0` の `st_ref_pic_set` を 1 個書き込む
///
/// 負 2 個 (`DeltaPocS0` が -1 / -2) と正 1 個 (`DeltaPocS1` が +1) で
/// `NumDeltaPocs = 3` になる。
fn push_explicit_st_ref_pic_set(w: &mut BitWriter) {
    w.push_ue(2); // num_negative_pics
    w.push_ue(1); // num_positive_pics
    w.push_ue(0); // delta_poc_s0_minus1[0] → DeltaPocS0[0] = -1
    w.push_bit(1); // used_by_curr_pic_s0_flag[0]
    w.push_ue(0); // delta_poc_s0_minus1[1] → DeltaPocS0[1] = -2
    w.push_bit(1); // used_by_curr_pic_s0_flag[1]
    w.push_ue(0); // delta_poc_s1_minus1[0] → DeltaPocS1[0] = +1
    w.push_bit(1); // used_by_curr_pic_s1_flag[0]
}

/// `vui_parameters` (E.2.1) を書き込む
fn push_vui_parameters(w: &mut BitWriter, p: &SpsParams, vui: &VuiParams) {
    w.push_bit(0); // aspect_ratio_info_present_flag
    w.push_bit(0); // overscan_info_present_flag
    w.push_bit(0); // video_signal_type_present_flag
    w.push_bit(0); // chroma_loc_info_present_flag
    // 次の 3 つは値が読み飛ばされるだけなので、ゼロ連続を避けて 1 を書く
    w.push_bit(1); // neutral_chroma_indication_flag
    w.push_bit(1); // field_seq_flag
    w.push_bit(1); // frame_field_info_present_flag
    w.push_bit(0); // default_display_window_flag

    match vui.timing {
        None => w.push_bit(0), // vui_timing_info_present_flag
        Some((num_units_in_tick, time_scale)) => {
            w.push_bit(1);
            w.push_bits(u64::from(num_units_in_tick), 32);
            w.push_bits(u64::from(time_scale), 32);
            w.push_bit(0); // vui_poc_proportional_to_timing_flag
            w.push_bit(u8::from(vui.hrd)); // vui_hrd_parameters_present_flag
            if vui.hrd {
                push_hrd_parameters(w, p.sps_max_sub_layers_minus1, vui.sub_pic_hrd);
            }
        }
    }

    match vui.min_spatial_segmentation_idc {
        None => w.push_bit(0), // bitstream_restriction_flag
        Some(value) => {
            w.push_bit(1);
            w.push_bit(0); // tiles_fixed_structure_flag
            w.push_bit(1); // motion_vectors_over_pic_boundaries_flag
            w.push_bit(0); // restricted_ref_pic_lists_flag
            w.push_ue(value); // min_spatial_segmentation_idc
            // 以降はパーサーが読まないが、実ストリームと同じく書いておく
            w.push_ue(0); // max_bytes_per_pic_denom
            w.push_ue(0); // max_bits_per_min_cu_denom
            w.push_ue(15); // log2_max_mv_length_horizontal
            w.push_ue(15); // log2_max_mv_length_vertical
        }
    }
}

/// `hrd_parameters(1, maxNumSubLayersMinus1)` (E.2.2) を書き込む
fn push_hrd_parameters(w: &mut BitWriter, max_num_sub_layers_minus1: u8, sub_pic_hrd: bool) {
    w.push_bit(1); // nal_hrd_parameters_present_flag
    w.push_bit(1); // vcl_hrd_parameters_present_flag
    w.push_bit(u8::from(sub_pic_hrd)); // sub_pic_hrd_params_present_flag
    if sub_pic_hrd {
        w.push_bits(0xFF, 8); // tick_divisor_minus2
        w.push_bits(0x1F, 5); // du_cpb_removal_delay_increment_length_minus1
        w.push_bit(1); // sub_pic_cpb_params_in_pic_timing_sei_flag
        w.push_bits(0x1F, 5); // dpb_output_delay_du_length_minus1
    }
    w.push_bits(0x0F, 4); // bit_rate_scale
    w.push_bits(0x0F, 4); // cpb_size_scale
    if sub_pic_hrd {
        w.push_bits(0x0F, 4); // cpb_size_du_scale
    }
    w.push_bits(0x1F, 5); // initial_cpb_removal_delay_length_minus1
    w.push_bits(0x1F, 5); // au_cpb_removal_delay_length_minus1
    w.push_bits(0x1F, 5); // dpb_output_delay_length_minus1

    for i in 0..=max_num_sub_layers_minus1 {
        // 先頭の sub-layer だけ fixed_pic_rate_general_flag = 1 とし、
        // 残りは low_delay_hrd_flag 経路を通す
        let fixed_pic_rate_general = i == 0;
        w.push_bit(u8::from(fixed_pic_rate_general));
        if fixed_pic_rate_general {
            // fixed_pic_rate_within_cvs_flag は 1 と推論される
            w.push_ue(0); // elemental_duration_in_tc_minus1
            w.push_ue(1); // cpb_cnt_minus1
            push_sub_layer_hrd_parameters(w, 1, sub_pic_hrd); // nal
            push_sub_layer_hrd_parameters(w, 1, sub_pic_hrd); // vcl
        } else {
            w.push_bit(0); // fixed_pic_rate_within_cvs_flag
            w.push_bit(1); // low_delay_hrd_flag → cpb_cnt_minus1 は 0 と推論される
            push_sub_layer_hrd_parameters(w, 0, sub_pic_hrd); // nal
            push_sub_layer_hrd_parameters(w, 0, sub_pic_hrd); // vcl
        }
    }
}

/// `sub_layer_hrd_parameters(i)` (E.2.3) を書き込む
fn push_sub_layer_hrd_parameters(w: &mut BitWriter, cpb_cnt_minus1: u32, sub_pic_hrd: bool) {
    for _ in 0..=cpb_cnt_minus1 {
        w.push_ue(1); // bit_rate_value_minus1
        w.push_ue(1); // cpb_size_value_minus1
        if sub_pic_hrd {
            w.push_ue(1); // cpb_size_du_value_minus1
            w.push_ue(1); // bit_rate_du_value_minus1
        }
        w.push_bit(1); // cbr_flag
    }
}

/// 3 バイト開始コードで NAL を連結した Annex B バイト列を作る
fn annexb_with_3(nals: &[&[u8]]) -> Vec<u8> {
    let mut out = Vec::new();
    for nal in nals {
        out.extend_from_slice(&[0x00, 0x00, 0x01]);
        out.extend_from_slice(nal);
    }
    out
}

/// 4 バイト開始コードで NAL を連結した Annex B バイト列を作る
fn annexb_with_4(nals: &[&[u8]]) -> Vec<u8> {
    let mut out = Vec::new();
    for nal in nals {
        out.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
        out.extend_from_slice(nal);
    }
    out
}

/// RBSP を EBSP 化する (`00 00` の後に emulation prevention byte `0x03` を挿入する)
///
/// ITU-T H.265 7.4.2.1 の挿入規則。テスト用に `parse_sps` の入力を作る
fn to_ebsp(rbsp: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    for &byte in rbsp {
        if out.len() >= 2 && out[out.len() - 2] == 0 && out[out.len() - 1] == 0 {
            out.push(0x03);
        }
        out.push(byte);
    }
    out
}

/// 大端序の長さフィールドを付けて length-prefixed バイト列を作る
fn length_prefixed(nals: &[&[u8]], length_size: LengthSize) -> Vec<u8> {
    let mut out = Vec::new();
    for nal in nals {
        match length_size {
            LengthSize::OneByte => out.push(nal.len() as u8),
            LengthSize::TwoBytes => out.extend_from_slice(&(nal.len() as u16).to_be_bytes()),
            LengthSize::FourBytes => out.extend_from_slice(&(nal.len() as u32).to_be_bytes()),
        }
        out.extend_from_slice(nal);
    }
    out
}

/// 有効な VPS (type 32 の非空 NAL)
fn valid_vps() -> Vec<u8> {
    vec![0x40, 0x01, 0x0C, 0x01, 0xFF, 0xFF]
}

/// 有効な PPS (type 34 の非空 NAL)
fn valid_pps() -> Vec<u8> {
    vec![0x44, 0x01, 0xC1, 0x72, 0xB4, 0x62, 0x40]
}

fn default_config() -> H265SampleEntryConfig {
    H265SampleEntryConfig {
        length_size: LengthSize::FourBytes,
        avg_frame_rate: H265SampleEntryConfig::AVG_FRAME_RATE_UNSPECIFIED,
        constant_frame_rate: H265ConstantFrameRate::Unknown,
    }
}

// ===== parse_annexb_nal_units: 受理系 =====

/// 4 バイト開始コードの単一 NAL を解析できる
#[test]
fn parse_annexb_single_nal_with_4byte_start_code() {
    let nal = [0x40, 0x01, 0x0C, 0x01];
    let input = annexb_with_4(&[&nal]);
    let nals = parse_annexb_nal_units(&input).expect("4 バイト開始コードの NAL は解析成功する");
    assert_eq!(nals.len(), 1);
    assert_eq!(nals[0].nal_unit_type, H265NalUnitType::Vps);
    assert_eq!(nals[0].nuh_layer_id, 0);
    assert_eq!(nals[0].nuh_temporal_id_plus1, 1);
    assert_eq!(nals[0].data, nal);
}

/// 3 バイト開始コードの単一 NAL を解析できる
#[test]
fn parse_annexb_single_nal_with_3byte_start_code() {
    let nal = [0x26, 0x01, 0xAF];
    let input = annexb_with_3(&[&nal]);
    let nals = parse_annexb_nal_units(&input).expect("3 バイト開始コードの NAL は解析成功する");
    assert_eq!(nals.len(), 1);
    assert_eq!(nals[0].nal_unit_type, H265NalUnitType::Other(19));
    assert_eq!(nals[0].data, nal);
}

/// 3 バイトと 4 バイトの開始コードが混在しても NAL 境界を正しく走査できる
#[test]
fn parse_annexb_mixed_start_code_lengths() {
    let nal1 = [0x40, 0x01, 0x0C];
    let nal2 = [0x42, 0x01, 0x01, 0x60];
    let nal3 = [0x44, 0x01, 0xC1];
    let mut input = Vec::new();
    input.extend_from_slice(&[0x00, 0x00, 0x01]);
    input.extend_from_slice(&nal1);
    input.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
    input.extend_from_slice(&nal2);
    input.extend_from_slice(&[0x00, 0x00, 0x01]);
    input.extend_from_slice(&nal3);
    let nals = parse_annexb_nal_units(&input).expect("混在開始コードは解析成功する");
    assert_eq!(nals.len(), 3);
    assert_eq!(nals[0].nal_unit_type, H265NalUnitType::Vps);
    assert_eq!(nals[0].data, nal1);
    assert_eq!(nals[1].nal_unit_type, H265NalUnitType::Sps);
    assert_eq!(nals[1].data, nal2);
    assert_eq!(nals[2].nal_unit_type, H265NalUnitType::Pps);
    assert_eq!(nals[2].data, nal3);
}

/// 空入力は NAL ユニット 0 個の成功 (開始コード欠落とは区別する)
#[test]
fn parse_annexb_empty_input_is_success() {
    let nals = parse_annexb_nal_units(&[]).expect("空入力は 0 個の成功");
    assert!(nals.is_empty());
}

/// 先頭のゼロ詰め (leading_zero_8bits) が NAL 本体に混ざらない
#[test]
fn parse_annexb_strips_leading_zero_padding() {
    let nal = [0x40, 0x01];
    let mut input = vec![0x00, 0x00];
    input.extend_from_slice(&annexb_with_3(&[&nal]));
    let nals = parse_annexb_nal_units(&input).expect("先頭ゼロ詰めは解析成功する");
    assert_eq!(nals.len(), 1);
    assert_eq!(nals[0].data, nal);
}

/// 末尾のゼロ詰め (trailing_zero_8bits) が NAL 本体に混ざらない
#[test]
fn parse_annexb_strips_trailing_zero_padding() {
    let nal = [0x40, 0x01];
    let mut input = annexb_with_3(&[&nal]);
    input.extend_from_slice(&[0x00, 0x00, 0x00]);
    let nals = parse_annexb_nal_units(&input).expect("末尾ゼロ詰めは解析成功する");
    assert_eq!(nals.len(), 1);
    assert_eq!(nals[0].data, nal);
}

/// NAL 間のゼロ詰めが直前の NAL 本体に混ざらない
///
/// 仕様 (ITU-T H.265 Annex B B.2) では NAL 本体は後続のバイトアラインされた
/// `0x000000` / `0x000001` の直前まで。末尾ゼロは次の 3 バイト開始コードの
/// 前置きゼロなので、1 個目は `[0x40, 0x01]` になる
#[test]
fn parse_annexb_strips_trailing_zeros_between_nals() {
    let input = [
        0x00, 0x00, 0x01, 0x40, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x42, 0x01,
    ];
    let nals = parse_annexb_nal_units(&input).expect("NAL 間ゼロ詰めは解析成功する");
    assert_eq!(nals.len(), 2);
    assert_eq!(nals[0].nal_unit_type, H265NalUnitType::Vps);
    assert_eq!(nals[0].data, [0x40, 0x01]);
    assert_eq!(nals[1].nal_unit_type, H265NalUnitType::Sps);
    assert_eq!(nals[1].data, [0x42, 0x01]);
}

/// NAL 間の詰め物が全てゼロだと空 NAL として Error
#[test]
fn parse_annexb_rejects_all_zero_span_between_start_codes() {
    // 1 個目の開始コードの後にゼロ 1 バイト、続けて 4 バイト開始コード。
    // ゼロ除去後は空になるため Error (黙って読み飛ばさない)
    let input = [0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x42, 0x01];
    let err = parse_annexb_nal_units(&input).expect_err("ゼロのみの NAL 間は空 NAL のため Error");
    assert_eq!(err.kind, ErrorKind::InvalidInput);
}

/// 開始コードが連続する (間に本体が一切無い) 空 NAL は Error
#[test]
fn parse_annexb_rejects_consecutive_start_codes() {
    let input = [0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x42, 0x01];
    let err = parse_annexb_nal_units(&input).expect_err("連続する開始コードの空 NAL は Error");
    assert_eq!(err.kind, ErrorKind::InvalidInput);
}

/// この列挙が名前を付けていない nal_unit_type はエラーにせず不透明な NAL として通す
#[test]
fn parse_annexb_unknown_nal_unit_type_is_opaque() {
    // nal_unit_type = 48 (未指定 41..=63)。0x60 = 0b0110_0000
    let nal = [0x60, 0x01, 0x02];
    let input = annexb_with_3(&[&nal]);
    let nals = parse_annexb_nal_units(&input).expect("未指定 nal_unit_type は通す");
    assert_eq!(nals[0].nal_unit_type, H265NalUnitType::Other(48));
    assert_eq!(nals[0].data, nal);
}

/// `nuh_layer_id != 0` の NAL は列挙では成功する (Annex A の ignore は契約にしない)
#[test]
fn parse_annexb_non_zero_nuh_layer_id_is_accepted() {
    // nuh_layer_id = 1: byte0 bit0 = 0、byte1 bit7..3 = 00001 (tid は 1)
    let nal = [0x40, 0x09];
    let input = annexb_with_3(&[&nal]);
    let nals = parse_annexb_nal_units(&input).expect("nuh_layer_id=1 の NAL は解析成功する");
    assert_eq!(nals[0].nal_unit_type, H265NalUnitType::Vps);
    assert_eq!(nals[0].nuh_layer_id, 1);
    assert_eq!(nals[0].data, nal);
}

// ===== parse_annexb_nal_units: 拒否系 =====

/// 非空入力に開始コードが 1 つも無い場合は Error
#[test]
fn parse_annexb_rejects_no_start_code() {
    let input = [0x40, 0x01, 0x0C];
    let err = parse_annexb_nal_units(&input).expect_err("開始コード無しは Error");
    assert_eq!(err.kind, ErrorKind::InvalidInput);
}

/// 開始コードの直後に別の開始コードが来る空 NAL は Error
#[test]
fn parse_annexb_rejects_empty_nal_between_start_codes() {
    let input = annexb_with_4(&[&[0x40, 0x01], &[]]);
    let err = parse_annexb_nal_units(&input).expect_err("空 NAL は Error");
    assert_eq!(err.kind, ErrorKind::InvalidInput);
}

/// 開始コードの直後に入力終端が来る空 NAL は Error
#[test]
fn parse_annexb_rejects_empty_nal_at_end() {
    let input = [0x00, 0x00, 0x00, 0x01];
    let err = parse_annexb_nal_units(&input).expect_err("末尾の空 NAL は Error");
    assert_eq!(err.kind, ErrorKind::InvalidInput);
}

/// 最初の開始コードより前に非ゼロバイトがある場合は Error
///
/// 詰め物 (ゼロ) でも NAL 本体でもないデータを黙って捨てない
#[test]
fn parse_annexb_rejects_non_zero_leading_bytes() {
    let nal = [0x40, 0x01];
    let mut input = vec![0x41];
    input.extend_from_slice(&annexb_with_3(&[&nal]));
    let err = parse_annexb_nal_units(&input).expect_err("非ゼロの先行バイトは Error");
    assert_eq!(err.kind, ErrorKind::InvalidInput);
}

/// forbidden_zero_bit が 1 の NAL は Error
#[test]
fn parse_annexb_rejects_forbidden_zero_bit() {
    // 0xC0 = 1100_0000: forbidden_zero_bit = 1 / nal_unit_type = 32
    let nal = [0xC0, 0x01];
    let input = annexb_with_3(&[&nal]);
    let err = parse_annexb_nal_units(&input).expect_err("forbidden_zero_bit=1 は Error");
    assert_eq!(err.kind, ErrorKind::InvalidInput);
}

/// nuh_temporal_id_plus1 が 0 の NAL は Error (shall not be equal to 0)
#[test]
fn parse_annexb_rejects_zero_nuh_temporal_id_plus1() {
    // byte1 = 0x00: nuh_temporal_id_plus1 = 0。末尾は非ゼロにして Annex B 走査の
    // trailing_zero_8bits 除去で NAL が切り詰められないようにする
    let nal = [0x40, 0x00, 0x01];
    let input = annexb_with_3(&[&nal]);
    let err = parse_annexb_nal_units(&input).expect_err("nuh_temporal_id_plus1=0 は Error");
    assert_eq!(err.kind, ErrorKind::InvalidInput);
}

/// nuh_layer_id が 63 の NAL は Error (0..=62 だけを受理)
#[test]
fn parse_annexb_rejects_nuh_layer_id_63() {
    // byte0 bit0 = 1 (layer の bit5)、byte1 bit7..3 = 11111 (layer の bit4..0)、tid = 1
    let nal = [0x41, 0xF9];
    let input = annexb_with_3(&[&nal]);
    let err = parse_annexb_nal_units(&input).expect_err("nuh_layer_id=63 は Error");
    assert_eq!(err.kind, ErrorKind::InvalidInput);
}

/// ヘッダー 2 バイトに満たない NAL は Error
#[test]
fn parse_annexb_rejects_short_nal_header() {
    let nal = [0x40];
    let input = annexb_with_3(&[&nal]);
    let err = parse_annexb_nal_units(&input).expect_err("ヘッダー 1 バイトの NAL は Error");
    assert_eq!(err.kind, ErrorKind::InvalidInput);
}

// ===== parse_length_prefixed_nal_units: 受理系 =====

/// 幅 1 / 2 / 4 の length-prefixed 列を解析できる
#[test]
fn parse_length_prefixed_widths_1_2_4() {
    let nal = [0x40, 0x01, 0x0C];
    for length_size in [
        LengthSize::OneByte,
        LengthSize::TwoBytes,
        LengthSize::FourBytes,
    ] {
        let input = length_prefixed(&[&nal], length_size);
        let nals = parse_length_prefixed_nal_units(&input, length_size)
            .unwrap_or_else(|_| panic!("幅 {:?} は解析成功する", length_size));
        assert_eq!(nals.len(), 1);
        assert_eq!(nals[0].nal_unit_type, H265NalUnitType::Vps);
        assert_eq!(nals[0].data, nal);
    }
}

/// 複数 NAL を入力順で返す
#[test]
fn parse_length_prefixed_multiple_nals() {
    let nal1 = [0x40, 0x01, 0x0C];
    let nal2 = [0x42, 0x01, 0x01];
    let input = length_prefixed(&[&nal1, &nal2], LengthSize::FourBytes);
    let nals = parse_length_prefixed_nal_units(&input, LengthSize::FourBytes)
        .expect("複数 NAL は解析成功する");
    assert_eq!(nals.len(), 2);
    assert_eq!(nals[0].nal_unit_type, H265NalUnitType::Vps);
    assert_eq!(nals[0].data, nal1);
    assert_eq!(nals[1].nal_unit_type, H265NalUnitType::Sps);
    assert_eq!(nals[1].data, nal2);
}

/// 空入力は NAL ユニット 0 個の成功
#[test]
fn parse_length_prefixed_empty_input_is_success() {
    let nals =
        parse_length_prefixed_nal_units(&[], LengthSize::FourBytes).expect("空入力は 0 個の成功");
    assert!(nals.is_empty());
}

// ===== parse_length_prefixed_nal_units: 拒否系 =====

/// 長さフィールドが入力末尾を超える場合は Error
#[test]
fn parse_length_prefixed_rejects_length_field_exceeding_end() {
    let input = [0x00, 0x00, 0x01];
    let err = parse_length_prefixed_nal_units(&input, LengthSize::FourBytes)
        .expect_err("長さフィールド超過は Error");
    assert_eq!(err.kind, ErrorKind::InvalidInput);
}

/// 宣言長が残バイトを超える (切り詰め) 場合は Error
#[test]
fn parse_length_prefixed_rejects_declared_length_exceeding_remaining() {
    let mut input = Vec::new();
    input.extend_from_slice(&5u32.to_be_bytes());
    input.extend_from_slice(&[0x40, 0x01]);
    let err = parse_length_prefixed_nal_units(&input, LengthSize::FourBytes)
        .expect_err("宣言長超過は Error (黙って打ち切らない)");
    assert_eq!(err.kind, ErrorKind::InvalidInput);
}

/// 宣言長が 0 の NAL は Error
#[test]
fn parse_length_prefixed_rejects_zero_length() {
    let input = [0x00, 0x00, 0x00, 0x00];
    let err = parse_length_prefixed_nal_units(&input, LengthSize::FourBytes)
        .expect_err("長さ 0 の NAL は Error");
    assert_eq!(err.kind, ErrorKind::InvalidInput);
}

/// forbidden_zero_bit が 1 の NAL は Error
#[test]
fn parse_length_prefixed_rejects_forbidden_zero_bit() {
    let nal = [0xC0, 0x01];
    let input = length_prefixed(&[&nal], LengthSize::OneByte);
    let err = parse_length_prefixed_nal_units(&input, LengthSize::OneByte)
        .expect_err("forbidden_zero_bit=1 は Error");
    assert_eq!(err.kind, ErrorKind::InvalidInput);
}

// ===== 相互変換 =====

/// Annex B → length-prefixed が幅どおりの長さフィールドを付けて変換する
#[test]
fn annexb_to_length_prefixed_adds_length_fields() {
    let nal1 = [0x40, 0x01, 0x0C];
    let nal2 = [0x42, 0x01, 0x01];
    let input = annexb_with_3(&[&nal1, &nal2]);
    let out =
        shiguredo_mp4::bitstream::h265::annexb_to_length_prefixed(&input, LengthSize::TwoBytes)
            .expect("Annex B → length-prefixed は成功する");
    let expected = length_prefixed(&[&nal1, &nal2], LengthSize::TwoBytes);
    assert_eq!(out, expected);
}

/// length-prefixed → Annex B が 4 バイト開始コードで変換する
#[test]
fn length_prefixed_to_annexb_adds_4byte_start_codes() {
    let nal1 = [0x40, 0x01, 0x0C];
    let nal2 = [0x42, 0x01, 0x01];
    let input = length_prefixed(&[&nal1, &nal2], LengthSize::OneByte);
    let out =
        shiguredo_mp4::bitstream::h265::length_prefixed_to_annexb(&input, LengthSize::OneByte)
            .expect("length-prefixed → Annex B は成功する");
    let expected = annexb_with_4(&[&nal1, &nal2]);
    assert_eq!(out, expected);
}

/// 変換 API は NAL ヘッダーを検証せず forbidden_zero_bit = 1 も通す
///
/// ヘッダー検証は parse 系 API だけの責務。変換はフレーミングのみを行う
#[test]
fn conversions_pass_through_forbidden_zero_bit() {
    // forbidden_zero_bit = 1 の NAL (0xC0 = 1100_0000 / nal_unit_type = 32)
    let nal = [0xC0, 0x01];
    let annexb = annexb_with_4(&[&nal]);
    let lp = length_prefixed(&[&nal], LengthSize::FourBytes);
    let out =
        shiguredo_mp4::bitstream::h265::annexb_to_length_prefixed(&annexb, LengthSize::FourBytes)
            .expect("Annex B → length-prefixed はヘッダー検証なしで成功する");
    assert_eq!(out, lp);
    let out = shiguredo_mp4::bitstream::h265::length_prefixed_to_annexb(&lp, LengthSize::FourBytes)
        .expect("length-prefixed → Annex B はヘッダー検証なしで成功する");
    assert_eq!(out, annexb);
}

/// NAL 本体が長さフィールド幅に収まらない場合は Error (黙った切り詰めをしない)
#[test]
fn annexb_to_length_prefixed_rejects_nal_too_long_for_width_1() {
    let nal = vec![0x40; 256];
    let input = annexb_with_3(&[&nal]);
    let err =
        shiguredo_mp4::bitstream::h265::annexb_to_length_prefixed(&input, LengthSize::OneByte)
            .expect_err("幅 1 に収まらない NAL は Error");
    assert_eq!(err.kind, ErrorKind::InvalidInput);
}

// ===== collect_nal_units =====

/// 指定した nal_unit_type の NAL だけを入力順で集める
#[test]
fn collect_nal_units_filters_by_type() {
    let input = annexb_with_3(&[&[0x40, 0x01], &[0x42, 0x01], &[0x44, 0x01], &[0x40, 0x09]]);
    let nals = parse_annexb_nal_units(&input).expect("Annex B は解析成功する");
    let vps = collect_nal_units(nals.iter().copied(), H265NalUnitType::Vps);
    assert_eq!(vps.len(), 2);
    assert_eq!(vps[0], [0x40, 0x01]);
    assert_eq!(vps[1], [0x40, 0x09]);
    let sps = collect_nal_units(nals.iter().copied(), H265NalUnitType::Sps);
    assert_eq!(sps.len(), 1);
    assert_eq!(sps[0], [0x42, 0x01]);
    // 一致が無ければ空 Vec
    let sei = collect_nal_units(nals.iter().copied(), H265NalUnitType::PrefixSei);
    assert!(sei.is_empty());
}

// ===== parse_sps: 受理系 =====

/// Main (general_profile_idc = 1) の SPS を解析できる
#[test]
fn parse_sps_main_profile() {
    let sps = parse_sps(&build_sps(&SpsParams::valid())).expect("Main SPS は解析成功する");
    assert_eq!(sps.general_profile_space, 0);
    assert_eq!(sps.general_tier_flag, 0);
    assert_eq!(sps.general_profile_idc, 1);
    assert_eq!(sps.general_profile_compatibility_flags, 0);
    assert_eq!(sps.general_constraint_indicator_flags, 0);
    assert_eq!(sps.general_level_idc, 90);
    assert_eq!(sps.sps_max_sub_layers_minus1, 0);
    assert_eq!(sps.sps_temporal_id_nesting_flag, 1);
    assert_eq!(sps.chroma_format_idc, 1);
    assert_eq!(sps.bit_depth_luma_minus8, 0);
    assert_eq!(sps.bit_depth_chroma_minus8, 0);
    assert_eq!(sps.width, 320);
    assert_eq!(sps.height, 240);
}

/// Main 10 (general_profile_idc = 2) の SPS を解析できる
#[test]
fn parse_sps_main10_profile() {
    let sps = parse_sps(&build_sps(&SpsParams {
        general_profile_idc: 2,
        bit_depth_luma_minus8: 2,
        bit_depth_chroma_minus8: 2,
        ..SpsParams::valid()
    }))
    .expect("Main 10 SPS は解析成功する");
    assert_eq!(sps.general_profile_idc, 2);
    assert_eq!(sps.bit_depth_luma_minus8, 2);
    assert_eq!(sps.bit_depth_chroma_minus8, 2);
    assert_eq!(sps.width, 320);
    assert_eq!(sps.height, 240);
}

/// プロファイル許可リスト外の general_profile_idc (例: 8) も受理する
///
/// Hisui の `H265_ALLOWED_PROFILE_IDCS` は移植しない。5 ビット値はそのまま受理する
#[test]
fn parse_sps_accepts_profile_idc_outside_hisui_allowlist() {
    let sps = parse_sps(&build_sps(&SpsParams {
        general_profile_idc: 8,
        ..SpsParams::valid()
    }))
    .expect("general_profile_idc=8 の SPS は解析成功する");
    assert_eq!(sps.general_profile_idc, 8);
    assert_eq!(sps.width, 320);
    assert_eq!(sps.height, 240);
}

/// general_profile_space / general_tier_flag / compatibility / constraint / level が反映される
#[test]
fn parse_sps_profile_tier_level_values() {
    let sps = parse_sps(&build_sps(&SpsParams {
        general_profile_space: 2,
        general_tier_flag: true,
        general_profile_idc: 2,
        general_profile_compatibility_flags: 0x6000_0001,
        general_constraint_indicator_flags: 0x1234_5678_9ABC,
        general_level_idc: 93,
        ..SpsParams::valid()
    }))
    .expect("profile_tier_level の値は解析成功する");
    assert_eq!(sps.general_profile_space, 2);
    assert_eq!(sps.general_tier_flag, 1);
    assert_eq!(sps.general_profile_idc, 2);
    assert_eq!(sps.general_profile_compatibility_flags, 0x6000_0001);
    assert_eq!(sps.general_constraint_indicator_flags, 0x1234_5678_9ABC);
    assert_eq!(sps.general_level_idc, 93);
}

/// sps_max_sub_layers_minus1 > 0 のとき reserved_zero_2bits を読み飛ばして解析できる
#[test]
fn parse_sps_with_sub_layers() {
    let sps = parse_sps(&build_sps(&SpsParams {
        sps_max_sub_layers_minus1: 2,
        ..SpsParams::valid()
    }))
    .expect("サブレイヤー宣言のある SPS は解析成功する");
    assert_eq!(sps.sps_max_sub_layers_minus1, 2);
    assert_eq!(sps.width, 320);
    assert_eq!(sps.height, 240);
}

/// sub-layer profile / level が present でも読み飛ばして解析できる
#[test]
fn parse_sps_with_sub_layer_profile_and_level() {
    let sps = parse_sps(&build_sps(&SpsParams {
        sps_max_sub_layers_minus1: 2,
        sub_layer_profile_present_flag: true,
        sub_layer_level_present_flag: true,
        ..SpsParams::valid()
    }))
    .expect("sub-layer profile / level 入りの SPS は解析成功する");
    assert_eq!(sps.width, 320);
    assert_eq!(sps.height, 240);
}

/// 4:4:4 (chroma_format_idc = 3) かつ separate_colour_plane_flag = 1 を解析できる
#[test]
fn parse_sps_chroma_444_separate_colour_plane() {
    let sps = parse_sps(&build_sps(&SpsParams {
        chroma_format_idc: 3,
        separate_colour_plane_flag: true,
        ..SpsParams::valid()
    }))
    .expect("4:4:4 separate colour plane の SPS は解析成功する");
    assert_eq!(sps.chroma_format_idc, 3);
    assert_eq!(sps.width, 320);
    assert_eq!(sps.height, 240);
}

/// クロップ無しで符号化寸法がそのまま出る
#[test]
fn parse_sps_no_crop_dimensions() {
    let sps = parse_sps(&build_sps(&SpsParams {
        pic_width_in_luma_samples: 1920,
        pic_height_in_luma_samples: 1088,
        ..SpsParams::valid()
    }))
    .expect("クロップ無し SPS は解析成功する");
    assert_eq!(sps.width, 1920);
    assert_eq!(sps.height, 1088);
}

/// クロップ後 1920x1080 になる
///
/// 符号化高さ 1088 から縦 8 ピクセルをクロップする。4:2:0 の SubHeightC = 2 なので
/// conf_win_bottom_offset = 4 が縦 8 ピクセルに相当する
#[test]
fn parse_sps_crop_to_1920x1080() {
    let sps = parse_sps(&build_sps(&SpsParams {
        pic_width_in_luma_samples: 1920,
        pic_height_in_luma_samples: 1088,
        conformance_window_flag: true,
        conf_win_bottom_offset: 4,
        ..SpsParams::valid()
    }))
    .expect("クロップ後 1920x1080 の SPS は解析成功する");
    assert_eq!(sps.width, 1920);
    assert_eq!(sps.height, 1080);
}

/// 4:2:2 (chroma_format_idc = 2) のクロップ (SubWidthC = 2 / SubHeightC = 1)
#[test]
fn parse_sps_chroma_422_crop() {
    // 幅 320 - 2 * (1 + 1) = 316、高さ 240 - 1 * (1 + 1) = 238
    let sps = parse_sps(&build_sps(&SpsParams {
        chroma_format_idc: 2,
        conformance_window_flag: true,
        conf_win_left_offset: 1,
        conf_win_right_offset: 1,
        conf_win_top_offset: 1,
        conf_win_bottom_offset: 1,
        ..SpsParams::valid()
    }))
    .expect("4:2:2 のクロップ SPS は解析成功する");
    assert_eq!(sps.chroma_format_idc, 2);
    assert_eq!(sps.width, 316);
    assert_eq!(sps.height, 238);
}

/// 4:4:4 (chroma_format_idc = 3、separate 無し) のクロップ (SubWidthC = 1 / SubHeightC = 1)
#[test]
fn parse_sps_chroma_444_crop() {
    // 幅 320 - 1 * (1 + 1) = 318、高さ 240 - 1 * (1 + 1) = 238
    let sps = parse_sps(&build_sps(&SpsParams {
        chroma_format_idc: 3,
        conformance_window_flag: true,
        conf_win_left_offset: 1,
        conf_win_right_offset: 1,
        conf_win_top_offset: 1,
        conf_win_bottom_offset: 1,
        ..SpsParams::valid()
    }))
    .expect("4:4:4 のクロップ SPS は解析成功する");
    assert_eq!(sps.chroma_format_idc, 3);
    assert_eq!(sps.width, 318);
    assert_eq!(sps.height, 238);
}

/// モノクロ (chroma_format_idc = 0) のクロップ (SubWidthC = 1 / SubHeightC = 1)
#[test]
fn parse_sps_monochrome_crop() {
    // 幅 320 - 1 * (1 + 1) = 318、高さ 240 - 1 * (1 + 1) = 238
    let sps = parse_sps(&build_sps(&SpsParams {
        chroma_format_idc: 0,
        conformance_window_flag: true,
        conf_win_left_offset: 1,
        conf_win_right_offset: 1,
        conf_win_top_offset: 1,
        conf_win_bottom_offset: 1,
        ..SpsParams::valid()
    }))
    .expect("モノクロのクロップ SPS は解析成功する");
    assert_eq!(sps.chroma_format_idc, 0);
    assert_eq!(sps.width, 318);
    assert_eq!(sps.height, 238);
}

/// RBSP の `00 00 03` が EBSP では `00 00 03 03` になり正しく復元される
///
/// emulation prevention byte の除去は入力側の位置で判定する。出力側の
/// 連続ゼロ数で判定すると 2 個目の `0x03` まで食って RBSP が壊れる。
/// profile / tier / idc を `00 00 03` にして RBSP に `00 00 03` を現させる
#[test]
fn parse_sps_restores_rbsp_with_00_00_03() {
    // general_profile_space / general_tier_flag / general_profile_idc が 0 になり、
    // general_profile_compatibility_flags の先頭が 00 00 03 になるようにする
    let rbsp = build_sps(&SpsParams {
        general_profile_compatibility_flags: 0x0000_0300,
        ..SpsParams::valid()
    });
    let ebsp = to_ebsp(&rbsp);
    assert!(
        ebsp.windows(4).any(|w| w == [0x00, 0x00, 0x03, 0x03]),
        "EBSP が `00 00 03 03` を含むこと"
    );
    let sps = parse_sps(&ebsp).expect("`00 00 03 03` 入り EBSP の SPS は解析成功する");
    assert_eq!(sps.width, 320);
    assert_eq!(sps.height, 240);
}

// ===== parse_sps: 拒否系 =====

/// NAL type が 33 以外の SPS は拒否する
#[test]
fn parse_sps_rejects_non_sps_nal_type() {
    // ヘッダーを type 32 (VPS) に書き換える
    let mut nal = build_sps(&SpsParams::valid());
    nal[0] = 0x40;
    let err = parse_sps(&nal).expect_err("type 32 の NAL は SPS として拒否される");
    assert_eq!(err.kind, ErrorKind::InvalidInput);
}

/// 空入力 (NAL ヘッダー 2 バイト未満) の SPS は拒否する
#[test]
fn parse_sps_rejects_empty_input() {
    let err = parse_sps(&[]).expect_err("空入力は SPS として拒否される");
    assert_eq!(err.kind, ErrorKind::InvalidInput);
}

/// ヘッダーの forbidden_zero_bit が 1 の SPS は拒否する
#[test]
fn parse_sps_rejects_forbidden_zero_bit() {
    let mut nal = build_sps(&SpsParams::valid());
    nal[0] = 0xC2;
    let err = parse_sps(&nal).expect_err("forbidden_zero_bit=1 の SPS は拒否される");
    assert_eq!(err.kind, ErrorKind::InvalidInput);
}

/// SPS の TemporalId が 0 以外なら拒否する (7.4.2.2)
#[test]
fn parse_sps_rejects_non_zero_temporal_id() {
    // byte1 の tid を 2 に書き換える
    let mut nal = build_sps(&SpsParams::valid());
    nal[1] = 0x02;
    let err = parse_sps(&nal).expect_err("TemporalId=1 の SPS は拒否される");
    assert_eq!(err.kind, ErrorKind::InvalidInput);
}

/// sps_max_sub_layers_minus1 > 6 は値域外として拒否する
#[test]
fn parse_sps_rejects_max_sub_layers_out_of_range() {
    let nal = build_sps(&SpsParams {
        sps_max_sub_layers_minus1: 7,
        ..SpsParams::valid()
    });
    let err = parse_sps(&nal).expect_err("sps_max_sub_layers_minus1=7 は拒否される");
    assert_eq!(err.kind, ErrorKind::InvalidInput);
}

/// sps_max_sub_layers_minus1 == 0 のとき sps_temporal_id_nesting_flag == 0 は拒否する
#[test]
fn parse_sps_rejects_temporal_id_nesting_flag_zero_without_sub_layers() {
    let nal = build_sps(&SpsParams {
        sps_temporal_id_nesting_flag: false,
        ..SpsParams::valid()
    });
    let err = parse_sps(&nal).expect_err("sps_temporal_id_nesting_flag=0 は拒否される");
    assert_eq!(err.kind, ErrorKind::InvalidInput);
}

/// chroma_format_idc > 3 は値域外として拒否する
#[test]
fn parse_sps_rejects_chroma_format_idc_out_of_range() {
    let nal = build_sps(&SpsParams {
        chroma_format_idc: 4,
        ..SpsParams::valid()
    });
    let err = parse_sps(&nal).expect_err("chroma_format_idc=4 は拒否される");
    assert_eq!(err.kind, ErrorKind::InvalidInput);
}

/// bit_depth_luma_minus8 > 7 は hvcC の 3 ビット欄に載せられないため拒否する
///
/// 仕様 (7.4.3.2.1) 上は 0..=8 だが、ISO/IEC 14496-15:2022 8.3.2.1.2 の
/// bitDepthLumaMinus8 は unsigned int(3) であり、8 (16-bit) を黙って
/// 8-bit として書き出すのを防ぐ
#[test]
fn parse_sps_rejects_bit_depth_luma_out_of_range() {
    let nal = build_sps(&SpsParams {
        bit_depth_luma_minus8: 8,
        ..SpsParams::valid()
    });
    let err = parse_sps(&nal).expect_err("bit_depth_luma_minus8=8 は拒否される");
    assert_eq!(err.kind, ErrorKind::InvalidInput);
}

/// bit_depth_chroma_minus8 > 7 は拒否する
#[test]
fn parse_sps_rejects_bit_depth_chroma_out_of_range() {
    let nal = build_sps(&SpsParams {
        bit_depth_chroma_minus8: 8,
        ..SpsParams::valid()
    });
    let err = parse_sps(&nal).expect_err("bit_depth_chroma_minus8=8 は拒否される");
    assert_eq!(err.kind, ErrorKind::InvalidInput);
}

/// 切り詰められた SPS は拒否する
#[test]
fn parse_sps_rejects_truncated() {
    let full = build_sps(&SpsParams::valid());
    let truncated = &full[..full.len() - 3];
    let err = parse_sps(truncated).expect_err("切り詰められた SPS は拒否される");
    assert_eq!(err.kind, ErrorKind::InvalidInput);
}

/// クロップが符号化幅以上 (7.4.3.2.1 は未満を要求) の場合は拒否する
#[test]
fn parse_sps_rejects_crop_eating_coded_width() {
    // 幅 320。4:2:0 の SubWidthC = 2 で 2 * (81 + 79) = 320 をクロップする
    let nal = build_sps(&SpsParams {
        conformance_window_flag: true,
        conf_win_left_offset: 81,
        conf_win_right_offset: 79,
        ..SpsParams::valid()
    });
    let err = parse_sps(&nal).expect_err("幅を食いつぶすクロップは拒否される");
    assert_eq!(err.kind, ErrorKind::InvalidInput);
}

/// クロップが符号化高さ以上の場合も拒否する
#[test]
fn parse_sps_rejects_crop_eating_coded_height() {
    // 高さ 240。4:2:0 の SubHeightC = 2 で 2 * (61 + 59) = 240 をクロップする
    let nal = build_sps(&SpsParams {
        conformance_window_flag: true,
        conf_win_top_offset: 61,
        conf_win_bottom_offset: 59,
        ..SpsParams::valid()
    });
    let err = parse_sps(&nal).expect_err("高さを食いつぶすクロップは拒否される");
    assert_eq!(err.kind, ErrorKind::InvalidInput);
}

/// クロップ後に幅 0 になる場合は拒否する
#[test]
fn parse_sps_rejects_zero_width_after_crop() {
    // pic_width_in_luma_samples = 0 (クロップ無し) で幅 0 になる
    let nal = build_sps(&SpsParams {
        pic_width_in_luma_samples: 0,
        ..SpsParams::valid()
    });
    let err = parse_sps(&nal).expect_err("クロップ後幅 0 は拒否される");
    assert_eq!(err.kind, ErrorKind::InvalidInput);
}

/// クロップ後に u16::MAX を超える幅は飽和せず拒否する
#[test]
fn parse_sps_rejects_width_exceeding_u16_max() {
    let nal = build_sps(&SpsParams {
        pic_width_in_luma_samples: 65536,
        ..SpsParams::valid()
    });
    let err = parse_sps(&nal).expect_err("u16::MAX 超過の幅は拒否される");
    assert_eq!(err.kind, ErrorKind::InvalidInput);
}

/// クロップ後に u16::MAX を超える高さは飽和せず拒否する
#[test]
fn parse_sps_rejects_height_exceeding_u16_max() {
    let nal = build_sps(&SpsParams {
        pic_height_in_luma_samples: 65536,
        ..SpsParams::valid()
    });
    let err = parse_sps(&nal).expect_err("u16::MAX 超過の高さは拒否される");
    assert_eq!(err.kind, ErrorKind::InvalidInput);
}

/// 32 ビットを超える Exp-Golomb コードは拒否する
#[test]
fn parse_sps_rejects_exp_golomb_code_too_long() {
    // ヘッダー + sps_video_parameter_set_id から general_level_idc まで (104 ビット) を
    // 0 で埋め、sps_seq_parameter_set_id の ue(v) が 0 を 32 個並べる
    // (leadingZeroBits = 32 で値域外)。
    // 先頭 1 バイトの sps_temporal_id_nesting_flag を 1 にして、単層 SPS の
    // 7.4.3.2.1 制約で先に弾かれないようにする
    let mut nal = vec![0x42, 0x01];
    nal.push(0x01); // sps_video_parameter_set_id (0) + sps_max_sub_layers_minus1 (0) + nesting (1)
    nal.extend_from_slice(&[0x00; 12]); // profile_tier_level (96 bits)
    nal.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // ue(v) の leadingZeroBits = 32
    nal.push(0x01); // 終端
    let err = parse_sps(&nal).expect_err("32 ビット超の Exp-Golomb は拒否される");
    assert_eq!(err.kind, ErrorKind::InvalidInput);
}

// ===== build_hev1_box / build_hvc1_box =====

/// build_hev1_box が array_completeness = 0 で hvcC を構築する
#[test]
fn build_hev1_box_array_completeness_is_zero() {
    let sps = build_sps(&SpsParams::valid());
    let hev1 = build_hev1_box(
        core::slice::from_ref(&valid_vps()),
        core::slice::from_ref(&sps),
        core::slice::from_ref(&valid_pps()),
        &default_config(),
    )
    .expect("有効な VPS / SPS / PPS は構築成功する");
    assert_eq!(hev1.hvcc_box.nalu_arrays.len(), 3);
    for array in &hev1.hvcc_box.nalu_arrays {
        assert_eq!(
            array.array_completeness.get(),
            0,
            "hev1 の completeness は 0"
        );
    }
}

/// build_hvc1_box が array_completeness = 1 で hvcC を構築する
#[test]
fn build_hvc1_box_array_completeness_is_one() {
    let sps = build_sps(&SpsParams::valid());
    let hvc1 = build_hvc1_box(
        core::slice::from_ref(&valid_vps()),
        core::slice::from_ref(&sps),
        core::slice::from_ref(&valid_pps()),
        &default_config(),
    )
    .expect("有効な VPS / SPS / PPS は構築成功する");
    assert_eq!(hvc1.hvcc_box.nalu_arrays.len(), 3);
    for array in &hvc1.hvcc_box.nalu_arrays {
        assert_eq!(
            array.array_completeness.get(),
            1,
            "hvc1 の completeness は 1"
        );
    }
}

/// 構築した hvcC の固定値 / ストリーム導出値 / 呼び出し側指定値を検証する
#[test]
fn build_hev1_box_fixed_and_derived_values() {
    let sps = build_sps(&SpsParams {
        general_profile_space: 1,
        general_tier_flag: true,
        general_profile_idc: 2,
        general_profile_compatibility_flags: 0x6000_0000,
        general_constraint_indicator_flags: 0x0000_0000_0001,
        general_level_idc: 93,
        pic_width_in_luma_samples: 1920,
        pic_height_in_luma_samples: 1088,
        conformance_window_flag: true,
        conf_win_bottom_offset: 4,
        ..SpsParams::valid()
    });
    let config = H265SampleEntryConfig {
        length_size: LengthSize::TwoBytes,
        avg_frame_rate: H265SampleEntryConfig::AVG_FRAME_RATE_UNSPECIFIED,
        constant_frame_rate: H265ConstantFrameRate::Unknown,
    };
    let hev1 = build_hev1_box(
        core::slice::from_ref(&valid_vps()),
        core::slice::from_ref(&sps),
        core::slice::from_ref(&valid_pps()),
        &config,
    )
    .expect("有効な VPS / SPS / PPS は構築成功する");
    let hvcc = &hev1.hvcc_box;

    // ストリーム導出値
    assert_eq!(hvcc.general_profile_space.get(), 1);
    assert_eq!(hvcc.general_tier_flag.get(), 1);
    assert_eq!(hvcc.general_profile_idc.get(), 2);
    assert_eq!(hvcc.general_profile_compatibility_flags, 0x6000_0000);
    assert_eq!(hvcc.general_constraint_indicator_flags.get(), 1);
    assert_eq!(hvcc.general_level_idc, 93);
    assert_eq!(hvcc.chroma_format_idc.get(), 1);
    assert_eq!(hvcc.bit_depth_luma_minus8.get(), 0);
    assert_eq!(hvcc.bit_depth_chroma_minus8.get(), 0);
    assert_eq!(hvcc.num_temporal_layers.get(), 1);
    assert_eq!(hvcc.temporal_id_nested.get(), 1);
    assert_eq!(hev1.visual.width, 1920);
    assert_eq!(hev1.visual.height, 1080);

    // 固定値
    assert_eq!(hvcc.min_spatial_segmentation_idc.get(), 0);
    assert_eq!(hvcc.parallelism_type.get(), 0);
    assert_eq!(
        hev1.visual.data_reference_index,
        VisualSampleEntryFields::DEFAULT_DATA_REFERENCE_INDEX
    );
    assert_eq!(
        hev1.visual.horizresolution,
        VisualSampleEntryFields::DEFAULT_HORIZRESOLUTION
    );
    assert_eq!(
        hev1.visual.vertresolution,
        VisualSampleEntryFields::DEFAULT_VERTRESOLUTION
    );
    assert_eq!(
        hev1.visual.frame_count,
        VisualSampleEntryFields::DEFAULT_FRAME_COUNT
    );
    assert_eq!(
        hev1.visual.compressorname,
        VisualSampleEntryFields::NULL_COMPRESSORNAME
    );
    assert_eq!(hev1.visual.depth, VisualSampleEntryFields::DEFAULT_DEPTH);
    assert!(hev1.unknown_boxes.is_empty());

    // 呼び出し側指定値 (長さ幅 2 → length_size_minus_one = 1、フレームレート
    // 0 / Unknown → avgFrameRate 0 / constantFrameRate 0)
    assert_eq!(hvcc.length_size_minus_one.get(), 1);
    assert_eq!(hvcc.avg_frame_rate, 0);
    assert_eq!(hvcc.constant_frame_rate.get(), 0);
}

/// 呼び出し側指定の avgFrameRate と constantFrameRate が HvccBox に写る
///
/// 非ゼロ avgFrameRate と 3 状態それぞれを hev1 / hvc1 の両方で検証する。
/// 構築 → encode → decode のラウンドトリップでも失われない
#[test]
fn build_sample_entry_reflects_frame_rate_config() {
    let sps = build_sps(&SpsParams::valid());
    let vps = valid_vps();
    let pps = valid_pps();
    for (constant_frame_rate, expected_bits) in [
        (H265ConstantFrameRate::Unknown, 0u8),
        (H265ConstantFrameRate::Constant, 1),
        (H265ConstantFrameRate::ConstantPerTemporalLayer, 2),
    ] {
        let config = H265SampleEntryConfig {
            length_size: LengthSize::FourBytes,
            avg_frame_rate: 3000,
            constant_frame_rate,
        };
        let hev1 = build_hev1_box(
            core::slice::from_ref(&vps),
            core::slice::from_ref(&sps),
            core::slice::from_ref(&pps),
            &config,
        )
        .expect("有効な VPS / SPS / PPS は構築成功する");
        assert_eq!(hev1.hvcc_box.avg_frame_rate, 3000, "avgFrameRate が写る");
        assert_eq!(
            hev1.hvcc_box.constant_frame_rate.get(),
            expected_bits,
            "constantFrameRate が写る"
        );
        let encoded = hev1.encode_to_vec().expect("encode 成功");
        let (decoded, size) = Hev1Box::decode(&encoded).expect("decode 成功");
        assert_eq!(size, encoded.len());
        assert_eq!(decoded, hev1);

        let hvc1 = build_hvc1_box(
            core::slice::from_ref(&vps),
            core::slice::from_ref(&sps),
            core::slice::from_ref(&pps),
            &config,
        )
        .expect("有効な VPS / SPS / PPS は構築成功する");
        assert_eq!(hvc1.hvcc_box.avg_frame_rate, 3000, "avgFrameRate が写る");
        assert_eq!(
            hvc1.hvcc_box.constant_frame_rate.get(),
            expected_bits,
            "constantFrameRate が写る"
        );
    }
}

/// nalu_arrays が VPS / SPS / PPS の順でヘッダー込み EBSP を格納する
#[test]
fn build_hev1_box_nalu_arrays_store_ebsp() {
    let vps1 = valid_vps();
    let vps2 = vec![0x40, 0x09, 0x0C];
    let sps = build_sps(&SpsParams::valid());
    let pps1 = valid_pps();
    let pps2 = vec![0x44, 0x01, 0xC1];
    let hev1 = build_hev1_box(
        &[vps1.clone(), vps2.clone()],
        core::slice::from_ref(&sps),
        &[pps1.clone(), pps2.clone()],
        &default_config(),
    )
    .expect("有効な VPS / SPS / PPS は構築成功する");

    assert_eq!(hev1.hvcc_box.nalu_arrays.len(), 3);
    assert_eq!(hev1.hvcc_box.nalu_arrays[0].nal_unit_type.get(), 32);
    assert_eq!(hev1.hvcc_box.nalu_arrays[0].nalus, vec![vps1, vps2]);
    assert_eq!(hev1.hvcc_box.nalu_arrays[1].nal_unit_type.get(), 33);
    assert_eq!(hev1.hvcc_box.nalu_arrays[1].nalus, vec![sps]);
    assert_eq!(hev1.hvcc_box.nalu_arrays[2].nal_unit_type.get(), 34);
    assert_eq!(hev1.hvcc_box.nalu_arrays[2].nalus, vec![pps1, pps2]);

    // 格納された NAL は 2 バイトヘッダー込みの EBSP のまま (開始コード無し)
    assert_eq!(
        hev1.hvcc_box.nalu_arrays[0].nalus[0],
        vec![0x40, 0x01, 0x0C, 0x01, 0xFF, 0xFF]
    );
}

/// 複数 SPS のとき、先頭 SPS の寸法を代表値にし、2 本目は配列に残す
#[test]
fn build_hev1_box_uses_first_sps_for_dimensions() {
    let sps_640 = build_sps(&SpsParams {
        pic_width_in_luma_samples: 640,
        pic_height_in_luma_samples: 480,
        ..SpsParams::valid()
    });
    let sps_320 = build_sps(&SpsParams::valid());
    let hev1 = build_hev1_box(
        &[valid_vps()],
        &[sps_640, sps_320.clone()],
        &[valid_pps()],
        &default_config(),
    )
    .expect("複数 SPS は構築成功する");

    // 構文解析して代表値にするのは先頭 SPS だけ
    assert_eq!(hev1.visual.width, 640, "先頭 SPS の幅が代表値になる");
    assert_eq!(hev1.visual.height, 480, "先頭 SPS の高さが代表値になる");
    assert_eq!(
        hev1.hvcc_box.general_level_idc, 90,
        "先頭 SPS の profile_tier_level が代表値になる"
    );

    // 2 本目もヘッダー検証を済ませたうえで入力順のまま配列に残る
    assert_eq!(hev1.hvcc_box.nalu_arrays[1].nalus.len(), 2);
    assert_eq!(hev1.hvcc_box.nalu_arrays[1].nalus[1], sps_320);
}

/// VPS リストが空なら拒否する
#[test]
fn build_hev1_box_rejects_empty_vps_list() {
    let sps = build_sps(&SpsParams::valid());
    let err = build_hev1_box(&[], &[sps], &[valid_pps()], &default_config())
        .expect_err("VPS 空リストは拒否される");
    assert_eq!(err.kind, ErrorKind::InvalidInput);
}

/// SPS リストが空なら拒否する
#[test]
fn build_hev1_box_rejects_empty_sps_list() {
    let err = build_hev1_box(&[valid_vps()], &[], &[valid_pps()], &default_config())
        .expect_err("SPS 空リストは拒否される");
    assert_eq!(err.kind, ErrorKind::InvalidInput);
}

/// PPS リストが空なら拒否する
#[test]
fn build_hev1_box_rejects_empty_pps_list() {
    let sps = build_sps(&SpsParams::valid());
    let err = build_hev1_box(&[valid_vps()], &[sps], &[], &default_config())
        .expect_err("PPS 空リストは拒否される");
    assert_eq!(err.kind, ErrorKind::InvalidInput);
}

/// VPS が NAL type 32 以外なら拒否する
#[test]
fn build_hev1_box_rejects_wrong_vps_type() {
    let sps = build_sps(&SpsParams::valid());
    // ヘッダーを type 33 (SPS) に書き換えた VPS
    let vps = vec![0x42, 0x01, 0x0C];
    let err = build_hev1_box(&[vps], &[sps], &[valid_pps()], &default_config())
        .expect_err("type 33 の VPS は拒否される");
    assert_eq!(err.kind, ErrorKind::InvalidInput);
}

/// PPS が NAL type 34 以外なら拒否する
#[test]
fn build_hev1_box_rejects_wrong_pps_type() {
    let sps = build_sps(&SpsParams::valid());
    // ヘッダーを type 33 (SPS) に書き換えた PPS
    let pps = vec![0x42, 0x01, 0xC1];
    let err = build_hev1_box(&[valid_vps()], &[sps], &[pps], &default_config())
        .expect_err("type 33 の PPS は拒否される");
    assert_eq!(err.kind, ErrorKind::InvalidInput);
}

/// 2 本目以降の SPS も非空・NAL type 33 であることを検証する
///
/// 空の 2 本目が hvcC に長さ 0 で載らないよう、先頭だけでなく全 SPS を
/// 検証する (構文解析して代表値にするのは先頭 SPS だけ)
#[test]
fn build_hev1_box_rejects_second_sps_not_type_33() {
    let sps = build_sps(&SpsParams::valid());
    // type 19 (IDR_W_RADL) の NAL。先頭は正当な SPS のまま
    let not_sps = vec![0x26, 0x01];
    let err = build_hev1_box(
        &[valid_vps()],
        &[sps, not_sps],
        &[valid_pps()],
        &default_config(),
    )
    .expect_err("type 33 以外の 2 本目 SPS は拒否される");
    assert_eq!(err.kind, ErrorKind::InvalidInput);
}

/// 2 本目以降の SPS が空なら拒否する
#[test]
fn build_hev1_box_rejects_empty_second_sps() {
    let sps = build_sps(&SpsParams::valid());
    let err = build_hev1_box(
        &[valid_vps()],
        &[sps, Vec::new()],
        &[valid_pps()],
        &default_config(),
    )
    .expect_err("空の 2 本目 SPS は拒否される");
    assert_eq!(err.kind, ErrorKind::InvalidInput);
}

/// 2 本目 SPS の forbidden_zero_bit が 1 なら拒否する
#[test]
fn build_hev1_box_rejects_second_sps_forbidden_zero_bit() {
    let sps = build_sps(&SpsParams::valid());
    // 0xC2 = 1100_0010: forbidden_zero_bit = 1 / nal_unit_type = 33
    let forbidden_sps = vec![0xC2, 0x01];
    let err = build_hev1_box(
        &[valid_vps()],
        &[sps, forbidden_sps],
        &[valid_pps()],
        &default_config(),
    )
    .expect_err("forbidden_zero_bit=1 の 2 本目 SPS は拒否される");
    assert_eq!(err.kind, ErrorKind::InvalidInput);
}

/// 2 本目 SPS の TemporalId が 0 以外なら拒否する
#[test]
fn build_hev1_box_rejects_second_sps_non_zero_temporal_id() {
    let sps = build_sps(&SpsParams::valid());
    let mut sps_tid2 = sps.clone();
    // byte1 の tid を 2 に書き換える (TemporalId = 1)。先頭 SPS は正当なまま
    sps_tid2[1] = 0x02;
    let err = build_hev1_box(
        &[valid_vps()],
        &[sps, sps_tid2],
        &[valid_pps()],
        &default_config(),
    )
    .expect_err("TemporalId=1 の 2 本目 SPS は拒否される");
    assert_eq!(err.kind, ErrorKind::InvalidInput);
}

/// 2 本目以降の VPS も非空・NAL type 32 であることを検証する
#[test]
fn build_hev1_box_rejects_second_vps_not_type_32() {
    let sps = build_sps(&SpsParams::valid());
    // ヘッダーを type 33 (SPS) に書き換えた 2 本目 VPS
    let not_vps = vec![0x42, 0x01, 0x0C];
    let err = build_hev1_box(
        &[valid_vps(), not_vps],
        &[sps],
        &[valid_pps()],
        &default_config(),
    )
    .expect_err("type 32 以外の 2 本目 VPS は拒否される");
    assert_eq!(err.kind, ErrorKind::InvalidInput);
}

/// 2 本目以降の VPS が空なら拒否する
#[test]
fn build_hev1_box_rejects_empty_second_vps() {
    let sps = build_sps(&SpsParams::valid());
    let err = build_hev1_box(
        &[valid_vps(), Vec::new()],
        &[sps],
        &[valid_pps()],
        &default_config(),
    )
    .expect_err("空の 2 本目 VPS は拒否される");
    assert_eq!(err.kind, ErrorKind::InvalidInput);
}

/// 2 本目以降の PPS も非空・NAL type 34 であることを検証する
#[test]
fn build_hev1_box_rejects_second_pps_not_type_34() {
    let sps = build_sps(&SpsParams::valid());
    // ヘッダーを type 33 (SPS) に書き換えた 2 本目 PPS
    let not_pps = vec![0x42, 0x01, 0xC1];
    let err = build_hev1_box(
        &[valid_vps()],
        &[sps],
        &[valid_pps(), not_pps],
        &default_config(),
    )
    .expect_err("type 34 以外の 2 本目 PPS は拒否される");
    assert_eq!(err.kind, ErrorKind::InvalidInput);
}

/// 2 本目以降の PPS が空なら拒否する
#[test]
fn build_hev1_box_rejects_empty_second_pps() {
    let sps = build_sps(&SpsParams::valid());
    let err = build_hev1_box(
        &[valid_vps()],
        &[sps],
        &[valid_pps(), Vec::new()],
        &default_config(),
    )
    .expect_err("空の 2 本目 PPS は拒否される");
    assert_eq!(err.kind, ErrorKind::InvalidInput);
}

/// VPS / SPS の TemporalId が 0 以外なら拒否する (PPS は 0 でなくてよい)
#[test]
fn build_hev1_box_rejects_non_zero_temporal_id_vps_sps() {
    let sps = build_sps(&SpsParams::valid());
    // VPS の tid を 2 に書き換える (TemporalId = 1)
    let vps = vec![0x40, 0x02, 0x0C];
    let err = build_hev1_box(
        &[vps],
        core::slice::from_ref(&sps),
        &[valid_pps()],
        &default_config(),
    )
    .expect_err("TemporalId=1 の VPS は拒否される");
    assert_eq!(err.kind, ErrorKind::InvalidInput);

    // SPS の tid を 2 に書き換える
    let mut sps_tid2 = build_sps(&SpsParams::valid());
    sps_tid2[1] = 0x02;
    let err = build_hev1_box(
        &[valid_vps()],
        &[sps_tid2],
        &[valid_pps()],
        &default_config(),
    )
    .expect_err("TemporalId=1 の SPS は拒否される");
    assert_eq!(err.kind, ErrorKind::InvalidInput);

    // PPS は tid = 2 でも受理する (NOTE 9 どおり 0 でなくてよい)
    let pps = vec![0x44, 0x02, 0xC1];
    let ok = build_hev1_box(&[valid_vps()], &[sps], &[pps], &default_config())
        .expect("TemporalId=1 の PPS は受理される");
    assert_eq!(
        ok.hvcc_box.nalu_arrays[2].nalus,
        vec![vec![0x44, 0x02, 0xC1]]
    );
}

/// パラメータセットの NAL が u16::MAX バイト超なら拒否する
#[test]
fn build_hev1_box_rejects_parameter_set_too_long() {
    let sps = build_sps(&SpsParams::valid());
    // 長さ検証は先頭 SPS の解析より前に行われるため、中身は不正でもよい
    let too_long_vps = vec![0x40; u16::MAX as usize + 1];
    let err = build_hev1_box(&[too_long_vps], &[sps], &[valid_pps()], &default_config())
        .expect_err("u16::MAX 超過のパラメータセットは拒否される");
    assert_eq!(err.kind, ErrorKind::InvalidInput);
}

/// パラメータセット配列が u16::MAX 個超なら拒否する
#[test]
fn build_hev1_box_rejects_too_many_parameter_sets() {
    let sps = build_sps(&SpsParams::valid());
    // hvcC の numNalus は unsigned int(16)。65536 個で超える
    let vps_list = vec![valid_vps(); u16::MAX as usize + 1];
    let err = build_hev1_box(&vps_list, &[sps], &[valid_pps()], &default_config())
        .expect_err("u16::MAX 超の配列は拒否される");
    assert_eq!(err.kind, ErrorKind::InvalidInput);
}

/// 構築した Hev1Box が encode → decode でラウンドトリップする
#[test]
fn build_hev1_box_roundtrip() {
    let sps = build_sps(&SpsParams::valid());
    let hev1 = build_hev1_box(
        core::slice::from_ref(&valid_vps()),
        core::slice::from_ref(&sps),
        core::slice::from_ref(&valid_pps()),
        &default_config(),
    )
    .expect("有効な VPS / SPS / PPS は構築成功する");
    let encoded = hev1.encode_to_vec().expect("encode 成功");
    let (decoded, size) = Hev1Box::decode(&encoded).expect("decode 成功");
    assert_eq!(size, encoded.len());
    assert_eq!(decoded, hev1);
}

/// 構築した Hvc1Box が encode → decode でラウンドトリップする
#[test]
fn build_hvc1_box_roundtrip() {
    let sps = build_sps(&SpsParams::valid());
    let hvc1 = build_hvc1_box(
        core::slice::from_ref(&valid_vps()),
        core::slice::from_ref(&sps),
        core::slice::from_ref(&valid_pps()),
        &default_config(),
    )
    .expect("有効な VPS / SPS / PPS は構築成功する");
    let encoded = hvc1.encode_to_vec().expect("encode 成功");
    let (decoded, size) = Hvc1Box::decode(&encoded).expect("decode 成功");
    assert_eq!(size, encoded.len());
    assert_eq!(decoded, hvc1);
}

// ===== build_hev1_box_from_annexb / build_hvc1_box_from_annexb =====

/// Annex B から VPS / SPS / PPS を抽出して Hev1Box を構築する
///
/// VCL や SEI 等の他種別 NAL は無視される
#[test]
fn build_hev1_box_from_annexb_ignores_non_parameter_sets() {
    let sps = build_sps(&SpsParams {
        pic_width_in_luma_samples: 640,
        pic_height_in_luma_samples: 480,
        ..SpsParams::valid()
    });
    let vps = valid_vps();
    let pps = valid_pps();
    // prefix SEI (type 39) とスライス (type 1) を混ぜる
    let input = annexb_with_3(&[
        &[0x26, 0x01, 0xAF],
        &vps,
        &[0x4E, 0x01, 0x05],
        &sps,
        &pps,
        &[0x02, 0x01, 0x88],
    ]);
    let hev1 =
        build_hev1_box_from_annexb(&input, &default_config()).expect("Annex B から構築成功する");
    assert_eq!(hev1.hvcc_box.general_profile_idc.get(), 1);
    assert_eq!(hev1.hvcc_box.general_level_idc, 90);
    assert_eq!(hev1.hvcc_box.nalu_arrays[0].nalus, vec![vps]);
    assert_eq!(hev1.hvcc_box.nalu_arrays[1].nalus, vec![sps]);
    assert_eq!(hev1.hvcc_box.nalu_arrays[2].nalus, vec![pps]);
    assert_eq!(hev1.visual.width, 640);
    assert_eq!(hev1.visual.height, 480);
}

/// Annex B に SPS が無ければ拒否する
#[test]
fn build_hev1_box_from_annexb_rejects_without_sps() {
    let input = annexb_with_3(&[&valid_vps(), &valid_pps()]);
    let err = build_hev1_box_from_annexb(&input, &default_config())
        .expect_err("SPS が無い Annex B は拒否される");
    assert_eq!(err.kind, ErrorKind::InvalidInput);
}

/// Annex B に PPS が無ければ拒否する
#[test]
fn build_hvc1_box_from_annexb_rejects_without_pps() {
    let sps = build_sps(&SpsParams::valid());
    let input = annexb_with_3(&[&valid_vps(), &sps]);
    let err = build_hvc1_box_from_annexb(&input, &default_config())
        .expect_err("PPS が無い Annex B は拒否される");
    assert_eq!(err.kind, ErrorKind::InvalidInput);
}

/// forbidden_zero_bit = 1 の VCL が混ざると、無視せず全体が失敗する
///
/// rustdoc は「VCL / SEI 等は無視する」と書くが、列挙時に全 NAL のヘッダーを
/// 検証するため、ヘッダー不正の NAL は無視されない
#[test]
fn build_hev1_box_from_annexb_rejects_invalid_slice() {
    let sps = build_sps(&SpsParams::valid());
    // 0x82 = 0b1000_0010: forbidden_zero_bit = 1 / nal_unit_type = 1 (スライス)
    let input = annexb_with_3(&[&valid_vps(), &sps, &valid_pps(), &[0x82, 0x01, 0x88]]);
    let err = build_hev1_box_from_annexb(&input, &default_config())
        .expect_err("forbidden_zero_bit=1 のスライスは拒否される");
    assert_eq!(err.kind, ErrorKind::InvalidInput);
}

// ===== 実データ fixture =====

/// 実エンコーダー (ffmpeg + x265) が生成した MP4 から抽出した VPS / SPS / PPS /
/// prefix SEI の Annex B 列
///
/// 抽出元: `tests/testdata/black-h265-video.mp4` の `hvcC` ボックス。
/// SPS は Main / level 90 / 640x480 で、emulation prevention byte を含む。
const REAL_H265_VPS_SPS_PPS_ANNEXB: &[u8] = include_bytes!("testdata/h265-vps-sps-pps-annexb.bin");

/// 実データの Annex B を解析して NAL 境界と種別を確認する
#[test]
fn real_h265_annexb_parses() {
    let nals = parse_annexb_nal_units(REAL_H265_VPS_SPS_PPS_ANNEXB)
        .expect("実 VPS / SPS / PPS の Annex B は解析成功する");
    assert_eq!(nals.len(), 4);
    assert_eq!(nals[0].nal_unit_type, H265NalUnitType::Vps);
    assert_eq!(nals[0].nuh_layer_id, 0);
    assert_eq!(nals[0].nuh_temporal_id_plus1, 1);
    assert_eq!(nals[1].nal_unit_type, H265NalUnitType::Sps);
    assert_eq!(nals[2].nal_unit_type, H265NalUnitType::Pps);
    assert_eq!(nals[3].nal_unit_type, H265NalUnitType::PrefixSei);
}

/// 実データの SPS を解析して hvcC 欄へ写す値が得られる
#[test]
fn real_h265_parse_sps() {
    let nals = parse_annexb_nal_units(REAL_H265_VPS_SPS_PPS_ANNEXB)
        .expect("実 VPS / SPS / PPS の Annex B は解析成功する");
    let sps = parse_sps(nals[1].data).expect("実 SPS は解析成功する");
    assert_eq!(sps.general_profile_space, 0);
    assert_eq!(sps.general_tier_flag, 0);
    assert_eq!(sps.general_profile_idc, 1);
    // 実 SPS の EBSP から独立に計算した期待値。合成ビルダーとパーサーが
    // 同じビット順バグを共有すると通ってしまうため、実データで固定する
    assert_eq!(sps.general_profile_compatibility_flags, 0x6000_0000);
    assert_eq!(sps.general_constraint_indicator_flags, 0x9000_0000_0000);
    assert_eq!(sps.general_level_idc, 90);
    assert_eq!(sps.sps_max_sub_layers_minus1, 0);
    assert_eq!(sps.sps_temporal_id_nesting_flag, 1);
    assert_eq!(sps.chroma_format_idc, 1);
    assert_eq!(sps.bit_depth_luma_minus8, 0);
    assert_eq!(sps.bit_depth_chroma_minus8, 0);
    assert_eq!(sps.width, 640);
    assert_eq!(sps.height, 480);
}

/// 実データから build_hev1_box_from_annexb で Hev1Box を構築できる
#[test]
fn real_h265_build_hev1_box_from_annexb() {
    let hev1 = build_hev1_box_from_annexb(REAL_H265_VPS_SPS_PPS_ANNEXB, &default_config())
        .expect("実データから構築成功する");
    let hvcc = &hev1.hvcc_box;
    assert_eq!(hvcc.general_profile_idc.get(), 1);
    assert_eq!(hvcc.general_level_idc, 90);
    assert_eq!(hvcc.chroma_format_idc.get(), 1);
    assert_eq!(hvcc.bit_depth_luma_minus8.get(), 0);
    assert_eq!(hvcc.bit_depth_chroma_minus8.get(), 0);
    assert_eq!(hvcc.num_temporal_layers.get(), 1);
    assert_eq!(hvcc.temporal_id_nested.get(), 1);
    assert_eq!(hev1.visual.width, 640);
    assert_eq!(hev1.visual.height, 480);

    // prefix SEI はパラメータセット配列に載らない (VPS / SPS / PPS の 3 配列だけ)
    assert_eq!(hvcc.nalu_arrays.len(), 3);
    assert_eq!(hvcc.nalu_arrays[0].nalus.len(), 1);
    assert_eq!(hvcc.nalu_arrays[1].nalus.len(), 1);
    assert_eq!(hvcc.nalu_arrays[2].nalus.len(), 1);

    // 格納された VPS / SPS / PPS は emulation prevention byte を残した EBSP のまま
    assert!(
        hvcc.nalu_arrays[1].nalus[0]
            .windows(3)
            .any(|w| w == [0x00, 0x00, 0x03]),
        "実 SPS は `00 00 03` を含む"
    );

    // 実データ由来の hev1 が encode → decode でラウンドトリップする
    let encoded = hev1.encode_to_vec().expect("encode 成功");
    let (decoded, size) = Hev1Box::decode(&encoded).expect("decode 成功");
    assert_eq!(size, encoded.len());
    assert_eq!(decoded, hev1);
}

/// 実データから build_hvc1_box_from_annexb で Hvc1Box を構築できる
#[test]
fn real_h265_build_hvc1_box_from_annexb() {
    let hvc1 = build_hvc1_box_from_annexb(REAL_H265_VPS_SPS_PPS_ANNEXB, &default_config())
        .expect("実データから構築成功する");
    assert_eq!(hvc1.hvcc_box.general_profile_idc.get(), 1);
    assert_eq!(hvc1.hvcc_box.general_level_idc, 90);
    assert_eq!(hvc1.visual.width, 640);
    assert_eq!(hvc1.visual.height, 480);
    // hvc1 は completeness が 1
    for array in &hvc1.hvcc_box.nalu_arrays {
        assert_eq!(array.array_completeness.get(), 1);
    }
}

/// 実 SPS の VUI から timing 情報を読める
///
/// 合成ビルダーとパーサーが同じ解釈の誤りを共有していると合成テストは
/// 通ってしまうため、実データで固定する。25 fps (1 / 25) のストリーム
#[test]
fn real_h265_parse_sps_vui_timing_info() {
    let nals = parse_annexb_nal_units(REAL_H265_VPS_SPS_PPS_ANNEXB)
        .expect("実 VPS / SPS / PPS の Annex B は解析成功する");
    let sps = parse_sps(nals[1].data).expect("実 SPS は解析成功する");

    let timing = sps
        .vui_timing_info
        .expect("実 SPS は vui_timing_info_present_flag = 1 である");
    assert_eq!(timing.num_units_in_tick.get(), 1);
    assert_eq!(timing.time_scale.get(), 25);
    // この SPS の VUI は bitstream_restriction_flag = 0
    assert_eq!(sps.min_spatial_segmentation_idc, None);
}

// ===== parse_sps: VUI =====

/// VUI の timing 情報を読める
#[test]
fn parse_sps_reads_vui_timing_info() {
    let nal = build_sps_with_tail(
        &SpsParams::valid(),
        &SpsTailParams::with_vui(VuiParams {
            timing: Some((1001, 60000)),
            ..VuiParams::empty()
        }),
    );
    let sps = parse_sps(&nal).expect("VUI 付き SPS は解析成功する");

    let timing = sps.vui_timing_info.expect("timing 情報が読めること");
    assert_eq!(timing.num_units_in_tick.get(), 1001);
    assert_eq!(timing.time_scale.get(), 60000);
    // 寸法は末尾の有無に影響されない
    assert_eq!(sps.width, 320);
    assert_eq!(sps.height, 240);
}

/// `vui_parameters_present_flag = 0` なら timing 情報は無い
#[test]
fn parse_sps_without_vui_has_no_timing_info() {
    let nal = build_sps_with_tail(
        &SpsParams::valid(),
        &SpsTailParams {
            vui: None,
            ..SpsTailParams::with_vui(VuiParams::empty())
        },
    );
    let sps = parse_sps(&nal).expect("VUI なし SPS は解析成功する");

    assert_eq!(sps.vui_timing_info, None);
    assert_eq!(sps.min_spatial_segmentation_idc, None);
}

/// `vui_timing_info_present_flag = 0` なら timing 情報は無い
#[test]
fn parse_sps_without_timing_info_flag_has_no_timing_info() {
    let nal = build_sps_with_tail(
        &SpsParams::valid(),
        &SpsTailParams::with_vui(VuiParams::empty()),
    );
    let sps = parse_sps(&nal).expect("timing なし VUI の SPS は解析成功する");

    assert_eq!(sps.vui_timing_info, None);
}

/// VUI より手前で終わる SPS も解析成功し、VUI 由来の値は None になる
#[test]
fn parse_sps_truncated_before_vui_has_no_vui_fields() {
    // build_sps は bit_depth_chroma_minus8 で打ち切る
    let sps = parse_sps(&build_sps(&SpsParams::valid())).expect("末尾欠落 SPS は解析成功する");

    assert_eq!(sps.vui_timing_info, None);
    assert_eq!(sps.min_spatial_segmentation_idc, None);
    assert_eq!(sps.width, 320);
}

/// `vui_time_scale = 0` の SPS は末尾の解析失敗として扱い、VUI 由来の値を落とす
///
/// E.2.1 は 0 より大きいことを要求する。0 を既定値で補完しない
#[test]
fn parse_sps_zero_time_scale_drops_vui_fields() {
    let nal = build_sps_with_tail(
        &SpsParams::valid(),
        &SpsTailParams::with_vui(VuiParams {
            timing: Some((1000, 0)),
            min_spatial_segmentation_idc: Some(15),
            ..VuiParams::empty()
        }),
    );
    let sps = parse_sps(&nal).expect("不正 timing でも SPS 全体は解析成功する");

    assert_eq!(sps.vui_timing_info, None);
    assert_eq!(sps.min_spatial_segmentation_idc, None);
    // 寸法と profile 系は影響を受けない
    assert_eq!(sps.width, 320);
    assert_eq!(sps.general_level_idc, 90);
}

/// `bitstream_restriction_flag` の `min_spatial_segmentation_idc` を読める
#[test]
fn parse_sps_reads_min_spatial_segmentation_idc() {
    let nal = build_sps_with_tail(
        &SpsParams::valid(),
        &SpsTailParams::with_vui(VuiParams {
            timing: Some((1, 30)),
            min_spatial_segmentation_idc: Some(15),
            ..VuiParams::empty()
        }),
    );
    let sps = parse_sps(&nal).expect("bitstream restriction 付き SPS は解析成功する");

    assert_eq!(sps.min_spatial_segmentation_idc, Some(15));
}

/// `min_spatial_segmentation_idc` が値域外 (4096 以上) なら VUI 由来の値を落とす
#[test]
fn parse_sps_min_spatial_segmentation_idc_out_of_range_drops_vui_fields() {
    let nal = build_sps_with_tail(
        &SpsParams::valid(),
        &SpsTailParams::with_vui(VuiParams {
            timing: Some((1, 30)),
            min_spatial_segmentation_idc: Some(4096),
            ..VuiParams::empty()
        }),
    );
    let sps = parse_sps(&nal).expect("値域外でも SPS 全体は解析成功する");

    assert_eq!(sps.min_spatial_segmentation_idc, None);
    // timing は同じ末尾の解析で落ちる (部分的な採用はしない)
    assert_eq!(sps.vui_timing_info, None);
}

/// `hrd_parameters` を跨いで bitstream restriction を読める
#[test]
fn parse_sps_skips_hrd_parameters() {
    let nal = build_sps_with_tail(
        &SpsParams::valid(),
        &SpsTailParams::with_vui(VuiParams {
            timing: Some((1, 30)),
            hrd: true,
            sub_pic_hrd: false,
            min_spatial_segmentation_idc: Some(7),
        }),
    );
    let sps = parse_sps(&nal).expect("hrd 付き SPS は解析成功する");

    assert_eq!(sps.vui_timing_info.expect("timing").time_scale.get(), 30);
    assert_eq!(sps.min_spatial_segmentation_idc, Some(7));
}

/// `sub_pic_hrd_params_present_flag = 1` の `hrd_parameters` も跨げる
#[test]
fn parse_sps_skips_sub_pic_hrd_parameters() {
    let nal = build_sps_with_tail(
        &SpsParams::valid(),
        &SpsTailParams::with_vui(VuiParams {
            timing: Some((1, 30)),
            hrd: true,
            sub_pic_hrd: true,
            min_spatial_segmentation_idc: Some(7),
        }),
    );
    let sps = parse_sps(&nal).expect("sub-pic hrd 付き SPS は解析成功する");

    assert_eq!(sps.min_spatial_segmentation_idc, Some(7));
}

/// sub-layer が複数ある `hrd_parameters` も跨げる
#[test]
fn parse_sps_skips_hrd_parameters_with_sub_layers() {
    let nal = build_sps_with_tail(
        &SpsParams {
            sps_max_sub_layers_minus1: 2,
            ..SpsParams::valid()
        },
        &SpsTailParams {
            sub_layer_ordering_info_present: true,
            ..SpsTailParams::with_vui(VuiParams {
                timing: Some((1, 30)),
                hrd: true,
                sub_pic_hrd: false,
                min_spatial_segmentation_idc: Some(3),
            })
        },
    );
    let sps = parse_sps(&nal).expect("sub-layer 付き SPS は解析成功する");

    assert_eq!(sps.sps_max_sub_layers_minus1, 2);
    assert_eq!(sps.min_spatial_segmentation_idc, Some(3));
}

/// `scaling_list_data` と `pcm` の構文を跨いで VUI を読める
#[test]
fn parse_sps_skips_scaling_list_and_pcm() {
    let nal = build_sps_with_tail(
        &SpsParams::valid(),
        &SpsTailParams {
            scaling_list: true,
            pcm: true,
            ..SpsTailParams::with_vui(VuiParams {
                timing: Some((1, 50)),
                min_spatial_segmentation_idc: Some(1),
                ..VuiParams::empty()
            })
        },
    );
    let sps = parse_sps(&nal).expect("scaling list / pcm 付き SPS は解析成功する");

    assert_eq!(sps.vui_timing_info.expect("timing").time_scale.get(), 50);
    assert_eq!(sps.min_spatial_segmentation_idc, Some(1));
}

/// 明示的な `st_ref_pic_set` と long-term 参照を跨いで VUI を読める
#[test]
fn parse_sps_skips_explicit_st_ref_pic_set_and_long_term() {
    let nal = build_sps_with_tail(
        &SpsParams::valid(),
        &SpsTailParams {
            st_ref_pic_sets: StRefPicSets::Single,
            long_term_ref_pics: 2,
            ..SpsTailParams::with_vui(VuiParams {
                timing: Some((1, 60)),
                ..VuiParams::empty()
            })
        },
    );
    let sps = parse_sps(&nal).expect("st_ref_pic_set 付き SPS は解析成功する");

    assert_eq!(sps.vui_timing_info.expect("timing").time_scale.get(), 60);
}

/// `inter_ref_pic_set_prediction_flag = 1` の `st_ref_pic_set` を跨いで VUI を読める
///
/// 予測された集合の `NumDeltaPocs` を 7.4.8 のとおり導出しないと、後続の集合の
/// フラグ個数がずれて VUI の読み出し位置が壊れる
#[test]
fn parse_sps_skips_predicted_st_ref_pic_sets() {
    let nal = build_sps_with_tail(
        &SpsParams::valid(),
        &SpsTailParams {
            st_ref_pic_sets: StRefPicSets::Predicted,
            ..SpsTailParams::with_vui(VuiParams {
                timing: Some((1001, 24000)),
                min_spatial_segmentation_idc: Some(4095),
                ..VuiParams::empty()
            })
        },
    );
    let sps = parse_sps(&nal).expect("予測 st_ref_pic_set 付き SPS は解析成功する");

    let timing = sps.vui_timing_info.expect("timing");
    assert_eq!(timing.num_units_in_tick.get(), 1001);
    assert_eq!(timing.time_scale.get(), 24000);
    assert_eq!(sps.min_spatial_segmentation_idc, Some(4095));
}

/// hvcC の min_spatial_segmentation_idc に SPS の VUI の値が載る
#[test]
fn build_hev1_box_uses_min_spatial_segmentation_idc_from_sps() {
    let sps = build_sps_with_tail(
        &SpsParams::valid(),
        &SpsTailParams::with_vui(VuiParams {
            timing: Some((1, 30)),
            min_spatial_segmentation_idc: Some(9),
            ..VuiParams::empty()
        }),
    );
    let hev1 = build_hev1_box(&[valid_vps()], &[sps], &[valid_pps()], &default_config())
        .expect("VUI 付き SPS から構築成功する");

    assert_eq!(hev1.hvcc_box.min_spatial_segmentation_idc.get(), 9);
}

/// VUI に bitstream restriction が無ければ hvcC の min_spatial_segmentation_idc は 0
#[test]
fn build_hev1_box_min_spatial_segmentation_idc_defaults_to_zero() {
    let hev1 = build_hev1_box(
        &[valid_vps()],
        &[build_sps(&SpsParams::valid())],
        &[valid_pps()],
        &default_config(),
    )
    .expect("末尾欠落 SPS から構築成功する");

    assert_eq!(hev1.hvcc_box.min_spatial_segmentation_idc.get(), 0);
}
