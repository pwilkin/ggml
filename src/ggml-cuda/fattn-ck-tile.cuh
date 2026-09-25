#pragma once

// GGML's RDNA macro collides with CK Tile's architecture enum.
#pragma push_macro("RDNA")
#undef RDNA
#include "ck_tile/ops/fmha_fwd.hpp"
#pragma pop_macro("RDNA")


struct ggml_ck_fmha_f16_f32 : FmhaFwdTypeConfig<FmhaFwdFp16> {
    using ODataType = float;
};

using ggml_ck_fmha_shape = ck_tile::TileFmhaShape<
    ck_tile::sequence<128, 64, 32, 128, 32, 128>,
    ck_tile::sequence<8, 1, 1>, ck_tile::sequence<16, 16, 16>,
    ck_tile::sequence<8, 1, 1>, ck_tile::sequence<16, 16, 16>, true>;
using ggml_ck_fmha_traits = ck_tile::TileFmhaTraits<
    true, true, false, false, false,
    ck_tile::BlockAttentionBiasEnum::NO_BIAS, false, false, false,
    ck_tile::BlockAttentionQuantScaleEnum::NO_SCALE, 6>;
using ggml_ck_fmha_problem = ck_tile::BlockFmhaPipelineProblem<
    ggml_ck_fmha_f16_f32::QDataType,
    ggml_ck_fmha_f16_f32::KDataType,
    ggml_ck_fmha_f16_f32::VDataType,
    ggml_ck_fmha_f16_f32::SaccDataType,
    ggml_ck_fmha_f16_f32::SMPLComputeDataType,
    ggml_ck_fmha_f16_f32::BiasDataType,
    ggml_ck_fmha_f16_f32::RandValOutputDataType,
    ggml_ck_fmha_f16_f32::LSEDataType,
    ggml_ck_fmha_f16_f32::PDataType,
    ggml_ck_fmha_f16_f32::OaccDataType,
    ggml_ck_fmha_f16_f32::ODataType,
    ggml_ck_fmha_shape, false, ck_tile::ComposedAttention<0, false>,
    FmhaMasks::NoMask, false, ggml_ck_fmha_traits>;
using ggml_ck_fmha_pipeline = ck_tile::BlockFmhaPipelineQRKSVS<ggml_ck_fmha_problem>;
using ggml_ck_fmha_epilogue = ck_tile::Default2DEpilogue<
    ck_tile::Default2DEpilogueProblem<float, float, true, false>>;
using ggml_ck_fmha_kernel = ck_tile::FmhaFwdKernel<ggml_ck_fmha_pipeline, ggml_ck_fmha_epilogue>;

static bool ggml_cuda_flash_attn_ext_ck_tile_eligible(const int cc, const ggml_tensor * dst) {
    if (cc != GGML_CUDA_CC_OFFSET_AMD + 0x1151 || dst->src[3] != nullptr || dst->src[4] != nullptr ||
            dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst)) {
        return false;
    }

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    return max_bias == 0.0f && logit_softcap == 0.0f &&
           Q->type == GGML_TYPE_F32 && K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16 &&
           Q->ne[0] == 128 &&
           Q->ne[1] >= 576 && Q->ne[1] <= 6400 && Q->ne[1] % 16 == 0 &&
           Q->ne[2] == 32 && Q->ne[3] == 1 &&
           K->ne[0] == 128 && K->ne[1] >= Q->ne[1] && K->ne[1] <= Q->ne[1] + 256 &&
           K->ne[2] == 32 && K->ne[3] == 1 &&
           V->ne[0] == 128 && V->ne[1] == K->ne[1] && V->ne[2] == 32 && V->ne[3] == 1 &&
           dst->ne[0] == 128 && dst->ne[1] == 32 && dst->ne[2] == Q->ne[1] && dst->ne[3] == 1 &&
           ggml_is_contiguously_allocated(Q) && ggml_is_contiguously_allocated(K) &&
           ggml_is_contiguously_allocated(V);
}

static void ggml_cuda_flash_attn_ext_ck_tile(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    ggml_cuda_pool_alloc<half> q_f16(ctx.pool());
    q_f16.alloc(ggml_nelements(Q));

    to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(GGML_TYPE_F32);
    to_fp16((const char *) Q->data, q_f16.ptr, ggml_nelements(Q), ctx.stream());

    fmha_fwd_args args{};
    args.q_ptr = q_f16.ptr;
    args.k_ptr = K->data;
    args.v_ptr = V->data;
    args.o_ptr = dst->data;
    args.seqlen_q = Q->ne[1];
    args.seqlen_k = K->ne[1];
    args.max_seqlen_q = Q->ne[1];
    args.batch = 1;
    args.hdim_q = Q->ne[0];
    args.hdim_v = V->ne[0];
    args.nhead_q = Q->ne[2];
    args.nhead_k = K->ne[2];
    memcpy(&args.scale_s, (const float *) dst->op_params, sizeof(float));
    args.stride_q = Q->ne[0];
    args.stride_k = K->ne[0];
    args.stride_v = V->ne[0];
    args.stride_o = dst->ne[0] * dst->ne[1];
    args.nhead_stride_q = Q->ne[0] * Q->ne[1];
    args.nhead_stride_k = K->ne[0] * K->ne[1];
    args.nhead_stride_v = V->ne[0] * V->ne[1];
    args.nhead_stride_o = dst->ne[0];
    args.batch_stride_q = ggml_nelements(Q);
    args.batch_stride_k = ggml_nelements(K);
    args.batch_stride_v = ggml_nelements(V);
    args.batch_stride_o = ggml_nelements(dst);
    args.drop_seed_offset = std::make_pair(uint64_t(0), uint64_t(0));

    auto [kargs, grids] = fmha_fwd_create_kargs_and_grids<ggml_ck_fmha_kernel>(args);
    ck_tile::stream_config stream_config{};
    stream_config.stream_id_ = ctx.stream();
    ck_tile::launch_kernel(stream_config,
        ck_tile::make_kernel<ggml_ck_fmha_kernel::kBlockPerCu>(
            ggml_ck_fmha_kernel{}, grids, ggml_ck_fmha_kernel::BlockSize(), 0, kargs));
    CUDA_CHECK(cudaGetLastError());
}
