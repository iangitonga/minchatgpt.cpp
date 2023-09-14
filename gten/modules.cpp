#include <cmath>
#include <iostream>
#include <thread>

#include "modules.h"
#include "module_ops.h"


#define GTEN_CHECK_DTYPE_EQUAL(inp_dtype, expected_dtype)     \
    GTEN_ASSERT(                                              \
        inp_dtype == expected_dtype,                          \
        "Expected tensor to have dtype=%s but got dtype=%s.", \
        dtype_str(expected_dtype),                            \
        dtype_str(inp_dtype))

#define GTEN_CHECK_NDIMS_EQUAL(inp_ndims, expected_ndims)    \
    GTEN_ASSERT(                                             \
        inp_ndims == expected_ndims,                         \
        "Expected a %d-dim tensor but got a %d-dim tensor.", \
        expected_ndims,                                      \
        inp_ndims)

#define GTEN_CHECK_DIMSIZE_EQUAL(dim, inp_dimsize, expected_dimsize)  \
    GTEN_ASSERT(                                                      \
        inp_dimsize == expected_dimsize,                              \
        "Expected tensor to have dim-%d=%d but got dim-%d=%d.",       \
        dim, expected_dimsize, dim, inp_dimsize)

// #define GTEN_CHECK_INP_CTX_SIZE(inp_ctx_size, max_ctx_size)                  
//     GTEN_ASSERT(                                                             
//         inp_ctx_size <= max_ctx_size,                                        
//         "The given input's context size=%d exceeds max context size of %d.", 
//         inp_ctx_size,                                                        
//         max_ctx_size)

#define GTEN_CHECK_INP_CTX_SIZE(inp_ctx_size, max_ctx_size)


namespace gten {

Embedding::Embedding(int n_vocab, int d_embed, int max_ctx)
    : weight{Tensor({n_vocab, d_embed}, kFloat16)},
      emb_acv_{Tensor({max_ctx, d_embed}, kFloat16)},
      proj_acv_{Tensor({n_vocab}, kFloat32)},
      max_ctx_(max_ctx)
{
}

Tensor Embedding::forward(const Tensor& tokens)
{
    Timer timer{&exec_time_emb_ms_};

    GTEN_CHECK_DTYPE_EQUAL(tokens.dtype(), kInt32);
    GTEN_CHECK_NDIMS_EQUAL(tokens.ndims(), 1);
    GTEN_CHECK_INP_CTX_SIZE(tokens.numel(), max_ctx_);
    
    if (emb_acv_cached_) {
        int cache_offset = tokens.numel() - 1;
        ops::embed_tokens(weight, tokens, emb_acv_, cache_offset);
    }
    else {
        emb_acv_cached_ = true;
        ops::embed_tokens(weight, tokens, emb_acv_);
    }

    return emb_acv_;
}


Tensor Embedding::forward_proj(const Tensor &inp)
{
    Timer timer(&exec_time_proj_ms_);

    GTEN_CHECK_NDIMS_EQUAL(inp.ndims(), 2);
    GTEN_CHECK_DIMSIZE_EQUAL(1, inp.size(1), weight.size(1));
    GTEN_CHECK_INP_CTX_SIZE(inp.size(0), max_ctx_);

    return forward_proj_impl(inp);
}

Tensor Embedding::forward_proj_impl(const Tensor& inp)
{
    ops::embedding_projection(inp, weight, proj_acv_);
    return proj_acv_;  
}

PosEmbedding::PosEmbedding(int max_ctx, int d_embed)
    : weight{Tensor({max_ctx, d_embed}, kFloat16)}, max_ctx_(max_ctx)
{
}

Tensor PosEmbedding::forward(int n_ctx)
{
    GTEN_CHECK_INP_CTX_SIZE(n_ctx, max_ctx_);

    Timer timer{&exec_time_ms_};
    
    return forward_impl(n_ctx);
}

Tensor PosEmbedding::forward_impl(int n_ctx)
{
    const Float16* weight_data = weight.data_ptr<Float16>();

    void* src_ptr = (void*)weight_data;
    const int d_embed = weight.size(1);

    // Shares the data with the weight.
    Tensor acv{src_ptr, {n_ctx, d_embed}, weight.dtype()};

    return acv;
}

LayerNorm::LayerNorm(int max_ctx, int d_embed)
    : weight{Tensor({d_embed}, kFloat16)},
      bias{Tensor({d_embed}, kFloat16)},
      acv_{Tensor({max_ctx, d_embed}, kFloat16)},
      max_ctx_{max_ctx}
{
}

Tensor LayerNorm::forward(const Tensor &inp)
{
    Timer timer(&exec_time_ms_);

    GTEN_CHECK_NDIMS_EQUAL(inp.ndims(), 2);
    const int d_embed = weight.size(0);
    GTEN_CHECK_DIMSIZE_EQUAL(1, inp.size(1), d_embed);

    return forward_impl(inp);
}

Tensor LayerNorm::forward_impl(const Tensor &inp)
{
    if (acv_cached_) {
        int ctx_start_idx = inp.size(0) - 1;
        ops::normalize(inp, weight, bias, acv_, ctx_start_idx);
    } else {
        acv_cached_ = true;
        ops::normalize(inp, weight, bias, acv_);
    }

    return acv_;
}


GELU::GELU(int max_ctx, int d_out, bool cache_ctx_acv)
    : acv_{Tensor({max_ctx, d_out}, kFloat16)}, cache_acv_{cache_ctx_acv}
{
}

Tensor GELU::forward(const Tensor& inp)
{
    Timer timer{&exec_time_ms_};

    GTEN_CHECK_NDIMS_EQUAL(inp.ndims(), 2);
    // GTEN_CHECK_DIMSIZE_EQUAL(1, inp.size(1), n_out);
    // Assert inp numel = acv numel.
    // Resize acv to inp shape.

    return forward_impl(inp);
}

Tensor GELU::forward_impl(const Tensor& inp)
{
    if (cache_acv_ && acv_cached_) {
        const int cache_offset = inp.size(0) - 1;
        ops::gelu(inp, acv_, cache_offset);
    } else {
        acv_cached_ = true;

        ops::gelu(inp, acv_);
    }
    
    return acv_;
}

Residual::Residual(int max_ctx, int d_out)
    : acv_{Tensor({max_ctx, d_out}, kFloat16)}
{
}

Tensor Residual::forward(const Tensor& inp0, const Tensor& inp1)
{
    Timer timer{&exec_time_ms_};

    GTEN_CHECK_DTYPE_EQUAL(inp0.dtype(), inp1.dtype());
    GTEN_CHECK_NDIMS_EQUAL(inp0.ndims(), 2);
    GTEN_CHECK_NDIMS_EQUAL(inp1.ndims(), 2);
    // TODO: Check shape inp1 == inp0

    return forward_impl(inp0, inp1);
}

Tensor Residual::forward_impl(const Tensor& inp0, const Tensor& inp1)
{
    if (acv_cached_) {
        const int cache_row_offset = inp0.size(0) - 1;
        ops::add(inp0, inp1, acv_, cache_row_offset);
    } else {
        acv_cached_ = true;

        ops::add(inp0, inp1, acv_);
    }

    return acv_;
}

Linear::Linear(int d_in, int d_out, int max_ctx)
    : weight{Tensor({d_out, d_in}, kFloat16)},
      bias{Tensor({d_out}, kFloat16)},
      acv_{Tensor({max_ctx, d_out}, kFloat16)}
{
}

Tensor Linear::forward(const Tensor &inp)
{
    Timer timer{&exec_time_ms_};

    GTEN_CHECK_NDIMS_EQUAL(inp.ndims(), 2);
    GTEN_CHECK_DIMSIZE_EQUAL(1, inp.size(1), weight.size(1));

    return forward_impl(inp);
}

Tensor Linear::forward_impl(const Tensor& inp)
{
    if (acv_cached_) {
        const int cache_row_offset = inp.size(0) - 1;
        ops::affine_proj_2d(inp, weight, bias, acv_, cache_row_offset);
        // ops::matmul_2d(inp, weight, acv_, cache_row_offset);
        // ops::bias_add_inplace(acv_, bias, cache_row_offset);
    } else {
        acv_cached_ = true;

        ops::affine_proj_2d(inp, weight, bias, acv_);
        // ops::matmul_2d(inp, weight, acv_);
        // ops::bias_add_inplace(acv_, bias);
    }

    return acv_;
}


MultiHeadSelfAttn::MultiHeadSelfAttn(int n_head, int d_embed, int max_ctx)
    : query{Linear(d_embed, d_embed, max_ctx)},
      key{Linear(d_embed, d_embed, max_ctx)},
      value{Linear(d_embed, d_embed, max_ctx)},
      qkv_proj{Linear(d_embed, d_embed, max_ctx)},
      qk_acv_{Tensor({n_head, max_ctx, max_ctx}, kFloat16)},
      qkv_acv_{Tensor({max_ctx, d_embed}, kFloat16)},
      n_heads_{n_head}
{
}

Tensor MultiHeadSelfAttn::forward(const Tensor &inp)
{
    GTEN_CHECK_NDIMS_EQUAL(inp.ndims(), 2);
    GTEN_CHECK_DIMSIZE_EQUAL(1, inp.size(1), qkv_acv_.size(1));

    Tensor q = query.forward(inp);
    Tensor k = key.forward(inp);
    Tensor v = value.forward(inp);

    const Tensor qkv = masked_qkv_attn(q, k, v);
    const Tensor out = qkv_proj.forward(qkv);
    return out;
}


Tensor MultiHeadSelfAttn::masked_qkv_attn(const Tensor &q, const Tensor &k, const Tensor &v)
{
    Timer timer{&time_attn_ms_};

    const int n_ctx = q.size(0);
    const int d_embed = q.size(1);
    qk_acv_.resize({n_heads_, n_ctx, n_ctx});
    qkv_acv_.resize({n_ctx, d_embed});

    if (qkv_cached_) {
        ops::qk_masked_attn_matmul(q ,k, qk_acv_, n_heads_, /*ctx_offset=*/n_ctx-1);
        ops::qk_softmax(qk_acv_, n_heads_, /*ctx_offset=*/n_ctx-1);
        ops::qkv_attn_matmul(qk_acv_, v, qkv_acv_, n_heads_, /*ctx_offset=*/n_ctx-1);
    }
    else {
        qkv_cached_ = true;
        
        ops::qk_masked_attn_matmul(q, k, qk_acv_, n_heads_);
        ops::qk_softmax(qk_acv_, n_heads_);
        ops::qkv_attn_matmul(qk_acv_, v, qkv_acv_, n_heads_);
    }

    return qkv_acv_;
}

ResidualAttnBlock::ResidualAttnBlock(int n_attn_heads, int d_embed, int d_mlp, int max_ctx)
    : attn_ln{LayerNorm(max_ctx, d_embed)},
      attn{MultiHeadSelfAttn(n_attn_heads, d_embed, max_ctx)},
      inp_res{Residual(max_ctx, d_embed)},
      mlp_ln{LayerNorm(max_ctx, d_embed)},
      mlp_fc{Linear(d_embed, d_mlp, max_ctx)},
      gelu{GELU(max_ctx, d_mlp, /*cache_ctx_acv=*/true)},
      mlp_proj{Linear(d_mlp, d_embed, max_ctx)},
      attn_res{Residual(max_ctx, d_embed)}
{
}

Tensor ResidualAttnBlock::forward(const Tensor &inp)
{
    Tensor attn_out = inp_res.forward(inp, attn.forward(attn_ln.forward(inp)));
    Tensor out = attn_res.forward(attn_out,
        mlp_proj.forward(gelu.forward(mlp_fc.forward(mlp_ln.forward(attn_out)))));
    return out;
}

} // namespace gten
