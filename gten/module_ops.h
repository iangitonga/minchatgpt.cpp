#include "tensor.h"
#include "simd_ops.h"


namespace gten {

namespace ops {

void embed_tokens(const Tensor& emb_table, const Tensor& tokens, Tensor& cache, int cache_offset = 0) {
    const int d_embed = emb_table.size(1);
    cache.resize({tokens.numel(), d_embed});

    const Float16* emb_table_data = emb_table.data_ptr<Float16>();
    const int* indices_data = tokens.data_ptr<int>();
    Float16* cache_data = cache.data_ptr<Float16>();

    const int n_vocab = emb_table.size(0);
    const size_t emb_vec_nbytes = d_embed * sizeof(Float16);
    const int n_indices = tokens.numel();
    for (int i = cache_offset; i < n_indices; i++) {
        int emb_index = indices_data[i];
        GTEN_ASSERT(
            emb_index >= 0 && emb_index < n_vocab,
            "Embedding index '%d' at index %d of the given tokens is outside the expected range.",
            emb_index, i);

        const int emb_offset = emb_index * d_embed;
        const int out_offset = i * d_embed;
        const void* src = reinterpret_cast<const void*>(emb_table_data + emb_offset);
        void* dest = reinterpret_cast<void*>(cache_data + out_offset);
        std::memcpy(dest, src, emb_vec_nbytes);
    }
}


void normalize_vec(const Float16* vec, int vec_size, const Float16* weight, const Float16* bias, Float16* out)
{
    // Calculate the mean.
    float sum = 0.0f;
    for (int i = 0; i < vec_size; i++) {
        sum += fpcvt_htos(vec[i]);
    }
    const float mean = sum / vec_size;

    // Calculate the std-dev.
    float sum_squares = 0.0f;
    for (int i = 0; i < vec_size; i++) {
        float x = fpcvt_htos(vec[i]);
        sum_squares += (x - mean) * (x - mean);
    }
    const float variance = sum_squares / (float)vec_size;
    const float stddev = std::sqrt(variance);

    // Normalize.
    float eps = 1e-05f;
    for (int i = 0; i < vec_size; i++) {
        float x = fpcvt_htos(vec[i]);
        float w = fpcvt_htos(weight[i]);
        float b = fpcvt_htos(bias[i]);

        // Epsilon added to standard deviation prevents div by zero.
        float normalized = ((x - mean) / (stddev + eps)) * w + b;
        out[i] = fpcvt_stoh(normalized);
    }
}


void normalize(const Tensor& inp, const Tensor& weight, const Tensor& bias, Tensor& cache, int cache_offset = 0)
{
    const int n_vectors = inp.size(0);
    const int vec_size = inp.size(1);

    cache.resize({n_vectors, vec_size});

    const Float16* inp_data = inp.data_ptr<Float16>();
    const Float16* weight_data = weight.data_ptr<Float16>();
    const Float16* bias_data = bias.data_ptr<Float16>();
    Float16* cache_data = cache.data_ptr<Float16>();

    for (int i = cache_offset; i < n_vectors; i++) {
        const int vec_offset = i * vec_size;
        const Float16* vec_ptr = inp_data + vec_offset;
        const int cache_offset = i * vec_size;
        Float16* cache_ptr = cache_data + cache_offset;

        normalize_vec(vec_ptr, vec_size, weight_data, bias_data, cache_ptr);
    }
}


void gelu(const Tensor& inp, Tensor& cache, int cache_offset = 0)
{
    // TODO: Replace with lookup table.
    const int n_vectors = inp.size(0);
    const int vec_size = cache.size(1);

    cache.resize({n_vectors, vec_size});
    
    const Float16* inp_data = inp.data_ptr<Float16>();
    Float16* cache_data = cache.data_ptr<Float16>();

    const int ne = inp.numel();
    const int start_idx = cache_offset * vec_size;
    for (int i = start_idx; i < ne; ++i) {
        float x = fpcvt_htos(inp_data[i]);
        float res = 0.5 * x 
                            * (1.0f + std::tanh(std::sqrt(2.0f / 3.141592653589793f)
                            * (x + 0.044715f * std::pow(x, 3.0f))));
        cache_data[i] = fpcvt_stoh(res);
    }
}


void vec_add(const Float16* a, const Float16* b, Float16* out, int vec_size)
{
    for (int i = 0; i < vec_size; i += 8)
    {
        Vec_f32x8 x0 = vec_f32x8_load(a + i);
        Vec_f32x8 x1 = vec_f32x8_load(b + i);
        Vec_f32x8 x_sum = vec_f32x8_add(x0, x1);
        vec_f32x8_store(x_sum, out + i);
    }
}

void add(const Tensor& a, const Tensor& b, Tensor& cache, int cache_offset = 0)
{
    const int nrows = a.size(0);
    const int ncols = a.size(1);

    cache.resize({nrows, ncols});

    const Float16* a_data = a.data_ptr<Float16>();
    const Float16* b_data = b.data_ptr<Float16>();
    Float16* cache_data = cache.data_ptr<Float16>();

    for (int r = cache_offset; r < nrows; r++)
    {
        const Float16* a_row_ptr = a_data + r * ncols;
        const Float16* b_row_ptr = b_data + r * ncols;
        Float16* cache_row_ptr = cache_data + r * ncols;
        vec_add(a_row_ptr, b_row_ptr, cache_row_ptr, ncols);
    }
}

/// TODO: VEC_STEP_SIZE, FMA
float dot_product(const Float16* vec_a, const Float16* vec_b, int vec_size) {
    Vec_f32x8 dot_prod = vec_f32x8_setzero();
    for (int i = 0; i < vec_size; i += 8) {
        Vec_f32x8 x0 = vec_f32x8_load(vec_a + i);
        Vec_f32x8 x1 = vec_f32x8_load(vec_b + i);
        // dot_prod += vec_f32x8_sum(vec_f32x8_mul(x0, x1));
        dot_prod = vec_f32x8_fma(x0, x1, dot_prod);
    }

    return vec_f32x8_sum(dot_prod);
}


void affine_proj_2d(const Tensor& a, const Tensor& b, const Tensor& bias, Tensor& cache, int cache_offset = 0)
{
    const int nrows0 = a.size(0);
    const int ncols = a.size(1);
    const int nrows1 = b.size(0);

    cache.resize({nrows0, nrows1});

    const Float16* a_data = a.data_ptr<Float16>();
    const Float16* b_data = b.data_ptr<Float16>();
    const Float16* bias_data = bias.data_ptr<Float16>();
    Float16* cache_data = cache.data_ptr<Float16>();

    for (int r0 = cache_offset; r0 < nrows0; r0++) {
        for (int r1 = 0; r1 < nrows1; r1++) {
            const Float16* a_row_ptr = a_data + r0 * ncols;
            const Float16* b_row_ptr = b_data + r1 * ncols;
            float dot_prod = dot_product(a_row_ptr, b_row_ptr, ncols);
            float bias_scalar = fpcvt_htos(bias_data[r1]);
            cache_data[r0 * nrows1 + r1] = fpcvt_stoh(dot_prod + bias_scalar);
        }
    }
}


void matmul_2d(const Tensor& a, const Tensor& b, Tensor& cache, int cache_offset = 0)
{
    const int nrows0 = a.size(0);
    const int ncols = a.size(1);
    const int nrows1 = b.size(0);

    cache.resize({nrows0, nrows1});

    const Float16* a_data = a.data_ptr<Float16>();
    const Float16* b_data = b.data_ptr<Float16>();
    Float16* cache_data = cache.data_ptr<Float16>();

    for (int r0 = cache_offset; r0 < nrows0; r0++) {
        for (int r1 = 0; r1 < nrows1; r1++) {
            const Float16* a_row_ptr = a_data + r0 * ncols;
            const Float16* b_row_ptr = b_data + r1 * ncols;
            float dot_prod = dot_product(a_row_ptr, b_row_ptr, ncols);
            cache_data[r0 * nrows1 + r1] = fpcvt_stoh(dot_prod);
        }
    }
}

void bias_add_inplace(Tensor& inp, const Tensor& bias, int cache_offset = 0)
{
    const int nrows = inp.size(0);
    const int ncols = inp.size(1);

    Float16* inp_data = inp.data_ptr<Float16>();
    const Float16* bias_data = bias.data_ptr<Float16>();

    for (int r = cache_offset; r < nrows; r++) {
        Float16* inp_row_ptr = inp_data + r * ncols;
        vec_add(inp_row_ptr, bias_data, inp_row_ptr, ncols);
    }
    
}


void embedding_projection(const Tensor& inp, const Tensor& emb_weight, Tensor& out)
{
    const int n_ctx = inp.size(0);
    const int n_vocab = emb_weight.size(0);
    const int n_embed = emb_weight.size(1);

    // Offsets input ptr to the start of the final vector in the inp tensor.
    const int inp_offset = (n_ctx - 1) * n_embed;
    const Float16* inp_data = inp.data_ptr<Float16>() + inp_offset;
    const Float16* emb_data = emb_weight.data_ptr<Float16>();
    float* out_data = out.data_ptr<float>();

    for (int i = 0; i < n_vocab; i++) {
        const Float16* emb_ptr = emb_data + i * n_embed;
        out_data[i] = dot_product(inp_data, emb_ptr, n_embed);
    }
}


static void qk_masked_attn_matmul(const Tensor& q, const Tensor& k, Tensor& qk_cache,
                           const int n_head, int cache_offset=0)
{
    const int q_nrows = q.size(0);
    const int ncols = q.size(1);
    const int k_nrows = k.size(0);
    const int d_head = ncols / n_head;
    const float scale_factor = 1.0f / std::sqrt((float)d_head);

    const Float16* q_data = q.data_ptr<Float16>();
    const Float16* k_data = k.data_ptr<Float16>();
    Float16* qk_data = qk_cache.data_ptr<Float16>();

    for (int h = 0; h < n_head; h++) {
        for (int qrow = cache_offset; qrow < q_nrows; qrow++) {
            const int q_row_offset = h * d_head + qrow * ncols;
            // We only compute dot products that are not masked. 'k_max' represents
            // the number of dot products that we should compute for the current q row.
            const int k_max = qrow + 1;
            for (int kcol = 0; kcol < k_max; kcol++) {
                const int k_row_offset = h * d_head + kcol * ncols;
                
                const float dot_prod = dot_product(q_data + q_row_offset, k_data + k_row_offset, d_head);

                int qk_i = h * q_nrows * k_nrows + qrow * k_nrows + kcol;
                qk_data[qk_i] = fpcvt_stoh(dot_prod * scale_factor);
            }
        }
    }

    // Do the masking.
    for (int head = 0; head < n_head; head++) {
        for (int qrow = cache_offset; qrow < q_nrows; qrow++) {
            const int kcol_start = qrow + 1;
            for (int kcol = kcol_start; kcol < k_nrows; kcol++) {
                int qk_i = head * q_nrows * k_nrows + qrow * k_nrows + kcol;
                // Use memset?
                qk_data[qk_i] = fpcvt_stoh(-std::numeric_limits<float>::infinity());
            }
        }
    }
}


static void qk_softmax(Tensor& qk_acv, int n_heads, int cache_offset=0)
{
    Float16* qk_data = qk_acv.data_ptr<Float16>();

    const int q_ctx = qk_acv.size(1);
    const int k_ctx = qk_acv.size(2);

    for (int head = 0; head < n_heads; head++) {
        for (int q_row = cache_offset; q_row < q_ctx; q_row++)
        {
            float max = -std::numeric_limits<float>::infinity();

            const int base_idx = head * q_ctx * k_ctx + q_row * k_ctx;

            for (int i = 0; i < k_ctx; i++) {
                float x = fpcvt_htos(qk_data[base_idx + i]);
                if (x > max)
                    max = x;
            }

            float sum_exp = 0;
            for (int i = 0; i < k_ctx; i++) {
                float x = fpcvt_htos(qk_data[base_idx + i]);
                float exp_val = std::exp(x - max);
                qk_data[base_idx + i] = fpcvt_stoh(exp_val);
                sum_exp += exp_val;
            }

            for (int i = 0; i < k_ctx; i++) {
                float qkw = fpcvt_htos(qk_data[base_idx + i]);
                qk_data[base_idx + i] = fpcvt_stoh(qkw / sum_exp);
            }
        }
    }
}

static void qkv_attn_matmul(const Tensor& qk, const Tensor& v, Tensor& cache, int n_heads, int cache_offset=0)
{
    const Float16* qk_data = qk.data_ptr<Float16>();
    const Float16* v_data = v.data_ptr<Float16>();
    Float16* qkv_data = cache.data_ptr<Float16>();

    const int qk_nrows = qk.size(1);
    const int qk_ncols = qk.size(2);
    const int d_embed = v.size(1);
    const int d_head = d_embed / n_heads;

    for (int h = 0; h < n_heads; h++) {
        for (int qkrow = cache_offset; qkrow < qk_nrows; qkrow++){
            for (int vcol = 0; vcol < d_head; vcol++) {
                
                float dot_prod = 0;
                for (int i = 0; i < qk_ncols; i++) {
                    int qk_i = h * qk_nrows * qk_ncols + qkrow * qk_ncols + i;
                    int v_i = h * d_head + i * d_embed + vcol;
                    float qkw = fpcvt_htos(qk_data[qk_i]);
                    float vw = fpcvt_htos(v_data[v_i]);
                    dot_prod += qkw * vw;
                }
                int qkv_data_i = h * d_head + qkrow * d_embed + vcol;
                qkv_data[qkv_data_i] = fpcvt_stoh(dot_prod);
            }
        }
    }
}


// static void qkv_attn_matmul_v2(const Tensor& qk, const Tensor& v, Tensor& cache, int n_heads, int cache_offset=0)
// {
//     const Float16* qk_data = qk.data_ptr<Float16>();
//     const Float16* v_data = v.data_ptr<Float16>();
//     Float16* cache_data = cache.data_ptr<Float16>();

//     const int qk_nrows = qk.size(1);
//     const int qk_ncols = qk.size(2);
//     const int d_embed = v.size(1);
//     const int d_head = d_embed / n_heads;

//     for (int h = 0; h < n_heads; h++) {
//         for (int qkrow = cache_offset; qkrow < qk_nrows; qkrow++){
//             for (int qkcol = 0; qkcol < qk_ncols; qkcol++) {
//                 const int qk_i = h * qk_nrows * qk_ncols + qkrow * qk_ncols + qkcol;
//                 const float qk_scalar = fpcvt_htos(qk_data[qk_i]);

//                 for (int vcol = 0; vcol < d_head; vcol++) {
//                     const int v_i = h * d_head + qkcol * d_embed + vcol;
//                     const int out_i = h * d_head + qkrow * n_heads * d_head + vcol;
//                     // This depends on cache_data being initialized to zero.
//                     float res = fpcvt_htos(cache_data[out_i]) + qk_scalar * fpcvt_htos(v_data[v_i]);
//                     cache_data[out_i] = fpcvt_stoh(res);
//                 }
//             }
//         }
//     }
// }

} // namespace ops

} // namespace gten

