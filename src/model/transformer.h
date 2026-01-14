#ifndef LAI_MODEL_TRANSFORMER_H
#define LAI_MODEL_TRANSFORMER_H

#include "../core/tensor.h"
#include "../core/allocator.h"
#include "config.h"
#include <vector>
#include <memory>

namespace lai {

// Forward declarations
class KVCache;
class Tokenizer;

// Transformer layer weights
struct TransformerLayerWeights {
    // Attention
    Tensor wq;          // [dim, dim]
    Tensor wk;          // [dim, kv_dim]
    Tensor wv;          // [dim, kv_dim]
    Tensor wo;          // [dim, dim]

    // FFN (SwiGLU)
    Tensor w_gate;      // [dim, hidden_dim]
    Tensor w_up;        // [dim, hidden_dim]
    Tensor w_down;      // [hidden_dim, dim]

    // Normalization
    Tensor attn_norm;   // [dim]
    Tensor ffn_norm;    // [dim]
};

// Full model weights
struct TransformerWeights {
    // Embeddings
    Tensor token_embed;     // [vocab_size, dim]
    Tensor output_weight;   // [vocab_size, dim] (tied or separate)

    // Layers
    std::vector<TransformerLayerWeights> layers;

    // Final normalization
    Tensor final_norm;      // [dim]
};

// KV Cache for efficient autoregressive generation
class KVCache {
public:
    KVCache() = default;

    void init(const ModelConfig& config) {
        config_ = config;
        const i32 kv_dim = config.kv_dim();

        // Allocate cache for all layers
        k_cache_.resize(config.n_layers);
        v_cache_.resize(config.n_layers);

        for (i32 l = 0; l < config.n_layers; ++l) {
            k_cache_[l] = Tensor(Shape(config.max_seq_len, kv_dim));
            v_cache_[l] = Tensor(Shape(config.max_seq_len, kv_dim));
            k_cache_[l].zero();
            v_cache_[l].zero();
        }

        pos_ = 0;
    }

    void reset() {
        pos_ = 0;
        for (auto& k : k_cache_) k.zero();
        for (auto& v : v_cache_) v.zero();
    }

    // Get K/V for a layer at current position
    TensorView k_at(i32 layer, i32 pos) {
        return TensorView(
            k_cache_[layer].data_f32() + pos * config_.kv_dim(),
            Shape(config_.kv_dim()),
            DType::F32
        );
    }

    TensorView v_at(i32 layer, i32 pos) {
        return TensorView(
            v_cache_[layer].data_f32() + pos * config_.kv_dim(),
            Shape(config_.kv_dim()),
            DType::F32
        );
    }

    // Get full K/V cache up to position
    TensorView k_cache(i32 layer, i32 seq_len) {
        return TensorView(
            k_cache_[layer].data_f32(),
            Shape(seq_len, config_.kv_dim()),
            DType::F32
        );
    }

    TensorView v_cache(i32 layer, i32 seq_len) {
        return TensorView(
            v_cache_[layer].data_f32(),
            Shape(seq_len, config_.kv_dim()),
            DType::F32
        );
    }

    i32 pos() const { return pos_; }
    void advance() { ++pos_; }
    void set_pos(i32 p) { pos_ = p; }

    i64 memory_bytes() const {
        return k_cache_.size() * 2 * config_.max_seq_len * config_.kv_dim() * sizeof(f32);
    }

private:
    ModelConfig config_;
    std::vector<Tensor> k_cache_;
    std::vector<Tensor> v_cache_;
    i32 pos_ = 0;
};

// Main Transformer model
class Transformer {
public:
    Transformer() = default;

    // Initialize with config
    void init(const ModelConfig& config) {
        config_ = config;
        kv_cache_.init(config);
        allocate_buffers();
    }

    // Load weights from file
    bool load(const std::string& path);

    // Forward pass for single token (autoregressive)
    void forward(Tensor& logits, i32 token, i32 pos);

    // Forward pass for sequence (prefill)
    void forward_sequence(Tensor& logits, const std::vector<i32>& tokens);

    // Reset state
    void reset() {
        kv_cache_.reset();
    }

    // Accessors
    const ModelConfig& config() const { return config_; }
    KVCache& kv_cache() { return kv_cache_; }

    // Memory usage
    i64 memory_bytes() const {
        return weights_bytes_ + kv_cache_.memory_bytes() + buffers_bytes_;
    }

private:
    void allocate_buffers();

    // Attention computation
    void attention(TensorView& output, const TensorView& input,
                   TransformerLayerWeights& layer, i32 layer_idx, i32 pos);

    // FFN computation
    void ffn(TensorView& output, const TensorView& input,
             TransformerLayerWeights& layer);

    ModelConfig config_;
    TransformerWeights weights_;
    KVCache kv_cache_;

    // Scratch buffers (reused across layers)
    Tensor buf_x_;          // [dim]
    Tensor buf_xb_;         // [dim]
    Tensor buf_q_;          // [dim]
    Tensor buf_k_;          // [kv_dim]
    Tensor buf_v_;          // [kv_dim]
    Tensor buf_attn_;       // [max_seq_len]
    Tensor buf_ffn_;        // [hidden_dim]
    Tensor buf_ffn2_;       // [hidden_dim]

    i64 weights_bytes_ = 0;
    i64 buffers_bytes_ = 0;
};

// Implementation

inline void Transformer::allocate_buffers() {
    const i32 dim = config_.dim;
    const i32 kv_dim = config_.kv_dim();
    const i32 hidden = config_.hidden_dim;
    const i32 max_seq = config_.max_seq_len;

    buf_x_ = Tensor(Shape(dim));
    buf_xb_ = Tensor(Shape(dim));
    buf_q_ = Tensor(Shape(dim));
    buf_k_ = Tensor(Shape(kv_dim));
    buf_v_ = Tensor(Shape(kv_dim));
    buf_attn_ = Tensor(Shape(max_seq));
    buf_ffn_ = Tensor(Shape(hidden));
    buf_ffn2_ = Tensor(Shape(hidden));

    buffers_bytes_ = (dim * 4 + kv_dim * 2 + max_seq + hidden * 2) * sizeof(f32);
}

inline void Transformer::attention(TensorView& output, const TensorView& input,
                                   TransformerLayerWeights& layer, i32 layer_idx, i32 pos) {
    const i32 dim = config_.dim;
    const i32 kv_dim = config_.kv_dim();
    const i32 n_heads = config_.n_heads;
    const i32 n_kv_heads = config_.n_kv_heads;
    const i32 head_dim = config_.head_dim();
    const i32 kv_mul = n_heads / n_kv_heads;  // GQA multiplier

    // Compute Q, K, V projections
    ops::matvec(buf_q_, layer.wq, input);
    ops::matvec(buf_k_, layer.wk, input);
    ops::matvec(buf_v_, layer.wv, input);

    // Apply RoPE to Q and K
    for (i32 h = 0; h < n_heads; ++h) {
        TensorView q_head(buf_q_.data_f32() + h * head_dim, Shape(head_dim), DType::F32);
        i32 kv_h = h / kv_mul;
        TensorView k_head(buf_k_.data_f32() + kv_h * head_dim, Shape(head_dim), DType::F32);
        ops::rope(q_head, k_head, pos, head_dim, config_.rope_theta);
    }

    // Store K, V in cache
    TensorView k_pos = kv_cache_.k_at(layer_idx, pos);
    TensorView v_pos = kv_cache_.v_at(layer_idx, pos);
    simd::copy_f32(k_pos.data_f32(), buf_k_.data_f32(), kv_dim);
    simd::copy_f32(v_pos.data_f32(), buf_v_.data_f32(), kv_dim);

    // Multi-head attention
    buf_xb_.zero();
    const f32 scale = 1.0f / std::sqrt(static_cast<f32>(head_dim));

    for (i32 h = 0; h < n_heads; ++h) {
        i32 kv_h = h / kv_mul;
        f32* q = buf_q_.data_f32() + h * head_dim;
        f32* out = buf_xb_.data_f32() + h * head_dim;

        // Compute attention scores for this head
        for (i32 t = 0; t <= pos; ++t) {
            f32* k = kv_cache_.k_at(layer_idx, t).data_f32() + kv_h * head_dim;
            buf_attn_.at(t) = simd::dot_f32(q, k, head_dim) * scale;
        }

        // Softmax over scores
        TensorView attn_scores(buf_attn_.data_f32(), Shape(pos + 1), DType::F32);
        ops::softmax(attn_scores, attn_scores);

        // Weighted sum of values
        for (i32 t = 0; t <= pos; ++t) {
            f32* v = kv_cache_.v_at(layer_idx, t).data_f32() + kv_h * head_dim;
            simd::fma_f32(out, v, buf_attn_.at(t), head_dim);
        }
    }

    // Output projection
    ops::matvec(output, layer.wo, buf_xb_);
}

inline void Transformer::ffn(TensorView& output, const TensorView& input,
                             TransformerLayerWeights& layer) {
    // SwiGLU: output = down(silu(gate(x)) * up(x))
    ops::matvec(buf_ffn_, layer.w_gate, input);
    ops::matvec(buf_ffn2_, layer.w_up, input);

    // SiLU activation on gate
    ops::silu(buf_ffn_, buf_ffn_);

    // Element-wise multiply
    ops::mul(buf_ffn_, buf_ffn_, buf_ffn2_);

    // Down projection
    ops::matvec(output, layer.w_down, buf_ffn_);
}

inline void Transformer::forward(Tensor& logits, i32 token, i32 pos) {
    const i32 dim = config_.dim;

    // Token embedding
    const f32* embed = weights_.token_embed.data_f32() + token * dim;
    simd::copy_f32(buf_x_.data_f32(), embed, dim);

    // Transformer layers
    for (i32 l = 0; l < config_.n_layers; ++l) {
        auto& layer = weights_.layers[l];

        // Pre-attention norm
        ops::rmsnorm(buf_xb_, buf_x_, layer.attn_norm, config_.norm_eps);

        // Attention (writes to buf_xb_)
        attention(buf_xb_, buf_xb_, layer, l, pos);

        // Residual
        ops::add(buf_x_, buf_x_, buf_xb_);

        // Pre-FFN norm
        ops::rmsnorm(buf_xb_, buf_x_, layer.ffn_norm, config_.norm_eps);

        // FFN
        ffn(buf_xb_, buf_xb_, layer);

        // Residual
        ops::add(buf_x_, buf_x_, buf_xb_);
    }

    // Final norm
    ops::rmsnorm(buf_x_, buf_x_, weights_.final_norm, config_.norm_eps);

    // Output projection (logits)
    ops::matvec(logits, weights_.output_weight, buf_x_);
}

inline void Transformer::forward_sequence(Tensor& logits, const std::vector<i32>& tokens) {
    for (size_t i = 0; i < tokens.size(); ++i) {
        forward(logits, tokens[i], static_cast<i32>(i));
    }
    kv_cache_.set_pos(static_cast<i32>(tokens.size()));
}

inline bool Transformer::load(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;

    // Read header
    ModelHeader header;
    if (fread(&header, sizeof(header), 1, f) != 1 || !header.is_valid()) {
        fclose(f);
        return false;
    }

    config_ = header.config;
    kv_cache_.init(config_);
    allocate_buffers();

    // Allocate weights
    const i32 dim = config_.dim;
    const i32 kv_dim = config_.kv_dim();
    const i32 hidden = config_.hidden_dim;
    const i32 vocab = config_.vocab_size;

    weights_.token_embed = Tensor(Shape(vocab, dim));
    weights_.output_weight = Tensor(Shape(vocab, dim));
    weights_.final_norm = Tensor(Shape(dim));

    weights_.layers.resize(config_.n_layers);
    for (i32 l = 0; l < config_.n_layers; ++l) {
        auto& layer = weights_.layers[l];
        layer.wq = Tensor(Shape(dim, dim));
        layer.wk = Tensor(Shape(dim, kv_dim));
        layer.wv = Tensor(Shape(dim, kv_dim));
        layer.wo = Tensor(Shape(dim, dim));
        layer.w_gate = Tensor(Shape(dim, hidden));
        layer.w_up = Tensor(Shape(dim, hidden));
        layer.w_down = Tensor(Shape(hidden, dim));
        layer.attn_norm = Tensor(Shape(dim));
        layer.ffn_norm = Tensor(Shape(dim));
    }

    // Read weights (simplified - real impl would handle quantization)
    fseek(f, header.weights_offset, SEEK_SET);

    // Read embeddings
    fread(weights_.token_embed.data(), sizeof(f32), vocab * dim, f);
    fread(weights_.output_weight.data(), sizeof(f32), vocab * dim, f);
    fread(weights_.final_norm.data(), sizeof(f32), dim, f);

    // Read layer weights
    for (i32 l = 0; l < config_.n_layers; ++l) {
        auto& layer = weights_.layers[l];
        fread(layer.wq.data(), sizeof(f32), dim * dim, f);
        fread(layer.wk.data(), sizeof(f32), dim * kv_dim, f);
        fread(layer.wv.data(), sizeof(f32), dim * kv_dim, f);
        fread(layer.wo.data(), sizeof(f32), dim * dim, f);
        fread(layer.w_gate.data(), sizeof(f32), dim * hidden, f);
        fread(layer.w_up.data(), sizeof(f32), dim * hidden, f);
        fread(layer.w_down.data(), sizeof(f32), hidden * dim, f);
        fread(layer.attn_norm.data(), sizeof(f32), dim, f);
        fread(layer.ffn_norm.data(), sizeof(f32), dim, f);
    }

    fclose(f);

    weights_bytes_ = config_.memory_bytes();
    return true;
}

} // namespace lai

#endif // LAI_MODEL_TRANSFORMER_H
