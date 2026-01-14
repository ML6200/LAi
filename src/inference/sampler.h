#ifndef LAI_INFERENCE_SAMPLER_H
#define LAI_INFERENCE_SAMPLER_H

#include "../core/types.h"
#include "../core/tensor.h"
#include "../model/config.h"
#include <random>
#include <algorithm>
#include <vector>
#include <cmath>

namespace lai {

// Token sampler with various strategies
class Sampler {
public:
    Sampler() : rng_(std::random_device{}()) {}

    explicit Sampler(u32 seed) : rng_(seed) {}

    void set_seed(u32 seed) {
        rng_.seed(seed);
    }

    // Sample next token from logits
    i32 sample(const TensorView& logits, const GenerationConfig& config,
               const std::vector<i32>& recent_tokens = {}) {
        const i64 vocab_size = logits.numel();
        const f32* logits_data = logits.data_f32();

        // Copy logits to work buffer
        std::vector<f32> probs(vocab_size);
        for (i64 i = 0; i < vocab_size; ++i) {
            probs[i] = logits_data[i];
        }

        // Apply repetition penalty
        if (config.repeat_penalty != 1.0f && !recent_tokens.empty()) {
            for (i32 token : recent_tokens) {
                if (token >= 0 && token < vocab_size) {
                    if (probs[token] > 0) {
                        probs[token] /= config.repeat_penalty;
                    } else {
                        probs[token] *= config.repeat_penalty;
                    }
                }
            }
        }

        // Temperature scaling
        if (config.temperature != 1.0f && config.temperature > 0.0f) {
            f32 inv_temp = 1.0f / config.temperature;
            for (i64 i = 0; i < vocab_size; ++i) {
                probs[i] *= inv_temp;
            }
        }

        // Greedy decoding (temperature = 0)
        if (config.temperature <= 0.0f) {
            return static_cast<i32>(std::max_element(probs.begin(), probs.end()) - probs.begin());
        }

        // Softmax
        f32 max_val = *std::max_element(probs.begin(), probs.end());
        f32 sum = 0.0f;
        for (i64 i = 0; i < vocab_size; ++i) {
            probs[i] = std::exp(probs[i] - max_val);
            sum += probs[i];
        }
        for (i64 i = 0; i < vocab_size; ++i) {
            probs[i] /= sum;
        }

        // Top-k filtering
        if (config.top_k > 0 && config.top_k < vocab_size) {
            apply_top_k(probs, config.top_k);
        }

        // Top-p (nucleus) filtering
        if (config.top_p < 1.0f && config.top_p > 0.0f) {
            apply_top_p(probs, config.top_p);
        }

        // Sample from distribution
        return sample_from_probs(probs);
    }

    // Argmax (greedy) sampling
    i32 argmax(const TensorView& logits) {
        const i64 n = logits.numel();
        const f32* data = logits.data_f32();

        i32 best = 0;
        f32 best_val = data[0];
        for (i64 i = 1; i < n; ++i) {
            if (data[i] > best_val) {
                best_val = data[i];
                best = static_cast<i32>(i);
            }
        }
        return best;
    }

private:
    void apply_top_k(std::vector<f32>& probs, i32 k) {
        // Find k-th largest value
        std::vector<f32> sorted = probs;
        std::partial_sort(sorted.begin(), sorted.begin() + k, sorted.end(), std::greater<f32>());
        f32 threshold = sorted[k - 1];

        // Zero out values below threshold
        f32 sum = 0.0f;
        for (size_t i = 0; i < probs.size(); ++i) {
            if (probs[i] < threshold) {
                probs[i] = 0.0f;
            } else {
                sum += probs[i];
            }
        }

        // Renormalize
        if (sum > 0.0f) {
            for (f32& p : probs) {
                p /= sum;
            }
        }
    }

    void apply_top_p(std::vector<f32>& probs, f32 top_p) {
        // Sort by probability
        std::vector<std::pair<f32, i32>> sorted;
        sorted.reserve(probs.size());
        for (size_t i = 0; i < probs.size(); ++i) {
            if (probs[i] > 0.0f) {
                sorted.push_back({probs[i], static_cast<i32>(i)});
            }
        }
        std::sort(sorted.begin(), sorted.end(), std::greater<std::pair<f32, i32>>());

        // Find cutoff
        f32 cumsum = 0.0f;
        size_t cutoff = sorted.size();
        for (size_t i = 0; i < sorted.size(); ++i) {
            cumsum += sorted[i].first;
            if (cumsum >= top_p) {
                cutoff = i + 1;
                break;
            }
        }

        // Zero out tokens beyond cutoff
        std::fill(probs.begin(), probs.end(), 0.0f);
        f32 sum = 0.0f;
        for (size_t i = 0; i < cutoff; ++i) {
            probs[sorted[i].second] = sorted[i].first;
            sum += sorted[i].first;
        }

        // Renormalize
        if (sum > 0.0f) {
            for (f32& p : probs) {
                p /= sum;
            }
        }
    }

    i32 sample_from_probs(const std::vector<f32>& probs) {
        std::uniform_real_distribution<f32> dist(0.0f, 1.0f);
        f32 r = dist(rng_);

        f32 cumsum = 0.0f;
        for (size_t i = 0; i < probs.size(); ++i) {
            cumsum += probs[i];
            if (r < cumsum) {
                return static_cast<i32>(i);
            }
        }

        // Fallback to last token (shouldn't happen)
        return static_cast<i32>(probs.size() - 1);
    }

    std::mt19937 rng_;
};

} // namespace lai

#endif // LAI_INFERENCE_SAMPLER_H
