#include "cli/repl.h"
#include "model/config.h"
#include <iostream>
#include <cstring>
#include <iomanip>

void print_usage(const char* prog) {
    std::cout << "LAi - Lightweight AI Assistant\n\n";
    std::cout << "Usage: " << prog << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -m, --model <path>    Path to model file (default: models/lai-mini.bin)\n";
    std::cout << "  -v, --vocab <path>    Path to vocabulary file (default: data/vocab.bin)\n";
    std::cout << "  -p, --prompt <text>   Single prompt mode (non-interactive)\n";
    std::cout << "  -t, --translate <text> Translate text (EN->HU)\n";
    std::cout << "  --to-en <text>        Translate text (HU->EN)\n";
    std::cout << "  --temp <float>        Temperature (default: 0.7)\n";
    std::cout << "  --tokens <int>        Max tokens to generate (default: 256)\n";
    std::cout << "  --test                Run self-tests\n";
    std::cout << "  --bench               Run benchmarks\n";
    std::cout << "  --info                Show model info and exit\n";
    std::cout << "  -h, --help            Show this help\n";
    std::cout << "\n";
    std::cout << "Examples:\n";
    std::cout << "  " << prog << "                      # Interactive mode\n";
    std::cout << "  " << prog << " -p \"Hello!\"          # Single prompt\n";
    std::cout << "  " << prog << " -t \"Hello world\"     # Translate to Hungarian\n";
    std::cout << "  " << prog << " --to-en \"Szia világ\" # Translate to English\n";
}

void run_tests() {
    std::cout << "Running tests...\n\n";

    // Test 1: Types
    std::cout << "Test 1: Types... ";
    {
        lai::Shape s(2, 3, 4);
        if (s.numel() != 24 || s.ndim != 3) {
            std::cout << "FAILED\n";
            return;
        }
    }
    std::cout << "OK\n";

    // Test 2: Tensor
    std::cout << "Test 2: Tensor... ";
    {
        lai::Tensor t(lai::Shape(4, 4));
        t.fill(1.0f);
        if (t.at(0, 0) != 1.0f || t.numel() != 16) {
            std::cout << "FAILED\n";
            return;
        }
    }
    std::cout << "OK\n";

    // Test 3: SIMD dot product
    std::cout << "Test 3: SIMD dot product... ";
    {
        float a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        float b[8] = {1, 1, 1, 1, 1, 1, 1, 1};
        float result = lai::simd::dot_f32(a, b, 8);
        if (std::abs(result - 36.0f) > 0.001f) {
            std::cout << "FAILED (got " << result << ", expected 36)\n";
            return;
        }
    }
    std::cout << "OK\n";

    // Test 4: Arena allocator
    std::cout << "Test 4: Arena allocator... ";
    {
        lai::Arena arena(1024);
        void* p1 = arena.alloc(100);
        void* p2 = arena.alloc(200);
        if (!p1 || !p2 || arena.used() < 300) {
            std::cout << "FAILED\n";
            return;
        }
        arena.reset();
        if (arena.used() != 0) {
            std::cout << "FAILED (reset)\n";
            return;
        }
    }
    std::cout << "OK\n";

    // Test 5: Model config
    std::cout << "Test 5: Model config... ";
    {
        auto cfg = lai::presets::lai_mini();
        if (cfg.dim != 512 || cfg.n_layers != 12 || cfg.head_dim() != 64) {
            std::cout << "FAILED\n";
            return;
        }
    }
    std::cout << "OK\n";

    // Test 6: Softmax
    std::cout << "Test 6: Softmax... ";
    {
        lai::Tensor x{lai::Shape{4}};
        lai::Tensor y{lai::Shape{4}};
        x.at(0) = 1.0f; x.at(1) = 2.0f; x.at(2) = 3.0f; x.at(3) = 4.0f;
        lai::ops::softmax(y, x);
        float sum = y.at(0) + y.at(1) + y.at(2) + y.at(3);
        if (std::abs(sum - 1.0f) > 0.001f) {
            std::cout << "FAILED (sum = " << sum << ")\n";
            return;
        }
    }
    std::cout << "OK\n";

    std::cout << "\nAll tests passed!\n";
}

void run_benchmarks() {
    std::cout << "Running benchmarks...\n\n";

    const int N = 512;
    const int K = 512;
    const int iters = 100;

    // Benchmark: Matrix-vector multiply
    std::cout << "Benchmark: Matrix-vector multiply (" << N << "x" << K << ")...\n";
    {
        lai::Tensor A{lai::Shape(N, K)};
        lai::Tensor x{lai::Shape(K)};
        lai::Tensor y{lai::Shape(N)};

        A.fill(0.01f);
        x.fill(1.0f);

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            lai::ops::matvec(y, A, x);
        }
        auto end = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        double ops_per_iter = 2.0 * N * K;  // multiply + add
        double gflops = (ops_per_iter * iters) / (ms * 1e6);

        std::cout << "  Time: " << std::fixed << std::setprecision(2)
                  << ms << " ms (" << iters << " iters)\n";
        std::cout << "  Performance: " << gflops << " GFLOP/s\n";
    }

    // Benchmark: Dot product
    std::cout << "\nBenchmark: Dot product (" << N * K << " elements)...\n";
    {
        std::vector<float> a(N * K, 0.01f);
        std::vector<float> b(N * K, 0.01f);

        auto start = std::chrono::high_resolution_clock::now();
        float result = 0;
        for (int i = 0; i < iters * 10; ++i) {
            result += lai::simd::dot_f32(a.data(), b.data(), N * K);
        }
        auto end = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        double ops_per_iter = 2.0 * N * K;
        double gflops = (ops_per_iter * iters * 10) / (ms * 1e6);

        std::cout << "  Time: " << std::fixed << std::setprecision(2)
                  << ms << " ms (" << iters * 10 << " iters)\n";
        std::cout << "  Performance: " << gflops << " GFLOP/s\n";
        (void)result;  // Prevent optimization
    }

    std::cout << "\nBenchmarks complete!\n";
}

int main(int argc, char* argv[]) {
    std::string model_path = "models/lai-mini.bin";
    std::string vocab_path = "data/vocab.bin";
    std::string prompt;
    std::string translate_text;
    bool to_hungarian = true;
    float temperature = 0.7f;
    int max_tokens = 256;
    bool run_test = false;
    bool run_bench = false;
    bool show_info = false;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        else if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) {
            if (++i < argc) model_path = argv[i];
        }
        else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--vocab") == 0) {
            if (++i < argc) vocab_path = argv[i];
        }
        else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--prompt") == 0) {
            if (++i < argc) prompt = argv[i];
        }
        else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--translate") == 0) {
            if (++i < argc) {
                translate_text = argv[i];
                to_hungarian = true;
            }
        }
        else if (strcmp(argv[i], "--to-en") == 0) {
            if (++i < argc) {
                translate_text = argv[i];
                to_hungarian = false;
            }
        }
        else if (strcmp(argv[i], "--temp") == 0) {
            if (++i < argc) temperature = std::stof(argv[i]);
        }
        else if (strcmp(argv[i], "--tokens") == 0) {
            if (++i < argc) max_tokens = std::stoi(argv[i]);
        }
        else if (strcmp(argv[i], "--test") == 0) {
            run_test = true;
        }
        else if (strcmp(argv[i], "--bench") == 0) {
            run_bench = true;
        }
        else if (strcmp(argv[i], "--info") == 0) {
            show_info = true;
        }
        else {
            std::cerr << "Unknown option: " << argv[i] << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // Run tests
    if (run_test) {
        run_tests();
        return 0;
    }

    // Run benchmarks
    if (run_bench) {
        run_benchmarks();
        return 0;
    }

    // Show info
    if (show_info) {
        auto cfg = lai::presets::lai_mini();
        std::cout << "LAi-Mini Model Configuration:\n";
        std::cout << "  Layers: " << cfg.n_layers << "\n";
        std::cout << "  Dimension: " << cfg.dim << "\n";
        std::cout << "  Heads: " << cfg.n_heads << "\n";
        std::cout << "  Head dim: " << cfg.head_dim() << "\n";
        std::cout << "  FFN hidden: " << cfg.hidden_dim << "\n";
        std::cout << "  Vocab size: " << cfg.vocab_size << "\n";
        std::cout << "  Max seq len: " << cfg.max_seq_len << "\n";
        std::cout << "  Parameters: ~" << (cfg.param_count() / 1000000) << "M\n";
        std::cout << "  Memory (F32): ~" << (cfg.memory_bytes() / (1024 * 1024)) << " MB\n";
        std::cout << "  Memory (Q4): ~" << (cfg.memory_bytes(lai::DType::Q4_0) / (1024 * 1024)) << " MB\n";
        return 0;
    }

    // Interactive mode or single prompt
    lai::REPL repl;

    // Check if model exists, if not, print helpful message
    FILE* f = fopen(model_path.c_str(), "rb");
    if (!f) {
        std::cout << "\n";
        std::cout << lai::color::YELLOW;
        std::cout << "Model not found at: " << model_path << "\n\n";
        std::cout << "To use LAi, you need to:\n";
        std::cout << "1. Train a model using the training scripts in training/\n";
        std::cout << "2. Or download a pre-trained model\n\n";
        std::cout << "For now, you can run:\n";
        std::cout << "  ./lai --test   # Run self-tests\n";
        std::cout << "  ./lai --bench  # Run benchmarks\n";
        std::cout << "  ./lai --info   # Show model specs\n";
        std::cout << lai::color::RESET << "\n";
        return 1;
    }
    fclose(f);

    if (!repl.init(model_path, vocab_path)) {
        return 1;
    }

    // Non-interactive modes
    if (!translate_text.empty()) {
        lai::Engine engine;
        engine.init(model_path, vocab_path);

        lai::GenerationConfig cfg;
        cfg.temperature = temperature;
        cfg.max_tokens = max_tokens;

        auto stream_cb = [](const std::string& token, lai::i32) -> bool {
            std::cout << token;
            std::cout.flush();
            return true;
        };

        engine.translate(translate_text, to_hungarian, cfg, stream_cb);
        std::cout << "\n";
        return 0;
    }

    if (!prompt.empty()) {
        repl.process(prompt);
        return 0;
    }

    // Interactive mode
    repl.run();

    return 0;
}
