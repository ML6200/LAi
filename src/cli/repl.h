#ifndef LAI_CLI_REPL_H
#define LAI_CLI_REPL_H

#include "../inference/engine.h"
#include "../model/config.h"
#include <string>
#include <iostream>
#include <sstream>
#include <cstring>
#include <vector>
#include <algorithm>

namespace lai {

// ANSI color codes
namespace color {
    constexpr const char* RESET = "\033[0m";
    constexpr const char* BOLD = "\033[1m";
    constexpr const char* DIM = "\033[2m";
    constexpr const char* RED = "\033[31m";
    constexpr const char* GREEN = "\033[32m";
    constexpr const char* YELLOW = "\033[33m";
    constexpr const char* BLUE = "\033[34m";
    constexpr const char* MAGENTA = "\033[35m";
    constexpr const char* CYAN = "\033[36m";
}

// Operating mode
enum class Mode {
    CHAT,
    TRANSLATE,
    CODE,
    TEXT
};

inline const char* mode_name(Mode mode) {
    switch (mode) {
        case Mode::CHAT: return "chat";
        case Mode::TRANSLATE: return "translate";
        case Mode::CODE: return "code";
        case Mode::TEXT: return "text";
    }
    return "unknown";
}

// REPL (Read-Eval-Print Loop) for interactive use
class REPL {
public:
    REPL() : mode_(Mode::CHAT), running_(false), verbose_(false) {}

    // Initialize with model
    bool init(const std::string& model_path, const std::string& vocab_path) {
        std::cout << color::CYAN << "Loading model..." << color::RESET << std::endl;

        if (!engine_.init(model_path, vocab_path)) {
            std::cout << color::RED << "Failed to load model!" << color::RESET << std::endl;
            return false;
        }

        print_info();
        return true;
    }

    // Run interactive loop
    void run() {
        running_ = true;
        print_welcome();

        std::string line;
        while (running_) {
            print_prompt();

            if (!std::getline(std::cin, line)) {
                break;  // EOF
            }

            line = trim(line);
            if (line.empty()) continue;

            if (line[0] == '/') {
                handle_command(line);
            } else {
                handle_input(line);
            }
        }

        std::cout << "\nGoodbye!\n";
    }

    // Process single input (non-interactive)
    void process(const std::string& input) {
        handle_input(input);
    }

private:
    void print_welcome() {
        std::cout << "\n";
        std::cout << color::BOLD << color::CYAN;
        std::cout << "╔═══════════════════════════════════════════════╗\n";
        std::cout << "║     LAi - Lightweight AI Assistant            ║\n";
        std::cout << "║     Hungarian + English | CPU Optimized       ║\n";
        std::cout << "╚═══════════════════════════════════════════════╝\n";
        std::cout << color::RESET << "\n";

        std::cout << color::DIM;
        std::cout << "Type /help for commands, /quit to exit\n";
        std::cout << "Current mode: " << mode_name(mode_) << "\n";
        std::cout << color::RESET << "\n";
    }

    void print_info() {
        const auto& cfg = engine_.config();
        std::cout << color::DIM;
        std::cout << "Model: " << cfg.n_layers << " layers, "
                  << cfg.dim << " dim, "
                  << cfg.n_heads << " heads\n";
        std::cout << "Memory: " << (engine_.memory_bytes() / (1024 * 1024)) << " MB\n";
        std::cout << "Vocab: " << engine_.tokenizer().vocab_size() << " tokens\n";
        std::cout << color::RESET;
    }

    void print_prompt() {
        std::cout << color::GREEN << "[" << mode_name(mode_) << "] "
                  << color::BOLD << "> " << color::RESET;
        std::cout.flush();
    }

    void print_help() {
        std::cout << color::BOLD << "\nCommands:\n" << color::RESET;
        std::cout << "  /chat      - Switch to chat mode\n";
        std::cout << "  /translate - Switch to translation mode\n";
        std::cout << "  /code      - Switch to code assistance mode\n";
        std::cout << "  /text      - Switch to text processing mode\n";
        std::cout << "  /hu <text> - Translate English to Hungarian\n";
        std::cout << "  /en <text> - Translate Hungarian to English\n";
        std::cout << "  /temp <n>  - Set temperature (0.0-2.0)\n";
        std::cout << "  /tokens <n>- Set max tokens\n";
        std::cout << "  /reset     - Reset conversation context\n";
        std::cout << "  /stats     - Toggle statistics display\n";
        std::cout << "  /info      - Show model information\n";
        std::cout << "  /help      - Show this help\n";
        std::cout << "  /quit      - Exit\n";
        std::cout << "\n";
    }

    void handle_command(const std::string& cmd) {
        std::istringstream iss(cmd);
        std::string command;
        iss >> command;

        if (command == "/quit" || command == "/exit" || command == "/q") {
            running_ = false;
        }
        else if (command == "/help" || command == "/?") {
            print_help();
        }
        else if (command == "/chat") {
            mode_ = Mode::CHAT;
            std::cout << "Switched to chat mode\n";
        }
        else if (command == "/translate") {
            mode_ = Mode::TRANSLATE;
            std::cout << "Switched to translation mode\n";
        }
        else if (command == "/code") {
            mode_ = Mode::CODE;
            std::cout << "Switched to code assistance mode\n";
        }
        else if (command == "/text") {
            mode_ = Mode::TEXT;
            std::cout << "Switched to text processing mode\n";
        }
        else if (command == "/hu") {
            std::string text;
            std::getline(iss >> std::ws, text);
            if (!text.empty()) {
                translate_text(text, true);
            } else {
                std::cout << "Usage: /hu <english text to translate>\n";
            }
        }
        else if (command == "/en") {
            std::string text;
            std::getline(iss >> std::ws, text);
            if (!text.empty()) {
                translate_text(text, false);
            } else {
                std::cout << "Usage: /en <hungarian text to translate>\n";
            }
        }
        else if (command == "/temp") {
            f32 temp;
            if (iss >> temp && temp >= 0.0f && temp <= 2.0f) {
                gen_config_.temperature = temp;
                std::cout << "Temperature set to " << temp << "\n";
            } else {
                std::cout << "Usage: /temp <0.0-2.0>\n";
            }
        }
        else if (command == "/tokens") {
            i32 tokens;
            if (iss >> tokens && tokens > 0 && tokens <= 2048) {
                gen_config_.max_tokens = tokens;
                std::cout << "Max tokens set to " << tokens << "\n";
            } else {
                std::cout << "Usage: /tokens <1-2048>\n";
            }
        }
        else if (command == "/reset") {
            engine_.reset();
            conversation_.clear();
            std::cout << "Context reset\n";
        }
        else if (command == "/stats") {
            verbose_ = !verbose_;
            std::cout << "Statistics " << (verbose_ ? "enabled" : "disabled") << "\n";
        }
        else if (command == "/info") {
            print_info();
        }
        else {
            std::cout << color::YELLOW << "Unknown command. Type /help for commands.\n"
                      << color::RESET;
        }
    }

    void handle_input(const std::string& input) {
        GenerationStats stats;
        std::string response;

        // Stream callback - print tokens as they're generated
        auto stream_cb = [](const std::string& token, i32 /*token_id*/) -> bool {
            std::cout << token;
            std::cout.flush();
            return true;  // Continue generation
        };

        std::cout << color::CYAN;

        switch (mode_) {
            case Mode::CHAT:
                response = engine_.chat(input, system_prompt_, gen_config_,
                                       stream_cb, verbose_ ? &stats : nullptr);
                conversation_.push_back({"user", input});
                conversation_.push_back({"assistant", response});
                break;

            case Mode::TRANSLATE:
                translate_text(input, true);  // Default: to Hungarian
                return;

            case Mode::CODE:
                response = engine_.code_assist(input, "", gen_config_,
                                              stream_cb, verbose_ ? &stats : nullptr);
                break;

            case Mode::TEXT:
                response = engine_.process_text(input, "", gen_config_,
                                               stream_cb, verbose_ ? &stats : nullptr);
                break;
        }

        std::cout << color::RESET << "\n";

        if (verbose_ && stats.generated_tokens > 0) {
            print_stats(stats);
        }
    }

    void translate_text(const std::string& text, bool to_hungarian) {
        GenerationStats stats;

        auto stream_cb = [](const std::string& token, i32 /*token_id*/) -> bool {
            std::cout << token;
            std::cout.flush();
            return true;
        };

        std::cout << color::CYAN;

        engine_.translate(text, to_hungarian, gen_config_,
                         stream_cb);

        std::cout << color::RESET << "\n";
    }

    void print_stats(const GenerationStats& stats) {
        std::cout << color::DIM;
        std::cout << "\n[Stats: "
                  << stats.prompt_tokens << " prompt, "
                  << stats.generated_tokens << " generated, "
                  << std::fixed << std::setprecision(1)
                  << stats.tokens_per_second() << " tok/s, "
                  << stats.total_time_ms << " ms total]\n";
        std::cout << color::RESET;
    }

    static std::string trim(const std::string& s) {
        size_t start = s.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) return "";
        size_t end = s.find_last_not_of(" \t\r\n");
        return s.substr(start, end - start + 1);
    }

    Engine engine_;
    Mode mode_;
    GenerationConfig gen_config_;
    bool running_;
    bool verbose_;

    std::string system_prompt_ = "You are a helpful AI assistant that speaks Hungarian and English fluently. "
                                 "Respond in the same language as the user's message.";

    std::vector<std::pair<std::string, std::string>> conversation_;
};

} // namespace lai

#endif // LAI_CLI_REPL_H
