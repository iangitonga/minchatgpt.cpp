#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>

#include "model.h"


using namespace gten;
// minchatgpt

const char *usage = R"(
usage:
minchatgpt [options]

Optional args.
-lg :      Use large model (762M) for inference. The default model is medium (345M).
-debug   : See debug-level information.
--temp T : Temperature to use during sampling. It must be greater than 0. [default=0.9].
--len  L : Number of words to generate. Minimum is 1 and max is 1000. [default=200].

Examples:
  ./minchatgpt
  ./minchatgpt --temp 0.5
)";


int main(int argc, char const *argv[])
{
    InferenceOptions options{};

    for (int i = 1; i < argc; i++)
    {
        std::string_view arg(argv[i]);
        if (arg == "-h" || arg == "--help") {
            std::cout << usage << "\n";
            return -1;
        }
        else if (arg == "-lg") {
            options.model_name = "minchatgpt-lg";
        }
        else if (arg == "-debug") {
            options.debug_mode = true;
        }
        else if (arg == "-greedy") {
            options.greedy = true;
        }
        else if (arg == "-stat") {
            options.showstat = true;
        }
        else if (arg == "--temp") {
            if (argc <= i+1) {
                std::cout << "Temp value is missing.\n";
                return -1;
            }
            float temp;
            try {
                temp = std::stof(argv[i+1]);
            } catch (...) {
                std::cout << "Invalid temp value.\n";
                return -1;
            }
            if (temp <= 0.0f) {
                std::cout << "Temp value must be greater than zero.\n";
                return -1;
            }
            options.temp = temp;
            i += 1; // skip parsed temp.
        }
        else if (arg == "--len") {
            if (argc <= i+1) {
                std::cout << "Length value is missing.\n";
                return -1;
            }
            int len;
            try {
                len = std::stoi(argv[i+1]);
            } catch (...) {
                std::cout << "Invalid Length value.\n";
                return -1;
            }
            if (len < 1 || len > 1024) {
                std::cout << "Length must be greater than 1 and less than 1000.\n";
                return -1;
            }
            options.gen_tokens = len;
            i += 1;
        }
        else {
            std::cout << "Unknown option: " << arg << "\n";
            return -1;
        }
    }

    options.print_debug_info();

    int res = std::system(options.get_dl_command().c_str());
    if (res != 0) {
        std::cout << "Error: Failed to download '" << options.model_name << "' model. Check your network connection.\n";
        return -1;
    }

    std::ifstream checkpoint{options.get_model_path(), std::ios::binary};
    GTEN_ASSERT(checkpoint.is_open(), "error opening model: %s", options.get_model_path().c_str());
    verify_magic_number(checkpoint);
    GPT2Config config;
    checkpoint.read(reinterpret_cast<char*>(&config), sizeof(config));
    if (options.debug_mode) {
        std::cout << config;
    }
    GPT2Tokenizer tokenizer = load_tokenizer(checkpoint);
    const int num_prompt_tokens = tokenizer.encode(options.prompt).size();
    const int max_ctx = options.calculate_max_ctx_size(num_prompt_tokens);

    GPT2 model{checkpoint, config, max_ctx};

    std::cout << "Chat interface. Write your prompt and press enter to submit. Enter q or press ctrl+c to quit.\n";
    std::string prompt;
    while (true) {
        std::cout << "\n\x1B[0m[You]: ";
        std::getline(std::cin, prompt);
        if (prompt == "q")
            break;

        options.prompt = "Below is an instruction that describes a task, paired with an input that provides further context."
                          " Write a response that appropriately completes the request.\n### Instruction: "
                          + prompt + "\n### Input: \n### Response: ";

        if (options.greedy)
            model.greedy_sample(options, tokenizer);
        else
            model.sample(options, tokenizer);

        model.reset_acv_caches();
    }

    return 0;
}
