#include <iostream>
#include <memory>
#include <vector>
#include <string>

#include <spdlog/spdlog.h>

// LiteRT C++ API headers
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_model.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_options.h"

int main(int argc, char* argv[]) {
    spdlog::info("Starting LiteRT C++ API Example");

    // 1. Initialize Environment
    // The environment uses global resources and must outlive other objects.
    auto env_result = litert::Environment::Create({});
    if (!env_result) {
        spdlog::error("Failed to create LiteRT environment: {}", env_result.Error().Message());
        return 1;
    }
    auto& env = *env_result;
    spdlog::info("LiteRT Environment created");

    // 2. Load Model
    // Ensure you have a valid .tflite model file.
    std::string model_path = "model.tflite"; 
    
    // Check if the file exists (simple check before calling LiteRT)
    FILE* f = fopen(model_path.c_str(), "rb");
    if (!f) {
        spdlog::warn("Model file '{}' not found. Please place a valid .tflite model in the working directory.", model_path);
        spdlog::warn("Skipping model loading and inference steps for this run.");
        return 0;
    }
    fclose(f);

    auto model_result = litert::Model::CreateFromFile(model_path);
    if (!model_result) {
        spdlog::error("Failed to load model from {}: {}", model_path, model_result.Error().Message());
        return 1;
    }
    auto& model = *model_result;
    spdlog::info("Model loaded successfully");

    // 3. Prepare Compilation Options
    // We can specify hardware accelerators here (e.g., NPU, GPU, CPU).
    auto options_result = litert::Options::Create();
    if (!options_result) {
        spdlog::error("Failed to create compilation options");
        return 1;
    }
    auto& options = *options_result;
    // Example: Set hardware accelerator
    // options.SetHardwareAccelerators(litert::HwAccelerators::kNpu);

    // 4. Compile Model
    // The compiled model is optimized for the target device/accelerator.
    // Note: We use the helper Create() method that takes the Model object directly.
    LiteRtCompiledModel compiled_model_handle;
    if (auto status = LiteRtCreateCompiledModel(env.Get(), model.Get(), options.Get(), &compiled_model_handle); status != kLiteRtStatusOk) {
        spdlog::error("Failed to compile model: Status {}", static_cast<int>(status));
        return 1;
    }
    auto compiled_model = litert::CompiledModel::WrapCObject(model.Get(), compiled_model_handle, litert::OwnHandle::kYes);
    spdlog::info("Model compiled successfully");

    // 5. Create Input Buffers
    // This helper creates buffers matching the model's input requirements.
    auto input_buffers_result = compiled_model.CreateInputBuffers();
    if (!input_buffers_result) {
        spdlog::error("Failed to create input buffers: {}", input_buffers_result.Error().Message());
        return 1;
    }
    auto input_buffers = std::move(*input_buffers_result);
    spdlog::info("Created {} input buffer(s)", input_buffers.size());

    // 6. Fill Input Data (Example)
    // Typically you would copy image data or other inputs into these buffers.
    // Example:
    // if (!input_buffers.empty()) {
    //     auto lock = input_buffers[0].Lock();
    //     if (lock) {
    //         void* data = lock->Data();
    //         size_t size = lock->Size();
    //         // memcpy(data, source_data, size);
    //     }
    // }

    // 7. Create Output Buffers
    auto output_buffers_result = compiled_model.CreateOutputBuffers();
    if (!output_buffers_result) {
        spdlog::error("Failed to create output buffers: {}", output_buffers_result.Error().Message());
        return 1;
    }
    auto output_buffers = std::move(*output_buffers_result);
    spdlog::info("Created {} output buffer(s)", output_buffers.size());

    // 8. Run Inference
    spdlog::info("Running inference...");
    auto run_result = compiled_model.Run(input_buffers, output_buffers);
    if (!run_result) {
        spdlog::error("Inference failed: {}", run_result.Error().Message());
        return 1;
    }
    spdlog::info("Inference completed successfully");

    // 9. Process Output
    // Read data from output_buffers similarly to input buffers.

    return 0;
}