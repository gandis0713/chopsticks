#include <spdlog/spdlog.h>

#include "litert/c/litert_common.h"

#include "litert/c/litert_model.h"

int main(int argc, char* argv[]) {
    spdlog::info("Hello World!");

    LiteRtModel model;
    LiteRtCreateModelFromFile("model.tflite", &model);

    // LiteRtEnvironment* env = LiteRtCreateEnvironment();
    // LiteRtDestroyEnvironment(env);
    
    return 0;
}