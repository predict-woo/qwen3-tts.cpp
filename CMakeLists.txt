cmake_minimum_required(VERSION 3.14)
project(qwen3-tts-ggml VERSION 0.1.0 LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
endif()
if(MSVC)
    add_compile_definitions(_USE_MATH_DEFINES NOMINMAX)
endif()

# Timing instrumentation option
option(QWEN3_TTS_TIMING "Enable detailed timing instrumentation" OFF)
if(QWEN3_TTS_TIMING)
    add_compile_definitions(QWEN3_TTS_TIMING)
endif()

# CoreML code predictor bridge (macOS only)
option(QWEN3_TTS_COREML "Enable CoreML code predictor bridge on macOS" ON)
if(APPLE AND QWEN3_TTS_COREML)
    enable_language(OBJCXX)
endif()

# Find threads
find_package(Threads REQUIRED)

# GGML library paths
set(GGML_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ggml")
set(GGML_BUILD_DIR "${GGML_DIR}/build")

# Detect Metal backend availability
if(APPLE AND EXISTS "${GGML_BUILD_DIR}/src/ggml-metal/libggml-metal.dylib")
    set(GGML_HAS_METAL ON)
    message(STATUS "GGML Metal backend found — GPU acceleration enabled")
else()
    set(GGML_HAS_METAL OFF)
endif()

# Common library (gguf_loader + coreml bridge — compiled once, shared by all)
set(COMMON_SOURCES
    src/common/backend_threads.cpp
    src/common/benchmark_json.cpp
    src/common/gguf_loader.cpp
    src/common/speaker_embedding_io.cpp
)
if(APPLE AND QWEN3_TTS_COREML)
    list(APPEND COMMON_SOURCES src/common/coreml_code_predictor.mm)
    set_source_files_properties(src/common/coreml_code_predictor.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
else()
    list(APPEND COMMON_SOURCES src/common/coreml_code_predictor_stub.cpp)
endif()

add_library(qwen3_tts_common STATIC ${COMMON_SOURCES})
target_include_directories(qwen3_tts_common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_directories(qwen3_tts_common PUBLIC
    ${GGML_BUILD_DIR}/src
    ${GGML_BUILD_DIR}/src/ggml-metal
)
target_link_libraries(qwen3_tts_common PUBLIC
    ggml
    ggml-base
    ggml-cpu
    Threads::Threads
)
if(GGML_HAS_METAL)
    target_link_libraries(qwen3_tts_common PUBLIC ggml-metal "-framework Metal" "-framework MetalKit")
endif()
if(APPLE AND QWEN3_TTS_COREML)
    target_link_libraries(qwen3_tts_common PUBLIC "-framework Foundation" "-framework CoreML")
endif()

# Text tokenizer library
add_library(text_tokenizer STATIC
    src/tokenizer/text_tokenizer.cpp
    src/tokenizer/tokenizer_unicode.cpp
    src/tokenizer/tokenizer_unicode_data.cpp
)
target_link_libraries(text_tokenizer PUBLIC qwen3_tts_common)

# TTS Transformer library
add_library(tts_transformer STATIC src/transformer/tts_transformer.cpp)
target_link_libraries(tts_transformer PUBLIC qwen3_tts_common)

# Audio tokenizer encoder library (ECAPA-TDNN x-vector speaker encoder)
add_library(audio_tokenizer_encoder STATIC src/encoder/audio_tokenizer_encoder.cpp)
target_link_libraries(audio_tokenizer_encoder PUBLIC qwen3_tts_common)

# Audio codec encoder library (Mimi codec for ICL voice cloning)
add_library(audio_codec_encoder STATIC src/encoder/audio_codec_encoder.cpp)
target_link_libraries(audio_codec_encoder PUBLIC qwen3_tts_common)

# Audio tokenizer decoder library
add_library(audio_tokenizer_decoder STATIC src/decoder/audio_tokenizer_decoder.cpp)
target_link_libraries(audio_tokenizer_decoder PUBLIC qwen3_tts_common)

# Qwen3 TTS library (full pipeline)
add_library(qwen3_tts STATIC src/pipeline/qwen3_tts.cpp)
target_link_libraries(qwen3_tts PUBLIC
    text_tokenizer
    tts_transformer
    audio_tokenizer_encoder
    audio_codec_encoder
    audio_tokenizer_decoder
)

# Shared library with C API (for Nim FFI)
add_library(qwen3tts_shared SHARED
    src/pipeline/qwen3tts_c_api.cpp
)
set_target_properties(qwen3tts_shared PROPERTIES
    OUTPUT_NAME "qwen3tts"
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
)
target_include_directories(qwen3tts_shared PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(qwen3tts_shared PRIVATE
    qwen3_tts
)

# CLI executable
add_executable(qwen3-tts-cli
    src/main.cpp
)
target_link_libraries(qwen3-tts-cli PRIVATE
    qwen3_tts
)

# Quantization tool
add_executable(qwen3-tts-quantize
    src/qwen3_tts_quantize.cpp
)
target_link_libraries(qwen3-tts-quantize PRIVATE
    qwen3_tts_common
)

# Test executable for tokenizer
add_executable(test_tokenizer
    tests/test_tokenizer.cpp
)
target_link_libraries(test_tokenizer PRIVATE
    text_tokenizer
    Threads::Threads
)

# Test executable for audio encoder
add_executable(test_encoder
    tests/test_encoder.cpp
)
target_link_libraries(test_encoder PRIVATE
    audio_tokenizer_encoder
    Threads::Threads
)

# Test executable for transformer
add_executable(test_transformer
    tests/test_transformer.cpp
)
target_link_libraries(test_transformer PRIVATE
    tts_transformer
    Threads::Threads
)

# Test executable for Mimi codec encoder (ICL voice cloning)
add_executable(test_codec_encoder
    tests/test_codec_encoder.cpp
)
target_link_libraries(test_codec_encoder PRIVATE
    audio_codec_encoder
    qwen3_tts
    Threads::Threads
)

# Test executable for audio decoder
add_executable(test_decoder
    tests/test_decoder.cpp
)
target_link_libraries(test_decoder PRIVATE
    audio_tokenizer_decoder
    Threads::Threads
)

# Test executable for benchmark JSON helper
add_executable(test_benchmark_json
    tests/test_benchmark_json.cpp
)
target_link_libraries(test_benchmark_json PRIVATE
    qwen3_tts_common
    Threads::Threads
)

# Test executable for backend thread helper
add_executable(test_backend_threads
    tests/test_backend_threads.cpp
)
target_link_libraries(test_backend_threads PRIVATE
    qwen3_tts
    Threads::Threads
)

# Install targets
install(TARGETS text_tokenizer tts_transformer audio_tokenizer_encoder audio_codec_encoder audio_tokenizer_decoder qwen3_tts qwen3tts_shared qwen3-tts-cli
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(FILES
    src/common/backend_threads.h
    src/common/gguf_loader.h
    src/common/coreml_code_predictor.h
    src/tokenizer/text_tokenizer.h
    src/transformer/tts_transformer.h
    src/encoder/audio_tokenizer_encoder.h
    src/decoder/audio_tokenizer_decoder.h
    src/pipeline/qwen3_tts.h
    src/pipeline/qwen3tts_c_api.h
    DESTINATION include
)

# CTest support
enable_testing()
add_test(NAME tokenizer_test
    COMMAND test_tokenizer
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME encoder_test
    COMMAND test_encoder --tokenizer models/qwen3-tts-0.6b-f16.gguf --audio clone.wav --reference reference/ref_audio_embedding.bin
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME transformer_test
    COMMAND test_transformer
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME decoder_test
    COMMAND test_decoder
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME benchmark_json_test
    COMMAND test_benchmark_json
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME backend_threads_test
    COMMAND test_backend_threads
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
