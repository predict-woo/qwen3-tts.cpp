#include "common/gguf_loader.h"
#include "common/backend_threads.h"

#include <cerrno>
#include <climits>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>

namespace qwen3_tts {

namespace {
struct shared_backend_state {
    ggml_backend_t backend = nullptr;
    int32_t ref_count = 0;
};

shared_backend_state & get_shared_backend_state() {
    static shared_backend_state state;
    return state;
}

enum class backend_mode {
    AUTO,
    CPU,
    CUDA,
};

bool iequals(const char * a, const char * b) {
    if (!a || !b) {
        return false;
    }
    while (*a && *b) {
        if (std::tolower((unsigned char) *a) != std::tolower((unsigned char) *b)) {
            return false;
        }
        ++a;
        ++b;
    }
    return *a == '\0' && *b == '\0';
}

bool parse_non_negative_int(const char * s, int & out) {
    if (!s || s[0] == '\0') {
        return false;
    }

    errno = 0;
    char * end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v < 0 || v > INT_MAX) {
        return false;
    }

    out = (int) v;
    return true;
}

backend_mode get_backend_mode_from_env() {
    const char * env = std::getenv("QWEN3_TTS_BACKEND");
    if (!env || env[0] == '\0' || iequals(env, "auto")) {
        return backend_mode::AUTO;
    }
    if (iequals(env, "cpu")) {
        return backend_mode::CPU;
    }
    if (iequals(env, "cuda")) {
        return backend_mode::CUDA;
    }

    fprintf(stderr, "  [backend] Unknown QWEN3_TTS_BACKEND=%s, using auto\n", env);
    return backend_mode::AUTO;
}

int get_cuda_device_index_from_env() {
    const char * env = std::getenv("QWEN3_TTS_DEVICE");
    if (!env || env[0] == '\0') {
        return -1; // first CUDA device
    }

    int parsed = -1;
    if (!parse_non_negative_int(env, parsed)) {
        fprintf(stderr, "  [backend] Invalid QWEN3_TTS_DEVICE=%s, using default CUDA device\n", env);
        return -1;
    }
    return parsed;
}

ggml_backend_t init_cuda_backend_from_env() {
    const int target_cuda_idx = get_cuda_device_index_from_env();
    int matched_cuda_devices = 0;

    const size_t n_devs = ggml_backend_dev_count();
    for (size_t i = 0; i < n_devs; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        const char * reg_name = reg ? ggml_backend_reg_name(reg) : nullptr;
        if (!reg_name || !iequals(reg_name, "CUDA")) {
            continue;
        }

        if (target_cuda_idx >= 0 && matched_cuda_devices != target_cuda_idx) {
            matched_cuda_devices++;
            continue;
        }

        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        if (backend) {
            return backend;
        }

        if (target_cuda_idx >= 0) {
            break;
        }
        matched_cuda_devices++;
    }

    if (target_cuda_idx >= 0) {
        fprintf(stderr, "  [backend] Requested CUDA device index %d not available\n", target_cuda_idx);
    }

    return nullptr;
}

ggml_backend_t init_tensor_loader_backend(enum ggml_backend_dev_type preferred_backend_type) {
    const backend_mode mode = get_backend_mode_from_env();

    if (mode == backend_mode::CPU) {
        return ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }
    if (mode == backend_mode::CUDA) {
        ggml_backend_t backend = init_cuda_backend_from_env();
        if (!backend) {
            backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        }
        return backend;
    }

    ggml_backend_t backend = ggml_backend_init_by_type(preferred_backend_type, nullptr);
    if (!backend && preferred_backend_type != GGML_BACKEND_DEVICE_TYPE_CPU) {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }
    return backend;
}
}

GGUFLoader::GGUFLoader() = default;

GGUFLoader::~GGUFLoader() {
    close();
}

ggml_backend_t init_preferred_backend(const char * component_name, std::string * error_msg) {
    if (error_msg) error_msg->clear();

    auto & shared = get_shared_backend_state();
    if (shared.backend) {
        apply_backend_n_threads(shared.backend, get_default_backend_n_threads());
        shared.ref_count++;
        return shared.backend;
    }

    ggml_backend_t backend = nullptr;
    const backend_mode mode = get_backend_mode_from_env();
    if (mode == backend_mode::CPU) {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    } else if (mode == backend_mode::CUDA) {
        backend = init_cuda_backend_from_env();
        if (!backend) {
            fprintf(stderr, "  [backend] CUDA requested but unavailable, falling back to CPU\n");
            backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        }
    } else {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr);
        if (!backend) {
            backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
        }
        if (!backend) {
            backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_ACCEL, nullptr);
        }
        if (!backend) {
            backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        }
    }

    if (!backend && error_msg) {
        const char * name = component_name ? component_name : "component";
        *error_msg = "Failed to initialize backend for " + std::string(name)
            + " (QWEN3_TTS_BACKEND=auto|cpu|cuda)";
    }

    if (backend) {
        apply_backend_n_threads(backend, get_default_backend_n_threads());
        shared.backend = backend;
        shared.ref_count = 1;
    }

    return backend;
}

bool apply_default_backend_threads_to_preferred_backend() {
    auto & shared = get_shared_backend_state();
    if (!shared.backend) {
        return false;
    }
    return apply_backend_n_threads(shared.backend, get_default_backend_n_threads());
}

void release_preferred_backend(ggml_backend_t backend) {
    if (!backend) {
        return;
    }

    auto & shared = get_shared_backend_state();
    if (shared.backend == backend) {
        shared.ref_count--;
        if (shared.ref_count <= 0) {
            ggml_backend_free(shared.backend);
            shared.backend = nullptr;
            shared.ref_count = 0;
        }
        return;
    }

    ggml_backend_free(backend);
}

bool GGUFLoader::open(const std::string & path) {
    close();  // Close any previously opened file
    
    file_path_ = path;
    
    struct gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &meta_ctx_,
    };
    
    ctx_ = gguf_init_from_file(path.c_str(), params);
    if (!ctx_) {
        error_msg_ = "Failed to open GGUF file: " + path;
        return false;
    }
    
    return true;
}

void GGUFLoader::close() {
    if (ctx_) {
        gguf_free(ctx_);
        ctx_ = nullptr;
    }
    if (meta_ctx_) {
        ggml_free(meta_ctx_);
        meta_ctx_ = nullptr;
    }
    file_path_.clear();
}

int64_t GGUFLoader::get_n_tensors() const {
    if (!ctx_) return 0;
    return gguf_get_n_tensors(ctx_);
}

const char * GGUFLoader::get_tensor_name(int64_t idx) const {
    if (!ctx_) return nullptr;
    return gguf_get_tensor_name(ctx_, idx);
}

enum ggml_type GGUFLoader::get_tensor_type(int64_t idx) const {
    if (!ctx_) return GGML_TYPE_F32;
    return gguf_get_tensor_type(ctx_, idx);
}

size_t GGUFLoader::get_tensor_offset(int64_t idx) const {
    if (!ctx_) return 0;
    return gguf_get_tensor_offset(ctx_, idx);
}

size_t GGUFLoader::get_tensor_size(int64_t idx) const {
    if (!ctx_) return 0;
    return gguf_get_tensor_size(ctx_, idx);
}

int32_t GGUFLoader::get_u32(const char * key, int32_t default_val) const {
    if (!ctx_) return default_val;
    int64_t idx = gguf_find_key(ctx_, key);
    if (idx < 0) return default_val;
    return (int32_t)gguf_get_val_u32(ctx_, idx);
}

float GGUFLoader::get_f32(const char * key, float default_val) const {
    if (!ctx_) return default_val;
    int64_t idx = gguf_find_key(ctx_, key);
    if (idx < 0) return default_val;
    return gguf_get_val_f32(ctx_, idx);
}

size_t GGUFLoader::get_data_offset() const {
    if (!ctx_) return 0;
    return gguf_get_data_offset(ctx_);
}

bool load_tensor_data_from_file(
    const std::string & path,
    struct gguf_context * ctx,
    struct ggml_context * model_ctx,
    const std::map<std::string, struct ggml_tensor *> & tensors,
    ggml_backend_buffer_t & buffer,
    std::string & error_msg,
    enum ggml_backend_dev_type preferred_backend_type
) {
    ggml_backend_t backend = init_tensor_loader_backend(preferred_backend_type);
    if (!backend) {
        error_msg = "Failed to initialize backend for GGUF tensor loader";
        return false;
    }
    
    // Allocate buffer for all tensors
    buffer = ggml_backend_alloc_ctx_tensors(model_ctx, backend);
    if (!buffer) {
        error_msg = "Failed to allocate tensor buffer";
        ggml_backend_free(backend);
        return false;
    }
    
    // Open file for reading tensor data
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        error_msg = "Failed to open file for reading: " + path;
        ggml_backend_free(backend);
        return false;
    }
    
    const size_t data_offset = gguf_get_data_offset(ctx);
    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    std::vector<uint8_t> read_buf;
    
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx, i);
        size_t offset = gguf_get_tensor_offset(ctx, i);
        
        auto it = tensors.find(name);
        if (it == tensors.end()) {
            continue;  // Skip tensors not in our map
        }
        
        struct ggml_tensor * tensor = it->second;
        size_t nbytes = ggml_nbytes(tensor);
        
        read_buf.resize(nbytes);
        
#ifdef _WIN32
        if (_fseeki64(f, (int64_t)(data_offset + offset), SEEK_SET) != 0) {
#else
        if (fseek(f, data_offset + offset, SEEK_SET) != 0) {
#endif
            error_msg = "Failed to seek to tensor data: " + std::string(name);
            fclose(f);
            ggml_backend_free(backend);
            return false;
        }
        
        if (fread(read_buf.data(), 1, nbytes, f) != nbytes) {
            error_msg = "Failed to read tensor data: " + std::string(name);
            fclose(f);
            ggml_backend_free(backend);
            return false;
        }
        
        ggml_backend_tensor_set(tensor, read_buf.data(), 0, nbytes);
    }
    
    fclose(f);
    ggml_backend_free(backend);
    
    return true;
}

void free_ggml_resources(struct ggml_context * ctx, ggml_backend_buffer_t buffer) {
    if (buffer) {
        ggml_backend_buffer_free(buffer);
    }
    if (ctx) {
        ggml_free(ctx);
    }
}

bool backend_is_gpu(ggml_backend_t backend) {
    if (!backend) return false;
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (!dev) return false;
    enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
    return type == GGML_BACKEND_DEVICE_TYPE_GPU ||
           type == GGML_BACKEND_DEVICE_TYPE_IGPU ||
           type == GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

} // namespace qwen3_tts
