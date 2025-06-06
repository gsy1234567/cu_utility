#include "cu_runtime_error.h"
#include <format>


namespace gsy {
    CuRuntimeError::CuRuntimeError(cudaError_t err, std::source_location loc)
        : std::runtime_error(std::format("====CUDA RUNTIME ERROR====\n\tposition: {}:{}\n\tdescription: {}", loc.file_name(), loc.line(), cudaGetErrorString(err)))
        {}

    const char* CuRuntimeError::what() const noexcept {
        return std::runtime_error::what();
    }
}