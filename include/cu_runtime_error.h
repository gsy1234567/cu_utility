#pragma once

#include <stdexcept>
#include <cuda_runtime.h>
#include <source_location>

namespace gsy {
    struct CuRuntimeError : public std::runtime_error {
        CuRuntimeError(cudaError_t err, std::source_location loc = std::source_location::current());
        virtual const char* what() const noexcept override;
    };
}