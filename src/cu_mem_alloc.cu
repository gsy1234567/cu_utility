#include "cu_mem_alloc.h"
#include "cuda_runtime.h"
#include <cu_runtime_error.h>

namespace gsy {
    template<>
    void* CuMemoryAdapter<DeviceMemory>::allocate(std::size_t size) {
        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, size);
        if(err != cudaSuccess) {
            throw CuRuntimeError(err);
        }
        return ptr;
    }

    template<>
    void CuMemoryAdapter<DeviceMemory>::deallocate(void* ptr) noexcept {
        cudaFree(ptr);
    }

    template<>
    void* CuMemoryAdapter<UnifiedMemory>::allocate(std::size_t size) {
        void* ptr = nullptr;
        cudaError_t err = cudaMallocManaged(&ptr, size);
        if(err != cudaSuccess) {
            throw CuRuntimeError(err);
        }
        return ptr;
    }

    template<>
    void CuMemoryAdapter<UnifiedMemory>::deallocate(void* ptr) noexcept {
        cudaFree(ptr);
    }
}