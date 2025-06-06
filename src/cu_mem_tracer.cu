#include <cuda_runtime.h>

#include "cu_mem_tracer.h"
#include "cu_runtime_error.h"

namespace gsy {
    CuMemTracer::memory CuMemTracer::used() {
        size_t free, total;
        auto err = cudaMemGetInfo(&free, &total);
        if(err != cudaSuccess) {
            throw CuRuntimeError(err);
        }
        auto used = total - free;
        return memory(used);
    }
}