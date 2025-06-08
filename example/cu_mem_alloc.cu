#include "cu_mem_alloc.h"
#include <iostream>
#include <assert.h>

using DevAlloc = gsy::CuMemoryAdapter<gsy::DeviceMemory>;
using UnifAlloc = gsy::CuMemoryAdapter<gsy::UnifiedMemory>;

__global__ void DataSetKernel(int* data, int size, int v);
__global__ void DataCopyKernel(const int* dSrc, int* unifDst, int size);

int main() {
    const std::size_t len = 4096;
    const std::uint32_t blockSize = 256;
    const int v = 0x1234'5678;
    const std::uint32_t gridSize = (len + blockSize - 1) / blockSize;
    int* pDev = (int*)DevAlloc::allocate(len*sizeof(int));
    int* pUnif = (int*)UnifAlloc::allocate(len*sizeof(int));
    DataSetKernel<<<gridSize, blockSize>>>(pDev, len, v);
    DataCopyKernel<<<gridSize, blockSize>>>(pDev, pUnif, len);
    cudaStreamSynchronize(0);
    for(std::size_t i=0 ; i<len ; ++i) {
        assert(pUnif[i] == v);
    }
    DevAlloc::deallocate(pDev);
    UnifAlloc::deallocate(pUnif);
    return 0;
}

__global__ void DataSetKernel(int* data, int size, int v) {
    const std::uint32_t n = blockIdx.x * blockDim.x + threadIdx.x;
    if(n < size) {
        data[n] = v;
    }
}

__global__ void DataCopyKernel(const int* dSrc, int* unifDst, int size) {
    const std::uint32_t n = blockIdx.x * blockDim.x + threadIdx.x;
    if(n < size) {
        unifDst[n] = dSrc[n];
    }
}