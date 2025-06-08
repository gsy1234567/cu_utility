#include "cu_ndarray.cuh"

using ndarray = gsy::CuNDArray<int, 3, true, gsy::DeviceMemory>;

__global__ void TestStoreKernel(const ndarray::MetaData __grid_constant__ params);
__global__ void TestLoadKernel(const ndarray::MetaData __grid_constant__ params);

int main() {
    ndarray arr (128, 128, 128);
    ndarray::MetaData metaData;
    arr.initMetaData(metaData);
    TestStoreKernel<<<dim3{4, 4, 128}, dim3{32, 32, 1}>>>(metaData);
    TestLoadKernel<<<dim3{4, 4, 128}, dim3{32, 32, 1}>>>(metaData);
    cudaDeviceSynchronize();
    return 0;
}

__global__ void TestStoreKernel(const ndarray::MetaData __grid_constant__ params) {
    const std::uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const std::uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;
    params.at(x, y, z) = x+y+z;
}

__global__ void TestLoadKernel(const ndarray::MetaData __grid_constant__ params) {
    const std::uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const std::uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;
    assert(params.at(x, y, z) == x+y+z);
}

