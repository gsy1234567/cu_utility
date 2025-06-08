#include "ndslice.h"
#include "cu_ndarray.cuh"
#include "cu_runtime_error.h"

static constexpr bool isDebug = false;

using MemPolicy = gsy::DeviceMemory;

using SliceT = gsy::Slice<isDebug>;

template<std::uint32_t Dim>
using CuNDArrayT = gsy::CuNDArray<int, Dim, isDebug, MemPolicy>;

template<gsy::is_ndslice<isDebug>... Slices>
using NDSliceT = gsy::NDSlice<isDebug, Slices...>;

template<std::uint32_t Dim, gsy::is_ndslice<isDebug>... Slices>
requires (sizeof...(Slices) == Dim)
__global__ void DataSetKernel(
    const typename gsy::detail::NDArrayMetaData<int, Dim, isDebug, MemPolicy> __grid_constant__ ndarray,
    const NDSliceT<Slices...> __grid_constant__ ndslice
);

template<std::uint32_t Dim, gsy::is_ndslice<isDebug>... Slices>
requires (sizeof...(Slices) == Dim)
__global__ void DataGetKernel(
    const typename gsy::detail::NDArrayMetaData<int, Dim, isDebug, MemPolicy> __grid_constant__ ndarray,
    const NDSliceT<Slices...> __grid_constant__ ndslice
);

int main() {
    CuNDArrayT<4> ndarray(16, 8, 4, 2);
    NDSliceT slice(SliceT(0, 16), SliceT(0,8), SliceT(0, 4), 1);
    CuNDArrayT<4>::MetaData metaData;
    ndarray.initMetaData(metaData);
    DataSetKernel<<<512, 512>>>(metaData, slice);
    DataGetKernel<<<512, 512>>>(metaData, slice);
    auto err = cudaStreamSynchronize(0);
    if (err != cudaSuccess) {
        throw gsy::CuRuntimeError(err);
    }
    return 0;
}

template<std::uint32_t Dim, gsy::is_ndslice<isDebug>... Slices>
requires (sizeof...(Slices) == Dim)
__global__ void DataSetKernel(
    const typename gsy::detail::NDArrayMetaData<int, Dim, isDebug, MemPolicy> __grid_constant__ ndarray,
    const NDSliceT<Slices...> __grid_constant__ ndslice
) {
    const std::uint32_t n = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t size = ndslice.getSize();
    const std::uint32_t stride = gridDim.x * blockDim.x;
    for(std::uint32_t i=n ; i < size ; i += stride) {
        auto index = ndslice.at(i);
        ndarray.at(index) = i;
    }
}

template<std::uint32_t Dim, gsy::is_ndslice<isDebug>... Slices>
requires (sizeof...(Slices) == Dim)
__global__ void DataGetKernel(
    const typename gsy::detail::NDArrayMetaData<int, Dim, isDebug, MemPolicy> __grid_constant__ ndarray, 
    const NDSliceT<Slices...> __grid_constant__ ndslice
) {
    const std::uint32_t n = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t size = ndslice.getSize();
    const std::uint32_t stride = gridDim.x * blockDim.x;
    for(std::uint32_t i=n ; i < size ; i += stride) {
        auto index = ndslice.at(i);
        assert(ndarray.at(index) == i);
    } 
}
