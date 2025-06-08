#include "ndslice.h"
#include <iostream>
#include "fancy_algorithm.cuh"
#include "cu_unified_ndarray.h"


static constexpr bool isDebug = true;
using ValueType = int;

template<std::uint32_t Dim>
using UnifiedNDArray = gsy::CuUnifiedNDArray<ValueType, Dim, isDebug>;

using Slice = gsy::Slice<isDebug>;

template<gsy::is_ndslice<isDebug>... Slices>
using NDSlice = gsy::NDSlice<isDebug, Slices...>;

constexpr std::uint32_t nx = 128;
constexpr std::uint32_t ny = 128;

template<typename MetaData, typename Slice>
__global__ void DataSetKernel(
    const MetaData __grid_constant__ metaData, 
    const Slice __grid_constant__ slice
) {
    std::uint32_t n  = blockDim.x*blockIdx.x+threadIdx.x;
    const std::uint32_t stride = blockDim.x*gridDim.x;
    const std::uint32_t size = slice.getSize();
    for( ; n<size ; n+=stride) {
        auto index = slice.at(n);
        metaData.at(index) = n;
    }
}

int main() {
    try {
        UnifiedNDArray<2> arr(nx, ny);
        typename UnifiedNDArray<2>::MetaData metaData;
        NDSlice slice(Slice(0, nx), Slice(0, ny));
        gsy::fancyFill(arr, slice, 20);
        for(std::uint32_t y=0 ; y<ny ; ++y) {
            for(std::uint32_t x=0 ; x<nx ; ++x) {
                assert(arr.at(x,y) == 201);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}