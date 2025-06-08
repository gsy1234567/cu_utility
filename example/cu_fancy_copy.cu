#include "fancy_algorithm.cuh"
#include <iostream>

static constexpr bool isDebug = true;

using SliceT = gsy::Slice<isDebug>;

template<std::uint32_t Dim>
using CuNDArraySrcT = gsy::CuNDArray<int, Dim, isDebug, gsy::DeviceMemory>;

template<std::uint32_t Dim>
using CuNDArrayDstT = gsy::CuNDArray<int, Dim, isDebug, gsy::UnifiedMemory>;

template<gsy::is_ndslice<isDebug>... Slices>
using NDSliceT = gsy::NDSlice<isDebug, Slices...>;

int main() {
    try {
        CuNDArraySrcT<3> srcArr (7, 16, 256);
        CuNDArrayDstT<2> dstArr (8, 64);
        NDSliceT srcSlice (1, SliceT(4, 12), SliceT(64, 128));
        NDSliceT dstSlice (SliceT(0,8), SliceT(0,64));
        gsy::fancyCopy(dstArr, dstSlice, srcArr, srcSlice);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    return 0;
}