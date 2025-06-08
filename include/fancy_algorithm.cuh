#pragma once

#include "ndslice.h"
#include "cu_ndarray.cuh"

namespace gsy {
    namespace detail {
        template<
            typename MetaDataDst, typename SliceDst, 
            typename MetaDataSrc, typename SliceSrc
        >
        __global__ void fancyCopyKernel(
            const MetaDataDst __grid_constant__ metaDataDst, const SliceDst __grid_constant__ sliceDst, 
            const MetaDataSrc __grid_constant__ metaDataSrc, const SliceSrc __grid_constant__ sliceSrc
        ) {
            std::uint32_t n = blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t size = sliceDst.getSize();
            const std::uint32_t stride = gridDim.x * blockDim.x;
            for( ; n < size ; n += stride ) {
                auto dstIdx = sliceDst.at(n);
                auto srcIdx = sliceSrc.at(n);
                metaDataDst.at(dstIdx) = metaDataSrc.at(srcIdx);
            }
        }

        template<
            typename MetaDataDst, typename SliceDst, typename ValueType
        >
        __global__ void fancyFillKernel(
            const MetaDataDst __grid_constant__ metaDataDst, const SliceDst __grid_constant__ sliceDst, const ValueType __grid_constant__ value
        ) {
            std::uint32_t n = blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t size = sliceDst.getSize();
            const std::uint32_t stride = gridDim.x * blockDim.x;
            for( ; n<size ; n+= stride) {
                auto dstIdx = sliceDst.at(n);
                metaDataDst.at(dstIdx) = value;
            }
        }
    }

    template<
        bool IsDebug, typename ValueType,
        std::uint32_t DimDst, typename MemPolicyDst, gsy::is_ndslice<IsDebug>... SlicesDst, 
        std::uint32_t DimSrc, typename MemPolicySrc, gsy::is_ndslice<IsDebug>... SlicesSrc
    >
    requires (DimDst == sizeof...(SlicesDst) and DimSrc == sizeof...(SlicesSrc))
    __host__ void fancyCopyAsync(
        const gsy::CuNDArray<ValueType, DimDst, IsDebug, MemPolicyDst>& ndarrayDst, const gsy::NDSlice<IsDebug, SlicesDst...>& slicesDst, 
        const gsy::CuNDArray<ValueType, DimSrc, IsDebug, MemPolicySrc>& ndarraySrc, const gsy::NDSlice<IsDebug, SlicesSrc...>& slicesSrc, 
        cudaStream_t stream = 0
    ) {
        assert(slicesDst.getSize() == slicesSrc.getSize());
        typename gsy::CuNDArray<ValueType, DimDst, IsDebug, MemPolicyDst>::MetaData metaDataDst;
        typename gsy::CuNDArray<ValueType, DimSrc, IsDebug, MemPolicySrc>::MetaData metaDataSrc;
        ndarrayDst.initMetaData(metaDataDst);
        ndarraySrc.initMetaData(metaDataSrc);
        detail::fancyCopyKernel<<<256, 1024, 0, stream>>>(metaDataDst, slicesDst, metaDataSrc, slicesSrc);
    }

    template<
        bool IsDebug, typename ValueType,
        std::uint32_t DimDst, typename MemPolicyDst, gsy::is_ndslice<IsDebug>... SlicesDst, 
        std::uint32_t DimSrc, typename MemPolicySrc, gsy::is_ndslice<IsDebug>... SlicesSrc
    >
    requires (DimDst == sizeof...(SlicesDst) and DimSrc == sizeof...(SlicesSrc))
    __host__ void fancyCopy(
        const gsy::CuNDArray<ValueType, DimDst, IsDebug, MemPolicyDst>& ndarrayDst, const gsy::NDSlice<IsDebug, SlicesDst...>& slicesDst, 
        const gsy::CuNDArray<ValueType, DimSrc, IsDebug, MemPolicySrc>& ndarraySrc, const gsy::NDSlice<IsDebug, SlicesSrc...>& slicesSrc
    ) {
        fancyCopyAsync(ndarrayDst, slicesDst, ndarraySrc, slicesSrc, 0);
        auto err = cudaStreamSynchronize(0);
        if(err != cudaSuccess) {
            throw CuRuntimeError(err);
        }
    }

    template<
        bool IsDebug, typename ValueType, 
        std::uint32_t DimDst, typename MemPloicyDst, gsy::is_ndslice<IsDebug>... SlicesDst
    >
    requires(DimDst == sizeof...(SlicesDst))
    __host__ void fancyFillAsync(
        const gsy::CuNDArray<ValueType, DimDst, IsDebug, MemPloicyDst>& ndarrayDst, 
        const gsy::NDSlice<IsDebug, SlicesDst...>& slicesDst, const ValueType& value, 
        cudaStream_t stream = 0
    ) {
        typename gsy::CuNDArray<ValueType, DimDst, IsDebug, MemPloicyDst>::MetaData metaData;
        ndarrayDst.initMetaData(metaData);
        detail::fancyFillKernel<<<256, 1024>>>(metaData, slicesDst, value);
    }

    template<
        bool IsDebug, typename ValueType, 
        std::uint32_t DimDst, typename MemPloicyDst, gsy::is_ndslice<IsDebug>... SlicesDst
    >
    requires(DimDst == sizeof...(SlicesDst))
    __host__ void fancyFill(
        const gsy::CuNDArray<ValueType, DimDst, IsDebug, MemPloicyDst>& ndarrayDst, const gsy::NDSlice<IsDebug, SlicesDst...>& slicesDst, const ValueType& value
    ) {
        fancyFillAsync(ndarrayDst, slicesDst, value);
        auto err = cudaStreamSynchronize(0);
        if(err != cudaSuccess) {
            throw CuRuntimeError(err);
        }
    }
}