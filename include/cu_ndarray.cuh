#pragma once

#include <array>
#include <cstdint>
#include <assert.h>
#include <type_traits>
#include "cu_runtime_error.h"

namespace gsy {

    template<typename ValueType, std::uint32_t Dim, bool IsDebug>
    class CuNDArray;

    namespace detail {
        template<typename ValueType, std::uint32_t Dim, bool IsDebug>
        class NDArrayMetaData;

        template<std::uint32_t I, std::uint32_t N, typename... Indices>
        __host__ __device__ std::uint32_t flatIndex1 (const std::array<std::uint32_t,N>& axisWeights, std::uint32_t n, Indices... indices) {
            static_assert((I + sizeof...(Indices)) == N);
            if constexpr(I == 0 and sizeof...(Indices) == 0) {
                return n;
            }
            else if constexpr(I == 0 and sizeof...(Indices) != 0) {
                return n + flatIndex1<I+1,N>(axisWeights, indices...);
            } else if constexpr(I > 0 and sizeof...(Indices) == 0) {
                return n*axisWeights[I-1];
            } else {
                return n*axisWeights[I-1] + flatIndex1<I+1,N>(axisWeights, indices...);
            }
        }

        template<std::uint32_t I, std::uint32_t N, typename... Indices>
        __host__ __device__ std::uint32_t flatIndex2 (const std::array<std::uint32_t, N>& shape, std::uint32_t n, Indices... indices) {
            static_assert((I+sizeof...(Indices)+1) == N);
            if constexpr(sizeof...(Indices) == 0) {
                return n;
            } else {
                return n + shape[I]*flatIndex2<I+1,N>(shape, indices...);
            }
        }

        template<std::uint32_t I, std::uint32_t N, typename... Indices>
        __host__ __device__ bool validateIndices(const std::array<std::uint32_t, N>& shape, std::uint32_t n, Indices... indices) {
            static_assert((I+sizeof...(Indices)+1) == N);
            if constexpr(sizeof...(Indices) == 0) {
                return n < shape[I];
            } else {
                return n < shape[I] and validateIndices<I+1,N>(shape, indices...);
            }
        }

        template<std::uint32_t N, std::uint32_t I=0>
        __host__ __device__ void getAxisWeights(std::array<std::uint32_t, N-1>& axisWeights, const std::array<std::uint32_t, N>& shape) {
            if constexpr (I == 0) {
                axisWeights[0] = shape[0];
            } 
            else if constexpr(I < N-1 ) {
                axisWeights[I] = shape[I] * axisWeights[I-1]; 
            }
            if constexpr(I < N-1) {
                getAxisWeights<N, I+1>(axisWeights, shape);
            }
        }
    }



    template<typename ValueType, std::uint32_t Dim, bool IsDebug>
    class CuNDArray {
        private:
            std::array<std::uint32_t, Dim> _shape;
            ValueType* _data;
        public:
            using MetaData = detail::NDArrayMetaData<ValueType, Dim, IsDebug>;
            __host__ CuNDArray() : _shape(), _data(nullptr) {}

            template<typename... Shape>
            __host__ CuNDArray(Shape... shape) 
                : _shape{static_cast<std::uint32_t>(shape)...} {
                    auto err = cudaMalloc((void**)&_data, sizeof(ValueType) * (shape * ...));
                    if(err != cudaSuccess)
                        throw CuRuntimeError(err);
                }

            __host__ ValueType* data() const { return _data; }

            __host__ bool isValid() const { return _data != nullptr; }

            __host__ void initMetaData(detail::NDArrayMetaData<ValueType, Dim, IsDebug>& metaData) const;

            __host__ ~CuNDArray() {
                if(isValid()) {
                    cudaFree(_data);
                    _data = nullptr;
                }
            }
    };

    template<typename ValueType, bool IsDebug>
    class CuNDArray<ValueType, 1, IsDebug> {
        private:
            std::uint32_t _shape;
            ValueType* _data;
        public:
            using MetaData = detail::NDArrayMetaData<ValueType, 1, IsDebug>;
            __host__ CuNDArray() : _shape(0), _data(nullptr) {}

            __host__ CuNDArray(std::uint32_t shape) : _shape(shape), _data(nullptr) {
                auto err = cudaMalloc((void**)&_data, sizeof(ValueType) * shape);
                if(err != cudaSuccess)
                    throw CuRuntimeError(err);
            }

            __host__ ValueType* data() const { return _data; }

            __host__ bool isValid() const {  return _data != nullptr;}

            __host__ void initMetaData(detail::NDArrayMetaData<ValueType,1, IsDebug>& metaData) const;

            __host__ ~CuNDArray() {
                if(isValid()) {
                    cudaFree(_data);
                    _data = nullptr;
                }
            }
    };

    namespace detail {
        template<typename ValueType>
        class NDArrayMetaData<ValueType, 1, false> {
            friend __host__ void CuNDArray<ValueType, 1, false>::initMetaData(NDArrayMetaData<ValueType, 1, false>& metaData) const;
            private:
                ValueType* _data;
            public:
                NDArrayMetaData() : _data(nullptr) {}

                __device__ ValueType& at(std::uint32_t n) const { return _data[n]; }
        };

        template<typename ValueType>
        class NDArrayMetaData<ValueType, 1, true> {
            friend __host__ void CuNDArray<ValueType, 1, true>::initMetaData(NDArrayMetaData<ValueType, 1, true>& metaData) const;
            private:
                std::uint32_t _shape;
                ValueType* _data;
            public:
                NDArrayMetaData() : _shape(0), _data(nullptr) {}

                __device__ ValueType& at(std::uint32_t n) const { assert(n < _shape); return _data[n]; }
        };

        template<typename ValueType, std::uint32_t Dim>
        class NDArrayMetaData<ValueType, Dim, false> {
            friend __host__ void CuNDArray<ValueType, Dim, false>::initMetaData(NDArrayMetaData<ValueType, Dim, false>& metaData) const;
            private:
                std::array<std::uint32_t, Dim-1> _axisWeights;
                ValueType* _data;
            public:
                NDArrayMetaData() : _axisWeights(), _data(nullptr) {}

                template<typename ... Indices>
                __device__ ValueType& at(Indices... indices) const {
                    return _data[detail::flatIndex1<0, Dim-1>(_axisWeights, indices...)];
                }
        };

        template<typename ValueType, std::uint32_t Dim>
        class NDArrayMetaData<ValueType, Dim, true> {
            friend __host__ void CuNDArray<ValueType, Dim, true>::initMetaData(NDArrayMetaData<ValueType, Dim, true>& metaData) const;
            private:
                std::array<std::uint32_t, Dim> _shape;
                ValueType* _data;
            public:
                NDArrayMetaData() : _shape(), _data(nullptr) {}

                template<typename... Indices>
                __device__ ValueType& at(Indices... indices) const {
                    bool valid = detail::validateIndices<0, Dim>(_shape, indices...);
                    assert(valid);
                    return _data[detail::flatIndex2<0, Dim>(_shape, indices...)];
                }
        };
    }

    template<typename ValueType, std::uint32_t Dim, bool IsDebug>
    __host__ void CuNDArray<ValueType, Dim, IsDebug>::initMetaData(detail::NDArrayMetaData<ValueType, Dim, IsDebug>& metaData) const {
        metaData._data = _data;
        if constexpr(IsDebug) {
            metaData._shape = _shape;
        } else {
            detail::getAxisWeights<Dim>(metaData._axisWeights, _shape);
        }
    }

    template<typename ValueType, bool IsDebug>
    __host__ void CuNDArray<ValueType, 1, IsDebug>::initMetaData(detail::NDArrayMetaData<ValueType, 1, IsDebug>& metaData) const {
        metaData._data = _data;
        if constexpr(IsDebug) {
            metaData._shape = _shape;
        }
    }
}