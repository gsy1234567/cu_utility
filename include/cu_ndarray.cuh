#pragma once

#include <array>
#include <cstdint>
#include <assert.h>
#include <type_traits>
#include "cu_runtime_error.h"
#include "cu_mem_alloc.h"

namespace gsy {

    template<typename ValueType, std::uint32_t Dim, bool IsDebug, memory_policy MemPolicy>
    class CuNDArray;

    namespace detail {
        template<typename ValueType, std::uint32_t Dim, bool IsDebug, memory_policy MemPolicy>
        class NDArrayMetaData;

        template<std::uint32_t I, std::uint32_t N, typename... Indices>
        constexpr std::uint32_t flatIndex1 (const std::array<std::uint32_t,N>& axisWeights, std::uint32_t n, Indices... indices) {
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

        template<std::uint32_t I, std::uint32_t N>
        constexpr std::uint32_t flatIndex1(const std::array<std::uint32_t,N>& axisWeights, const std::array<std::uint32_t, N+1>& index) {
            if constexpr (I == N) {
                return axisWeights[I-1] * index[I];
            }
            else {
                if constexpr (I == 0) {
                    return index[I] + flatIndex1<I+1,N>(axisWeights, index);
                }
                else {
                    return axisWeights[I-1] * index[I] + flatIndex1<I+1,N>(axisWeights, index);
                }
            }
        } 

        template<std::uint32_t I, std::uint32_t N, typename... Indices>
        constexpr bool validateIndices(const std::array<std::uint32_t, N>& shape, std::uint32_t n, Indices... indices) {
            static_assert((I+sizeof...(Indices)+1) == N);
            if constexpr(sizeof...(Indices) == 0) {
                return n < shape[I];
            } else {
                return (n < shape[I]) and validateIndices<I+1,N>(shape, indices...);
            }
        }

        template<std::uint32_t I, std::uint32_t N>
        constexpr bool validateIndices(const std::array<std::uint32_t, N>& shape, const std::array<std::uint32_t, N>& index) {
            if constexpr (I == N-1) {
                return shape[I] > index[I];
            }
            else {
                return (shape[I] > index[I]) and validateIndices<I+1, N>(shape, index);
            }
        }

        template<std::uint32_t I, std::uint32_t N, typename... Indices>
        constexpr std::uint32_t flatIndex2 (const std::array<std::uint32_t, N>& shape, std::uint32_t n, Indices... indices) {
            static_assert((I+sizeof...(Indices)+1) == N);
            if constexpr(sizeof...(Indices) == 0) {
                return n;
            } 
            else {
                return n+shape[I]*flatIndex2<I+1,N>(shape, indices...);
            }
        }

        template<std::uint32_t I, std::uint32_t N>
        constexpr std::uint32_t flatIndex2(const std::array<std::uint32_t, N>& shape, const std::array<std::uint32_t, N>& index) {
            if constexpr (I == N-1) {
                return index[I];
            }
            else {
                return index[I]+shape[I]*flatIndex2<I+1, N>(shape, index);
            }
        }

        template<std::uint32_t N, std::uint32_t I=0>
        constexpr void getAxisWeights(std::array<std::uint32_t, N-1>& axisWeights, const std::array<std::uint32_t, N>& shape) {
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



    template<typename ValueType, std::uint32_t Dim, bool IsDebug, memory_policy MemPolicy>
    class CuNDArray {
        protected:
            std::array<std::uint32_t, Dim> _shape;
            ValueType* _data;
        public:
            using MetaData = detail::NDArrayMetaData<ValueType, Dim, IsDebug, MemPolicy>;
            
            __host__ CuNDArray() : _shape(), _data(nullptr) {}

            template<typename... Shape>
            __host__ CuNDArray(Shape... shape) 
                : _shape{static_cast<std::uint32_t>(shape)...} {
                    _data = (ValueType*)CuMemoryAdapter<MemPolicy>::allocate(sizeof(ValueType)* (shape * ...));
                }

            __host__ ValueType* data() const { return _data; }

            __host__ bool isValid() const { return _data != nullptr; }

            __host__ std::array<std::uint32_t, Dim> shape() const { return _shape; }

            __host__ void initMetaData(MetaData& metaData) const;

            __host__ ~CuNDArray() {
                if(isValid()) {
                    CuMemoryAdapter<MemPolicy>::deallocate((void*)_data);
                    _data = nullptr;
                }
            }
    };

    template<typename ValueType, bool IsDebug, memory_policy MemPolicy>
    class CuNDArray<ValueType, 1, IsDebug, MemPolicy>{
        protected:
            std::uint32_t _shape;
            ValueType* _data;
        public:
            using MetaData = detail::NDArrayMetaData<ValueType, 1, IsDebug, MemPolicy>;

            __host__ CuNDArray() : _shape(0), _data(nullptr) {}

            __host__ CuNDArray(std::uint32_t shape) : _shape(shape), _data(nullptr) {
                _data = (ValueType*)CuMemoryAdapter<MemPolicy>::allocate(sizeof(ValueType)*_shape);
            }

            __host__ ValueType* data() const { return _data; }

            __host__ bool isValid() const {  return _data != nullptr; }

            __host__ std::uint32_t shape() const { return _shape; }

            __host__ void initMetaData(MetaData& metaData) const;

            __host__ ~CuNDArray() {
                if(isValid()) {
                    CuMemoryAdapter<MemPolicy>::deallocate((void*)_data);
                    _data = nullptr;
                }
            }
    };

    namespace detail {
        template<typename ValueType, memory_policy MemPolicy>
        class NDArrayMetaData<ValueType, 1, false, MemPolicy> {
            friend __host__ void CuNDArray<ValueType, 1, false, MemPolicy>::initMetaData(NDArrayMetaData<ValueType, 1, false, MemPolicy>& metaData) const;
            private:
                ValueType* _data;
            public:
                NDArrayMetaData() : _data(nullptr) {}

                __device__ ValueType& at(std::uint32_t n) const { return _data[n]; }
                __device__ ValueType& at(const std::array<std::uint32_t, 1>& index) const { return _data[index[0]]; }
        };

        template<typename ValueType, memory_policy MemPolicy>
        class NDArrayMetaData<ValueType, 1, true, MemPolicy> {
            friend __host__ void CuNDArray<ValueType, 1, true, MemPolicy>::initMetaData(NDArrayMetaData<ValueType, 1, true, MemPolicy>& metaData) const;
            private:
                std::uint32_t _shape;
                ValueType* _data;
            public:
                NDArrayMetaData() : _shape(0), _data(nullptr) {}

                __device__ ValueType& at(std::uint32_t n) const { assert(n < _shape); return _data[n]; }
                __device__ ValueType& at(const std::array<std::uint32_t, 1>& index) const { assert(index[0] < _shape); return _data[index[0]]; }
        };

        template<typename ValueType, std::uint32_t Dim, memory_policy MemPolicy>
        class NDArrayMetaData<ValueType, Dim, false, MemPolicy> {
            friend __host__ void CuNDArray<ValueType, Dim, false, MemPolicy>::initMetaData(NDArrayMetaData<ValueType, Dim, false, MemPolicy>& metaData) const;
            private:
                std::array<std::uint32_t, Dim-1> _axisWeights;
                ValueType* _data;
            public:
                NDArrayMetaData() : _axisWeights(), _data(nullptr) {}

                template<typename ... Indices>
                __device__ ValueType& at(Indices... indices) const {
                    return _data[detail::flatIndex1<0, Dim-1>(_axisWeights, indices...)];
                }

                __device__ ValueType& at(const std::array<std::uint32_t, Dim>& index) const {
                    return _data[detail::flatIndex1<0, Dim-1>(_axisWeights, index)];
                }
        };

        template<typename ValueType, std::uint32_t Dim, memory_policy MemPolicy>
        class NDArrayMetaData<ValueType, Dim, true, MemPolicy> {
            friend __host__ void CuNDArray<ValueType, Dim, true, MemPolicy>::initMetaData(NDArrayMetaData<ValueType, Dim, true, MemPolicy>& metaData) const;
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

                __device__ ValueType& at(const std::array<std::uint32_t, Dim>& index) const {
                    bool valid = detail::validateIndices<0, Dim>(_shape, index);
                    assert(valid);
                    return _data[detail::flatIndex2<0, Dim>(_shape, index)];
                }
        };
    }

    template<typename ValueType, std::uint32_t Dim, bool IsDebug, memory_policy MemPolicy>
    __host__ void CuNDArray<ValueType, Dim, IsDebug, MemPolicy>::initMetaData(detail::NDArrayMetaData<ValueType, Dim, IsDebug, MemPolicy>& metaData) const {
        metaData._data = _data;
        if constexpr(IsDebug) {
            metaData._shape = _shape;
        } else {
            detail::getAxisWeights<Dim>(metaData._axisWeights, _shape);
        }
    }

    template<typename ValueType, bool IsDebug, memory_policy MemPolicy>
    __host__ void CuNDArray<ValueType, 1, IsDebug, MemPolicy>::initMetaData(detail::NDArrayMetaData<ValueType, 1, IsDebug, MemPolicy>& metaData) const {
        metaData._data = _data;
        if constexpr(IsDebug) {
            metaData._shape = _shape;
        }
    }
}