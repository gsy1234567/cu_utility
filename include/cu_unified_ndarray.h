#pragma once

#include "cu_ndarray.cuh"

namespace gsy {
    template<typename ValueType, std::uint32_t Dim, bool IsDebug>
    class CuUnifiedNDArray : public CuNDArray<ValueType, Dim, IsDebug, UnifiedMemory> {
        protected:
            using base = CuNDArray<ValueType, Dim, IsDebug, UnifiedMemory>;

            template<std::uint32_t I, typename... Other>
            constexpr std::uint32_t _flatIndex(std::uint32_t idx, Other... other) const {
                static_assert(I + 1 + sizeof...(Other) == Dim);
                if constexpr (sizeof...(Other) == 0) {
                    return idx;
                }
                else {
                    return idx+base::_shape[I]*_flatIndex<I+1>(other...);
                }
            }

            template<std::uint32_t I>
            constexpr std::uint32_t _flatIndex(const std::array<std::uint32_t, Dim>& index) const {
                if constexpr (I == Dim-1) {
                    return index[I];
                }
                else {
                    return index[I]+base::_shape[I]*_flatIndex<I+1>(index);
                }
            }

            template<std::uint32_t I, typename... Other>
            constexpr bool _validateIndices(std::uint32_t idx, Other... other) const {
                static_assert(I + 1 + sizeof...(Other) == Dim);
                if constexpr (sizeof...(Other) == 0) {
                    if constexpr (Dim == 1) {
                        return idx < base::_shape;
                    }
                    else {
                        return idx < base::_shape[I];
                    }
                }
                else {
                    return (idx < base::_shape[I]) and _validateIndices<I+1>(other...);
                }
            }

            template<std::uint32_t I>
            constexpr bool _validateIndices(const std::array<std::uint32_t, Dim>& index) const {
                if constexpr (I == Dim-1) {
                    return index[I] < base::_shape[I];
                }
                else {
                    return (index[I] < base::_shape[I]) and _validateIndices<I+1>(index);
                }
            }
        public:
            using typename base::MetaData;
            
            CuUnifiedNDArray() = default;

            template<typename... Shape>
            __host__ CuUnifiedNDArray(Shape... shape)
                : base(shape...) {}

            template<typename... Indices>
            __host__ ValueType& at(Indices... indices) const {
                if constexpr (IsDebug) {
                    assert(_validateIndices<0>(indices...));
                }
                return base::_data[_flatIndex<0>(indices...)];
            }

            __host__ ValueType& at(const std::array<std::uint32_t, Dim>& index) const {
                if constexpr (IsDebug) {
                    assert(_validateIndices(index));
                }
                return base::_data[_flatIndex<0>(index)];
            }
    };
}