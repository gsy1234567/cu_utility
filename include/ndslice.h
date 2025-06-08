#pragma once

#include <array>
#include <tuple>
#include <assert.h>
#include <cstdint>
#include <concepts>

namespace gsy {
    template<bool IsDebug = false>
    class Slice {
        private:
            std::uint32_t _start, _end, _step;
        public:
            constexpr Slice(std::uint32_t start, std::uint32_t end, std::uint32_t step = 1)
                : _start(start), _end(end), _step(step) {}
            constexpr auto start() const { return _start; }
            constexpr auto end() const { return _end; }
            constexpr auto step() const { return _step; }
            constexpr std::uint32_t size() const {
                if (_start >= _end) {
                    return 0;
                }
                return (_end - _start - 1) / _step + 1;
            }
            constexpr auto operator[](std::uint32_t n) const {
                return _start + n * _step;
            }
            constexpr auto at(std::uint32_t n) const {
                if constexpr (IsDebug) {
                    assert(n < size());
                }
                return _start + n * _step;
            }
    };

    template<typename T, bool IsDebug>
    concept is_ndslice = 
        std::integral<T> || 
        std::is_same_v<T, Slice<IsDebug>>;

    template<bool IsDebug, is_ndslice<IsDebug>... Slices>
    class NDSlice {
        public:
            static constexpr std::uint32_t dim = sizeof...(Slices);
        private:
            std::tuple<Slices...> _slices;
        public:
            template<std::uint32_t I>
            using nth_slice = typename std::tuple_element<I, decltype(_slices)>::type;
        private:
            template<std::uint32_t I=0>
            constexpr std::uint32_t _getSize() const {
                if constexpr (std::is_integral_v<nth_slice<I>>) {
                    if constexpr (I == dim-1) {
                        return 1;
                    } 
                    else {
                        return _getSize<I+1>();
                    }
                }
                else {
                    if constexpr (I == dim-1) {
                        return std::get<I>(_slices).size();
                    }
                    else {
                        return std::get<I>(_slices).size() * _getSize<I+1>();
                    }
                }
            }
            template<std::uint32_t I=0>
            static constexpr std::uint32_t _getSliceNum() {
                if constexpr (I == dim-1) {
                    return std::is_same_v<nth_slice<I>, Slice<IsDebug>> ? 1 : 0;
                }
                else {
                    return (std::is_same_v<nth_slice<I>, Slice<IsDebug>> ? 1 : 0) + _getSliceNum<I+1>();
                }
            }
        public:
            static constexpr std::uint32_t shapeDim = _getSliceNum();
        private:
            template<std::uint32_t InsertPtr=0, std::uint32_t I=0>
            constexpr void _getShape(std::array<std::uint32_t, shapeDim-1>& shape) const {
                if constexpr (I == dim-1 and InsertPtr < shapeDim-1){
                    if constexpr (std::is_same_v<nth_slice<I>, Slice<IsDebug>> ) {
                        shape[InsertPtr] = std::get<I>(_slices).size();
                    }
                }
                else if constexpr(I < dim-1 and InsertPtr < shapeDim-1) {
                    if constexpr (std::is_same_v<nth_slice<I>, Slice<IsDebug>>) {
                        shape[InsertPtr] = std::get<I>(_slices).size();
                        _getShape<InsertPtr+1, I+1>(shape);
                    }
                    else {
                        _getShape<InsertPtr, I+1>(shape);
                    }
                }
            }
            template<std::uint32_t ShapeReadPtr=0, std::uint32_t I=0>
            constexpr void _getIndex(const std::array<std::uint32_t, shapeDim-1>& shape, std::uint32_t& n, std::array<std::uint32_t, dim>& index) const {
                if constexpr (I < dim) {
                    if constexpr (std::is_same_v<nth_slice<I>, Slice<IsDebug>>) {
                        if constexpr (ShapeReadPtr >= shapeDim-1) {
                            index[I] = std::get<I>(_slices).at(n);
                            _getIndex<ShapeReadPtr, I+1>(shape, n, index);
                        }
                        else {
                            index[I] = std::get<I>(_slices).at(n % shape[ShapeReadPtr]);
                            n /= shape[ShapeReadPtr];
                            _getIndex<ShapeReadPtr+1, I+1>(shape, n, index);
                        }
                    }
                    else {
                        index[I] = std::get<I>(_slices);
                        _getIndex<ShapeReadPtr, I+1>(shape, n, index);
                    }
                }
            }

            std::array<std::uint32_t, shapeDim-1> _shape;
        public:
            constexpr explicit NDSlice(Slices... slices) : _slices(slices...) { _getShape(_shape); }
            constexpr std::uint32_t getDim() const { return dim; }
            constexpr std::uint32_t getSize() const { return _getSize(); }
            constexpr std::array<std::uint32_t, dim> at(std::uint32_t n) const {
                std::array<std::uint32_t, dim> index;
                _getIndex(_shape, n, index);
                return index;
            }
    };
}