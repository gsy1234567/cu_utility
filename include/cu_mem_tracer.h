#pragma once

#include <ratio>
#include <cstdint>

namespace gsy {
    template<typename Rep, typename Mag = std::ratio<1,1>>
    class Memory {
        private:
            Rep _count;
        public:
            using rep = Rep;
            using mag = Mag;
            constexpr Memory() : _count(0) {};
            constexpr Memory(Rep count) : _count(count) {};

            template<typename OtherRep, typename OtherMag>
            explicit constexpr Memory(const Memory<OtherRep, OtherMag>& other)
                : _count(static_cast<Rep>(other.count()*OtherMag::num/Mag::num)) {}

            auto count() const { return _count; }

            constexpr auto& operator+=(const Memory& other) {
                _count += other._count;
                return *this;
            }

            constexpr auto& operator-=(const Memory& other) {
                _count -= other._count;
                return *this;
            }

            constexpr auto operator+(const Memory& other) const {
                auto tmp = *this;
                return tmp += other;
            }

            constexpr auto operator-(const Memory& other) const {
                auto tmp = *this;
                return tmp -= other;
            }
    };

    using B = Memory<std::size_t>;
    using KB = Memory<std::size_t, std::ratio<1024ULL, 1>>;
    using MB = Memory<std::size_t, std::ratio<1024ULL*1024, 1>>;
    using GB = Memory<std::size_t, std::ratio<1024ULL*1024*1024, 1>>;
    using TB = Memory<std::size_t, std::ratio<1024LL*1024*1024*1024, 1>>;

    template<typename ToMem, typename FromRep, typename FromMag>
    constexpr ToMem cast(const Memory<FromRep, FromMag>& from) {
        return ToMem(from);
    }

    struct CuMemTracer {
        using rep = std::size_t;
        using mag = std::ratio<1,1>;
        using memory = Memory<rep, mag>;

        static memory used();
    };
}