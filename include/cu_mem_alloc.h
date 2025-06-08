#pragma once

#include <cstdint>

namespace gsy {
    struct DeviceMemory {};
    struct UnifiedMemory {};

    template<typename T>
    concept memory_policy = std::is_same_v<T, DeviceMemory> || std::is_same_v<T, UnifiedMemory>;

    template<memory_policy MemoryPolicy>
    class CuMemoryAdapter {
        public:
            static void* allocate(std::size_t size);
            static void deallocate(void* ptr) noexcept;
    };
}