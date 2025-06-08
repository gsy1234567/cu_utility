#include "ndslice.h"
#include <iostream>

constexpr bool isDebug = true;

using SliceT = gsy::Slice<isDebug>;
template<gsy::is_ndslice<isDebug>... Slices>
using NDSliceT = gsy::NDSlice<isDebug, Slices...>;

template<std::size_t N>
void printArray(const std::array<std::uint32_t, N>& arr) {
    for(int i=0 ; i<N ; ++i) {
        printf("%3u ", arr[i]);
    }
    printf("\n");
}

int main() {
    constexpr NDSliceT ndslice(SliceT(0, 5), SliceT(0,10));
    auto size = ndslice.getSize();
    printf("ndslice size: %u\n", size);
    for(auto i=0 ; i<size ; ++i) {
        auto idx = ndslice.at(i);
        printArray(idx);
    }
    return 0;
}