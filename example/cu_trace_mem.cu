#include <thrust/device_vector.h>
#include "cu_mem_tracer.h" 

int main() {
    auto usedBefore = gsy::CuMemTracer::used();
    thrust::device_vector<int> vec(1024*1024*10, 1);
    auto usedAfter = gsy::CuMemTracer::used();
    printf("Used %lu MB\n", gsy::cast<gsy::MB>(usedAfter - usedBefore).count());
    return 0;
}