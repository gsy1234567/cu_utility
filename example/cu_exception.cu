#include "cu_runtime_error.h"
#include <iostream>

int main() {
    try {
        throw gsy::CuRuntimeError(cudaError::cudaErrorAssert);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    return 0;
}