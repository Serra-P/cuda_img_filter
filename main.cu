#include <iostream>
#include <cuda_runtime.h>

__global__ void helloWorldKernel() {
    printf("Hello, World! from GPU\n");
}

int main() {
    std::cout << "Hello, World! from CPU" << std::endl;

    // Lancer le kernel CUDA
    helloWorldKernel<<<1, 1>>>();

    // Synchroniser pour s'assurer que le kernel a terminé
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    return 0;
}
