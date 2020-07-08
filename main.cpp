#include <iostream>
#include <cuda_runtime.h>

#include "etudes/add.h"
#include "etudes/matmul.h"
#include "etudes/gauss_jordan.h"

int cuda_devices(void)
{
    int result = 0;
    cudaGetDeviceCount(&result);
    return result;
}

int main(void)
{
    const int n = cuda_devices();
    if (n == 0)
    {
        std::cout << "No CUDA hardware found. Exiting.\n";
        return 0;
    }

    add();
    matmul();
    gauss_jordan();

    return 0;
}
