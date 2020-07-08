#include <iostream>
#include <cuda_runtime.h>

#include "etudes/add.h"
#include "etudes/matmul.h"
#include "etudes/gauss_jordan.h"
#include "etudes/vector.h"

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

    vector<double> m(10, 0.1);
  for (int i = 0; i <10 ; ++i) {
    std::cout << m(i) << "\n";
  }



    return 0;
}
