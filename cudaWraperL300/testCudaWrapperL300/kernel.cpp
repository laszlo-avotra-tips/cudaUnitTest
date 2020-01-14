#include "../cudaWraperL300/cudaWrapperL300.h"
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

#include <stdio.h>

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    bool cudaStatus(false);

#ifdef __CUDACC__
    cudaStatus = addWithCuda(c, a, b, arraySize);
#endif

    if (!cudaStatus) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    resetCuda();

    return 0;
}

