#include "../cudaWraperL300/cudaWrapperL300.h"

#include <stdio.h>
#include <vector>

int main()
{

    const size_t arraySize = 5;
    const int a[arraySize] { 1, 2, 3, 4, 7 };
    const int b[arraySize] { 10, 20, 30, 40, 50 };
    int c[arraySize] {};

    // Add vectors in parallel.
    if(!addTwoVectors(c, a, b, arraySize)){
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    resetCuda();

    return 0;
}

