#pragma once

#include "cudaWrapperL300.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <vector>
#include <algorithm>

#define MINIMUN_VAL(x_,y_) (x_>y_)?x_:y_

#ifndef __CUDACC__
struct dim3 {
    dim3(int x_, int y_, int z_) :x(x_), y(y_), z(z_) {}
    int x;
    int y;
    int z;
};
#endif

#ifdef __CUDACC__

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
}
#endif

// Helper function for using CUDA to add vectors in parallel.
bool addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

 //    dim3 blocksPerGrid(size / 256, 1, 1);
    dim3 blocksPerGrid(MINIMUN_VAL(size/256,1), 1, 1);
    dim3 threadsPerBlock(256, 1, 1);

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
#ifdef __CUDACC__
    addKernel<<<blocksPerGrid, threadsPerBlock >>>(dev_c, dev_a, dev_b);
#endif

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus == cudaSuccess;
}

bool resetCuda() {
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    return (cudaSuccess == cudaStatus);
}

bool addTwoVectors(int* c, const int* a, const int* b, unsigned int size)
{
    bool success{ false };

    success = addWithCuda(c, a, b, size);

    return success;
}
bool allocateInputMemObj(unsigned short* devMem, unsigned int linesPerFrame, unsigned int recordLength)
{
    cudaError_t cudaStatus;

    const int size = linesPerFrame * recordLength * sizeof(unsigned short);
    cudaStatus = cudaMalloc((void**)&devMem, size);
    return cudaStatus == cudaSuccess;
}

bool initializeInputMemObj(unsigned short* devMem, unsigned short* data, unsigned int size)
{
    cudaError_t cudaStatus;
    cudaStatus = cudaMemcpy(devMem, data, size, cudaMemcpyHostToDevice);
    return cudaStatus != cudaSuccess;
}

bool allocateOutputMemObj(float* devMem, unsigned int linesPerFrame, unsigned int rescalingDataLength)
{
    cudaError_t cudaStatus;

    const int size = linesPerFrame * rescalingDataLength * sizeof(float);
    cudaStatus = cudaMalloc((void**)&devMem, size);
    return cudaStatus == cudaSuccess;
}

bool allocateFracSamplesMemObj(float* devMem, unsigned int rescalingDataLength)
{
    cudaError_t cudaStatus;

    const int size = rescalingDataLength * sizeof(float);
    cudaStatus = cudaMalloc((void**)&devMem, size);
    return cudaStatus == cudaSuccess;
}

bool initializeFracSamplesMemObj(float* devMem, float* data, unsigned int size)
{
    if (data) {
        cudaError_t cudaStatus;
        cudaStatus = cudaMemcpy(devMem, data, size, cudaMemcpyHostToDevice);
        return cudaStatus != cudaSuccess;
    }
    return false;
}

bool cudaRescale(unsigned short* data, unsigned int size,
    float* wholeSamples,
    float* fractionalSamples,
    char* errorMsg,
    unsigned int linesPerFrame, unsigned int recordLength, unsigned int rescalingDataLength)
{
    bool success{ false };

    if (data && size) {
        success = true;
        if (errorMsg) {
            *errorMsg = 0;
        }
    } else {
        if (errorMsg) {
            sprintf(errorMsg, "Invalid arguments");
        }
    }

    unsigned short* rescaleInputMemoryObject(0);
    float* rescaleOutputMemObj(0);
    float* fracSamplesMemObj(0);

    if (success) {
        success = allocateInputMemObj(rescaleInputMemoryObject, linesPerFrame, recordLength);
    }

    if (success) {
        success = initializeInputMemObj(rescaleInputMemoryObject, data, size);
    } 

    if (success) {
        cudaFree(rescaleInputMemoryObject);
    }

    if (success) {
        success = allocateOutputMemObj(rescaleOutputMemObj, linesPerFrame, rescalingDataLength);
    }

    if (success) {
        success = allocateFracSamplesMemObj(fracSamplesMemObj, rescalingDataLength);
    }

    if (success) {
        success = initializeFracSamplesMemObj(fracSamplesMemObj, fractionalSamples, rescalingDataLength);
    }

    return success;
}

