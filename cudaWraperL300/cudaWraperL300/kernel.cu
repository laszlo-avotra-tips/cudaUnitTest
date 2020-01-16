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

__global__ void rescale_kernel(const unsigned short* input,
    float* output,
    const float* fractionalSamples,
    const float* wholeSamples,
    const unsigned int inputLength,
    const unsigned int outputLength)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    float interpSample;
    int write_offset = j * outputLength;
    int read_offset = j * inputLength;
    int input_pos;
    unsigned int sampleIndex;
    sampleIndex = (int)wholeSamples[i];

    if ((sampleIndex + 1) > inputLength) {
        return;
    }

    input_pos = read_offset + sampleIndex;
    interpSample = (float)(input[input_pos + 1] - input[input_pos]) *
        fractionalSamples[i];
    interpSample = interpSample + input[input_pos];

    // Apply the passed in window function and set the output
    output[i + write_offset] = interpSample;
}

#endif

unsigned short* rescaleInputMemoryObject(0);
float* rescaleOutputMemObj(0);
float* fracSamplesMemObj(0);
float* wholeSamplesMemObj(0);

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

bool allocateWholeSamplesMemObj(float* devMem, unsigned int rescalingDataLength)
{
    cudaError_t cudaStatus;

    const int size = rescalingDataLength * sizeof(float);
    cudaStatus = cudaMalloc((void**)&devMem, size);
    return cudaStatus == cudaSuccess;
}

bool initializeWholeSamplesMemObj(float* devMem, float* data, unsigned int size)
{
    if (data) {
        cudaError_t cudaStatus;
        cudaStatus = cudaMemcpy(devMem, data, size, cudaMemcpyHostToDevice);
        return cudaStatus != cudaSuccess;
    }
    return false;
}


bool cudaRescale(float* output, unsigned short* data, unsigned int inputSize,
    float* wholeSamples,
    float* fractionalSamples,
    char* errorMsg,
    unsigned int linesPerFrame, unsigned int recordLength, unsigned int rescalingDataLength)
{
    bool success{ false };

    if (data && inputSize && wholeSamples && fractionalSamples) {
        success = true;
        if (errorMsg) {
            errorMsg[0] = 0;
        }
    }

    if (success) {
        success = allocateInputMemObj(rescaleInputMemoryObject, linesPerFrame, recordLength);
    }
    else {
        if (errorMsg) {
            sprintf(errorMsg, "Invalid arguments");
        }
        return false;
    }

    if (success) {
        success = initializeInputMemObj(rescaleInputMemoryObject, data, inputSize);
    } 
    else {
        if (errorMsg) {
            sprintf(errorMsg, "failed to allocate rescaleInputMemoryObject");
        }
        return false;
    }
    if (success) {
        success = allocateOutputMemObj(rescaleOutputMemObj, linesPerFrame, rescalingDataLength);
    }
    else {
        if (errorMsg) {
            sprintf(errorMsg, "failed to initialize rescaleInputMemoryObject");
        }
        return false;
    }

    if (success) {
        success = allocateFracSamplesMemObj(fracSamplesMemObj, rescalingDataLength);
    }
    else {
        if (errorMsg) {
            sprintf(errorMsg, "failed to allocate rescaleOutputMemObj");
        }
        return false;
    }

    if (success) {
        success = initializeFracSamplesMemObj(fracSamplesMemObj, fractionalSamples, rescalingDataLength);
    }
    else {
        if (errorMsg) {
            sprintf(errorMsg, "failed to allocate fracSamplesMemObj");
        }
        return false;
    }

    if (success) {
        success = allocateWholeSamplesMemObj(wholeSamplesMemObj, rescalingDataLength);
    }
    else {
        if (errorMsg) {
            sprintf(errorMsg, "failed to initialize fracSamplesMemObj");
        }
        return false;
    }

    if (success) {
        success = initializeWholeSamplesMemObj(wholeSamplesMemObj, wholeSamples, rescalingDataLength);
    }
    else {
        if (errorMsg) {
            sprintf(errorMsg, "failed to allocate wholeSamplesMemObj");
        }
        return false;
    }

    if (!success) {
        if (errorMsg) {
            sprintf(errorMsg, "failed to initialize wholeSamplesMemObj");
        }
    }


    dim3 blocksPerGrid(inputSize/16, inputSize/16, 1);
    dim3 threadsPerBlock(16, 16, 1);

#ifdef __CUDACC__
    rescale_kernel << <blocksPerGrid, threadsPerBlock >> > (
        rescaleInputMemoryObject, rescaleOutputMemObj, 
        fracSamplesMemObj, wholeSamplesMemObj,
        recordLength, rescalingDataLength
        );
#endif
    // Check for any errors launching the kernel
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    }

    // Copy output vector from GPU buffer to host memory.
    if (output) {
        cudaStatus = cudaMemcpy(output, rescaleOutputMemObj, inputSize * sizeof(float), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
        }
    }

//    success = cudaStatus == cudaSuccess;

    if (success) {
        cudaFree(rescaleInputMemoryObject);
    }

    return success;
}

