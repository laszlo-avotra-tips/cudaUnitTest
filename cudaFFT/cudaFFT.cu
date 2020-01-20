/*
 * Copyright 1993-2017 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* Example showing the use of CUFFT for fast 1D-convolution using FFT. */

// includes, system
#include <algorithm>
#include <iostream>
#include <complex>
#include <vector>

// includes, project
#include <cuda_runtime.h>
#include <cufft.h>
#include <helper_cuda.h>

#ifndef __CUDACC__
struct dim3 {
    dim3(int x_, int y_, int z_) :x(x_), y(y_), z(z_) {}
    int x;
    int y;
    int z;
};
#endif

// Complex data type
using Complex = std::complex<float>;

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char **argv);

//// The filter size is assumed to be a number smaller than the signal size
int cudaFFt(Complex* data, const size_t signalSize, bool isFFT /*versus iFFT*/);

bool compareData(const Complex* originalData, const Complex* transformedData, const size_t SIGNAL_SIZE);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) { 
    runTest(argc, argv); 
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char** argv) {

    bool success{ false };
    std::cout << "[cudaFFT] is starting..." << std::endl;

    findCudaDevice(argc, (const char**)argv);

    const size_t SIGNAL_SIZE{ 100 };

    // Allocate host memory for the signal
    Complex* h_signal = new Complex[SIGNAL_SIZE];
    Complex* original = new Complex[SIGNAL_SIZE];

    Complex* h_signal_fft_ifft = new Complex[SIGNAL_SIZE];

    // Initialize the memory for the signal
    for (unsigned int i = 0; i < SIGNAL_SIZE; ++i) {
        original[i] = { rand() / static_cast<float>(RAND_MAX), 0.0 };
        h_signal[i] = original[i];
    }

    //perform a direct Fourier
    cudaFFt(h_signal, SIGNAL_SIZE, true);

    if (false) {
        // Copy the fourier coefs into h_signal_fft_ifft
        for (unsigned int i = 0; i < SIGNAL_SIZE; ++i) {
            h_signal_fft_ifft[i] = h_signal[i];
        }

        //perform an inverse Fourier
        cudaFFt(h_signal_fft_ifft, SIGNAL_SIZE, false);

        //result scaling 
        for (int i = 0; i < SIGNAL_SIZE; ++i) {
            h_signal_fft_ifft[i] = { h_signal_fft_ifft[i].real() / 8.0f / SIGNAL_SIZE, 0 };
        }
        success = compareData(original, h_signal_fft_ifft, SIGNAL_SIZE);
    }
    else {
        //perform an inverse Fourier
        cudaFFt(h_signal, SIGNAL_SIZE, false);

        //result scaling 
        for (int i = 0; i < SIGNAL_SIZE; ++i) {
            h_signal[i] = { h_signal[i].real() / 8.0f / SIGNAL_SIZE, 0 };
        }
        success = compareData(original, h_signal, SIGNAL_SIZE);
    }


    // cleanup memory
    delete h_signal;
    delete h_signal_fft_ifft;

    exit((success) ? EXIT_SUCCESS : EXIT_FAILURE);  
    // Destroy CUFFT context
}

int cudaFFt(Complex* h_signal, const size_t signalSize, bool isFFT /*versus iFFT*/)
{

    int mem_size = sizeof(Complex) * signalSize;

    // Allocate device memory for signal
    Complex* d_signal;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_signal), mem_size));

    // Copy host memory to device
    checkCudaErrors(cudaMemcpy(d_signal, h_signal, mem_size, cudaMemcpyHostToDevice));

    // CUFFT plan simple API
    cufftHandle plan;
    checkCudaErrors(cufftPlan1d(&plan, mem_size, CUFFT_C2C, 1));

    // Transform signal and kernel
    if(isFFT)
        std::cout << "Direct Transform with cufftExecC2" << std::endl;
    else
        std::cout << "Inverse Transform with cufftExecC2C" << std::endl;


    if (isFFT) {
        checkCudaErrors(cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(d_signal),
            reinterpret_cast<cufftComplex*>(d_signal),
            CUFFT_FORWARD));
    }
    else {
        checkCudaErrors(cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(d_signal),
            reinterpret_cast<cufftComplex*>(d_signal),
            CUFFT_INVERSE));
    }

    // Check if kernel execution generated and error
    getLastCudaError("Kernel execution failed [ ComplexPointwiseMulAndScale ]");

    // Copy device memory to host
    checkCudaErrors(cudaMemcpy(h_signal, d_signal, mem_size,
        cudaMemcpyDeviceToHost));

    checkCudaErrors(cufftDestroy(plan));
    checkCudaErrors(cudaFree(d_signal));

    return 0;
}

bool compareData(const Complex* originalData, const Complex* transformedData, const size_t dataSize)
{
    // check result
    int iTestResult = 0;

    for (int i = 0; i < dataSize; ++i) {
        if (std::abs(transformedData[i].real() - originalData[i].real()) > 1e-3f)
            iTestResult += 1;
    }

    std::cout << "The first 10 real values:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << originalData[i].real() << " ";
    }
    std::cout << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << transformedData[i].real() << " ";
    }
    std::cout << std::endl;

    return !bool(iTestResult);
}
