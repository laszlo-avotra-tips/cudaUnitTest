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
using ComplexVector = std::vector<Complex>;

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char **argv);
void addjustCoefficientMagnitude(Complex* h_data, size_t dataSize);
int isOriginalEqualToTheTransformedAndInverseTransformenData(
    const Complex* original, const Complex* transformed, size_t dataSize);
void printTheData(const Complex* original, const Complex* transformed, size_t dataSize);
void initializeTheSignals(Complex* fft, Complex* invfft, size_t dataSize);
void ComputeTheFFT(Complex* h_signal, Complex* h_signal_fft_ifft, size_t dataSize);


//// The filter size is assumed to be a number smaller than the signal size

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) { 
    runTest(argc, argv); 
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char **argv) {
  std::cout << "[cudaFFT] is starting..." << std::endl;

  findCudaDevice(argc, (const char **)argv);

  const size_t SIGNAL_SIZE{ 256 };

  // Allocate host memory for the signal
  Complex* h_signal = new Complex[SIGNAL_SIZE];
  Complex* h_signal_fft_ifft = new Complex[SIGNAL_SIZE];
  initializeTheSignals(h_signal, h_signal_fft_ifft, SIGNAL_SIZE);
  
  ComputeTheFFT(h_signal, h_signal_fft_ifft, SIGNAL_SIZE);

  // check result
  int iTestResult = 0;

  //result scaling
  addjustCoefficientMagnitude(h_signal_fft_ifft, SIGNAL_SIZE);

  iTestResult = isOriginalEqualToTheTransformedAndInverseTransformenData(h_signal, h_signal_fft_ifft, SIGNAL_SIZE);

  printTheData(h_signal, h_signal_fft_ifft, 10);


  // cleanup memory
  delete h_signal;
  delete h_signal_fft_ifft;

  exit((iTestResult == 0) ? EXIT_SUCCESS : EXIT_FAILURE);
}

void addjustCoefficientMagnitude(Complex* h_data, size_t dataSize)
{
    for (size_t i = 0; i < dataSize; ++i) {
        h_data[i] = { h_data[i].real() / 8.0f / dataSize, 0 };
    }
}

int isOriginalEqualToTheTransformedAndInverseTransformenData(
    const Complex* original, const Complex* transformed, size_t dataSize)
{
    int iTestResult = 0;
    for (int i = 0; i < dataSize; ++i) {
        if (std::abs(transformed[i].real() - original[i].real()) > abs(original[i].real() * 1e-5f))
            iTestResult += 1;
    }
    return iTestResult;
}

void printTheData(const Complex* original, const Complex* transformed, size_t dataSize)
{
    std::cout << "The first " << dataSize << " real values:" << std::endl;
    for (int i = 0; i < dataSize; ++i) {
        std::cout << original[i].real() << " ";
    }
    std::cout << std::endl;
    for (int i = 0; i < dataSize; ++i) {
        std::cout << transformed[i].real() << " ";
    }
    std::cout << std::endl;
}

void initializeTheSignals(Complex* fft, Complex* invfft, size_t dataSize)
{
    for (size_t i = 0; i < dataSize; ++i) {
        fft[i] = { rand() / static_cast<float>(RAND_MAX), 0 };
        invfft[i] = { float(i), 1000.f * i };
    }
}

void ComputeTheFFT(Complex* h_signal, Complex* h_signal_fft_ifft, size_t dataSize)
{
   int mem_size = sizeof(Complex) * dataSize;

   Complex* d_signal{ nullptr };
   cufftHandle plan;
   
   // Allocate device memory for signal
   checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_signal), mem_size));
   
   if (h_signal) {

        // Copy host memory to device
        checkCudaErrors(cudaMemcpy(d_signal, h_signal, mem_size, cudaMemcpyHostToDevice));

        // CUFFT plan simple API
        checkCudaErrors(cufftPlan1d(&plan, mem_size, CUFFT_C2C, 1));

        // Transform signal and kernel
        std::cout << "Transforming signal cufftExecC2" << std::endl;
        checkCudaErrors(cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(d_signal),
            reinterpret_cast<cufftComplex*>(d_signal),
            CUFFT_FORWARD));
        //h_signal has the original coefficients
        //d_signal has the direct FFT coefficients

        // Check if kernel execution generated and error
        getLastCudaError("Kernel execution failed [ ComplexPointwiseMulAndScale ]");
    }

    if (h_signal_fft_ifft) {
        // Transform signal back
        std::cout << "Transforming signal back cufftExecC2C" << std::endl;

        checkCudaErrors(cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(d_signal),
            reinterpret_cast<cufftComplex*>(d_signal),
            CUFFT_INVERSE));
        //h_signal has the original coefficients
        //d_signal has the FFT --> iFFT coefficients

        // Copy device memory to host
        checkCudaErrors(cudaMemcpy(h_signal_fft_ifft, d_signal, mem_size,
            cudaMemcpyDeviceToHost));
        //h_signal has the original coefficients
        //h_signal_fft_ifft has the FFT --> iFFT coefficients
    }

    // Destroy CUFFT context
    checkCudaErrors(cufftDestroy(plan));

    checkCudaErrors(cudaFree(d_signal));
}

