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
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// includes, project
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#ifndef __CUDACC__
struct dim3 {
    dim3(int x_, int y_, int z_) :x(x_), y(y_), z(z_) {}
    int x;
    int y;
    int z;
};
#endif

// Complex data type
typedef float2 Complex;

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char **argv);

// The filter size is assumed to be a number smaller than the signal size
#define SIGNAL_SIZE 256

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
  printf("[cudaFFT] is starting...\n");

  findCudaDevice(argc, (const char **)argv);

  // Allocate host memory for the signal
  Complex *h_signal =
      reinterpret_cast<Complex *>(malloc(sizeof(Complex) * SIGNAL_SIZE));

  Complex* h_signal_fft_ifft =
      reinterpret_cast<Complex*>(malloc(sizeof(Complex) * SIGNAL_SIZE));

  // Initialize the memory for the signal
  for (unsigned int i = 0; i < SIGNAL_SIZE; ++i) {
    h_signal[i].x = rand() / static_cast<float>(RAND_MAX);
    h_signal[i].y = 0;
    h_signal_fft_ifft[i].x = i;
    h_signal_fft_ifft[i].y = 1000 * i;
  }

  int mem_size = sizeof(Complex) * SIGNAL_SIZE;

  // Allocate device memory for signal
  Complex *d_signal;
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_signal), mem_size));
  // Copy host memory to device
  checkCudaErrors(
      cudaMemcpy(d_signal, h_signal, mem_size, cudaMemcpyHostToDevice));

  // CUFFT plan simple API
  cufftHandle plan;
  checkCudaErrors(cufftPlan1d(&plan, mem_size, CUFFT_C2C, 1));

  // Transform signal and kernel
  printf("Transforming signal cufftExecC2C\n");
  checkCudaErrors(cufftExecC2C(plan, reinterpret_cast<cufftComplex *>(d_signal),
                               reinterpret_cast<cufftComplex *>(d_signal),
                               CUFFT_FORWARD));

  // Check if kernel execution generated and error
  getLastCudaError("Kernel execution failed [ ComplexPointwiseMulAndScale ]");

  // Transform signal back
  printf("Transforming signal back cufftExecC2C\n");
  checkCudaErrors(cufftExecC2C(plan, reinterpret_cast<cufftComplex *>(d_signal),
                               reinterpret_cast<cufftComplex *>(d_signal),
                               CUFFT_INVERSE));

  // Copy device memory to host
  checkCudaErrors(cudaMemcpy(h_signal_fft_ifft, d_signal, mem_size,
                             cudaMemcpyDeviceToHost));
  // check result
  bool bTestResult = true;

  //result scaling 
  for (int i = 0; i < SIGNAL_SIZE; ++i) {
      h_signal_fft_ifft[i].x = h_signal_fft_ifft[i].x / 8.0f / SIGNAL_SIZE;
  }

  bTestResult = sdkCompareL2fe(
      reinterpret_cast<float *>(h_signal),
      reinterpret_cast<float *>(h_signal_fft_ifft), SIGNAL_SIZE, 1e-3f);

  //for (int i = 0; i < SIGNAL_SIZE; ++i) {
  //    printf("h_signal = %f, h_signal_fft_ifft = %f k = %f\n\r", h_signal[i].x, h_signal_fft_ifft[i].x, h_signal_fft_ifft[i].x/ h_signal[i].x);
  //}

  // Destroy CUFFT context
  checkCudaErrors(cufftDestroy(plan));

  // cleanup memory
  free(h_signal);
  free(h_signal_fft_ifft);
  checkCudaErrors(cudaFree(d_signal));

  exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}
