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
#include <memory>


#include "../../cudaFFT/cudaFFTwrapper.h"

#ifndef __CUDACC__ //this not functional it only silences IntelliSense
struct dim3 {
    dim3(int x_, int y_, int z_) noexcept : x(x_), y(y_), z(z_) {}
    int x;
    int y;
    int z;
};
#endif

// Complex data type
using Complex = std::complex<float>;

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char** argv);
void addjustCoefficientMagnitude(Complex* h_data, long dataSize) noexcept;
int isOriginalEqualToTheTransformedAndInverseTransformenData(
    const Complex* original, const Complex* transformed, long dataSize) noexcept;
void printTheData(const Complex* original, const Complex* transformed, long dataSize);
void initializeTheSignals(Complex* fft, Complex* invfft, long dataSize) noexcept;

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
    runTest(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char** argv) {

    std::cout << "[cudaFFT] is starting..." << std::endl;

    constexpr long SIGNAL_SIZE(256);

    // Allocate host memory for the signal
    auto h_signal = std::move(std::make_unique<std::complex<float>[]>(SIGNAL_SIZE));
    auto h_signal_fft_ifft = std::move(std::make_unique<std::complex<float>[]>(SIGNAL_SIZE));

    initializeTheSignals(h_signal.get(), h_signal_fft_ifft.get(), SIGNAL_SIZE);

    ComputeTheFFT(h_signal.get(), h_signal_fft_ifft.get(), SIGNAL_SIZE);

    // check result
    int iTestResult = 0;

    //result scaling
    addjustCoefficientMagnitude(h_signal_fft_ifft.get(), SIGNAL_SIZE);

    iTestResult = isOriginalEqualToTheTransformedAndInverseTransformenData(h_signal.get(), h_signal_fft_ifft.get(), SIGNAL_SIZE);

    printTheData(h_signal.get(), h_signal_fft_ifft.get(), 10);

    exit((iTestResult == 0) ? EXIT_SUCCESS : EXIT_FAILURE);
}

void addjustCoefficientMagnitude(Complex* h_data, long dataSize) noexcept
{

    if (h_data) {
        for (long i = 0; i < dataSize; ++i) {
            h_data[i] = { h_data[i].real() / 8.0f / dataSize, 0 };
        }
    }
}

int isOriginalEqualToTheTransformedAndInverseTransformenData(
    const Complex* original, const Complex* transformed, long dataSize) noexcept
{
    int iTestResult = 1;
    if (original && transformed) {
        iTestResult = 0;
        for (int i = 0; i < dataSize; ++i) {
            if (std::abs(transformed[i].real() - original[i].real()) > abs(original[i].real() * 1e-5f))
                iTestResult += 1;
        }
    }
    return iTestResult;
}

void printTheData(const Complex* original, const Complex* transformed, long dataSize)
{
    std::cout << "The first " << dataSize << " real values:" << std::endl;
    if (original) {
        for (int i = 0; i < dataSize; ++i) {
            std::cout << original[i].real() << " ";
        }
        std::cout << std::endl;
    }
    if (transformed) {
        for (int i = 0; i < dataSize; ++i) {
            std::cout << transformed[i].real() << " ";
        }
        std::cout << std::endl;
    }
}

void initializeTheSignals(Complex* fft, Complex* invfft, long dataSize) noexcept
{
    for (long i = 0; i < dataSize; ++i) {
        if (fft)
            fft[i] = { rand() / static_cast<float>(RAND_MAX), 0 };
        if (invfft)
            invfft[i] = { float(i), 1000.f * i };
    }
}
