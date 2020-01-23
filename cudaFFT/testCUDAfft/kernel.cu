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

 // includes, system
#include <algorithm>
#include <iostream>
#include <memory>

#include "../../cudaFFT/cudaFFTwrapper.h"

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char** argv);

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

    constexpr int fftSize(2048);
    constexpr int batchSize(160);

    constexpr long SIGNAL_SIZE(fftSize*batchSize);

    // Allocate host memory for the signal
    auto h_signal = std::make_unique<std::complex<float>[]>(SIGNAL_SIZE);
    auto h_signal_fft_ifft = std::make_unique<std::complex<float>[]>(SIGNAL_SIZE);

    auto pHs = h_signal.get();
    auto pHt = nullptr; // h_signal_fft_ifft.get();

    initializeTheSignals(pHs, SIGNAL_SIZE);

    ComputeTheFFT(pHs, pHt, fftSize, batchSize);
    //ComputeTheFFTdev(pHs, pHt, SIGNAL_SIZE, batchSize);

    // check result
    int iTestResult = 0;

    //result scaling
    addjustCoefficientMagnitude(pHt, SIGNAL_SIZE);

    //iTestResult = isOriginalEqualToTheTransformedAndInverseTransformenData(pHs, pHt, SIGNAL_SIZE);

    printf("iTestResult: %d\n\r", iTestResult);

    printTheData(pHs, pHt, 8, fftSize * batchSize - 9);

    exit((iTestResult == 0) ? EXIT_SUCCESS : EXIT_FAILURE);
}
