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
    constexpr int batchSize(1024);

    constexpr long dataSize(fftSize*batchSize);

    // Allocate host memory for the signal
    auto h_signalIn = std::make_unique<std::complex<float>[]>(dataSize);
    auto h_signalOut = std::make_unique<std::complex<float>[]>(dataSize);

    auto pHin = h_signalIn.get();
    auto pHout = h_signalOut.get();

    initializeTheSignals(pHin, dataSize);

    ComputeTheFFT(pHout, pHin, fftSize, batchSize);

    // check result

    int iTestResult = 0;

    //result scaling
    addjustCoefficientMagnitude(pHout, dataSize);

    printf("iTestResult: %d\n\r", iTestResult);

    printTheData(pHin, pHout, 8, fftSize * batchSize - 8);
//    printTheData(pHin, pHout, 8, 0);

    exit((iTestResult == 0) ? EXIT_SUCCESS : EXIT_FAILURE);
}
