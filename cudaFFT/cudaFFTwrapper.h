#pragma once

#include <complex>

void ComputeTheFFT(std::complex<float>* h_signalOut, const std::complex<float>* h_signalIn, size_t dataSize, int batch);
double ComputeTheFFTdev(std::complex<float>* h_signalOut, const std::complex<float>* h_signalIn, size_t dataSize, int batch);
void addjustCoefficientMagnitude(std::complex<float>* h_data, long dataSize) noexcept;
int isOriginalEqualToTheTransformedAndInverseTransformenData(
    const std::complex<float>* original, const std::complex<float>* transformed, long dataSize) noexcept;
void printTheData(const std::complex<float>* hIn, const std::complex<float>* hOut, long dataSize, const rsize_t printOffset);
void initializeTheSignals(std::complex<float>* fft, long dataSize) noexcept;
