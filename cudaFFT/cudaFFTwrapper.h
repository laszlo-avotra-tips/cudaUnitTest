#pragma once

#include <complex>

void ComputeTheFFT(std::complex<float>* h_signal, std::complex<float>* h_signal_fft_ifft, const long dataSize, const int batch = 1);
void addjustCoefficientMagnitude(std::complex<float>* h_data, long dataSize) noexcept;
int isOriginalEqualToTheTransformedAndInverseTransformenData(
    const std::complex<float>* original, const std::complex<float>* transformed, long dataSize) noexcept;
void printTheData(const std::complex<float>* original, const std::complex<float>* transformed, long dataSize, const int printOffset = 0);
void initializeTheSignals(std::complex<float>* fft, long dataSize) noexcept;
