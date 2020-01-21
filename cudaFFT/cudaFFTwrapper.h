#pragma once

#include <complex>

void ComputeTheFFT(const std::complex<float>* h_signal, std::complex<float>* h_signal_fft_ifft, const long dataSize);
