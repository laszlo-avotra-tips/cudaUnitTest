#pragma once

#include <complex>

void ComputeTheFFT(std::complex<float>* h_signal, std::complex<float>* h_signal_fft_ifft, const size_t dataSize);
