#pragma once

#include <complex>
#include <stdint.h>

void ComputeTheFFT(std::complex<float>* h_signal, std::complex<float>* h_signal_fft_ifft, const long dataSize, const int batch = 1);
void ComputeTheFFT(std::complex<float>* dataOut, const uint16_t* h_signal, std::complex<float>* h_signal_fft_ifft, const long dataSize, const int batch);