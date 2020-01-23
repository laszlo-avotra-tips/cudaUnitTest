#include <cudaFFTwrapper.h>

#include <cuda_runtime.h>
#include <cufft.h>
#include <helper_cuda.h>
#include <stdint.h>
#include <memory>
#include <iostream>

void checkGpuMem()

{
    double free_m, total_m, used_m;

    size_t free_t, total_t;

    cudaMemGetInfo(&free_t, &total_t);

    free_m = free_t / 1048576.0;

    total_m = total_t / 1048576.0;

    used_m = total_m - free_m;

    printf("  mem free %zd .... %f MB mem total %zd....%f MB mem used %f MB\n", free_t, free_m, total_t, total_m, used_m);

}

void ComputeTheFFT(std::complex<float>* h_signal, std::complex<float>* h_signal_fft_ifft, const long dataSize, const int batch)
{
    const long mem_size = sizeof(std::complex<float>) * dataSize;

    std::complex<float>* d_signal{ nullptr };
    cufftHandle plan{ -1 };

    const char* argv{ "test" };
    findCudaDevice(1, &argv);

    checkGpuMem();
    // Allocate device memory for signal
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_signal), mem_size));
    checkGpuMem();
    if (h_signal) {

        // Copy host memory to device
        checkCudaErrors(cudaMemcpy(d_signal, h_signal, mem_size, cudaMemcpyHostToDevice));

        // CUFFT plan simple API
        checkCudaErrors(cufftPlan1d(&plan, mem_size, CUFFT_C2C, batch));

        // Transform signal and kernel
        //std::cout << "Transforming signal cufftExecC2" << std::endl;
        checkCudaErrors(cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(d_signal),
            reinterpret_cast<cufftComplex*>(d_signal),
            CUFFT_FORWARD));       

        //h_signal has the original coefficients
        //d_signal has the direct FFT coefficients

        // Check if kernel execution generated and error
    }

    if (h_signal_fft_ifft) {
        // Transform signal back
        //std::cout << "Transforming signal back cufftExecC2C" << std::endl;

        cudaDeviceSynchronize();

        checkCudaErrors(cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(d_signal),
            reinterpret_cast<cufftComplex*>(d_signal),
            CUFFT_INVERSE));
        //h_signal has the original coefficients
        //d_signal has the FFT --> iFFT coefficients
        //cudaDeviceSynchronize();

        // Copy device memory to host
        checkCudaErrors(cudaMemcpy(h_signal_fft_ifft, d_signal, mem_size,
            cudaMemcpyDeviceToHost));
        //h_signal has the original coefficients
        //h_signal_fft_ifft has the FFT --> iFFT coefficients

    }
    checkGpuMem();

    cudaDeviceReset();
}
void addjustCoefficientMagnitude(std::complex<float>* h_data, long dataSize) noexcept
{

    if (h_data) {
        for (long i = 0; i < dataSize; ++i) {
            h_data[i] = { h_data[i].real() / 8.0f / dataSize, 0 };
        }
    }
}

int isOriginalEqualToTheTransformedAndInverseTransformenData(
    const std::complex<float>* original, const std::complex<float>* transformed, long dataSize) noexcept
{
    int iTestResult = 0;
    if (original && transformed) {
        for (int i = 0; i < dataSize; ++i) {
            float e = transformed[i].real() - original[i].real();
            //float e = std::abs(transformed[i].real() / original[i].real()) - 1.0;
            if ( std::abs(e) >  1e-5f) {
                iTestResult += 1;
                printf("iTestesult=%d , i=%d, o=%f, t=%f e=%f\n\r", iTestResult, i, original[i].real(), transformed[i].real(), e);
            }
        }
    }
    return iTestResult;
}

void printTheData(const std::complex<float>* original, const std::complex<float>* transformed, long dataSize, const rsize_t printOffset)
{
    std::cout << "The first " << dataSize << " real values with offset [" << printOffset << "] :" << std::endl;
    if (original) {
        for (int i = 0; i < dataSize; ++i) {
            std::cout << original[i + printOffset].real() << " ";
        }
        std::cout << std::endl;
    }
    if (transformed) {
        for (int i = 0; i < dataSize; ++i) {
            std::cout << transformed[i + printOffset].real() << " ";
        }
        std::cout << std::endl;
    }
}


void initializeTheSignals(std::complex<float>* fft, long dataSize) noexcept
{
    for (long i = 0; i < dataSize; ++i) {
        if (fft)
            fft[i] = { rand() / static_cast<float>(RAND_MAX), 0 };
    }
}
