#include <cudaFFTwrapper.h>

#include <cuda_runtime.h>
#include <cufft.h>
#include <helper_cuda.h>
#include <stdint.h>
#include <memory>


// Complex data type
using Complex = std::complex<float>;

void ComputeTheFFT(Complex* dataOut, const uint16_t* dataIn, std::complex<float>* h_signal_fft_ifft, const long dataSize, const int batch)
{
    for (long i = 0; i < dataSize; ++i) {
        dataOut[i] = Complex(dataIn[i], 0.0f);
    }

    ComputeTheFFT(dataOut, nullptr, dataSize, batch);
}

void ComputeTheFFT(std::complex<float>* h_signal, std::complex<float>* h_signal_fft_ifft, const long dataSize, const int batch)
{
    const long mem_size = sizeof(Complex) * dataSize;

    Complex* d_signal{ nullptr };
    cufftHandle plan{ -1 };

    const char* argv{ "test" };
    findCudaDevice(1, &argv);


    // Allocate device memory for signal
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_signal), mem_size));

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
        getLastCudaError("Kernel execution failed [ ComplexPointwiseMulAndScale ]");
    }

    if (h_signal_fft_ifft) {
        // Transform signal back
        //std::cout << "Transforming signal back cufftExecC2C" << std::endl;

        checkCudaErrors(cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(d_signal),
            reinterpret_cast<cufftComplex*>(d_signal),
            CUFFT_INVERSE));
        //h_signal has the original coefficients
        //d_signal has the FFT --> iFFT coefficients

        // Copy device memory to host
        checkCudaErrors(cudaMemcpy(h_signal_fft_ifft, d_signal, mem_size,
            cudaMemcpyDeviceToHost));
        //h_signal has the original coefficients
        //h_signal_fft_ifft has the FFT --> iFFT coefficients
    }

    // Destroy CUFFT context
    checkCudaErrors(cufftDestroy(plan));

    checkCudaErrors(cudaFree(d_signal));
}
