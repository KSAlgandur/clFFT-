#define CL_TARGET_OPENCL_VERSION 300
#include <iostream>
#include <vector>
#include <boost/compute/core.hpp>
#include <boost/compute/container/vector.hpp>
#include <clFFT.h>


namespace compute = boost::compute;

void PrintOutVec(const std::vector<std::complex<float>>& vec)
{
    for( auto v : vec)
    {
        std::cout <<"("<< v.imag() <<";"<<v.real()<<")"<<",";
    }
}

void clFFT_lib(const size_t N,compute::context& context,compute::command_queue& queue,  compute::vector<std::complex<float>>& buff)
{
    cl_int err;
    /* FFT library realted declarations */
    clfftPlanHandle planHandle;
    clfftDim dim = CLFFT_1D;
    size_t clLengths[1] = {N};

    /* Setup clFFT. */
    clfftSetupData fftSetup;
    err = clfftInitSetupData(&fftSetup);
    err = clfftSetup(&fftSetup);

    /* Create a default plan for a complex FFT. */
    err = clfftCreateDefaultPlan(&planHandle, context, dim, clLengths);

    /* Set plan parameters. */
    clfftSetPlanPrecision(planHandle, CLFFT_SINGLE);
    clfftSetLayout(planHandle, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
    clfftSetResultLocation(planHandle, CLFFT_INPLACE); // CLFFT_INPLACE CLFFT_OUTOFPLACE

    /* Bake the plan. */
    clfftBakePlan(planHandle, 1, reinterpret_cast<cl_command_queue*>(&queue), NULL, NULL);

        /* Execute the plan. */
    clfftEnqueueTransform(planHandle, CLFFT_FORWARD, 1, reinterpret_cast<cl_command_queue*>(&queue), 0, NULL, NULL, reinterpret_cast<cl_mem*>(&buff),NULL, NULL);
    queue.finish();

    /* Release the plan. */
    clfftDestroyPlan( &planHandle );
    /* Release clFFT library. */
    clfftTeardown();


}
int main()

{
     // get the default device

    size_t N = 16;
    compute::device device = compute::system::default_device();
    compute::platform platform = device.platform();
    compute::context context(device);
    compute::command_queue queue(context,device);

    // print the device's name
    std::cout << "platform: " << platform.name() << std::endl;
    std::cout << "device:\t  " << device.name() << std::endl;


    std::vector<std::complex<float>>params(N);// исходный вектор
    std::fill(params.begin(),params.end(),std::complex<float>(1.0f, 1.0f)); //заполние вектора на хосте значениями


    compute::vector<std::complex<float>> buff(params.size(), context); // создаем вектор комплексных чисел на устройстве

    compute::copy(params.begin(),params.end(),buff.begin(),queue);

    clFFT_lib(N,context,queue,buff);

    compute::copy(buff.begin(),buff.end(),params.begin(), queue);

    PrintOutVec(params);

}
