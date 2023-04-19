#include "CL/cl.h"
#include "CL/cl2.hpp"
#include <clFFT.h>
#include <iostream>
#include <boost/compute.hpp>
#include <vector>
#include <string>
#include <clFFT.h>



void PrintOutVec(const std::vector<float>& vec)
{
    for( auto v : vec)
    {
        std::cout << v << " ";
    }
}

void CreateVector(std::vector<float>& vec, size_t N)
{
    printf("\nPerforming fft on an one dimensional array of size N = %lu\n", (unsigned long)N);
    for(size_t i = 0; i < vec.size() ;i++ )
        {
            float x = 1;//i;
            float y = 4;//i * 3;
            vec[i] = x;
            vec[i + 1] = y;
            std::cout <<std::fixed << "(" << vec[i] << "," << vec[i+1] << ")" ;

        }
    printf("\n\nfft result: \n");
}

void DevicesOnThePlatform(std::vector<cl::Platform>& platforms,std::vector<cl::Device>& devices)
{
    cl::Platform::get(&platforms);
    for(const auto& platform : platforms)
    {
        std::cout << "Platform found:   "<< platform.getInfo<CL_PLATFORM_NAME>()<<std::endl;
    }
    platforms.front().getDevices(CL_DEVICE_TYPE_GPU,&devices);

    for(const auto& device : devices)
    {
        std::cout <<"Device:\t\t " <<device.getInfo<CL_DEVICE_NAME>()<<std::endl;
    }
}

void clFFT_lib(const size_t N,cl::Context& context,cl::CommandQueue& queue,cl::Buffer& bufVec,std::vector<float>& params)
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

    //===================================================================================
    /* Create a default plan for a complex FFT. */
        err = clfftCreateDefaultPlan(&planHandle, context(), dim, clLengths);

        /* Set plan parameters. */
        err = clfftSetPlanPrecision(planHandle, CLFFT_SINGLE);
        err = clfftSetLayout(planHandle, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
        err = clfftSetResultLocation(planHandle, CLFFT_INPLACE); // CLFFT_INPLACE CLFFT_OUTOFPLACE

        /* Bake the plan. */
        err = clfftBakePlan(planHandle, 1, &queue(), NULL, NULL);

        /* Execute the plan. */
        err = clfftEnqueueTransform(planHandle, CLFFT_FORWARD, 1, &queue(), 0, NULL, NULL, &bufVec(),NULL, NULL);
        queue.finish();

        //считывание результата
        queue.enqueueReadBuffer(bufVec,CL_TRUE, 0, params.size() * sizeof(float), params.data());



        /* Release the plan. */
        err = clfftDestroyPlan( &planHandle );
        /* Release clFFT library. */
        clfftTeardown();
}

int main()
{
    size_t N = 16;
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device>devices;
    //std::vector<float>params(N * 2,1);
    std::vector<float>params(N * 2);
    params[0] = 16;
    params[1] = 16;


    DevicesOnThePlatform(platforms,devices); // перебор и вывод всех доступных платформ Opencl и устройств на ПК

    cl::Context context(devices[0]);
    cl::CommandQueue queue(context,devices[0]);

    // создание буфера
    cl::Buffer bufVec(context,CL_MEM_READ_WRITE, params.size() * sizeof(float));
    //cl::Buffer bufOut(context,CL_MEM_READ_WRITE, out.size() * sizeof(float));

    // запись данных в буферы
    queue.enqueueWriteBuffer(bufVec,CL_TRUE,0,params.size() * sizeof(float),params.data());


    clFFT_lib(N,context,queue,bufVec,params);
    PrintOutVec(params);

    return 0;
}
