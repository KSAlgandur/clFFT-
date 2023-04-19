#include "CL/cl.h"
#include "CL/cl2.hpp"
#include <clFFT.h>
#include <iostream>
#include <boost/compute.hpp>
#include <vector>
#include <string>
#include <clFFT.h>

float* CreateArray(int N)
{
    //X = (float *)malloc(N * 2 * sizeof(*X));
    float *X = new float[N * 2];

    printf("\nPerforming fft on an one dimensional array of size N = %lu\n", (unsigned long)N);
      int print_iter = 0;
      while(print_iter < N) {
          float x = (float)print_iter;
          float y = (float)print_iter*3;
          X[2*print_iter  ] = x;
          X[2*print_iter+1] = y;
          printf("(%f, %f) ", x, y);
          print_iter++;
      }
      printf("\n\nfft result: \n");

      return X;
}

void PrintOutArr(float*X,int N)
{
    // Print the output array
    for(int i = 0; i < N; i++) {
        std::cout << "(" << X[2 * i] << ", " << X[2 * i + 1] << ") ";
    }

    std::cout << std::endl;
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

float* clFFT_lib(const size_t N,cl::Context& context,cl::CommandQueue& queue,cl::Buffer& bufX,float* X)
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
        err = clfftSetResultLocation(planHandle, CLFFT_INPLACE);

        /* Bake the plan. */
        err = clfftBakePlan(planHandle, 1, &queue(), NULL, NULL);

        /* Execute the plan. */
        err = clfftEnqueueTransform(planHandle, CLFFT_FORWARD, 1, &queue(), 0, NULL, NULL, &bufX(), NULL, NULL);

        queue.finish();

        //считывание результата

        queue.enqueueReadBuffer(bufX,CL_TRUE,0,N * 2 * sizeof (*X),X);

        /* Release the plan. */
        err = clfftDestroyPlan( &planHandle );
        /* Release clFFT library. */
        clfftTeardown();

    return X;
}

int main()
{

    float *X;
    size_t N = 16;
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device>devices;

    DevicesOnThePlatform(platforms,devices); // перебор и вывод всех доступных платформ Opencl и устройств на ПК

    cl::Context context(devices[0]);
    cl::CommandQueue queue(context,devices[0]);

    X = CreateArray(N);

    // создание буфера
    cl::Buffer bufX(context,CL_MEM_READ_WRITE, N * 2 * sizeof(float));
    // запись данных с векторов в буферы
    queue.enqueueWriteBuffer(bufX,CL_TRUE,0,N * 2 * sizeof (float),X);

    X = clFFT_lib(N,context,queue,bufX,X);
    PrintOutArr(X,N);

    delete [] X;
    return 0;
}
