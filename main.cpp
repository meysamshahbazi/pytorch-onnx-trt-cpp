#include <iostream>
#include <fstream>
#include <NvInfer.h>
#include <memory>
#include <NvOnnxParser.h>
#include <vector>
#include <cuda_runtime_api.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <algorithm>
#include <numeric>



struct  TRTDestroy
{
    template<class T>
    void operator()(T* obj) const 
    {
        if (obj)
            obj->destroy();
    }
};


int main(int argc, const char ** argv) 
{
    if (argc < 3)
    {
        std::cerr<<"usage: " << argv[0] << " model.onnx image.jpg"<<std::endl;
        return -1;
    }

    std::string model_path = argv[1];
    std::string image_path = argv[2];
    const int batch_size = 1;



    return 0;
}
