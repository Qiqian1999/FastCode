#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <iterator>
#include <time.h>
#include "Matrix.h"
#include "Tensor.h"
#include "Filters.h"
#include "Utility.h"

using namespace std;


void exp_effect_of_optimization_baseline() {
    // Testing convolution on 64x64x3 Tensor with 64 filters: 3x3x3 and stride 1, pad 1
    int padding = 1;
    int stride = 1;
    int bias = 0;

    // 64x64x3 input layer
    Tensor data_layer = Tensor(64, 64);
    data_layer.addLayer(Utility::createMatrixFromFile("layers/input_layer_64x64"));
    data_layer.addLayer(Utility::createMatrixFromFile("layers/input_layer_64x64"));
    data_layer.addLayer(Utility::createMatrixFromFile("layers/input_layer_64x64"));

    // // 3x3x3 x 64 filters
    Filters kernel_conv1_1 = Filters(3, 3, 3, 64);

    cout << "______ exp_effect_of_optimization_baseline Test Start_______________________\n" << endl;
    cout << "---- Test [Fwd Convolution] ----" << endl;

    Tensor conv1_1_layer_baseline = data_layer.fwdConv_baseline(kernel_conv1_1, stride, bias, padding);
}

void exp_effect_of_optimization_simd() {
    // Testing convolution on 64x64x3 Tensor with 64 filters: 3x3x3 and stride 1, pad 1
    int padding = 1;
    int stride = 1;
    int bias = 0;

    // 64x64x3 input layer
    Tensor data_layer = Tensor(64, 64);
    data_layer.addLayer(Utility::createMatrixFromFile("layers/input_layer_64x64"));
    data_layer.addLayer(Utility::createMatrixFromFile("layers/input_layer_64x64"));
    data_layer.addLayer(Utility::createMatrixFromFile("layers/input_layer_64x64"));

    // // 3x3x3 x 64 filters
    Filters kernel_conv1_1 = Filters(3, 3, 3, 64);

    cout << "______ exp_effect_of_optimization_simd Test Start_______________________\n" << endl;
    cout << "---- Test [Fwd Convolution] ----" << endl;

    Tensor conv1_1_layer_simd = data_layer.fwdConv_simd(kernel_conv1_1, stride, bias, padding);
}

void exp_effect_of_optimization_simd_openmp() {
    // Testing convolution on 64x64x3 Tensor with 64 filters: 3x3x3 and stride 1, pad 1
    int padding = 1;
    int stride = 1;
    int bias = 0;

    // 64x64x3 input layer
    Tensor data_layer = Tensor(64, 64);
    data_layer.addLayer(Utility::createMatrixFromFile("layers/input_layer_64x64"));
    data_layer.addLayer(Utility::createMatrixFromFile("layers/input_layer_64x64"));
    data_layer.addLayer(Utility::createMatrixFromFile("layers/input_layer_64x64"));

    // // 3x3x3 x 64 filters
    Filters kernel_conv1_1 = Filters(3, 3, 3, 64);

    cout << "______ exp_effect_of_optimization_simd_openmp Test Start_______________________\n" << endl;
    cout << "---- Test [Fwd Convolution] ----" << endl;

    Tensor conv1_1_layer_simd_openmp = data_layer.fwdConv_simd_openmp(kernel_conv1_1, stride, bias, padding);
}

void exp_comparison_with_different_problem_sizes_64x64x16() {
    int padding = 1;
    int stride = 1;
    int bias = 0;

    Tensor data_layer = Tensor(64, 64);
    data_layer.addLayer(Utility::createMatrixFromFile("layers/input_layer_64x64"));
    data_layer.addLayer(Utility::createMatrixFromFile("layers/input_layer_64x64"));
    data_layer.addLayer(Utility::createMatrixFromFile("layers/input_layer_64x64"));

    Filters kernel_conv1_1 = Filters(3, 3, 3, 16);

    cout << "______ exp_comparison_with_different_problem_sizes_64x64x16 Test Start_______________________\n" << endl;
    cout << "---- Test [Fwd Convolution] ----" << endl;

    Tensor conv1_1_layer_simd_openmp = data_layer.fwdConv_simd_openmp(kernel_conv1_1, stride, bias, padding);
}

void exp_comparison_with_different_problem_sizes_64x64x32() {
    int padding = 1;
    int stride = 1;
    int bias = 0;

    Tensor data_layer = Tensor(64, 64);
    data_layer.addLayer(Utility::createMatrixFromFile("layers/input_layer_64x64"));
    data_layer.addLayer(Utility::createMatrixFromFile("layers/input_layer_64x64"));
    data_layer.addLayer(Utility::createMatrixFromFile("layers/input_layer_64x64"));

    Filters kernel_conv1_1 = Filters(3, 3, 3, 32);

    cout << "______ exp_comparison_with_different_problem_sizes_64x64x32 Test Start_______________________\n" << endl;
    cout << "---- Test [Fwd Convolution] ----" << endl;

    Tensor conv1_1_layer_simd_openmp = data_layer.fwdConv_simd_openmp(kernel_conv1_1, stride, bias, padding);
}

void exp_comparison_with_different_problem_sizes_64x64x64() {
    int padding = 1;
    int stride = 1;
    int bias = 0;

    Tensor data_layer = Tensor(64, 64);
    data_layer.addLayer(Utility::createMatrixFromFile("layers/input_layer_64x64"));
    data_layer.addLayer(Utility::createMatrixFromFile("layers/input_layer_64x64"));
    data_layer.addLayer(Utility::createMatrixFromFile("layers/input_layer_64x64"));

    Filters kernel_conv1_1 = Filters(3, 3, 3, 64);

    cout << "______ exp_comparison_with_different_problem_sizes_64x64x64 Test Start_______________________\n" << endl;
    cout << "---- Test [Fwd Convolution] ----" << endl;

    Tensor conv1_1_layer_simd_openmp = data_layer.fwdConv_simd_openmp(kernel_conv1_1, stride, bias, padding);
}

void exp_inference_with_deep_convolution_neural_nets_baseline() {
    int padding = 1;
    int stride = 1;
    int bias = 0;

    Tensor data_layer = Tensor(224, 224);
    data_layer.addLayer(Utility::createMatrixFromFile("layers/input_layer_224x224"));
    data_layer.addLayer(Utility::createMatrixFromFile("layers/input_layer_224x224"));
    data_layer.addLayer(Utility::createMatrixFromFile("layers/input_layer_224x224"));

    Filters kernel_conv1_1 = Filters(3, 3, 3, 64);
    Filters kernel_conv1_2 = Filters(3, 3, 64, 64);
    Filters kernel_conv2_1 = Filters(3, 3, 64, 128);
    Filters kernel_conv2_2 = Filters(3, 3, 128, 128);
    Filters kernel_conv3_1 = Filters(3, 3, 128, 256);
    Filters kernel_conv3_2 = Filters(3, 3, 256, 256);
    Filters kernel_conv3_3 = Filters(3, 3, 256, 256);
    Filters kernel_conv4_1 = Filters(3, 3, 256, 512);
    Filters kernel_conv4_2 = Filters(3, 3, 512, 512);
    Filters kernel_conv4_3 = Filters(3, 3, 512, 512);
    Filters kernel_conv5_1 = Filters(3, 3, 512, 512);
    Filters kernel_conv5_2 = Filters(3, 3, 512, 512);
    Filters kernel_conv5_3 = Filters(3, 3, 512, 512);

    cout << "______ exp_inference_with_deep_convolution_neural_nets_baseline Test Start_______________________\n" << endl;
    cout << "---- Test [Fwd Convolution] ----" << endl;

    cout << "---- conv1_1 ----" << endl;
    Tensor conv1_1 = data_layer.fwdConv_baseline(kernel_conv1_1, stride, bias, padding);
    cout << "---- conv1_2 ----" << endl;
    Tensor conv1_2 = conv1_1.fwdConv_baseline(kernel_conv1_2, stride, bias, padding);
    cout << "---- pool1 ----" << endl;
    Tensor pool1 = conv1_2.fwdMaxPool(2, 2, 2, 0);
    cout << "---- conv2_1 ----" << endl;
    Tensor conv2_1 = pool1.fwdConv_baseline(kernel_conv2_1, stride, bias, padding);
    cout << "---- conv2_2 ----" << endl;
    Tensor conv2_2 = conv2_1.fwdConv_baseline(kernel_conv2_2, stride, bias, padding);
    cout << "---- pool2 ----" << endl;
    Tensor pool2 = conv2_2.fwdMaxPool(2, 2, 2, 0);
    cout << "---- conv3_1 ----" << endl;
    Tensor conv3_1 = pool2.fwdConv_baseline(kernel_conv3_1, stride, bias, padding);
    cout << "---- conv3_2 ----" << endl;
    Tensor conv3_2 = conv3_1.fwdConv_baseline(kernel_conv3_2, stride, bias, padding);
    cout << "---- conv3_3 ----" << endl;
    Tensor conv3_3 = conv3_2.fwdConv_baseline(kernel_conv3_3, stride, bias, padding);
    cout << "---- pool3 ----" << endl;
    Tensor pool3 = conv3_3.fwdMaxPool(2, 2, 2, 0);
    cout << "---- conv4_1 ----" << endl;
    Tensor conv4_1 = pool3.fwdConv_baseline(kernel_conv4_1, stride, bias, padding);
    cout << "---- conv4_2 ----" << endl;
    Tensor conv4_2 = conv4_1.fwdConv_baseline(kernel_conv4_2, stride, bias, padding);
    cout << "---- conv4_3 ----" << endl;
    Tensor conv4_3 = conv4_2.fwdConv_baseline(kernel_conv4_3, stride, bias, padding);
    cout << "---- pool4 ----" << endl;
    Tensor pool4 = conv4_3.fwdMaxPool(2, 2, 2, 0);
    cout << "---- conv5_1 ----" << endl;
    Tensor conv5_1 = pool4.fwdConv_baseline(kernel_conv5_1, stride, bias, padding);
    cout << "---- conv5_2 ----" << endl;
    Tensor conv5_2 = conv5_1.fwdConv_baseline(kernel_conv5_2, stride, bias, padding);
    cout << "---- conv5_3 ----" << endl;
    Tensor conv5_3 = conv5_2.fwdConv_baseline(kernel_conv5_3, stride, bias, padding);
}

void exp_inference_with_deep_convolution_neural_nets_simd() {
    int padding = 1;
    int stride = 1;
    int bias = 0;

    Tensor data_layer = Tensor(224, 224);
    data_layer.addLayer(Utility::createMatrixFromFile("layers/input_layer_224x224"));
    data_layer.addLayer(Utility::createMatrixFromFile("layers/input_layer_224x224"));
    data_layer.addLayer(Utility::createMatrixFromFile("layers/input_layer_224x224"));

    Filters kernel_conv1_1 = Filters(3, 3, 3, 64);
    Filters kernel_conv1_2 = Filters(3, 3, 64, 64);
    Filters kernel_conv2_1 = Filters(3, 3, 64, 128);
    Filters kernel_conv2_2 = Filters(3, 3, 128, 128);
    Filters kernel_conv3_1 = Filters(3, 3, 128, 256);
    Filters kernel_conv3_2 = Filters(3, 3, 256, 256);
    Filters kernel_conv3_3 = Filters(3, 3, 256, 256);
    Filters kernel_conv4_1 = Filters(3, 3, 256, 512);
    Filters kernel_conv4_2 = Filters(3, 3, 512, 512);
    Filters kernel_conv4_3 = Filters(3, 3, 512, 512);
    Filters kernel_conv5_1 = Filters(3, 3, 512, 512);
    Filters kernel_conv5_2 = Filters(3, 3, 512, 512);
    Filters kernel_conv5_3 = Filters(3, 3, 512, 512);

    cout << "______ exp_inference_with_deep_convolution_neural_nets_simd Test Start_______________________\n" << endl;
    cout << "---- Test [Fwd Convolution] ----" << endl;

    cout << "---- conv1_1 ----" << endl;
    Tensor conv1_1 = data_layer.fwdConv_simd_openmp(kernel_conv1_1, stride, bias, padding);
    cout << "---- conv1_2 ----" << endl;
    Tensor conv1_2 = conv1_1.fwdConv_simd_openmp(kernel_conv1_2, stride, bias, padding);
    cout << "---- pool1 ----" << endl;
    Tensor pool1 = conv1_2.fwdMaxPool(2, 2, 2, 0);
    cout << "---- conv2_1 ----" << endl;
    Tensor conv2_1 = pool1.fwdConv_simd_openmp(kernel_conv2_1, stride, bias, padding);
    cout << "---- conv2_2 ----" << endl;
    Tensor conv2_2 = conv2_1.fwdConv_simd_openmp(kernel_conv2_2, stride, bias, padding);
    cout << "---- pool2 ----" << endl;
    Tensor pool2 = conv2_2.fwdMaxPool(2, 2, 2, 0);
    cout << "---- conv3_1 ----" << endl;
    Tensor conv3_1 = pool2.fwdConv_simd_openmp(kernel_conv3_1, stride, bias, padding);
    cout << "---- conv3_2 ----" << endl;
    Tensor conv3_2 = conv3_1.fwdConv_simd_openmp(kernel_conv3_2, stride, bias, padding);
    cout << "---- conv3_3 ----" << endl;
    Tensor conv3_3 = conv3_2.fwdConv_simd_openmp(kernel_conv3_3, stride, bias, padding);
    cout << "---- pool3 ----" << endl;
    Tensor pool3 = conv3_3.fwdMaxPool(2, 2, 2, 0);
    cout << "---- conv4_1 ----" << endl;
    Tensor conv4_1 = pool3.fwdConv_simd_openmp(kernel_conv4_1, stride, bias, padding);
    cout << "---- conv4_2 ----" << endl;
    Tensor conv4_2 = conv4_1.fwdConv_simd_openmp(kernel_conv4_2, stride, bias, padding);
    cout << "---- conv4_3 ----" << endl;
    Tensor conv4_3 = conv4_2.fwdConv_simd_openmp(kernel_conv4_3, stride, bias, padding);
    cout << "---- pool4 ----" << endl;
    Tensor pool4 = conv4_3.fwdMaxPool(2, 2, 2, 0);
    cout << "---- conv5_1 ----" << endl;
    Tensor conv5_1 = pool4.fwdConv_simd_openmp(kernel_conv5_1, stride, bias, padding);
    cout << "---- conv5_2 ----" << endl;
    Tensor conv5_2 = conv5_1.fwdConv_simd_openmp(kernel_conv5_2, stride, bias, padding);
    cout << "---- conv5_3 ----" << endl;
    Tensor conv5_3 = conv5_2.fwdConv_simd_openmp(kernel_conv5_3, stride, bias, padding);
}

int main(int argc, char* argv[]) {
    // srand(time(0)); //used for setting random values for filters
    srand(1); //used for setting random values for filters

    /*
     Experiment: Effects of Optimization
    */
    // exp_effect_of_optimization_baseline();
    // exp_effect_of_optimization_simd();
    // exp_effect_of_optimization_simd_openmp();

    /*
     Experiment: Comparison with different Problem Sizes
    */
    // exp_comparison_with_different_problem_sizes_64x64x16();
    // exp_comparison_with_different_problem_sizes_64x64x32();
    // exp_comparison_with_different_problem_sizes_64x64x64();

    /*
     Experiment: Inference with Deep Convolution Neural Nets
    */
    // exp_inference_with_deep_convolution_neural_nets_baseline();
    // exp_inference_with_deep_convolution_neural_nets_simd();

    return 0;
}	
