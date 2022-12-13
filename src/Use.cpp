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

void test_pack_filters(){
    // 224x224x3 input layer
    Tensor data_layer = Tensor(64, 64);
    data_layer.addLayer(Utility::createMatrixFromFile("layers/input_layer_64x64"));
    data_layer.addLayer(Utility::createMatrixFromFile("layers/input_layer_64x64"));
    data_layer.addLayer(Utility::createMatrixFromFile("layers/input_layer_64x64"));

    // // 3x3x3 x 64 filters
    Filters kernel_conv1_1 = Filters(3, 3, 3, 64);

    cout << "______test_224x224_Conv Test Start_______________________\n" << endl;
    cout << "---- Test [Fwd Convolution] ----" << endl;

    double* output = new double[3*3*3*64];

    data_layer.pack_filters(output,kernel_conv1_1,64,3);

    for(int i = 0; i < 4; i++){
        cout<<"---this is the "<<i<< "kernel----"<<endl;
        for(int j = 0; j < 3; j++){
            Matrix matrix = kernel_conv1_1.getFilter(i).getLayer(j);
            cout<<matrix.getIndexValue(0,0)<<matrix.getIndexValue(0,1)<<matrix.getIndexValue(0,2)<<endl;
            cout<<matrix.getIndexValue(1,0)<<matrix.getIndexValue(1,1)<<matrix.getIndexValue(1,2)<<endl;
            cout<<matrix.getIndexValue(2,0)<<matrix.getIndexValue(2,1)<<matrix.getIndexValue(2,2)<<endl;
        }
        
        
    }
    cout<<"---this is the output----"<<endl;
    for(int i = 0; i<27; i++){
        cout<<output[i*4]<<output[i*4+1]<<output[i*4+2]<<output[i*4+3]<<endl;
    }
}

//Testing convolution on 224x224x3 Tensor with 64 filters: 3x3x3 and stride 1, pad 1
void test_224x224_Conv() {
    int padding = 1;
    int stride = 1;
    int bias = 0;

    // 224x224x3 input layer
    Tensor data_layer = Tensor(64, 64);
    data_layer.addLayer(Utility::createMatrixFromFile("layers/input_layer_64x64"));
    data_layer.addLayer(Utility::createMatrixFromFile("layers/input_layer_64x64"));
    data_layer.addLayer(Utility::createMatrixFromFile("layers/input_layer_64x64"));

    // // 3x3x3 x 64 filters
    Filters kernel_conv1_1 = Filters(3, 3, 3, 64);

    cout << "______test_224x224_Conv Test Start_______________________\n" << endl;
    cout << "---- Test [Fwd Convolution] ----" << endl;

    Tensor conv1_1_layer = data_layer.fwdConv(kernel_conv1_1, stride, bias, padding);
    Tensor conv1_1_layer_baseline = data_layer.fwdConv_baseline(kernel_conv1_1, stride, bias, padding); // TURBO Cycles Taken for Baseline: 186363473.250000
    Tensor conv1_1_layer_simd = data_layer.fwdConv_simd(kernel_conv1_1, stride, bias, padding); // 

    cout << "conv1_1_layer output: " << endl;
    conv1_1_layer.getLayer(3).print();
    cout << "== end conv1_1_layer output: " << endl;

    cout << "conv1_1_layer_baseline output: " << endl;
    conv1_1_layer_baseline.getLayer(3).print();
    cout << "== end conv1_1_layer_baseline output: " << endl;

    cout << "conv1_1_layer_simd output: " << endl;
    conv1_1_layer_simd.getLayer(3).print();
    cout << "== end conv1_1_layer_simd output: " << endl;

    cout << "\n[Input volume]: 224x224x3 --> Convolution (filter: 3x3x3 * 64 @ stride=1, padding=1) --> [Output volume]: "
         << conv1_1_layer.getHeight() << "x"
         << conv1_1_layer.getWidth() << "x"
         << conv1_1_layer.getDepth() << endl;

    // printing first layer of output volume to see if it actually convolved input layer
    cout << "---- [Input matrix layer 0] ----" << endl;
    data_layer.getLayer(0).print();
    // cout << "---- [Input matrix layer 1] ----" << endl;
    // data_layer.getLayer(1).print();
    // cout << "---- [Input matrix layer 2] ----" << endl;
    // data_layer.getLayer(2).print();

    cout << "###### FILTER 0  #####" << endl;
    kernel_conv1_1.getFilter(0).getLayer(0).print();
    // cout << "###### FILTER 0  #####" << endl;
    // kernel_conv1_1.getFilter(0).getLayer(1).print();
    // cout << "###### FILTER 0  #####" << endl;
    // kernel_conv1_1.getFilter(0).getLayer(2).print();

    cout << "######################" << endl;
    conv1_1_layer.getLayer(0).print();
    cout << "Output volume ->  " << conv1_1_layer.getHeight() << "x" << conv1_1_layer.getWidth() << "x" << conv1_1_layer.getDepth() << endl;

    cout << "\n___________________Test End_________________________\n" << endl;
}

int main(int argc, char* argv[]) {
    // srand(time(0)); //used for setting random values for filters
    srand(1); //used for setting random values for filters
    //test_224x224_Conv();
    test_pack_filters();
    return 0;
}	
