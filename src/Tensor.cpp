#include <vector>
#include <ctime>
#include <stdexcept>
#include <cmath>
#include "Filters.h"
#include "Tensor.h"

#include <x86intrin.h>
#include <immintrin.h>

#define MAX_FREQ 3.4
#define BASE_FREQ 2.4

static __inline__ unsigned long long rdtsc(void) {
    unsigned hi, lo;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}

using namespace std;

Tensor::Tensor(){}

Tensor::Tensor(int height, int width)
{
    this->height = height;
    this->width = width;
    this->depth = 0;
    this->layers = vector<Matrix> (depth);
}

Tensor::Tensor(int height, int width, int depth)
{
    this->height = height;
    this->width = width;
    this->depth = depth;
    this->layers = vector<Matrix> (depth);
}

Tensor::Tensor(vector<Matrix> const &layers)
{
    this->height = layers[0].getHeight();
    this->width = layers[0].getWidth();
    this->depth = layers.size();
    this->layers = layers;
}

int Tensor::getHeight() const
{   
    return height;
}       
        
int Tensor::getWidth() const
{           
    return width;
}

int Tensor::getDepth() const
{
    return depth;
}

void Tensor::addLayer(Matrix layer)
{
    layers.push_back(layer);
    depth++;
}

Matrix Tensor::getLayer(int index) const
{
    return layers[index];
}

void Tensor::randomValueInit(int low, int high)
{

    for (int i=0; i<depth; i++){
        vector<vector<double>> layer;
        for (int y=0; y<height; y++){
            vector<double> rows;
            for (int x=0; x<width; x++){
                double temp = low + (rand() % (high - low + 1));
                rows.push_back(temp);
            }
            layer.push_back(rows);
        }
        layers[i] = layer;
    }
}

Tensor Tensor::fwdConv(Filters setOfFilters, int stride, int bias)
{
    int F = setOfFilters.getWidth();
    float f_W = (float)width;
    float f_F = (float)F;
    float f_S = (float)stride;
    int output_size = ceil((f_W-f_F)/f_S)+1;
    
    if (output_size < 1)
        throw logic_error("Invalid: Output matrix size 0.");    
    
    Tensor outputVolume = Tensor(output_size, output_size);
    
    for (int filterNumber=0; filterNumber<setOfFilters.getNumberOfFilters(); filterNumber++) {
        //temporarily doing addition of blank matrix in first iteration -- will fix later
        Matrix result = Matrix(output_size, output_size);
        for (int i=0; i<depth; i++){
            result.add(layers[i].filterSlide(setOfFilters.getFilter(filterNumber).getLayer(i), stride, bias));
        }

        if (bias > 0) {
            vector<vector<double>> bias_filter(output_size, vector<double>(output_size, bias));
            result.add(bias_filter);
        }

        cout << "Convolution->[Tensor layer]: " << filterNumber << endl;

        outputVolume.addLayer(result);                
    }

    return outputVolume;
}

Tensor Tensor::fwdConv(Filters setOfFilters, int stride, int bias, int padding)
{
    int F = setOfFilters.getWidth();
    float f_W = (float)width;
    float f_F = (float)F;
    float f_S = (float)stride;
    float f_P = (float)padding;
    int output_size = ceil((f_W-f_F+2*f_P)/f_S)+1;

    if (output_size < 1)
        throw logic_error("Invalid: Output matrix size 0.");    
    
    Tensor outputVolume = Tensor(output_size, output_size);
    
    for (int filterNumber=0; filterNumber<setOfFilters.getNumberOfFilters(); filterNumber++) {
        //temporarily doing addition of blank matrix in first iteration -- will fix later
        Matrix result = Matrix(output_size, output_size);
        for (int i=0; i<depth; i++){
            Matrix filter = setOfFilters.getFilter(filterNumber).getLayer(i);
            Matrix result_depth_i = layers[i].filterSlide(filter, stride, bias, padding);
            result.add(result_depth_i);
        }

        if (bias > 0) {
            vector<vector<double>> bias_filter(output_size, vector<double>(output_size, bias));
            result.add(bias_filter);
        }

        outputVolume.addLayer(result);                
    }

    return outputVolume;
}

double Tensor::kernel(double* A, double* B, int y, int x, int z, int F, int f_W_padded)
{
    double output = 0;

    for (int k = 0; k < depth; k++) {
        for (int i = y; i < (y+F); i++) {
            for (int j = x; j < (x+F); j++) {
                double a = A[f_W_padded*f_W_padded*k + f_W_padded*i + j];
                double b = B[F*F*depth*z + F*F*k + F*(i-y) + (j-x)];
                output += a*b;
            }
        }
    }

    return output;
}

Tensor Tensor::fwdConv_baseline(Filters setOfFilters, int stride, int bias, int padding)
{
    int F = setOfFilters.getWidth(); // filter_size
    float f_W = (float)width;
    float f_F = (float)F;
    float f_S = (float)stride;
    float f_P = (float)padding;
    int output_size = ceil((f_W-f_F+2*f_P)/f_S)+1;

    if (output_size < 1)
        throw logic_error("Invalid: Output matrix size 0.");    
    
    Tensor outputVolume = Tensor(output_size, output_size);

    int numberOfFilters = setOfFilters.getNumberOfFilters();

    // A
    int f_W_padded = f_W+2*f_P;
    double* inputs = new double[depth*f_W_padded*f_W_padded];

    for (int k = 0; k < depth; k++) {
        std::vector<std::vector<double>> padded_matrix;
        if (padding > 0) {
            padded_matrix = layers[k].getPadMatrix(padding);
        } else {
            padded_matrix = layers[k].matrix;
        }

        for (int i = 0; i < f_W_padded; i++) {
            for (int j = 0; j < f_W_padded; j++) {
                inputs[f_W_padded*f_W_padded*k + f_W_padded*i + j] = (double)padded_matrix[i][j];
            }
        }
    }

    // B
    double* filters = new double[numberOfFilters*depth*F*F]; // 64x3x3x3
    for (int l = 0; l < numberOfFilters; l++) {
        for (int k = 0; k < depth; k++) {
            Matrix filter = setOfFilters.getFilter(l).getLayer(k);

            for (int i = 0; i < F; i++) {
                for (int j = 0; j < F; j++) {
                    filters[F*F*depth*l + F*F*k + F*i + j] = (double)filter.getIndexValue(i, j); // matrix[i][j]
                }
            }
        }
    }

    unsigned long long t0, t1;

    // C
    double* flatten_output_tensor = new double[output_size*output_size*numberOfFilters]; // 224x224x64
    t0 = rdtsc();
    for (int y = 0; y < output_size; y++) {
        for (int x = 0; x < output_size; x++) {
            for (int z = 0; z < numberOfFilters; z++) {
                flatten_output_tensor[numberOfFilters*output_size*y + numberOfFilters*x + z] = kernel(inputs, filters, y, x, z, F, f_W_padded);
            }
        }
    }
    t1 = rdtsc();
    printf("TURBO Cycles Taken for Baseline: %lf\n\r", (double)(t1-t0)*MAX_FREQ/BASE_FREQ);

    // unpack C
    for (int z = 0; z < numberOfFilters; z++) {
        Matrix result = Matrix(output_size, output_size);

        for (int y = 0; y < output_size; y++) {
            for (int x = 0; x < output_size; x++) {
                result.matrix[y][x] = flatten_output_tensor[numberOfFilters*output_size*y + numberOfFilters*x + z];
            }
        }

        outputVolume.addLayer(result);
    }

    return outputVolume;
}

Tensor Tensor::fwdMaxPool(int pool_filter_height, int pool_filter_width, int stride, int bias)
{
    float f_W = (float)width;
    float f_P = (float)pool_filter_width;
    float f_S = (float)stride;
    int pool_output_size = ceil((f_W-f_P)/f_S)+1;
    
    if (pool_output_size < 1)
        throw logic_error("Invalid: Output matrix size 0.");    
    
    Tensor output_volume = Tensor(pool_output_size, pool_output_size);

    for (int i=0; i<depth; i++){
        Matrix result = Matrix(pool_output_size, pool_output_size);
        result = layers[i].maxSlide(pool_filter_height, pool_filter_width, stride, bias);

        output_volume.addLayer(result);

        if (bias > 0) {
            vector<vector<double>> bias_filter(pool_output_size, vector<double>(pool_output_size, bias));
            result.add(bias_filter);
        }
    }

    return output_volume;
}

