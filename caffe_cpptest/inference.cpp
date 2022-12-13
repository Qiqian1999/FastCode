#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace std;
using namespace caffe;
using namespace cv;
using std::string;

void wrapInputLayer(Blob<float>* input_layer, std::vector<cv::Mat>* input_channels) {
    // Blob<float>* input_layer = net_->input_blobs()[0];
    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}

void preprocess(const cv::Mat& img,
                std::vector<cv::Mat>* input_channels) {
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;

    img.convertTo(sample, CV_32FC3);

    /* This operation will write the separate BGR planes directly to the
    * input layer of the network because it is wrapped by the cv::Mat
    * objects in input_channels. */
    cv::split(sample, *input_channels);
}

int main(int argc, char** argv) {
    const string model_file = argv[1]; // deploy.prototxt
    const string trained_file = argv[2]; // VGG_ILSVRC_16_layers.caffemodel
    const string imagePath = argv[3]; // cat.jpg
    const string layer_name = argv[4]; // conv1_1
    const string output = argv[5]; // conv1_1.txt

    std::ofstream file(output);
    cv::Mat img = imread(imagePath);

    /* Load the network. */
    Caffe::set_mode(Caffe::CPU);
    boost::shared_ptr<Net<float> > net_;
    net_.reset(new Net<float>(model_file, TEST));
    net_->CopyTrainedLayersFrom(trained_file);

    /* Input */
    std::vector<cv::Mat> input_channels;
    caffe::Blob<float>* inputLayer = net_->input_blobs()[0];
    inputLayer->Reshape(1, 3, 224, 224);
    net_->Reshape();
    wrapInputLayer(inputLayer, &input_channels);
    preprocess(img, &input_channels);

    /* Forward */
    net_->Forward();

    /* Output */
    boost::shared_ptr<caffe::Blob<float>> output_layer = net_->blob_by_name(layer_name);
    const float* begin = output_layer->cpu_data();
    const float* end = begin + output_layer->channels();
    std::vector<float> output_flatten(begin, end);

    for (auto it = output_flatten.begin(); it != output_flatten.end(); ++it) {
        if (it == output_flatten.begin()) {
            file << *it;
        } else {
            file << " " << *it;
        }
    }
    file.close();
}
