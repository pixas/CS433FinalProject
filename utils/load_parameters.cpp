
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace cv;

cv::dnn::Net load_model(const char* model_name){
    const cv::String read_file_name = (cv::String)model_name;
    cv::dnn::Net net = cv::dnn::readNetFromONNX(read_file_name);
    return net;
}

std::unordered_map<String, std::unordered_map<String, cv::Mat>> obtain_layer_info(const dnn::Net& net) {
    std::vector<String> layer_names = net.getLayerNames();
    std::unordered_map<String, std::unordered_map<String, cv::Mat>> params;
    for (int i = 0; i < layer_names.size(); ++i) {
        int id = net.getLayerId(layer_names[i]);
        auto layer = net.getLayer(id);
        
        std::cout << "layer id: " << id << ", type: " << layer->type.c_str() << ", name: " << layer->name.c_str() << std::endl;
        auto layer_blobs = layer->blobs;
        
        for (int j = 0; j < layer_blobs.size(); ++j) {
            auto param_size = layer_blobs[j].size;
            String size_string = "";
            params[layer->name][j == 0 ? "weight" : "bias"] = layer_blobs[j];
            for (int k = 0; k < param_size.dims(); ++k) {
                size_string += std::to_string(param_size[k]) + ",";
            }
            printf("\tLayer %d's param %d's dims %d and size (%s) \n", id, j, param_size.dims(), size_string.c_str());
        }
    }
    return params;
}

std::unordered_map<String, std::unordered_map<String, cv::Mat>> obtain_layer_info(const String& file_name) {
    auto net = load_model(file_name.c_str());
    return obtain_layer_info(net);
}

std::unordered_map<String, std::unordered_map<String, cv::Mat>> obtain_layer_info(const char * file_name) {
    auto net = load_model(file_name);
    return obtain_layer_info(net);
}

int main(int argc, char const *argv[])
{
    auto net = load_model("./resnet18.proto");
    auto params = obtain_layer_info(net);
    for (auto idx = params.begin(); idx != params.end(); ++idx) {
        printf("%s\n", idx->first.c_str());
        for (auto j = idx->second.begin(); j != idx->second.end(); ++j) {
            printf("\t%s\n", j->first.c_str());
        }
    }
    return 0;
}
