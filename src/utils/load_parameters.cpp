
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>

using namespace cv;



int get_mat_size(cv::Mat src) {
    auto param_size = src.size;
    int total_size = 1;
    for (int i = 0; i < param_size.dims(); ++i) {
        total_size *= param_size[i];
    }
    return total_size;
}

cv::dnn::Net load_model(const char* model_name) {
    const cv::String read_file_name = (cv::String)model_name;
    // try
    // {
    //     /* code */cv::dnn::Net net = cv::dnn::readNetFromTorch(read_file_name);
    //     return net;
    // }
    // catch(const std::exception& e)
    // {
    //     std::cerr << e.what() << '\n';
    // }
    
    cv::dnn::Net net = cv::dnn::readNetFromONNX(read_file_name);
    return net;

}

std::unordered_map<String, std::unordered_map<String, float *>> obtain_layer_info(const dnn::Net& net) {
    std::vector<String> layer_names = net.getLayerNames();
    std::unordered_map<String, std::unordered_map<String, float *>> params;
    for (int i = 0; i < layer_names.size(); ++i) {
        int id = net.getLayerId(layer_names[i]);
        auto layer = net.getLayer(id);
        
        auto layer_blobs = layer->blobs;
        
        for (int j = 0; j < layer_blobs.size(); ++j) {
            auto param_size = layer_blobs[j].size;
            String size_string = "";

            // mat2array
            float * temp;
            int total_size = get_mat_size(layer_blobs[j]);
            temp = new float[total_size];
            memcpy(temp, layer_blobs[j].data, total_size * sizeof(float));

            params[layer->name][j == 0 ? "weight" : "bias"] = temp;
            for (int k = 0; k < param_size.dims(); ++k) {
                size_string += std::to_string(param_size[k]) + ",";
            }
            printf("\tLayer %d's param %d's dims %d and size (%s) \n", id, j, param_size.dims(), size_string.c_str());
        }
    }
    return params;
}

std::unordered_map<String, std::unordered_map<String, float *>> obtain_layer_info(const String& file_name) {
    auto net = load_model(file_name.c_str());
    return obtain_layer_info(net);
}

std::unordered_map<String, std::unordered_map<String, float *>> obtain_layer_info(const char * file_name) {
    auto net = load_model(file_name);
    return obtain_layer_info(net);
}




// int main(int argc, char const *argv[])
// {
//     int batch_size = 128;
//     // ResNet18 model(batch_size);
//     // model.cuda();
//     // model.load_from_statedict(state_dict);
//     std::vector<std::string> image_names;
//     DataLoader dt("/home/group14/CS433FinalProject/select_file_list.txt", batch_size);
//     float * batched_images;
//     dt.load_batch_data(&batched_images, image_names);
//     for (const auto& file_name: image_names) {
//         std::cout << file_name << std::endl;
//     }
//     return 0;
// }
