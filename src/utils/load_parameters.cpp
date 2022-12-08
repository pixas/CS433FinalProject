
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <fstream>
using namespace cv;



class DataLoader {
    private:
        int bsz;
        int n_workers;
        const int max_file = 10000;
        int file_ptr = 0;
        std::vector<std::string> file_list;
    public:
        DataLoader(int batch_size=1, int num_workers=1) {
            bsz = batch_size;
            n_workers = num_workers;
            file_list.reserve(max_file);
            std::fstream f_cin;
            f_cin.open("../select_file_list.txt", std::ios::out | std::ios::in);
            for (int i = 0; i < max_file; ++i) {
                std::getline(f_cin, file_list[i]);
                int cur_size = file_list[i].size();
                file_list[i].substr(1, cur_size - 3);
            }


            f_cin.close();
        }

        ~DataLoader() {

        }
        void load_batch_data(float * input_batch_images) {
            std::vector<cv::Mat> batched_images(bsz);
            int i;
            int height, width;
            for (i = 0; i < bsz; ++i) {
                auto temp = cv::imread(file_list[file_ptr]);
                height = temp.size[1];
                width = temp.size[0];
                cv::cvtColor(temp, batched_images[i], cv::COLOR_BGR2RGB);

                file_ptr += 1;
                if (file_ptr == max_file) {
                    break ;
                }
            }

            std::vector<cv::Mat> output_images;
            output_images.assign(batched_images.begin(), batched_images.begin() + i);
            int cur_batch_size = output_images.size();
            input_batch_images = new float[cur_batch_size * 3 * height * width];
            for (int b = 0; b < cur_batch_size; ++b) {
                for (int i = 0; i < height; ++i) {
                    for (int j = 0; j < width; ++j) {
                        for (int k = 0; k < 3; ++k) {
                            input_batch_images[b * 3 * height * width + k * height * width + i * width + j] = (float)output_images[b].at<cv::Vec3b>(i, j)[k];
                        }
                    }
                }
            }
            return;
        }
};




cv::dnn::Net load_model(const char* model_name) {
    const cv::String read_file_name = (cv::String)model_name;
    try
    {
        /* code */cv::dnn::Net net = cv::dnn::readNetFromTorch(read_file_name);
        return net;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
    cv::dnn::Net net = cv::dnn::readNetFromONNX("resnet18.proto");
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
    auto net = load_model("/home/group14/CS433FinalProject/resnet18.t7");
    auto params = obtain_layer_info(net);
    for (auto idx = params.begin(); idx != params.end(); ++idx) {
        printf("%s\n", idx->first.c_str());
        for (auto j = idx->second.begin(); j != idx->second.end(); ++j) {
            printf("\t%s\n", j->first.c_str());
        }
    }
    return 0;
}
