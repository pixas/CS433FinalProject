#ifndef _GROUP14_DATA_UTILS_HPP_
#define _GROUP14_DATA_UTILS_HPP_

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
cv::dnn::Net load_model(const char* model_name);
std::unordered_map<cv::String, std::unordered_map<cv::String, float *>> obtain_layer_info(const cv::dnn::Net& net);
std::unordered_map<cv::String, std::unordered_map<cv::String, float *>> obtain_layer_info(const cv::String& file_name);
std::unordered_map<cv::String, std::unordered_map<cv::String, float *>> obtain_layer_info(const char * file_name);
class DataLoader {
    private:
        int bsz;
        int n_workers;
        int max_file = 10000;
        int file_ptr = 0;
        std::vector<std::string> file_list;
    public:
        DataLoader(const std::string file_name, int batch_size=1, int num_workers=1) {
            bsz = batch_size;
            n_workers = num_workers;
            file_list.resize(max_file);
            std::fstream f_cin;
            f_cin.open(file_name.c_str(), std::ios::out | std::ios::in);
            for (int i = 0; i < max_file; ++i) {
                if (!std::getline(f_cin, file_list[i])) {
                    max_file = i;
                    break;
                }
                // std::getline(f_cin, file_list[i]);
                int cur_size = file_list[i].size();
                file_list[i] = file_list[i].substr(1, cur_size - 3);
            }

            f_cin.close();
        }

        ~DataLoader() {

        }
        int load_batch_data(float *input_batch_images, std::vector<std::string>& image_names, int* output_batch_size) {
            // resize to 224 x 224
            std::vector<cv::Mat> batched_images(bsz);
            int i;
            int height = 224, width = 224;
            int channels = 3;
            int begin_ptr = file_ptr;
            for (i = 0; i < bsz; ++i) {
                if (file_ptr == max_file) {
                    break;
                }
                auto temp = cv::imread(file_list[file_ptr]);
                cv::Mat temp2;

                cv::resize(temp, temp2, cv::Size(224, 224));
                cv::cvtColor(temp2, batched_images[i], cv::COLOR_BGR2RGB);
                file_ptr += 1;
            }
            int end_ptr = file_ptr;
            image_names.assign(file_list.begin() + begin_ptr, file_list.begin() + end_ptr);
            if (i == 0) {
                return -1;
            }
            
            int cur_batch_size = end_ptr - begin_ptr;
            *output_batch_size = cur_batch_size;
            
            // copy the data to the input_batch_images
            for (int b = 0; b < cur_batch_size; ++b) {
                for (int c = 0; c < channels; ++c) {
                    for (int h = 0; h < height; ++h) {
                        for (int w = 0; w < width; ++w) {
                            input_batch_images[b * channels * height * width + c * height * width + h * width + w] = batched_images[b].at<cv::Vec3b>(h, w)[c] / (float)255;
                        }
                    }
                }
            }
            
            return 0;
        }
};

#endif 