#ifndef _GROUP14_DATA_UTILS_HPP_
#define _GROUP14_DATA_UTILS_HPP_

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

cv::dnn::Net load_model(const char* model_name);
std::unordered_map<cv::String, std::unordered_map<cv::String, cv::Mat>> obtain_layer_info(const cv::dnn::Net& net);
std::unordered_map<cv::String, std::unordered_map<cv::String, cv::Mat>> obtain_layer_info(const cv::String& file_name);
std::unordered_map<cv::String, std::unordered_map<cv::String, cv::Mat>> obtain_layer_info(const char * file_name);
class DataLoader;
#endif 