#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <fstream>


cv::dnn::Net load_model(const char* model_name) {
    const cv::String read_file_name = (cv::String)model_name;
    
    cv::dnn::Net net = cv::dnn::readNetFromONNX(read_file_name);
    return net;
}

int main(int argc, char const *argv[])
{
    cv::dnn::Net net = load_model("/home/group14/CS433FinalProject/resnet18.onnx");
    std::vector<std::string> file_list;
    const int max_file = 10000;
    file_list.resize(max_file);
    std::fstream f_cin;
    std::string file_name = "/home/group14/CS433FinalProject/select_file_list.txt";
    f_cin.open(file_name.c_str(), std::ios::out | std::ios::in);
    for (int i = 0; i < max_file; ++i) {
        std::getline(f_cin, file_list[i]);
        int cur_size = file_list[i].size();
        file_list[i] = file_list[i].substr(1, cur_size - 3);
    }

    
    f_cin.close();

    for (auto & file_name : file_list) {
        cv::Mat image = cv::imread(file_name);
        cv::Mat resized_image;
        cv::resize(image, resized_image, cv::Size(224, 224));
        cv::Mat blob;
        cv::dnn::blobFromImage(resized_image, blob, 1.0, cv::Size(224, 224));
        double t = cv::getTickCount(); 
        net.setInput(blob);
        cv::Mat output = net.forward();
        t = (cv::getTickCount() - t) / cv::getTickFrequency();
        double max_value;
        cv::Point max_index;
        cv::minMaxLoc(output, NULL, &max_value, NULL, &max_index);
        std::cout << file_name << " "  << max_value << " " << max_index.x << std::endl;
    }
    return 0;
}
