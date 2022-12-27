#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <unordered_set>
#include <vector>
#include <fstream>
#include <iomanip>


cv::dnn::Net load_model(const char* model_name) {
    const cv::String read_file_name = (cv::String)model_name;
    
    cv::dnn::Net net = cv::dnn::readNetFromONNX(read_file_name);
    return net;
}

void print_mat(const cv::Mat & output) {
    std::fstream f_cin;
    f_cin.open("/home/group14/CS433FinalProject/oracle_err_output.txt", std::ios::out | std::ios::in);
    f_cin.flags(f_cin.fixed);
    f_cin.precision(3);
    for (int i = 0; i < output.size[1]; ++i) {
        f_cin << output.at<float>(0, i) << (i == output.size[1] - 1 ? "\n" : " ");
    }
    f_cin.close();
}

std::unordered_set<std::string> get_error_file(const char * error_file) {
    std::fstream f_cin;
    f_cin.open(error_file, std::ios::out | std::ios::in);
    std::string temp_file;
    std::unordered_set<std::string> output; 
    while (std::getline(f_cin, temp_file)) {
        output.insert(temp_file);
    }
    f_cin.close();
    return output;
}

int main(int argc, char const *argv[])
{
    cv::dnn::Net net = load_model("/home/group14/CS433FinalProject/task1/resnet18.onnx");
    std::vector<std::string> file_list;
    const int max_file = 5000;
    file_list.resize(max_file);
    std::fstream f_cin;
    std::string file_name = "/home/group14/CS433FinalProject/task1/select_file_list.txt";
    f_cin.open(file_name.c_str(), std::ios::out | std::ios::in);
    for (int i = 0; i < max_file; ++i) {
        std::getline(f_cin, file_list[i]);
        int cur_size = file_list[i].size();
        file_list[i] = file_list[i].substr(1, cur_size - 3);
    }
    
    f_cin.close();
    std::unordered_set<std::string> error_file_list = get_error_file("/home/group14/CS433FinalProject/task1/error_file_list.txt");
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    std::chrono::duration<double> forward_time;
    for (auto & file_name : file_list) {
        cv::Mat image = cv::imread(file_name);
        cv::Mat resized_image;
        cv::resize(image, resized_image, cv::Size(224, 224));

        cv::Mat resized_image2;
        cv::cvtColor(resized_image, resized_image2, cv::COLOR_BGR2RGB);
        cv::Mat resized_image3;
        resized_image2.convertTo(resized_image3, CV_32F);
        cv::Mat resized_image4 = resized_image3 / 255.0;

        cv::Mat blob;
        cv::dnn::blobFromImage(resized_image4, blob, 1.0, cv::Size(224, 224));
        
        auto start_time = std::chrono::high_resolution_clock::now();
        net.setInput(blob);
        cv::Mat output = net.forward();
        auto end_time = std::chrono::high_resolution_clock::now();
        forward_time += end_time - start_time;
        
        double max_value;
        cv::Point max_index;
        cv::minMaxLoc(output, NULL, &max_value, NULL, &max_index);
        std::cout << file_name << " "  << max_value << " " << max_index.x << std::endl;
        if (error_file_list.count(file_name) != 0) {
            print_mat(output);
        }
    }

    std::ofstream benchmark_output("/home/group14/CS433FinalProject/task1/target/benchmark/benchmark_oracle.txt");
    benchmark_output << forward_time.count() << "s" << std::endl;
    benchmark_output.close();  

    return 0;
}
