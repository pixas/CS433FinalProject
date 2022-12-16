
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>

using namespace cv;

int main() {
    std::string read_file_name = "/home/group14/CS433FinalProject/resnet18.onnx";
    cv::dnn::Net net = cv::dnn::readNetFromONNX(read_file_name);
    Mat image = imread("/home/cs433/files/benchmark/ImageNet/val/n03379051/ILSVRC2012_val_00014944.JPEG");
    Mat resized_image;        
    resize(image, resized_image, Size(224, 224));
    Mat blob;
    cv::dnn::blobFromImage(resized_image, blob, 1.0, cv::Size(224, 224));
    std::cout << blob.size << std::endl;
    for (int i = 1; i <= 1; ++i) {
        auto layer = net.getLayer(i);
        auto blobs = layer->blobs;
        int inpGroupCn = blobs.empty() ? blob.size[1] : blobs[0].size[1];
        std::cout << blobs[0].size[1] << " " << inpGroupCn << std::endl;
    //     Mat output;       
    //     Mat internals;
    //     // layer->forward()
    //     layer->forward(blob, output, internals);
    //     std::cout << output.size << std::endl;
    //     std::cout << layer->type << std::endl;

    }
    return 0;
}