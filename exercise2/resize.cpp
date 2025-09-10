#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int main() {
    // Load the image
    cv::Mat img = cv::imread("images/pexels-christian-heitz.jpg");
    if (img.empty()) {
        std::cout << "Error: image not found!\n";
        return -1;
    }
    
    int resolutionsNum[] = {4, 8, 16};
    // Define the target resolutions
    std::vector<cv::Size> resolutions = {
        cv::Size(3840, 2160), // 4K
        cv::Size(7680, 4320), // 8K
        cv::Size(15360, 8640) // 16K
    };

    // Resize and save images
    for (size_t i = 0; i < resolutions.size(); i++) {
        cv::Mat resized;
        // Resize the image
        cv::resize(img, resized, resolutions[i]);

        // Save the resized image
        std::string outname = "images/resized_pexels-christian-heitz_" + std::to_string(resolutionsNum[i]) + "K.png";
        cv::imwrite(outname, resized);
        std::cout << "Salvata: " + outname << "\n";
    }


}

