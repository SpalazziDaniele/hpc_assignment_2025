#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <fstream>

// Kernel CUDA to apply the filter
__global__ void filterKernel(unsigned char* input, unsigned char* output, int width, int height, const float* filter, float norm) {
    // Standard way to calculate pixel position
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check to avoid overflow
    if (x >= width || y >= height) return;


    // Apply filter
    float sum = 0.0f;
    int filterSize = 3;
    int half = filterSize / 2;

    for (int fy = -half; fy <= half; fy++) {
        for (int fx = -half; fx <= half; fx++) {
            int ix = min(max(x + fx, 0), width - 1);
            int iy = min(max(y + fy, 0), height - 1);
            float coeff = filter[(fy + half) * filterSize + (fx + half)];
            sum += input[iy * width + ix] * coeff;
        }
    }

    sum /= norm; // normalization
    output[y * width + x] = min(max(int(sum), 0), 255);
}

// Function that call the CUDA kernel and set up memory
void applyFilterCUDA(std::vector<cv::Mat>& channels, std::vector<cv::Mat>& outChannels, const float* h_filter, float norm, int blockSize, size_t* freeMem, size_t* totalMem) {
    // Define the dimensions of the image
    int width = channels[0].cols;
    int height = channels[0].rows;
    // Define the size of the image in bytes for the memory allocation
    int size = width * height * sizeof(unsigned char);

    // Define the device pointers for input and output images and filter
    unsigned char* d_input, * d_output;
    float* d_filter;

    // Allocate device memory
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMalloc(&d_filter, 9 * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_filter, h_filter, 9 * sizeof(float), cudaMemcpyHostToDevice);

    // Get memory info
    cudaMemGetInfo(freeMem, totalMem);

    // Define block and grid sizes in a 2D configuration being sure that for each dimension the grid size is rounded up
    dim3 block(blockSize, blockSize);
    dim3 grid((width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y);

    // Process each channel
    for(int c = 0; c < channels.size(); c++){
        outChannels[c] = cv::Mat(channels[c].size(), CV_8UC1);
        // Upload this channel
        cudaMemcpy(d_input, channels[c].data, size, cudaMemcpyHostToDevice);

        // Launch the kernel    
        filterKernel<<<grid, block>>> (d_input, d_output, width, height, d_filter, norm);

        // Download result
        cudaMemcpy(outChannels[c].data, d_output, size, cudaMemcpyDeviceToHost);
    }   

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_filter);
}

// Function to extract the file name from a path without extension
std::string getFileName(const std::string& path) {
    size_t pos = path.find_last_of("/\\");
    std::string filename = (pos == std::string::npos) ? path : path.substr(pos + 1);

    // Remove extension
    size_t dotPos = filename.find_last_of('.');
    if (dotPos != std::string::npos) {
        filename = filename.substr(0, dotPos);
    }

    return filename;
}

int main(int argc, char* argv[]) {
    // Check arguments
    if(argc < 4) {
        std::cout << "Usage: ./decomposeAndFilterNew <psnrMode> <inputImageNum> <image1> ... <imageN> [referenceImage]\n";
        return -1;
    }

    // Get PSNR mode and input image number
    int psnrMode = atoi(argv[1]);
    int inputImageNum = atoi(argv[2]);

    // Validate input image number
    if(inputImageNum < 1 || inputImageNum > (argc - 3)) {
        std::cout << "Invalid number of images!\n";
        return -1;
    }

    // Load input images
    std::vector<cv::Mat> inputImages;
    std::vector<std::string> inputImageNames;
    for(int i = 0; i < inputImageNum; i++){
        cv::Mat img = cv::imread(argv[i + 3]);
        if (img.empty()) {
            std::cout << "Error: image " << argv[i + 3] << " not found!\n";
            return -1;
        }
        inputImages.push_back(img);
        inputImageNames.push_back(argv[i + 3]);
    }

    // Load reference image if PSNR mode is 1
    cv::Mat referenceImg;
    if(psnrMode == 1) {
        if(argc < 4 + inputImageNum) {
            std::cout << "Error: PSNR mode 1 requires a reference image!\n";
            return -1;
        }
        referenceImg = cv::imread(argv[3 + inputImageNum]);
        if(referenceImg.empty()){
            std::cout << "Error: reference image not found!\n";
            return -1;
        }
    }

    // CUDA events for timing
    cudaEvent_t start, stop;
    // Variables for memory info
    size_t freeMem, totalMem;

    // Define CSV file for results
    std::ofstream csvFile("results/execution_result.csv");
    csvFile << "Image,BlockSize,Time_ms,GPU_Used_Mem[MB],PSNR,PSNR_Filtered\n";

    // Definition of the filter and its normalization factor
    float h_filter[9] = {
        1, 2, 1,
        3, 4, 3,
        1, 2, 1
    };
    float norm = 16.0f;

    // Cycle over input images
    for (size_t i = 0; i < inputImages.size(); i++) {

        // Split RGB
        std::vector<cv::Mat> channels;
        cv::split(inputImages[i], channels);

        std::vector<int> blockSizes = { 8, 16, 24, 32};

        // Cycle over block sizes
        for(int bs : blockSizes){
            float elapsedTime = 0.0f;
            // Create time events to measure execution time
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            std::vector<cv::Mat> outChannels(3);

            // Start timing
            cudaEventRecord(start);

            // Apply filter the channels
            applyFilterCUDA(channels, outChannels, h_filter, norm, bs, &freeMem, &totalMem);

            // Stop timing
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            // Calculate elapsed time
            cudaEventElapsedTime(&elapsedTime, start, stop);    

            // Merge channels back
            cv::Mat filtered;
            cv::merge(outChannels, filtered);

            // Calculate PSNR comparing with input or reference
            cv::Mat ref = (psnrMode == 0) ? inputImages[i] : referenceImg;
            double psnrNotFiltered = cv::PSNR(ref, inputImages[i]);
            double psnrFiltered = cv::PSNR(ref, filtered); 

            // Save filtered image
            std::string baseName = getFileName(inputImageNames[i]);
            std::string outName = "images/filtered_" + baseName + "_bs" + std::to_string(bs) + ".png";
            cv::imwrite(outName, filtered);
            std::cout << "Saved: " << outName << "\n";

            size_t usedMem = totalMem - freeMem;

            // Write results to CSV
            csvFile << baseName << "," 
            << bs << "," 
            << elapsedTime << "," 
            << static_cast<double>(usedMem) / (1024.0 * 1024.0) << ","  // used memory in MB
            << psnrNotFiltered << "," 
            << psnrFiltered << "\n";

            // Destroy events
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

    }

    // Close CSV file
    csvFile.close();
    return 0;
}
