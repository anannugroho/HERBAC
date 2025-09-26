#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>   // Diperlukan untuk std::exp
#include <limits>  // Diperlukan untuk std::numeric_limits

/**
 * @brief Membuat kernel Gaussian 2D yang identik dengan fspecial('gaussian') di MATLAB.
 * @param shape Ukuran kernel (misal: cv::Size(3, 3)).
 * @param sigma Nilai standar deviasi (sigma) dari fungsi Gaussian.
 * @return cv::Mat dengan tipe CV_64F (double) yang berisi kernel Gaussian yang sudah dinormalisasi.
 */
cv::Mat matlab_style_gauss2D(cv::Size shape = cv::Size(3, 3), double sigma = 0.5) {
    // Menghitung koordinat titik tengah
    // Python: m,n = [(ss-1.)/2. for ss in shape]
    double centerX = (shape.width - 1.0) / 2.0;
    double centerY = (shape.height - 1.0) / 2.0;

    // Membuat matriks kernel dengan tipe data double untuk presisi tinggi
    cv::Mat kernel(shape, CV_64F);

    // Loop ini menggantikan np.ogrid untuk menghitung nilai di setiap koordinat
    for (int y = 0; y < shape.height; ++y) {
        for (int x = 0; x < shape.width; ++x) {
            double dist_x = static_cast<double>(x) - centerX;
            double dist_y = static_cast<double>(y) - centerY;
            
            // Menghitung nilai Gaussian
            // Python: h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
            kernel.at<double>(y, x) = std::exp(-(dist_x * dist_x + dist_y * dist_y) / (2.0 * sigma * sigma));
        }
    }

    // Mengatur nilai yang sangat kecil (mendekati nol) menjadi nol
    // Python: h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    double maxVal;
    cv::minMaxLoc(kernel, nullptr, &maxVal);
    double threshold = std::numeric_limits<double>::epsilon() * maxVal;
    cv::threshold(kernel, kernel, threshold, 0, cv::THRESH_TOZERO); // Nilai < threshold menjadi 0

    // Normalisasi kernel agar total nilainya adalah 1
    // Python: sumh = h.sum(); if sumh != 0: h /= sumh
    double sum = cv::sum(kernel)[0]; // Ambil elemen pertama dari cv::Scalar
    if (sum != 0) {
        kernel /= sum;
    }

    return kernel;
}


// Fungsi main untuk demonstrasi
int main() {
    cv::Mat kernel3x3 = matlab_style_gauss2D(cv::Size(3, 3), 1);
    std::cout << "Kernel Gaussian 3x3, sigma=0.5:\n" << kernel3x3 << std::endl;

    std::cout << "\n-----------------------------------\n" << std::endl;

    cv::Mat kernel5x5 = matlab_style_gauss2D(cv::Size(5, 5), 1.0);
    std::cout << "Kernel Gaussian 5x5, sigma=1.0:\n" << kernel5x5 << std::endl;

    return 0;
}