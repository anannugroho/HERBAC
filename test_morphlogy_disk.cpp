#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    // Ekuivalen dari skimage.morphology.disk(1)
    cv::Mat seMorf = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));

    // Cetak hasilnya untuk melihat bentuk matriksnya
    std::cout << "Structuring Element (seMorf):" << std::endl;
    std::cout << seMorf << std::endl;

    return 0;
}