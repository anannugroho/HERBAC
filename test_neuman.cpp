#include <iostream>
#include <opencv2/opencv.hpp>

/**
 * @brief Menerapkan kondisi batas Neumann pada matriks.
 * @param phi Matriks input/output dengan tipe CV_64F (double). Matriks ini akan dimodifikasi secara langsung.
 */
void neumann(cv::Mat& phi) {
    // Pastikan matriks tidak terlalu kecil untuk operasi ini
    if (phi.rows < 5 || phi.cols < 5) {
        std::cerr << "Error: Matriks terlalu kecil untuk operasi Neumann." << std::endl;
        return;
    }

    int rows = phi.rows;
    int cols = phi.cols;

    // Menangani 4 sudut (corners)
    // Python: (g[0,0],g[0,ncol],g[nrow,0],g[nrow,ncol]) = (g[2,2],g[2,ncol-3],g[nrow-3,2],g[nrow-3,ncol-3])
    phi.at<double>(0, 0) = phi.at<double>(2, 2);
    phi.at<double>(0, cols - 1) = phi.at<double>(2, cols - 4);
    phi.at<double>(rows - 1, 0) = phi.at<double>(rows - 4, 2);
    phi.at<double>(rows - 1, rows - 1) = phi.at<double>(rows - 4, cols - 4);

    // Menangani sisi atas dan bawah (top and bottom edges)
    // Python: (g[0,1:-1],g[nrow,1:-1]) = (g[2,1:-1],g[nrow-3,1:-1])
    phi(cv::Rect(1, 2, cols - 2, 1)).copyTo(phi(cv::Rect(1, 0, cols - 2, 1)));
    phi(cv::Rect(1, rows - 4, cols - 2, 1)).copyTo(phi(cv::Rect(1, rows - 1, cols - 2, 1)));
    
    // Menangani sisi kiri dan kanan (left and right edges)
    // Python: (g[1:-1,1],g[1:-1,ncol]) = (g[1:-1,2],g[1:-1,ncol-3])
    // CATATAN: Ini adalah terjemahan literal dari kode Python Anda.
    // Lihat penjelasan di bawah mengenai potensi kesalahan logika.
    phi(cv::Rect(2, 1, 1, rows - 2)).copyTo(phi(cv::Rect(1, 1, 1, rows - 2)));
    phi(cv::Rect(cols - 4, 1, 1, rows - 2)).copyTo(phi(cv::Rect(cols - 1, 1, 1, rows - 2)));
}

// Fungsi main untuk demonstrasi
int main() {
    // Membuat matriks contoh 10x10 dengan angka berurutan
    cv::Mat matrix = cv::Mat(10, 10, CV_64F);
    for (int i = 0; i < matrix.rows; ++i) {
        for (int j = 0; j < matrix.cols; ++j) {
            matrix.at<double>(i, j) = 10 * i + j;
        }
    }

    std::cout << "## Matriks Sebelum Neumann:\n" << matrix << std::endl;

    neumann(matrix);

    std::cout << "\n## Matriks Sesudah Neumann:\n" << matrix << std::endl;

    return 0;
}