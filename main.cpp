#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
/**
 * @brief Struct untuk menampung hasil dari pengecekan konvergensi.
 */
struct ConvergenceResult
{
  bool hasConverged;
  double currentArea;
  double currentLength;
  double errorArea;
  double errorLength;
};

struct DetectionResult
{
  cv::Mat phi;
  cv::Mat g;
  int init;
};

/**
 * @brief Meniru skimage.segmentation.clear_border
 * Menghapus objek yang menyentuh batas gambar.
 */
cv::Mat clearBorder(const cv::Mat &binaryImage)
{
  cv::Mat labeledImage, stats, centroids;
  int nLabels = cv::connectedComponentsWithStats(binaryImage, labeledImage, stats, centroids);

  cv::Mat result = binaryImage.clone();
  int lastRow = binaryImage.rows - 1;
  int lastCol = binaryImage.cols - 1;

  for (int i = 1; i < nLabels; ++i)
  { // Mulai dari 1 untuk lewati background
    int x = stats.at<int>(i, cv::CC_STAT_LEFT);
    int y = stats.at<int>(i, cv::CC_STAT_TOP);
    int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
    int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);

    if (x == 0 || y == 0 || (x + w) == binaryImage.cols || (y + h) == binaryImage.rows)
    {
      // Jika objek menyentuh batas, hapus
      cv::Mat mask = (labeledImage == i);
      result.setTo(0, mask);
    }
  }
  return result;
}

/**
 * @brief Meniru scipy.ndimage.binary_fill_holes
 * Mengisi lubang di dalam objek biner.
 */
cv::Mat fillHoles(const cv::Mat &binaryImage)
{
  // Algoritma: temukan kontur eksternal, buat mask, lalu gambar semua kontur
  cv::Mat filled = cv::Mat::zeros(binaryImage.size(), CV_8U);
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;

  cv::findContours(binaryImage, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

  if (contours.empty())
  {
    return filled;
  }

  // Gambar semua level kontur untuk mengisi lubang
  for (int i = 0; i >= 0; i = hierarchy[i][0])
  {
    cv::drawContours(filled, contours, i, cv::Scalar(255), cv::FILLED, 8, hierarchy);
  }
  return filled;
}
/**
 * @brief Fungsi utama untuk deteksi objek.
 * @param phi Matriks level set input.
 * @param imgResult Matriks gambar tempat menggambar hasil (diterima by reference).
 * @param imageShape Ukuran gambar asli untuk membuat mask.
 * @return Struct DetectionResult.
 */
DetectionResult obDetection(const cv::Mat &phi, cv::Mat &imgResult, cv::Size imageShape)
{
  // Python: g = np.where(phi<=0, 1, 0)
  cv::Mat g_float;
  cv::threshold(phi, g_float, 0, 1, cv::THRESH_BINARY_INV);

  // Python: g1 = img_as_ubyte(g)
  cv::Mat g1;
  g_float.convertTo(g1, CV_8U, 255.0);

  // Python: se = estructurant(3) -> Asumsi kernel 3x3
  cv::Mat se = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

  // Python: opening = cv2.morphologyEx(g1, cv2.MORPH_OPEN, se)
  cv::Mat opening;
  cv::morphologyEx(g1, opening, cv::MORPH_OPEN, se);

  // Python: clearObj = segmentation.clear_border(opening)
  cv::Mat clearObj = clearBorder(opening);

  // Python: fillObj = ndimage.binary_fill_holes(clearObj)
  cv::Mat fillObj = fillHoles(clearObj);

  // Python: labelObj = measure.label(fillObj), propObj = measure.regionprops(labelObj)
  cv::Mat labels, stats, centroids;
  int nLabels = cv::connectedComponentsWithStats(fillObj, labels, stats, centroids);

  cv::Mat maskBW = cv::Mat::zeros(imageShape, CV_8U);
  std::vector<double> minorAxisLengths;
  std::vector<cv::RotatedRect> ellipses;

  // Loop pertama: Hitung minor axis length untuk semua objek
  for (int i = 1; i < nLabels; ++i)
  {
    cv::Mat mask = (labels == i);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty() || contours[0].size() < 5)
    {
      ellipses.push_back(cv::RotatedRect()); // elips kosong
      minorAxisLengths.push_back(0);
      continue;
    }
    cv::RotatedRect ellipse = cv::fitEllipse(contours[0]);
    ellipses.push_back(ellipse);
    minorAxisLengths.push_back(std::min(ellipse.size.width, ellipse.size.height));
  }

  // Cari panjang minor axis maksimum
  double max_minor_axis = 0;
  if (!minorAxisLengths.empty())
  {
    max_minor_axis = *std::max_element(minorAxisLengths.begin(), minorAxisLengths.end());
  }

  // Loop kedua: Filter objek dan gambar hasilnya
  if (max_minor_axis > 0)
  {
    for (int i = 1; i < nLabels; ++i)
    {
      if ((minorAxisLengths[i - 1] / max_minor_axis) >= 0.7)
      {
        int minc = stats.at<int>(i, cv::CC_STAT_LEFT);
        int minr = stats.at<int>(i, cv::CC_STAT_TOP);
        int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        int maxc = minc + w;
        int maxr = minr + h;
        int area = stats.at<int>(i, cv::CC_STAT_AREA);

        cv::rectangle(maskBW, cv::Point(minc, minr), cv::Point(maxc, maxr), cv::Scalar(255), -1);
        cv::rectangle(imgResult, cv::Point(minc - 20, minr - 20), cv::Point(maxc + 20, maxr + 20), cv::Scalar(0, 255, 0), 2);
        cv::putText(imgResult, "Area: " + std::to_string(area), cv::Point(minc, minr - 30), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
      }
    }
  }

  // Python: phi = -2*maskBW+1
  // PENTING: maskBW (0-255) harus dinormalisasi ke 0-1 dulu
  cv::Mat mask_float;
  maskBW.convertTo(mask_float, CV_64F, 1.0 / 255.0);
  cv::Mat new_phi = -2.0 * mask_float + 1.0;

  // Kembalikan hasilnya
  DetectionResult result;
  result.phi = new_phi;
  result.g = fillObj; // Python: g = fillObj
  result.init = 0;
  return result;
}
cv::Mat dirac(const cv::Mat &phi)
{
  CV_Assert(phi.depth() == CV_64F || phi.depth() == CV_32F);

  // a = 1 + phi**2
  cv::Mat phi_sq;
  cv::multiply(phi, phi, phi_sq); // phi^2
  cv::Mat a = 1.0 + phi_sq;

  // Dir = (1/pi) / a
  cv::Mat Dir;
  cv::divide((1.0 / CV_PI), a, Dir);

  return Dir;
}

ConvergenceResult convergence(const cv::Mat &phi, int iteration, const cv::Mat &absR,
                              double teta = 0.1, int maxs = 50,
                              double preArea = 0.0, double preLength = 0.0)
{

  cv::Mat phip;
  cv::threshold(phi, phip, 0, 1, cv::THRESH_BINARY_INV);

  double Area = cv::sum(phip)[0];

  double ErrorArea = std::abs(Area - preArea);

  cv::Mat dPhi = dirac(phi);

  cv::Mat product;
  cv::multiply(absR, dPhi, product);
  double Length = cv::sum(product)[0];

  double ErrorLength = std::abs(Length - preLength);

  bool hasConverged = ((ErrorArea <= teta) && (ErrorLength <= teta)) || (iteration >= maxs);

  ConvergenceResult result;
  result.hasConverged = hasConverged;
  result.currentArea = Area;
  result.currentLength = Length;
  result.errorArea = ErrorArea;
  result.errorLength = ErrorLength;

  return result;
}

cv::Mat heaviside(const cv::Mat &phi)
{
  CV_Assert(phi.depth() == CV_32F || phi.depth() == CV_64F);

  cv::Mat H = cv::Mat(phi.size(), phi.type());
  if (phi.depth() == CV_64F)
  {
    phi.forEach<double>([&](double &p, const int *position) -> void
                        { H.at<double>(position[0], position[1]) = (1.0 / CV_PI) * std::atan(p) + 0.5; });
  }
  else
  {
    phi.forEach<float>([&](float &p, const int *position) -> void
                       { H.at<float>(position[0], position[1]) = (1.0f / CV_PI) * std::atan(p) + 0.5f; });
  }

  return H;
}

std::pair<double, double> fittingAverage(const cv::Mat &img, const cv::Mat &phi)
{
  CV_Assert(img.depth() == CV_32F || img.depth() == CV_64F);
  CV_Assert(phi.depth() == CV_32F || phi.depth() == CV_64F);

  cv::Mat Hphi = heaviside(phi);

  cv::Mat cHphi = 1.0 - Hphi;
  cv::Mat img_double;
  img.convertTo(img_double, Hphi.type());
  double ca = cv::sum(img_double.mul(Hphi))[0];
  double cb = cv::sum(img_double.mul(cHphi))[0];
  double sumH = cv::sum(Hphi)[0];
  double sumCH = cv::sum(cHphi)[0];
  double c1 = 0.0, c2 = 0.0;
  if (sumH > 0)
  {
    c1 = ca / sumH;
  }
  if (sumCH > 0)
  {
    c2 = cb / sumCH;
  }
  return std::make_pair(c1, c2);
}

std::pair<cv::Mat, cv::Mat> curvature(const cv::Mat &phi)
{
  CV_Assert(phi.depth() == CV_32F || phi.depth() == CV_64F);
  int ddepth = phi.depth();
  cv::Mat nx, ny;
  cv::Sobel(phi, nx, ddepth, 1, 0, 3);
  cv::Sobel(phi, ny, ddepth, 0, 1, 3);
  cv::Mat absR;
  cv::magnitude(nx, ny, absR);
  cv::Mat zero_mask = (absR == 0);
  double epsilon = (ddepth == CV_64F) ? std::numeric_limits<double>::epsilon() : std::numeric_limits<double>::epsilon();
  absR.setTo(epsilon, zero_mask);
  cv::Mat norm_nx, norm_ny;
  cv::divide(nx, absR, norm_nx);
  cv::divide(ny, absR, norm_ny);
  cv::Mat nxx1;
  cv::Sobel(norm_nx, nxx1, ddepth, 1, 0, 3);
  cv::Mat nyy1;
  cv::Sobel(norm_ny, nyy1, ddepth, 0, 1, 3);
  cv::Mat Kappa = nxx1 + nyy1;

  return std::make_pair(Kappa, absR);
}

void neumann(cv::Mat &phi)
{
  if (phi.rows < 5 || phi.cols < 5)
  {
    std::cerr << "Error: Matriks terlalu kecil untuk operasi Neumann." << std::endl;
    return;
  }

  int rows = phi.rows;
  int cols = phi.cols;
  phi.at<double>(0, 0) = phi.at<double>(2, 2);
  phi.at<double>(0, cols - 1) = phi.at<double>(2, cols - 4);
  phi.at<double>(rows - 1, 0) = phi.at<double>(rows - 4, 2);
  phi.at<double>(rows - 1, rows - 1) = phi.at<double>(rows - 4, cols - 4);
  phi(cv::Rect(1, 2, cols - 2, 1)).copyTo(phi(cv::Rect(1, 0, cols - 2, 1)));
  phi(cv::Rect(1, rows - 4, cols - 2, 1)).copyTo(phi(cv::Rect(1, rows - 1, cols - 2, 1)));
  phi(cv::Rect(2, 1, 1, rows - 2)).copyTo(phi(cv::Rect(1, 1, 1, rows - 2)));
  phi(cv::Rect(cols - 4, 1, 1, rows - 2)).copyTo(phi(cv::Rect(cols - 1, 1, 1, rows - 2)));
}

cv::Mat matlab_style_gauss2D(cv::Size shape = cv::Size(3, 3), double sigma = 0.5)
{
  double centerX = (shape.width - 1.0) / 2.0;
  double centerY = (shape.height - 1.0) / 2.0;
  cv::Mat kernel(shape, CV_64F);
  for (int y = 0; y < shape.height; ++y)
  {
    for (int x = 0; x < shape.width; ++x)
    {
      double dist_x = static_cast<double>(x) - centerX;
      double dist_y = static_cast<double>(y) - centerY;
      kernel.at<double>(y, x) = std::exp(-(dist_x * dist_x + dist_y * dist_y) / (2.0 * sigma * sigma));
    }
  }

  double maxVal;
  cv::minMaxLoc(kernel, nullptr, &maxVal);
  double threshold = std::numeric_limits<double>::epsilon() * maxVal;
  cv::threshold(kernel, kernel, threshold, 0, cv::THRESH_TOZERO);

  double sum = cv::sum(kernel)[0];
  if (sum != 0)
  {
    kernel /= sum;
  }

  return kernel;
}

cv::Mat initLS(const cv::Mat &image)
{
  int height = image.rows;
  int width = image.cols;

  double centerX = std::floor(static_cast<double>(width) / 2.0);
  double centerY = std::floor(static_cast<double>(height) / 2.0);
  double radius = std::floor(std::min(0.2 * width, 0.2 * height));
  cv::Mat phi0 = cv::Mat(height, width, CV_64F);

  for (int y = 0; y < height; ++y)
  {
    for (int x = 0; x < width; ++x)
    {
      double dist = std::sqrt(std::pow(x - centerX, 2) + std::pow(y - centerY, 2));
      double val = dist - radius;
      double sign = (val > 0.0) - (val < 0.0);
      phi0.at<double>(y, x) = sign * 2.0;
    }
  }

  return phi0;
}

std::string type2str(int type)
{
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch (depth)
  {
  case CV_8U:
    r = "8U";
    break;
  case CV_8S:
    r = "8S";
    break;
  case CV_16U:
    r = "16U";
    break;
  case CV_16S:
    r = "16S";
    break;
  case CV_32S:
    r = "32S";
    break;
  case CV_32F:
    r = "32F";
    break;
  case CV_64F:
    r = "64F";
    break;
  default:
    r = "User";
    break;
  }

  r += "C";
  r += std::to_string(chans);

  return r;
}

int main(int argc, char *argv[])
{
  using namespace cv;

  int maxs = 150;
  int dt = 10;
  int teta = 1;
  cv::Mat seMorf = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
  cv::Mat kernel = matlab_style_gauss2D(cv::Size(3, 3), 1);

  if (argc != 2)
  {
    std::cout << "Error: Path gambar tidak diberikan." << std::endl;
    std::cout << "Cara pakai: " << argv[0] << " <path_ke_gambar>" << std::endl;
    return -1;
  }

  std::string image_path = argv[1];

  Mat img = imread(image_path, IMREAD_GRAYSCALE);
  if (img.empty())
  {
    std::cout << "Error: Tidak dapat membuka atau menemukan gambar!" << std::endl;
    return -1;
  }

  Mat img_calculate;
  img.convertTo(img_calculate, CV_64F, 1.0);
  Mat img_calculate_ori = img_calculate.clone();
  img_calculate = initLS(img_calculate);
  int height = img.rows;
  int width = img.cols;
  int channel = img.channels();
  cv::Mat gShrink;
  cv::Mat g = cv::Mat::zeros(height, width, CV_64F);
  cv::Mat error = cv::Mat::zeros(2, 1, CV_64F);
  cv::Mat Beta = cv::Mat::zeros(1, 1, CV_64F);
  double preLength = 0.0;
  double preArea = 0.0;
  double beta = 0.0;
  int i = 1;

  std::cout << "Dimensi Gambar:" << std::endl;
  std::cout << "Tinggi (Height): " << height << " piksel" << std::endl;
  std::cout << "Lebar (Width):   " << width << " piksel" << std::endl;
  std::cout << "Jumlah Kanal:    " << channel << std::endl;
  namedWindow("Jendela Gambar", WINDOW_AUTOSIZE);
  imshow("Gambar Input", img);
  waitKey(1);
  while (i > 0)
  {
    neumann(img_calculate);
    std::pair<cv::Mat, cv::Mat> result_curvature = curvature(img_calculate);
    cv::Mat div = result_curvature.first;
    cv::Mat absR = result_curvature.second;
    std::pair<double, double> averages = fittingAverage(img_calculate_ori, img_calculate);
    double c1 = averages.first;
    double c2 = averages.second;
    double avg_intensity = (c1 + c2) / 2.0;
    cv::Mat term1 = div.mul(absR);
    cv::Mat term2 = (1.0 - std::abs(beta)) * (img_calculate_ori - avg_intensity);
    cv::Mat term3 = beta * g.mul(absR);
    cv::Mat AACMR = term1 + term2 + term3;
    img_calculate = img_calculate + static_cast<double>(dt) * AACMR;
    cv::Mat signed_mat = cv::Mat::zeros(img_calculate.size(), img_calculate.type());
    img_calculate.forEach<double>([&](double &pixel, const int *position) -> void
                                  { signed_mat.at<double>(position[0], position[1]) = (pixel > 0) - (pixel < 0); });
    cv::Mat convolved_mat;
    cv::filter2D(signed_mat, convolved_mat, -1, kernel);
    cv::threshold(convolved_mat, img_calculate, 0, 1, cv::THRESH_BINARY);
    cv::morphologyEx(img_calculate, img_calculate, cv::MORPH_OPEN, seMorf);
    cv::morphologyEx(img_calculate, img_calculate, cv::MORPH_CLOSE, seMorf);
    cv::Mat pos_mask = (img_calculate > 0);
    cv::Mat nonpos_mask = (img_calculate <= 0);
    img_calculate.setTo(1.0, pos_mask);
    img_calculate.setTo(-1.0, nonpos_mask);
    std::cout << "Iterasi ke-" << i << std::endl;
    ConvergenceResult result = convergence(img_calculate, i, absR, teta, 500, preArea, preLength);
    preArea = result.currentArea;
    preLength = result.currentLength;
    if (result.hasConverged)
    {
      if(beta!=0){
        break;
      }
    }
    if (beta == 0)
    {
      DetectionResult detection_result = obDetection(img_calculate, img_calculate, img.size());
      cv::Mat phiShrink = detection_result.phi;
      cv::Mat gShrink = detection_result.g;
      gShrink.convertTo(gShrink, CV_64F, 1.0 / 255.0);
      g = 1.0 - gShrink;
      img_calculate = phiShrink;
      beta=1;
      std::cout << "Terdeteksi!"<< std::endl;
    }else{
      imshow("Proses Shrink", img_calculate);
      waitKey(1);
    }
    i +=1;
  }

  // Only for Test
  // neumann(img_calculate);
  // std::pair<cv::Mat, cv::Mat> result_curvature = curvature(img_calculate);
  // cv::Mat kappa = result_curvature.first;
  // cv::Mat absR = result_curvature.second;
  // std::pair<double, double> averages = fittingAverage(img_calculate_ori, img_calculate);
  // double c1 = averages.first;
  // double c2 = averages.second;
  // double avg_intensity = (c1 + c2) / 2.0;
  // cv::Mat term1 = kappa.mul(absR);
  // cv::Mat term2 = (1.0 - std::abs(beta)) * (img_calculate_ori - avg_intensity);
  // cv::Mat term3 = beta * g.mul(absR);
  // cv::Mat AACMR = term1 + term2 + term3;
  // img_calculate = img_calculate + static_cast<double>(dt) * AACMR;
  // cv::Mat signed_mat = cv::Mat::zeros(img_calculate.size(), img_calculate.type());
  // img_calculate.forEach<double>([&](double &pixel, const int *position) -> void
  //                               { signed_mat.at<double>(position[0], position[1]) = (pixel > 0) - (pixel < 0); });
  // cv::Mat convolved_mat;
  // cv::filter2D(signed_mat, convolved_mat, -1, kernel);
  // cv::threshold(convolved_mat, img_calculate, 0, 1, cv::THRESH_BINARY);
  // namedWindow("Nueman", WINDOW_AUTOSIZE);
  // imshow("Nueman", img_calculate);
  // std::cout << "\n## Matriks Neumann:\n"
  //           << img_calculate.at<double>(100, 100) << std::endl;
  // namedWindow("kappa", WINDOW_AUTOSIZE);
  // imshow("kappa", kappa);
  // std::cout << "\n## Matriks kappa:\n"
  //           << kappa.at<double>(100, 100) << std::endl;
  // namedWindow("absR", WINDOW_AUTOSIZE);
  // imshow("absR", absR);
  // std::cout << "\n## Matriks absR:\n"
  //           << absR.at<double>(100, 100) << std::endl;
  // std::cout << "Rata-rata intensitas dalam (c1): " << averages.first << std::endl;
  // std::cout << "Rata-rata intensitas luar (c2): " << averages.second << std::endl;
  imshow("Jendela Gambar", img_calculate);
  waitKey(0);
  destroyAllWindows();

  return 0;
}