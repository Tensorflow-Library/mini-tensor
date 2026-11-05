// Treats a 2000 x 3000 RGB image as a 3-D tensor. It blurs the image with sliced
// views and broadcasting, then converts it to grayscale with a matrix product.
#include <chrono>
#include <iostream>

#include "minitensor/minitensor.hpp"

using Tensor = mt::Tensor<float>;

namespace {

// Averages each pixel with its four neighbours. Each neighbour is the same
// image, shifted by one position, and none of these views copies any pixels.
Tensor blur(const Tensor& image) {
    const std::ptrdiff_t height = static_cast<std::ptrdiff_t>(image.dim(0));
    const std::ptrdiff_t width = static_cast<std::ptrdiff_t>(image.dim(1));
    const Tensor middle = image.slice(0, 1, height - 1).slice(1, 1, width - 1);
    const Tensor up = image.slice(0, 0, height - 2).slice(1, 1, width - 1);
    const Tensor down = image.slice(0, 2, height).slice(1, 1, width - 1);
    const Tensor left = image.slice(0, 1, height - 1).slice(1, 0, width - 2);
    const Tensor right = image.slice(0, 1, height - 1).slice(1, 2, width);

    Tensor out = image.clone();
    out.slice(0, 1, height - 1)
        .slice(1, 1, width - 1)
        .assign((middle + up + down + left + right) / 5.0f);
    return out;
}

}  // namespace

int main() {
    const std::size_t height = 2000;
    const std::size_t width = 3000;
    Tensor image = Tensor::rand({height, width, 3}, 1);

    // A bright square, to give the blur something to spread out.
    image.slice(0, 900, 1100).slice(1, 1400, 1600).fill(1.0f);

    const auto start = std::chrono::steady_clock::now();
    Tensor blurred = image;
    for (int pass = 0; pass < 10; ++pass) blurred = blur(blurred);

    // Each pixel's gray value is a weighted sum of its colours.
    const Tensor weights({3}, {0.299f, 0.587f, 0.114f});
    const Tensor gray = mt::matmul(blurred.reshape({-1, 3}), weights).reshape({height, width});
    const auto stop = std::chrono::steady_clock::now();

    std::cout << "input shape:  " << image.shape().to_string() << "\n"
              << "output shape: " << gray.shape().to_string() << "\n"
              << "brightness before: " << image.mean() << ", after: " << gray.mean() << "\n"
              << "time: " << std::chrono::duration<double, std::milli>(stop - start).count()
              << " ms for 10 blur passes and the conversion to gray\n";
    return 0;
}
