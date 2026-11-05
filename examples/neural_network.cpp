// Trains a small neural network to tell apart two interleaved spirals of
// points, which a straight line can't separate. The forward pass and the
// gradients are written out by hand, using only tensor operations.
#include <cmath>
#include <iostream>

#include "minitensor/minitensor.hpp"

using Tensor = mt::Tensor<double>;

namespace {

Tensor sigmoid(const Tensor& x) {
    return mt::map(x, [](double value) { return 1.0 / (1.0 + std::exp(-value)); });
}

Tensor relu(const Tensor& x) {
    return mt::map(x, [](double value) { return value > 0.0 ? value : 0.0; });
}

Tensor is_positive(const Tensor& x) {
    return mt::map(x, [](double value) { return value > 0.0 ? 1.0 : 0.0; });
}

// Points on two spirals, half a turn apart.
void make_spirals(std::size_t points_per_class, Tensor& points, Tensor& labels) {
    const std::size_t total = 2 * points_per_class;
    points = Tensor({total, 2});
    labels = Tensor({total, 1});
    const Tensor noise = Tensor::randn({total, 2}, 7) * 0.02;
    for (std::size_t i = 0; i < points_per_class; ++i) {
        const double progress = static_cast<double>(i) / static_cast<double>(points_per_class);
        const double radius = 0.1 + progress;
        const double angle = 3.0 * M_PI * progress;
        points(i, 0) = radius * std::cos(angle) + noise(i, 0);
        points(i, 1) = radius * std::sin(angle) + noise(i, 1);
        const std::size_t j = i + points_per_class;
        points(j, 0) = -radius * std::cos(angle) + noise(j, 0);
        points(j, 1) = -radius * std::sin(angle) + noise(j, 1);
        labels(j, 0) = 1.0;
    }
}

}  // namespace

int main() {
    Tensor x, y;
    make_spirals(1000, x, y);
    const std::size_t samples = x.dim(0);

    // Layer sizes: 2 inputs, 64 and 64 hidden units, 1 output.
    const std::size_t hidden = 64;
    Tensor w1 = Tensor::randn({2, hidden}, 1) * 0.5;
    Tensor b1({1, hidden});
    Tensor w2 = Tensor::randn({hidden, hidden}, 2) * (1.0 / std::sqrt(hidden));
    Tensor b2({1, hidden});
    Tensor w3 = Tensor::randn({hidden, 1}, 3) * (1.0 / std::sqrt(hidden));
    Tensor b3({1, 1});

    const double learning_rate = 0.5;
    const auto scaled_mean = [&](const Tensor& t) { return t * (learning_rate / samples); };

    double accuracy = 0.0;
    for (int epoch = 0; epoch <= 4000; ++epoch) {
        // Forward pass.
        const Tensor z1 = mt::matmul(x, w1) + b1;
        const Tensor h1 = relu(z1);
        const Tensor z2 = mt::matmul(h1, w2) + b2;
        const Tensor h2 = relu(z2);
        const Tensor logits = mt::matmul(h2, w3) + b3;
        const Tensor probability = sigmoid(logits);

        const Tensor error = probability - y;
        const Tensor correct = mt::map(mt::map(error, [](double e) { return std::fabs(e); }),
                                       [](double e) { return e < 0.5 ? 1.0 : 0.0; });
        accuracy = correct.mean();

        // Backward pass, from the output to the input.
        const Tensor mask2 = is_positive(z2);
        const Tensor mask1 = is_positive(z1);
        const Tensor grad_h2 = mt::matmul(error, w3.transpose()) * mask2;
        const Tensor grad_h1 = mt::matmul(grad_h2, w2.transpose()) * mask1;

        w3 -= scaled_mean(mt::matmul(h2.transpose(), error));
        b3 -= scaled_mean(error.sum(0).unsqueeze(0));
        w2 -= scaled_mean(mt::matmul(h1.transpose(), grad_h2));
        b2 -= scaled_mean(grad_h2.sum(0).unsqueeze(0));
        w1 -= scaled_mean(mt::matmul(x.transpose(), grad_h1));
        b1 -= scaled_mean(grad_h1.sum(0).unsqueeze(0));

        if (epoch % 800 == 0) {
            std::cout << "epoch " << epoch << ": accuracy = " << accuracy * 100.0 << "%\n";
        }
    }

    std::cout << "\nfinal accuracy = " << accuracy * 100.0 << "%\n";
    return 0;
}
