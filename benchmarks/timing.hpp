// Timing helpers shared by the efficiency benchmark and the efficiency checks.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <vector>

namespace bench {

// Writing a result here stops the compiler removing the work being timed.
inline volatile double sink = 0.0;

struct Timing {
    double min_ms;
    double median_ms;
};

template <typename Work>
double run_ms(Work& work, std::size_t count) {
    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    for (std::size_t i = 0; i < count; ++i) work();
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// Repeats `work` until `budget_ms` has been spent, with at least `min_runs`
// samples. The minimum is the run least disturbed by the rest of the machine,
// and the median shows what a typical run looks like.
template <typename Work>
Timing measure(Work&& work, double budget_ms = 200.0, std::size_t min_runs = 5) {
    // A clock can't time very short work, so each sample is a batch of calls
    // that takes long enough to measure.
    std::size_t batch = 1;
    while (batch < (std::size_t{1} << 20) && run_ms(work, batch) < 0.02) batch *= 2;

    std::vector<double> samples;
    double spent = 0.0;
    while (samples.size() < min_runs || (spent < budget_ms && samples.size() < 5000)) {
        const double elapsed = run_ms(work, batch);
        samples.push_back(elapsed / static_cast<double>(batch));
        spent += elapsed;
    }
    std::sort(samples.begin(), samples.end());
    return {samples.front(), samples[samples.size() / 2]};
}

}  // namespace bench
