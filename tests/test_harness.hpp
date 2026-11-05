// A very small test framework, so the tests need nothing but a compiler.
#pragma once

#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace test {

struct Case {
    const char* name;
    void (*run)();
};

inline std::vector<Case>& cases() {
    static std::vector<Case> all;
    return all;
}

inline int& failed_checks() {
    static int count = 0;
    return count;
}

struct Registrar {
    Registrar(const char* name, void (*run)()) { cases().push_back({name, run}); }
};

inline void fail(const char* file, int line, const std::string& message) {
    ++failed_checks();
    std::cerr << file << ":" << line << ": " << message << "\n";
}

inline int run_all() {
    int failed_cases = 0;
    for (const Case& test_case : cases()) {
        const int before = failed_checks();
        try {
            test_case.run();
        } catch (const std::exception& error) {
            fail(test_case.name, 0, std::string("unexpected exception: ") + error.what());
        }
        if (failed_checks() != before) {
            ++failed_cases;
            std::cerr << "FAILED " << test_case.name << "\n";
        }
    }
    std::cout << cases().size() - static_cast<std::size_t>(failed_cases) << " of " << cases().size()
              << " tests passed\n";
    return failed_cases == 0 ? 0 : 1;
}

}  // namespace test

#define TEST(name)                                              \
    static void name();                                         \
    static const test::Registrar registrar_##name(#name, name); \
    static void name()

#define CHECK(condition)                                                         \
    do {                                                                         \
        if (!(condition)) test::fail(__FILE__, __LINE__, "failed: " #condition); \
    } while (false)

#define CHECK_EQ(actual, expected)                                                     \
    do {                                                                               \
        const auto actual_value = (actual);                                            \
        const auto expected_value = (expected);                                        \
        if (!(actual_value == expected_value)) {                                       \
            test::fail(__FILE__, __LINE__,                                             \
                       #actual " is " + std::to_string(actual_value) + ", expected " + \
                           std::to_string(expected_value));                            \
        }                                                                              \
    } while (false)

#define CHECK_NEAR(actual, expected, tolerance)                                        \
    do {                                                                               \
        const double actual_value = static_cast<double>(actual);                       \
        const double expected_value = static_cast<double>(expected);                   \
        if (!(std::fabs(actual_value - expected_value) <= (tolerance))) {              \
            test::fail(__FILE__, __LINE__,                                             \
                       #actual " is " + std::to_string(actual_value) + ", expected " + \
                           std::to_string(expected_value));                            \
        }                                                                              \
    } while (false)

#define CHECK_THROWS(expression, exception_type)                                                   \
    do {                                                                                           \
        bool threw = false;                                                                        \
        try {                                                                                      \
            (void)(expression);                                                                    \
        } catch (const exception_type&) {                                                          \
            threw = true;                                                                          \
        }                                                                                          \
        if (!threw) test::fail(__FILE__, __LINE__, #expression " did not throw " #exception_type); \
    } while (false)

#define TEST_MAIN()             \
    int main() {                \
        return test::run_all(); \
    }
