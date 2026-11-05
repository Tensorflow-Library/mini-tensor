#!/usr/bin/env bash
# Runs the whole efficiency test and writes a report. It takes about a minute.
#
#   cmake -B build
#   cmake --build build -j
#   benchmarks/run_efficiency_test.sh [build-folder] [report-file]
set -euo pipefail

build_dir=${1:-build}
report=${2:-efficiency-report.md}
benchmark="$build_dir/benchmarks/bench_tensor"

if [ ! -x "$benchmark" ]; then
    echo "$benchmark not found. Build the project first:" >&2
    echo "  cmake -B $build_dir && cmake --build $build_dir -j" >&2
    exit 1
fi

echo "Running the efficiency checks..." >&2
ctest --test-dir "$build_dir" -R '^test_efficiency$' --output-on-failure >&2

cpu_field() {
    lscpu | sed -n "s/^$1: *//p" | head -1
}

echo "Running the benchmark, which takes about a minute..." >&2
{
    echo "# mini-tensor efficiency report"
    echo
    echo "Generated on $(date -u '+%Y-%m-%d at %H:%M UTC') from commit" \
        "$(git rev-parse --short HEAD 2>/dev/null || echo unknown)."
    echo
    echo "## Machine"
    echo
    echo "- CPU: $(cpu_field 'Model name')"
    echo "- Cores: $(cpu_field 'Core(s) per socket') per socket," \
        "$(cpu_field 'Thread(s) per core') threads per core"
    echo "- Memory: $(free -h | awk '/^Mem:/ {print $2}')"
    echo "- OS: $(. /etc/os-release && echo "$PRETTY_NAME"), kernel $(uname -r)"
    echo "- Compiler: $(c++ --version | head -1)"
    echo
    echo "## Benchmark"
    echo
    "$benchmark"
    echo
    echo "## SIMD levels"
    echo
    echo "Each row is a 1024 x 1024 \`double\` matrix multiplication on one thread. A level this"
    echo "CPU doesn't support runs the best level it does support, so it's only listed once."
    echo
    echo "| SIMD | Time | Billion operations/s |"
    echo "|---|---:|---:|"
    {
        MINITENSOR_ISA=generic OMP_NUM_THREADS=1 "$benchmark" --simd
        MINITENSOR_ISA=avx2 OMP_NUM_THREADS=1 "$benchmark" --simd
        OMP_NUM_THREADS=1 "$benchmark" --simd
    } | awk '!seen[$2]++'
} | tee "$report"

echo >&2
echo "Report written to $report" >&2
