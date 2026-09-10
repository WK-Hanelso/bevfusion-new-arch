#include "statistics.hpp"

#include "common.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

namespace thor_multisensor {
namespace {

double percentile(std::vector<float> const& sorted, double fraction) {
  double const position = fraction * static_cast<double>(sorted.size() - 1);
  auto const low = static_cast<std::size_t>(std::floor(position));
  auto const high = static_cast<std::size_t>(std::ceil(position));
  double const weight = position - static_cast<double>(low);
  return sorted[low] * (1.0 - weight) + sorted[high] * weight;
}

}  // namespace

Summary summarize(std::vector<float> values) {
  require(!values.empty(), "cannot summarize empty latency samples");
  double const mean = std::accumulate(values.begin(), values.end(), 0.0) /
                      static_cast<double>(values.size());
  std::sort(values.begin(), values.end());
  return {percentile(values, 0.50), percentile(values, 0.90),
          percentile(values, 0.95), percentile(values, 0.99), mean};
}

void printSummary(std::string const& name, Summary const& value) {
  std::cout << name << " p50=" << value.p50 << " ms"
            << " p90=" << value.p90 << " ms"
            << " p95=" << value.p95 << " ms"
            << " p99=" << value.p99 << " ms"
            << " mean=" << value.mean << " ms\n";
}

}  // namespace thor_multisensor
