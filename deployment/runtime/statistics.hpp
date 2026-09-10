#pragma once

#include <string>
#include <vector>

namespace thor_multisensor {

struct Summary {
  double p50;
  double p90;
  double p95;
  double p99;
  double mean;
};

Summary summarize(std::vector<float> values);
void printSummary(std::string const& name, Summary const& value);

}  // namespace thor_multisensor
