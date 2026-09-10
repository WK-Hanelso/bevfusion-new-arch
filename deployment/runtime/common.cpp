#include "common.hpp"

#include <stdexcept>

namespace thor_multisensor {

void require(bool condition, std::string const& message) {
  if (!condition) throw std::runtime_error(message);
}

void cudaCheck(cudaError_t status, char const* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

}  // namespace thor_multisensor
