#pragma once

#include <cuda_runtime_api.h>

#include <string>

namespace thor_multisensor {

void require(bool condition, std::string const& message);
void cudaCheck(cudaError_t status, char const* operation);

}  // namespace thor_multisensor
