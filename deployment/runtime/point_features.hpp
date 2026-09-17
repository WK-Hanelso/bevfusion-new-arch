#pragma once

#include <vector>
#include <string>

namespace thor_multisensor {

void zeroPointFeatureChannels(std::vector<float>& points, int featureCount,
                              std::vector<int> const& channels);
std::vector<int> zeroFeatureChannelsFromManifest(std::string const& path,
                                                 int featureCount);

}  // namespace thor_multisensor
