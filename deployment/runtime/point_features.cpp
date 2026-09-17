#include "point_features.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace thor_multisensor {

std::vector<int> zeroFeatureChannelsFromManifest(std::string const& path,
                                                 int featureCount) {
  std::ifstream stream(path);
  if (!stream) {
    throw std::runtime_error("failed to open LiDAR manifest: " + path);
  }
  std::string const payload((std::istreambuf_iterator<char>(stream)),
                            std::istreambuf_iterator<char>());
  std::string const key = "\"zero_feature_channels\"";
  std::size_t cursor = payload.find(key);
  if (cursor == std::string::npos) {
    throw std::runtime_error("LiDAR manifest is missing zero_feature_channels");
  }
  cursor = payload.find('[', cursor + key.size());
  std::size_t const end = payload.find(']', cursor);
  if (cursor == std::string::npos || end == std::string::npos) {
    throw std::runtime_error("invalid zero_feature_channels in LiDAR manifest");
  }

  std::vector<int> channels;
  ++cursor;
  while (cursor < end) {
    while (cursor < end &&
           (std::isspace(static_cast<unsigned char>(payload[cursor])) ||
            payload[cursor] == ',')) {
      ++cursor;
    }
    if (cursor == end) break;
    if (!std::isdigit(static_cast<unsigned char>(payload[cursor]))) {
      throw std::runtime_error("invalid zero_feature_channels in LiDAR manifest");
    }
    std::size_t consumed = 0;
    int const channel = std::stoi(payload.substr(cursor, end - cursor), &consumed);
    if (channel < 0 || channel >= featureCount) {
      throw std::out_of_range("point feature channel is out of range");
    }
    if (std::find(channels.begin(), channels.end(), channel) != channels.end()) {
      throw std::runtime_error("duplicate point feature channel in LiDAR manifest");
    }
    channels.push_back(channel);
    cursor += consumed;
  }
  return channels;
}

void zeroPointFeatureChannels(std::vector<float>& points, int featureCount,
                              std::vector<int> const& channels) {
  if (featureCount <= 0 ||
      points.size() % static_cast<std::size_t>(featureCount) != 0) {
    throw std::invalid_argument("point feature buffer has an invalid shape");
  }
  for (int channel : channels) {
    if (channel < 0 || channel >= featureCount) {
      throw std::out_of_range("point feature channel is out of range");
    }
  }
  for (std::size_t offset = 0; offset < points.size(); offset += featureCount) {
    for (int channel : channels) {
      points[offset + static_cast<std::size_t>(channel)] = 0.0F;
    }
  }
}

}  // namespace thor_multisensor
