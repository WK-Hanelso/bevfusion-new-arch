#include "point_features.hpp"

#include <cassert>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
  assert(argc == 2);
  auto const channels =
      thor_multisensor::zeroFeatureChannelsFromManifest(argv[1], 5);
  assert((channels == std::vector<int>{4}));
  std::vector<float> points{
      1.0F, 2.0F, 3.0F, 4.0F, 5.0F,
      6.0F, 7.0F, 8.0F, 9.0F, 10.0F,
  };
  thor_multisensor::zeroPointFeatureChannels(points, 5, channels);
  assert((points == std::vector<float>{
                        1.0F, 2.0F, 3.0F, 4.0F, 0.0F,
                        6.0F, 7.0F, 8.0F, 9.0F, 0.0F,
                    }));

  bool rejected = false;
  try {
    thor_multisensor::zeroPointFeatureChannels(points, 5, {5});
  } catch (std::out_of_range const&) {
    rejected = true;
  }
  assert(rejected);
  return 0;
}
