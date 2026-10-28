#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include "detect/box.h"

namespace vc {

// A velocity in px per frame index.
struct Velocity {
  double x = 0;
  double y = 0;
};

// What matching needs of a track: where it is on the previous keyframe, its last real detection (whose score
// decides whether the loose pass may take it, spec 2026-10-19 §3.8) and its velocity, when it has one.
struct MatchTrack {
  Box box{};
  Box anchor{};
  std::optional<Velocity> velocity{};
};

struct MatchParams {
  double maxDistance = 1.0;       // first pass: max |predicted centre - centre| / mean box side
  double maxDistanceLoose = 6.0;  // second pass: max |dx| and |dy| / the largest side of the two boxes
  // Per class index: boxes scoring below their class's entry need confirming; a class without an entry does not.
  std::vector<float> confirmScore{};
};

struct MatchPair {
  size_t track = 0;  // index into the tracks
  size_t next = 0;   // index into the next keyframe's boxes
};

// Matches tracks to the boxes of the next keyframe, `steps` frames after the previous one, in two greedy,
// one-to-one passes over same-class pairs, cheapest first; both try the target staying put and keeping its
// velocity. The first pass takes pairs whose centres are at most maxDistance mean box sides apart. The second
// pass, over what is left, takes pairs of similar size (each side within a factor of 3 of the other's) whose
// centres are at most maxDistanceLoose times the largest side of the two boxes apart along each axis, and only if
// neither the track's anchor nor the box is below its class's confirmScore. Pairs come in the order they were taken.
std::vector<MatchPair> matchBoxes(const std::vector<MatchTrack>& tracks, const std::vector<Box>& next, double steps,
                                  const MatchParams& params);

// Whether `box` scores below its class's entry of confirmScore (a class without an entry needs no confirmation).
bool needsConfirmation(const Box& box, const std::vector<float>& confirmScore);

}  // namespace vc
