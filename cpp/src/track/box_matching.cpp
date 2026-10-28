#include "track/box_matching.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include "track/box_geometry.h"

namespace vc {
namespace {

// The second matching pass pairs boxes only when each side is within this factor of the other's.
constexpr double kMaxLooseSizeRatio = 3.0;

bool similarSizes(const Box& a, const Box& b) {
  const auto within = [](double p, double q) { return p <= kMaxLooseSizeRatio * q && q <= kMaxLooseSizeRatio * p; };
  return within(widthOf(a), widthOf(b)) && within(heightOf(a), heightOf(b));
}

}  // namespace

bool needsConfirmation(const Box& box, const std::vector<float>& confirmScore) {
  if (box.cls < 0 || static_cast<size_t>(box.cls) >= confirmScore.size()) return false;
  return box.score < confirmScore[static_cast<size_t>(box.cls)];
}

std::vector<MatchPair> matchBoxes(const std::vector<MatchTrack>& tracks, const std::vector<Box>& next, double steps,
                                  const MatchParams& params) {
  struct Candidate {
    double cost;
    size_t i;
    size_t j;
  };
  // The cheaper of "stayed put" and "kept its velocity" (a target that stops still matches).
  const auto cheapest = [&](const MatchTrack& track, const auto& costFrom) {
    const Point from = centerOf(track.box);
    double cost = costFrom(from);
    if (const auto& v = track.velocity) {
      cost = std::min(cost, costFrom(Point{from.x + v->x * steps, from.y + v->y * steps}));
    }
    return cost;
  };
  std::vector<bool> usedTrack(tracks.size(), false);
  std::vector<bool> usedNext(next.size(), false);
  std::vector<MatchPair> pairs;
  const auto takeGreedily = [&](std::vector<Candidate> candidates) {
    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const Candidate& p, const Candidate& q) { return p.cost < q.cost; });
    for (const Candidate& c : candidates) {
      if (usedTrack[c.i] || usedNext[c.j]) continue;
      usedTrack[c.i] = true;
      usedNext[c.j] = true;
      pairs.push_back(MatchPair{c.i, c.j});
    }
  };

  // First pass: centre distance in mean box sides.
  std::vector<Candidate> strict;
  for (size_t i = 0; i < tracks.size(); ++i) {
    const Box& a = tracks[i].box;
    for (size_t j = 0; j < next.size(); ++j) {
      const Box& b = next[j];
      if (a.cls != b.cls) continue;
      const Point actual = centerOf(b);
      const double size = std::max(1.0, (widthOf(a) + heightOf(a) + widthOf(b) + heightOf(b)) / 4.0);
      const double cost =
          cheapest(tracks[i], [&](Point p) { return std::hypot(actual.x - p.x, actual.y - p.y) / size; });
      if (cost <= params.maxDistance) strict.push_back({cost, i, j});
    }
  }
  takeGreedily(std::move(strict));

  // Second pass, over what is left: both axes in the largest side of the two boxes, similar sizes only. A pair
  // with either side below its class's confirm score is left to the first pass (spec 2026-10-19 §3.8): two
  // far-apart low-score boxes are more likely two false positives than one target, and joining them sweeps a mask
  // between them. A track's side is its anchor, its last real detection.
  std::vector<Candidate> loose;
  for (size_t i = 0; i < tracks.size(); ++i) {
    if (usedTrack[i]) continue;
    const Box& a = tracks[i].box;
    for (size_t j = 0; j < next.size(); ++j) {
      const Box& b = next[j];
      if (usedNext[j] || a.cls != b.cls || !similarSizes(a, b)) continue;
      if (needsConfirmation(tracks[i].anchor, params.confirmScore) || needsConfirmation(b, params.confirmScore)) {
        continue;
      }
      const Point actual = centerOf(b);
      const double side = std::max({1.0, widthOf(a), heightOf(a), widthOf(b), heightOf(b)});
      const double cost = cheapest(tracks[i], [&](Point p) {
        return std::max(std::abs(actual.x - p.x), std::abs(actual.y - p.y)) / side;
      });
      if (cost <= params.maxDistanceLoose) loose.push_back({cost, i, j});
    }
  }
  takeGreedily(std::move(loose));
  return pairs;
}

}  // namespace vc
