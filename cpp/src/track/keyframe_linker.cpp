#include "track/keyframe_linker.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <stdexcept>

#include "track/box_geometry.h"

namespace vc {
namespace {

// A backfilled box moves back along its track's velocity by at most this fraction of its width (x) and height (y)
// per keyframe interval, like a held one (spec 2026-10-27 §2.2).
constexpr double kMaxShiftPerInterval = 0.5;

bool finiteBox(const Box& b) {
  return std::isfinite(b.x1) && std::isfinite(b.y1) && std::isfinite(b.x2) && std::isfinite(b.y2);
}

}  // namespace

KeyframeLinker::KeyframeLinker(const LinkerConfig& config) : config_(config) {
  if (config.lookahead < 1) throw std::invalid_argument("KeyframeLinker: lookahead must be >= 1");
  if (config.backfill < 0 || config.backfill > config.lookahead - 1) {
    throw std::invalid_argument("KeyframeLinker: backfill must be in [0, lookahead - 1]");
  }
}

Keyframe& KeyframeLinker::at(int64_t seq) { return pending_.at(static_cast<size_t>(seq - baseSeq_)); }

std::vector<Keyframe> KeyframeLinker::push(const Keyframe& key) {
  if (!std::isfinite(key.time.ptsMs) || !std::all_of(key.boxes.begin(), key.boxes.end(), finiteBox)) {
    throw std::invalid_argument("KeyframeLinker: keyframe timestamp or box is not finite");
  }
  if (prevTime_ && key.time.index <= prevTime_->index) {
    throw std::invalid_argument("KeyframeLinker: keyframe indices must increase");
  }
  const int64_t seq = nextSeq_++;
  pending_.push_back(Keyframe{key.time, key.boxes, std::vector<BoxLink>(key.boxes.size())});

  // Each track where it would be on the previous keyframe: its last detection moved along its velocity.
  std::vector<MatchTrack> candidates;
  candidates.reserve(tracks_.size());
  for (const Track& t : tracks_) {
    Box box = t.last;
    if (t.velocity && prevTime_) {
      const double since = static_cast<double>(prevTime_->index - t.lastIndex);
      box = shifted(t.last, t.velocity->x * since, t.velocity->y * since);
    }
    candidates.push_back(MatchTrack{box, t.last, t.velocity});
  }
  const double steps = prevTime_ ? static_cast<double>(key.time.index - prevTime_->index) : 0.0;
  std::vector<std::optional<size_t>> matchedTo(tracks_.size());
  std::vector<bool> taken(key.boxes.size(), false);
  for (const MatchPair& m : matchBoxes(candidates, key.boxes, steps, config_.match)) {
    matchedTo[m.track] = m.next;
    taken[m.next] = true;
  }

  std::vector<Track> tracks;
  tracks.reserve(tracks_.size() + key.boxes.size());
  for (size_t i = 0; i < tracks_.size(); ++i) {
    Track t = tracks_[i];
    if (!matchedTo[i]) {
      ++t.missed;
      // a confirmed track waits up to W keyframes to be found again; an unconfirmed one ends at its first miss
      if (t.confirmed && t.missed <= config_.lookahead) tracks.push_back(std::move(t));
      continue;
    }
    const size_t j = *matchedTo[i];
    const Box& box = key.boxes[j];
    if (t.missed > 0) bridge(t, box, key.time.index);
    const Point a = centerOf(t.last);
    const Point b = centerOf(box);
    const double span = static_cast<double>(key.time.index - t.lastIndex);
    const bool firstVelocity = !t.velocity;
    t.velocity = Velocity{(b.x - a.x) / span, (b.y - a.y) / span};
    t.last = box;
    t.lastSeq = seq;
    t.lastIndex = key.time.index;
    t.missed = 0;
    pending_.back().links[j] = BoxLink{t.id, LinkKind::Real, true};
    if (!t.confirmed) {
      confirm(t);  // matched on the very next keyframe: an unconfirmed track never survives a miss
    } else if (firstVelocity) {
      placeBackfills(t);
    }
    tracks.push_back(std::move(t));
  }
  for (size_t j = 0; j < key.boxes.size(); ++j) {
    if (taken[j]) continue;
    Track t;
    t.id = nextId_++;
    t.first = key.boxes[j];
    t.firstSeq = seq;
    t.firstBox = j;
    t.firstIndex = key.time.index;
    t.last = key.boxes[j];
    t.lastSeq = seq;
    t.lastIndex = key.time.index;
    pending_.back().links[j] = BoxLink{t.id, LinkKind::Real, false};
    if (!needsConfirmation(key.boxes[j], config_.match.confirmScore)) confirm(t);
    tracks.push_back(std::move(t));
  }
  tracks_ = std::move(tracks);
  prevTime_ = key.time;

  std::vector<Keyframe> out;
  while (pending_.size() > static_cast<size_t>(config_.lookahead)) {
    out.push_back(std::move(pending_.front()));
    pending_.pop_front();
    ++baseSeq_;
  }
  return out;
}

void KeyframeLinker::bridge(const Track& track, const Box& to, int64_t toIndex) {
  // the keyframes after its last detection, up to the one just pushed (nextSeq_ - 1)
  for (int64_t s = track.lastSeq + 1; s < nextSeq_ - 1; ++s) {
    Keyframe& k = at(s);
    const double f =
        static_cast<double>(k.time.index - track.lastIndex) / static_cast<double>(toIndex - track.lastIndex);
    k.boxes.push_back(lerp(track.last, to, f));
    k.links.push_back(BoxLink{track.id, LinkKind::Bridged, true});
  }
}

void KeyframeLinker::confirm(Track& track) {
  track.confirmed = true;
  at(track.firstSeq).links.at(track.firstBox).masked = true;
  for (int b = 1; b <= config_.backfill; ++b) {
    const int64_t s = track.firstSeq - b;
    if (s < baseSeq_) break;  // returned already, or before the first keyframe
    Keyframe& k = at(s);
    k.boxes.push_back(backfillBox(track, s));
    k.links.push_back(BoxLink{track.id, LinkKind::Backfilled, true});
    track.backfills.emplace_back(s, k.boxes.size() - 1);
  }
}

void KeyframeLinker::placeBackfills(const Track& track) {
  for (const auto& [s, j] : track.backfills) {
    if (s >= baseSeq_) at(s).boxes.at(j) = backfillBox(track, s);
  }
}

Box KeyframeLinker::backfillBox(const Track& track, int64_t seq) {
  if (!track.velocity) return track.first;
  const double since = static_cast<double>(track.firstIndex - at(seq).time.index);
  const double intervals = static_cast<double>(track.firstSeq - seq);
  const double capX = kMaxShiftPerInterval * intervals * widthOf(track.first);
  const double capY = kMaxShiftPerInterval * intervals * heightOf(track.first);
  return shifted(track.first, std::clamp(-track.velocity->x * since, -capX, capX),
                 std::clamp(-track.velocity->y * since, -capY, capY));
}

std::vector<Keyframe> KeyframeLinker::finish() {
  std::vector<Keyframe> out(std::make_move_iterator(pending_.begin()), std::make_move_iterator(pending_.end()));
  pending_.clear();
  baseSeq_ = 0;
  nextSeq_ = 0;
  prevTime_.reset();
  tracks_.clear();
  nextId_ = 1;
  return out;
}

}  // namespace vc
