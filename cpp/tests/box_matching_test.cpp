#include "track/box_matching.h"

#include <gtest/gtest.h>

#include <vector>

namespace vc {
namespace {

constexpr int kFace = 0;

Box face(float x1, float x2, float score = 0.9f) { return Box{kFace, x1, 100, x2, 180, score}; }

MatchParams params() { return MatchParams{1.0, 6.0, {0.5f}}; }

TEST(BoxMatching, TakesTheCheapestPairsFirstOneToOne) {
  const std::vector<MatchTrack> tracks{{face(100, 180), face(100, 180), std::nullopt},
                                       {face(130, 210), face(130, 210), std::nullopt}};
  // box 1 is 5 px from track 0, box 0 6 px from track 1: the cheaper pair is taken first
  const std::vector<MatchPair> pairs = matchBoxes(tracks, {face(136, 216), face(95, 175)}, 6, params());
  ASSERT_EQ(pairs.size(), 2u);
  EXPECT_EQ(pairs[0].track, 0u);
  EXPECT_EQ(pairs[0].next, 1u);
  EXPECT_EQ(pairs[1].track, 1u);
  EXPECT_EQ(pairs[1].next, 0u);
}

TEST(BoxMatching, FollowsTheVelocityInTheStrictPass) {
  // 120 px on in 6 frames is 1.5 mean sides: too far for the strict pass unless the track keeps its velocity of
  // 20 px per frame, and the low scores keep the pair out of the loose pass
  const std::vector<MatchTrack> moving{{face(100, 180, 0.3f), face(100, 180, 0.3f), Velocity{20, 0}}};
  const std::vector<MatchTrack> still{{face(100, 180, 0.3f), face(100, 180, 0.3f), std::nullopt}};
  EXPECT_EQ(matchBoxes(moving, {face(220, 300, 0.3f)}, 6, params()).size(), 1u);
  EXPECT_TRUE(matchBoxes(still, {face(220, 300, 0.3f)}, 6, params()).empty());
}

TEST(BoxMatching, LeavesLowScorePairsToTheStrictPass) {
  const std::vector<MatchTrack> strong{{face(100, 180), face(100, 180), std::nullopt}};
  const std::vector<MatchTrack> weakAnchor{{face(100, 180), face(100, 180, 0.3f), std::nullopt}};
  // 240 px away: 3 sides, too far for the strict pass, near enough for the loose one
  EXPECT_EQ(matchBoxes(strong, {face(340, 420)}, 6, params()).size(), 1u);
  EXPECT_TRUE(matchBoxes(weakAnchor, {face(340, 420)}, 6, params()).empty());
  EXPECT_TRUE(matchBoxes(strong, {face(340, 420, 0.3f)}, 6, params()).empty());
}

TEST(BoxMatching, NeedsConfirmationGoesByClass) {
  EXPECT_TRUE(needsConfirmation(face(0, 10, 0.3f), {0.5f}));
  EXPECT_FALSE(needsConfirmation(face(0, 10, 0.5f), {0.5f}));
  EXPECT_FALSE(needsConfirmation(Box{1, 0, 0, 10, 10, 0.01f}, {0.5f}));  // a class without an entry
}

}  // namespace
}  // namespace vc
