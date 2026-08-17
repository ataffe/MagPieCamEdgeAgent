
//
// Unit tests for the IoU / assignment helpers in matching.cpp.

#include <gtest/gtest.h>

#include <memory>
#include <utility>
#include <vector>

#include "tracking/matching.h"
#include "tracking/strack.h"

using namespace byte_track;

// --- iou() ---------------------------------------------------------------

TEST(Iou, IdenticalBoxesIsOne) {
    const Vec4 box = {0, 0, 10, 10};
    EXPECT_DOUBLE_EQ(iou(box, box), 1.0);
}

TEST(Iou, DisjointBoxesIsZero) {
    EXPECT_DOUBLE_EQ(iou({0, 0, 10, 10}, {20, 20, 30, 30}), 0.0);
}

TEST(Iou, TouchingButNonOverlappingIsZero) {
    // The cython_bbox +1 convention means edge-touching boxes still report 0
    // once the x interval is empty (min - max + 1 <= 0).
    EXPECT_DOUBLE_EQ(iou({0, 0, 10, 10}, {11, 0, 21, 10}), 0.0);
}

TEST(Iou, PartialOverlapKnownValue) {
    // A and B are both 11x11 (with the +1 convention) and overlap in a 6x11
    // region: inter = 66, union = 121 + 121 - 66 = 176.
    EXPECT_DOUBLE_EQ(iou({0, 0, 10, 10}, {5, 0, 15, 10}), 66.0 / 176.0);
}

TEST(Iou, IsSymmetric) {
    const Vec4 a = {0, 0, 10, 10};
    const Vec4 b = {5, 0, 15, 10};
    EXPECT_DOUBLE_EQ(iou(a, b), iou(b, a));
}

// --- iou_distance() ------------------------------------------------------

namespace {
// STrack stores tlwh internally; build one whose tlbr() equals the given box.
STrackPtr make_track(const Vec4& tlbr) {
    return std::make_shared<STrack>(STrack::tlbr_to_tlwh(tlbr), 1.0, 0);
}
}  // namespace

TEST(IouDistance, DiagonalIsZeroForIdenticalBoxes) {
    std::vector<STrackPtr> a = {make_track({0, 0, 10, 10}), make_track({20, 20, 30, 30})};
    const Mat cost = iou_distance(a, a);

    ASSERT_EQ(cost.r, 2);
    ASSERT_EQ(cost.c, 2);
    EXPECT_DOUBLE_EQ(cost(0, 0), 0.0);  // 1 - iou(self) == 0
    EXPECT_DOUBLE_EQ(cost(1, 1), 0.0);
    EXPECT_DOUBLE_EQ(cost(0, 1), 1.0);  // disjoint -> 1 - 0 == 1
}

TEST(IouDistance, EmptyInputsProduceEmptyMatrix) {
    std::vector<STrackPtr> empty;
    std::vector<STrackPtr> one = {make_track({0, 0, 10, 10})};

    const Mat c1 = iou_distance(empty, one);
    EXPECT_EQ(c1.r, 0);
    EXPECT_EQ(c1.c, 1);

    const Mat c2 = iou_distance(one, empty);
    EXPECT_EQ(c2.r, 1);
    EXPECT_EQ(c2.c, 0);
}

// --- linear_assignment() -------------------------------------------------

TEST(LinearAssignment, MatchesCheapDiagonal) {
    Mat cost(2, 2);
    cost(0, 0) = 0.1; cost(0, 1) = 0.9;
    cost(1, 0) = 0.9; cost(1, 1) = 0.1;

    std::vector<std::pair<int, int>> matches;
    std::vector<int> ua, ub;
    linear_assignment(cost, 0.8, matches, ua, ub);

    ASSERT_EQ(matches.size(), 2u);
    EXPECT_EQ(matches[0], std::make_pair(0, 0));
    EXPECT_EQ(matches[1], std::make_pair(1, 1));
    EXPECT_TRUE(ua.empty());
    EXPECT_TRUE(ub.empty());
}

TEST(LinearAssignment, DropsPairsAboveThreshold) {
    // Optimal assignment is still the diagonal, but every cost exceeds the
    // threshold, so both rows and both columns come back unmatched.
    Mat cost(2, 2);
    cost(0, 0) = 0.9; cost(0, 1) = 0.95;
    cost(1, 0) = 0.95; cost(1, 1) = 0.9;

    std::vector<std::pair<int, int>> matches;
    std::vector<int> ua, ub;
    linear_assignment(cost, 0.8, matches, ua, ub);

    EXPECT_TRUE(matches.empty());
    EXPECT_EQ(ua, (std::vector<int>{0, 1}));
    EXPECT_EQ(ub, (std::vector<int>{0, 1}));
}

TEST(LinearAssignment, RectangleLeavesExtraColumnUnmatched) {
    Mat cost(1, 2);
    cost(0, 0) = 0.1;
    cost(0, 1) = 0.5;

    std::vector<std::pair<int, int>> matches;
    std::vector<int> ua, ub;
    linear_assignment(cost, 0.8, matches, ua, ub);

    ASSERT_EQ(matches.size(), 1u);
    EXPECT_EQ(matches[0], std::make_pair(0, 0));
    EXPECT_TRUE(ua.empty());
    EXPECT_EQ(ub, (std::vector<int>{1}));
}

TEST(LinearAssignment, EmptyCostMarksAllColumnsUnmatched) {
    Mat cost(0, 3);
    std::vector<std::pair<int, int>> matches;
    std::vector<int> ua, ub;
    linear_assignment(cost, 0.8, matches, ua, ub);

    EXPECT_TRUE(matches.empty());
    EXPECT_TRUE(ua.empty());
    EXPECT_EQ(ub, (std::vector<int>{0, 1, 2}));
}
