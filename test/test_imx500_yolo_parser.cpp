
//
// Unit tests for parse_imx500_detections(). The parser decodes two raw buffers:
//   - `info`: the CnnOutputTensorInfo blob (network name + per-tensor shapes),
//   - `tensor`: the flat, dequantized float output (boxes / scores / classes / count).
// The helpers below build byte-exact versions of both so the tests exercise the
// real decode path rather than a mock.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "parsers/imx500_yolo_parser.h"

using namespace byte_track;

namespace {

constexpr size_t NAME_LEN = 64;
constexpr size_t MAX_DIMS = 16;

void put_u32(std::vector<uint8_t>& b, uint32_t v) {
    uint8_t tmp[4];
    std::memcpy(tmp, &v, 4);  // native-endian, matching the parser's memcpy read
    b.insert(b.end(), tmp, tmp + 4);
}

void put_u16(std::vector<uint8_t>& b, uint16_t v) {
    uint8_t tmp[2];
    std::memcpy(tmp, &v, 2);
    b.insert(b.end(), tmp, tmp + 2);
}

// Build a CnnOutputTensorInfo blob for the given per-tensor dimension lists.
// Layout: char name[64]; u32 num_tensors; OutputTensorInfo[ ] (40 bytes each:
//   u32 tensor_data_num; u32 num_dimensions; u16 size[16]); u8 frameCount.
std::vector<uint8_t> build_info(const std::vector<std::vector<int>>& shapes) {
    std::vector<uint8_t> b(NAME_LEN, 0);          // network_name (zeroed)
    put_u32(b, static_cast<uint32_t>(shapes.size()));
    for (const auto& dims : shapes) {
        uint32_t data_num = 1;
        for (int d : dims) data_num *= static_cast<uint32_t>(d);
        put_u32(b, data_num);                     // tensor_data_num (unused by reader)
        put_u32(b, static_cast<uint32_t>(dims.size()));  // num_dimensions
        for (size_t k = 0; k < MAX_DIMS; ++k)     // fixed 16-slot size[] array
            put_u16(b, k < dims.size() ? static_cast<uint16_t>(dims[k]) : 0);
    }
    b.push_back(0);  // frameCount
    return b;
}

// The standard 4-tensor YOLO export shape: boxes (R x 4), scores (R), classes (R), count (1).
std::vector<uint8_t> standard_info(int R) {
    return build_info({{R, 4}, {R}, {R}, {1}});
}

struct Det {
    double x1, y1, x2, y2, score;
    int cls;
};

// Build the flat float tensor for R rows, filling `dets` and the count field.
std::vector<float> build_tensor(int R, const std::vector<Det>& dets, int count) {
    const size_t off_boxes = 0, off_scores = static_cast<size_t>(R) * 4,
                 off_classes = off_scores + R, off_count = off_classes + R;
    std::vector<float> t(off_count + 1, 0.0f);
    for (size_t i = 0; i < dets.size(); ++i) {
        // Boxes are Fortran/column-major: element (i, c) = flat[c * R + i].
        t[off_boxes + 0 * R + i] = static_cast<float>(dets[i].x1);
        t[off_boxes + 1 * R + i] = static_cast<float>(dets[i].y1);
        t[off_boxes + 2 * R + i] = static_cast<float>(dets[i].x2);
        t[off_boxes + 3 * R + i] = static_cast<float>(dets[i].y2);
        t[off_scores + i] = static_cast<float>(dets[i].score);
        t[off_classes + i] = static_cast<float>(dets[i].cls);
    }
    t[off_count] = static_cast<float>(count);
    return t;
}

}  // namespace

// --- happy path ----------------------------------------------------------

TEST(Imx500Parser, ParsesBoxesScoresAndClasses) {
    const int R = 2;
    const auto info = standard_info(R);
    const auto t = build_tensor(R, {{10, 20, 30, 40, 0.9, 1}, {50, 60, 70, 80, 0.5, 2}}, 2);

    auto out = parse_imx500_detections(t.data(), t.size(), info.data(), info.size());

    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0].tlbr, (Vec4{10, 20, 30, 40}));
    EXPECT_FLOAT_EQ(out[0].score, 0.9f);  // score round-trips through float
    EXPECT_EQ(out[0].label, 1);
    EXPECT_EQ(out[1].tlbr, (Vec4{50, 60, 70, 80}));
    EXPECT_EQ(out[1].label, 2);
}

TEST(Imx500Parser, BoxComponentsAreNotTransposed) {
    // Distinct values per coordinate so a row/column-major mixup would be caught.
    const int R = 1;
    const auto info = standard_info(R);
    const auto t = build_tensor(R, {{1, 2, 3, 4, 0.7, 0}}, 1);

    auto out = parse_imx500_detections(t.data(), t.size(), info.data(), info.size());
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].tlbr, (Vec4{1, 2, 3, 4}));
}

// --- count handling ------------------------------------------------------

TEST(Imx500Parser, CountCapsNumberOfDetections) {
    const int R = 3;
    const auto info = standard_info(R);
    // Three rows present in the buffer, but count says only the first is valid.
    const auto t = build_tensor(R, {{1, 1, 2, 2, 0.9, 0},
                                    {3, 3, 4, 4, 0.9, 0},
                                    {5, 5, 6, 6, 0.9, 0}}, 1);

    auto out = parse_imx500_detections(t.data(), t.size(), info.data(), info.size());
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].tlbr, (Vec4{1, 1, 2, 2}));
}

TEST(Imx500Parser, CountClampedToRowCount) {
    const int R = 2;
    const auto info = standard_info(R);
    const auto t = build_tensor(R, {{1, 1, 2, 2, 0.9, 0}, {3, 3, 4, 4, 0.9, 0}}, 99);

    auto out = parse_imx500_detections(t.data(), t.size(), info.data(), info.size());
    EXPECT_EQ(out.size(), 2u);  // clamped to R, not 99
}

TEST(Imx500Parser, NegativeCountYieldsNoDetections) {
    const int R = 2;
    const auto info = standard_info(R);
    const auto t = build_tensor(R, {{1, 1, 2, 2, 0.9, 0}}, -1);

    auto out = parse_imx500_detections(t.data(), t.size(), info.data(), info.size());
    EXPECT_TRUE(out.empty());
}

// --- value transforms ----------------------------------------------------

TEST(Imx500Parser, NegativeCoordinatesAreClampedToZero) {
    const int R = 1;
    const auto info = standard_info(R);
    const auto t = build_tensor(R, {{-5, -10, 30, 40, 0.9, 0}}, 1);

    auto out = parse_imx500_detections(t.data(), t.size(), info.data(), info.size());
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].tlbr, (Vec4{0, 0, 30, 40}));
}

TEST(Imx500Parser, NormalizationDividesByInputHeight) {
    const int R = 1;
    const auto info = standard_info(R);
    const auto t = build_tensor(R, {{120, 240, 360, 480, 0.9, 0}}, 1);

    auto out = parse_imx500_detections(t.data(), t.size(), info.data(), info.size(),
                                       /*bbox_normalization=*/true, /*input_h=*/480.0);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].tlbr, (Vec4{0.25, 0.5, 0.75, 1.0}));
}

// --- guard rails ---------------------------------------------------------

TEST(Imx500Parser, NullPointersReturnEmpty) {
    const int R = 1;
    const auto info = standard_info(R);
    const auto t = build_tensor(R, {{1, 1, 2, 2, 0.9, 0}}, 1);

    EXPECT_TRUE(parse_imx500_detections(nullptr, t.size(), info.data(), info.size()).empty());
    EXPECT_TRUE(parse_imx500_detections(t.data(), t.size(), nullptr, info.size()).empty());
}

TEST(Imx500Parser, ShortInfoReturnsEmpty) {
    const int R = 1;
    const auto t = build_tensor(R, {{1, 1, 2, 2, 0.9, 0}}, 1);
    std::vector<uint8_t> tiny(10, 0);  // shorter than NAME_LEN + 4
    EXPECT_TRUE(parse_imx500_detections(t.data(), t.size(), tiny.data(), tiny.size()).empty());
}

TEST(Imx500Parser, FewerThanFourTensorsReturnsEmpty) {
    const auto info = build_info({{2, 4}, {2}, {2}});  // missing the count tensor
    const auto t = build_tensor(2, {{1, 1, 2, 2, 0.9, 0}}, 1);
    EXPECT_TRUE(parse_imx500_detections(t.data(), t.size(), info.data(), info.size()).empty());
}

TEST(Imx500Parser, BoxTensorWithFewerThanFourColumnsReturnsEmpty) {
    const int R = 2;
    const auto info = build_info({{R, 3}, {R}, {R}, {1}});  // boxes only 3 wide
    const auto t = build_tensor(R, {{1, 1, 2, 2, 0.9, 0}}, 1);
    EXPECT_TRUE(parse_imx500_detections(t.data(), t.size(), info.data(), info.size()).empty());
}

TEST(Imx500Parser, TensorShorterThanShapesReturnsEmpty) {
    const int R = 2;
    const auto info = standard_info(R);
    const auto t = build_tensor(R, {{1, 1, 2, 2, 0.9, 0}}, 1);
    // Claim a length smaller than the shapes require -> the size guard trips.
    EXPECT_TRUE(parse_imx500_detections(t.data(), t.size() - 3, info.data(), info.size()).empty());
}
