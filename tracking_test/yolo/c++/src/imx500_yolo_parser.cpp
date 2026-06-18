#include "imx500_yolo_parser.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace byte_track {

// CnnOutputTensorInfoExported layout (little-endian, from picamera2/libcamera):
//   char     network_name[64];
//   uint32_t num_tensors;
//   OutputTensorInfo info[16];   // each: u32 tensor_data_num; u32 num_dimensions; u16 size[16]; (40 bytes)
//   uint8_t  frameCount;
namespace {
constexpr size_t NAME_LEN = 64;
constexpr size_t MAX_TENSORS = 16;
constexpr size_t MAX_DIMS = 16;
constexpr size_t OTI_SIZE = 4 + 4 + 2 * MAX_DIMS;  // 40

uint32_t rd_u32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
uint16_t rd_u16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }
}  // namespace

std::vector<Object> parse_imx500_detections(const float* tensor, size_t tensor_len,
                                            const uint8_t* info, size_t info_len,
                                            bool bbox_normalization, double input_h) {
    std::vector<Object> out;
    if (!tensor || !info || info_len < NAME_LEN + 4) return out;

    uint32_t num_tensors = rd_u32(info + NAME_LEN);
    if (num_tensors < 4) return out;  // need boxes, scores, classes, count
    num_tensors = std::min<uint32_t>(num_tensors, MAX_TENSORS);

    // Per-tensor dimension lists.
    std::vector<std::vector<int>> shapes;
    for (uint32_t t = 0; t < num_tensors; ++t) {
        const size_t base = NAME_LEN + 4 + static_cast<size_t>(t) * OTI_SIZE;
        if (base + 8 > info_len) return out;
        uint32_t nd = std::min<uint32_t>(rd_u32(info + base + 4), MAX_DIMS);
        std::vector<int> dims;
        for (uint32_t k = 0; k < nd; ++k) dims.push_back(static_cast<int>(rd_u16(info + base + 8 + 2 * k)));
        shapes.push_back(std::move(dims));
    }

    // Sequential offsets into the flat float buffer (matches get_outputs split order).
    auto prod = [](const std::vector<int>& d) {
        size_t p = 1;
        for (int x : d) p *= static_cast<size_t>(std::max(1, x));
        return p;
    };
    std::vector<size_t> offset(num_tensors, 0);
    size_t acc = 0;
    for (uint32_t t = 0; t < num_tensors; ++t) { offset[t] = acc; acc += prod(shapes[t]); }
    if (acc > tensor_len) return out;

    // tensor[0] = boxes (R x C, C>=4), Fortran-ordered.
    if (shapes[0].size() < 2 || shapes[0][1] < 4) return out;
    const int R = shapes[0][0];
    const size_t off_boxes = offset[0], off_scores = offset[1], off_classes = offset[2], off_count = offset[3];

    int num = static_cast<int>(std::lround(tensor[off_count]));
    num = std::max(0, std::min(num, R));
    const double denom = bbox_normalization ? input_h : 1.0;

    out.reserve(num);
    for (int i = 0; i < num; ++i) {
        // Fortran 2-D indexing: element (i, c) = flat[c * R + i].
        double x1 = std::max(0.0, static_cast<double>(tensor[off_boxes + 0 * R + i]));
        double y1 = std::max(0.0, static_cast<double>(tensor[off_boxes + 1 * R + i]));
        double x2 = std::max(0.0, static_cast<double>(tensor[off_boxes + 2 * R + i]));
        double y2 = std::max(0.0, static_cast<double>(tensor[off_boxes + 3 * R + i]));
        if (denom != 1.0) { x1 /= denom; y1 /= denom; x2 /= denom; y2 /= denom; }
        const double score = tensor[off_scores + i];
        const int cls = static_cast<int>(std::lround(tensor[off_classes + i]));
        out.push_back({Vec4{x1, y1, x2, y2}, score, cls});
    }
    return out;
}

}  // namespace byte_track