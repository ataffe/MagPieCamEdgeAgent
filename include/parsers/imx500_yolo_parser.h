// Copyright © 2026 Alexander Taffe

#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include "tracking/byte_tracker.h"  // Object, Vec4

namespace byte_track {

// Parse the IMX500 4-tensor detection output (the format produced by an Ultralytics
// IMX export with NMS): tensor[0]=boxes (R x 4), [1]=scores (R), [2]=classes (R),
// [3]=count (1). This mirrors picamera2 IMX500.get_outputs + IMX500YOLOBoundingBoxParser:
//   - the float tensor is already dequantized by the IPA,
//   - the 2-D box tensor is laid out Fortran/column-major (box i, coord c = flat[c*R + i]),
//   - boxes are tlbr in input-pixel coords, clamped to >= 0, optionally /input_h normalized,
//   - 'count' caps the number of valid detections.
//
// tensor / info come straight from libcamera controls rpi::CnnOutputTensor and
// rpi::CnnOutputTensorInfo. Pointers may be null (no inference this frame) -> empty result.
std::vector<Object> parse_imx500_detections(const float* tensor, size_t tensor_len,
                                            const uint8_t* info, size_t info_len,
                                            bool bbox_normalization = false,
                                            double input_h = 480.0);

}  // namespace byte_track