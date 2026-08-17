

#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include "tracking/byte_tracker.h"  // Object, Vec4

/**
 * @file imx500_yolo_parser.h
 * @brief Decoder for the Sony IMX500 on-sensor YOLO detection output.
 */
namespace byte_track {

/**
 * @brief Parse the IMX500 4-tensor detection output into tracker @ref Object%s.
 *
 * Decodes the format produced by an Ultralytics IMX export with NMS:
 *   - `tensor[0]` = boxes (R x 4),
 *   - `tensor[1]` = scores (R),
 *   - `tensor[2]` = classes (R),
 *   - `tensor[3]` = count (1).
 *
 * This mirrors picamera2's `IMX500.get_outputs` + `IMX500YOLOBoundingBoxParser`:
 *   - the float tensor is already dequantized by the IPA,
 *   - the 2-D box tensor is laid out Fortran/column-major
 *     (box i, coord c = `flat[c * R + i]`),
 *   - boxes are tlbr in input-pixel coords, clamped to `>= 0`, and optionally
 *     normalized by dividing by @p input_h,
 *   - the `count` tensor caps the number of valid detections.
 *
 * @p tensor and @p info come straight from the libcamera controls
 * `rpi::CnnOutputTensor` and `rpi::CnnOutputTensorInfo`. Either pointer may be
 * null (e.g. no inference ran this frame), in which case an empty result is
 * returned. Malformed or truncated buffers also yield an empty result.
 *
 * @param tensor             Pointer to the flat, dequantized float output, or
 *                           null.
 * @param tensor_len         Number of floats available at @p tensor.
 * @param info               Pointer to the `CnnOutputTensorInfo` blob, or null.
 * @param info_len           Number of bytes available at @p info.
 * @param bbox_normalization If true, divide box coordinates by @p input_h.
 * @param input_h            Network input height used for normalization.
 * @return The decoded detections, or an empty vector on missing/invalid input.
 */
std::vector<DetectedObject> parse_imx500_detections(const float *tensor, size_t tensor_len,
                                                    const uint8_t *info, size_t info_len,
                                                    bool bbox_normalization = false,
                                                    double input_h = 480.0);

}  // namespace byte_track
