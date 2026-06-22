#!/usr/bin/env bash
# Build the byte_track rpicam-apps post-processing stage as a .so and stage it in
# a user-owned postproc dir (alongside symlinks to the system stages, since
# --post-process-libs replaces the search dir). No sudo required.
set -u
cd "$(dirname "$0")"

SYS=/usr/lib/aarch64-linux-gnu/rpicam-apps-postproc
OUT=postproc_libs
mkdir -p build "$OUT"

FLAGS="-std=c++17 -O2 -fPIC -shared -I include -I/usr/include/rpicam-apps $(pkg-config --cflags libcamera) $(pkg-config --cflags opencv4)"
LIBS="$(pkg-config --libs libcamera) $(pkg-config --libs opencv4)"

printf "[cc]   byte_track_stage.so ... "
S=$(date +%s)
g++ $FLAGS \
    src/tracking/byte_track_stage.cpp src/parsers/imx500_yolo_parser.cpp src/streaming/mjpeg_server.cpp \
    src/tracking/kalman_filter.cpp src/tracking/strack.cpp src/tracking/matching.cpp src/tracking/byte_tracker.cpp \
    -o "$OUT/byte_track_stage.so" $LIBS || { echo "FAILED"; exit 1; }
echo "$(( $(date +%s) - S ))s"

# Mirror the system stages so imx500_object_detection etc. remain available.
for so in "$SYS"/*.so; do ln -sf "$so" "$OUT/"; done

echo "staged in $(pwd)/$OUT:"
ls -la "$OUT"