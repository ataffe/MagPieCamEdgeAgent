#!/usr/bin/env bash
# Fast correctness build of the ByteTrack core smoke test (no external deps now,
# so it compiles in seconds even on the Pi Zero 2 W). Compiles per-file with
# timing, links, and runs.
set -u
cd "$(dirname "$0")"

mkdir -p build/obj
FLAGS="-std=c++17 -O2 -I include"
SRCS="kalman_filter strack matching byte_tracker main_smoke"

OBJS=""
for s in $SRCS; do
    obj="build/obj/$s.o"
    OBJS="$OBJS $obj"
    if [ -f "$obj" ] && [ "$obj" -nt "src/$s.cpp" ]; then
        echo "[skip] $s.o up to date"
        continue
    fi
    printf "[cc]   %-16s ... " "$s.cpp"
    S=$(date +%s)
    g++ $FLAGS -c "src/$s.cpp" -o "$obj" || { echo "FAILED"; exit 1; }
    echo "$(( $(date +%s) - S ))s"
done

printf "[link] smoke ... "
g++ $OBJS -o build/smoke || { echo "FAILED"; exit 1; }
echo "ok"

echo "=== run smoke ==="
./build/smoke