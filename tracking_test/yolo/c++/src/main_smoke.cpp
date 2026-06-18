// Standalone sanity check for the ByteTrack core (no camera needed).
// Feeds two synthetic objects moving across the frame and prints the resulting
// track IDs + a per-frame tracker timing, so we can confirm the port runs and
// produces stable IDs before wiring up the camera.
#include "byte_tracker.h"
#include <chrono>
#include <cstdio>

using namespace byte_track;

int main() {
    BYTETracker tracker(0.5, 0.8, 30, 30.0);

    for (int f = 0; f < 12; ++f) {
        std::vector<Object> objs;
        // Object A drifts right; Object B drifts right+down. Both high score.
        objs.push_back({Vec4{10.0 + f * 5, 20.0, 50.0 + f * 5, 80.0}, 0.90, 0});
        objs.push_back({Vec4{100.0 + f * 3, 50.0 + f * 2, 140.0 + f * 3, 110.0 + f * 2}, 0.85, 1});

        const auto t0 = std::chrono::steady_clock::now();
        const auto out = tracker.update(objs);
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::printf("frame %2d: %zu tracks (%.3f ms): ", f, out.size(), ms);
        for (const auto& t : out) {
            const Vec4 w = t->tlwh();
            std::printf("[id=%d cls=%d %.1f,%.1f,%.1f,%.1f] ",
                        t->track_id, t->label, w[0], w[1], w[2], w[3]);
        }
        std::printf("\n");
    }
    return 0;
}