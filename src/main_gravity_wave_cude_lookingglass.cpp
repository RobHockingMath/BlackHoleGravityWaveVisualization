#include "BuildScene.h"
#include "Camera.h"
#include "Image.h"
#include "Params.h"
#include "RendererCuda.h"
#include "Vec3.h"

#include <cmath>
#include <chrono>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {
constexpr double PI = 3.141592653589793238462643383279502884;

// -----------------------------------------------------------------------------
// Hardcoded Looking Glass / light-field shot control panel.
// Edit these values directly. No command-line parameters.
// -----------------------------------------------------------------------------

constexpr bool RENDER_ANIMATION = true;

// Still-frame settings, used only if RENDER_ANIMATION is false.
constexpr double STILL_TIME = 1415.8;

// Animation settings.
constexpr int NUM_FRAMES = 600;
constexpr double ANIM_START_TIME = 700 - 1 * 192 - 0 * 100;
constexpr double ANIM_END_TIME   = 700 + 600 - 1 * 192 - 0 * 100;

// Looking Glass / arc settings.
// N_LK is the number of views rendered at each time step.
// DTHETA_LK_DEGREES is the total angular camera arc. Each time step renders
// from -DTHETA_LK_DEGREES/2 to +DTHETA_LK_DEGREES/2 around the nominal camera.
constexpr int N_LK = 48;
constexpr double DTHETA_LK_DEGREES = 60.0;

// Output naming: one directory, filenames encode both time index and view index.
const std::string FRAME_DIR = "frames_cuda_r214_colorfalloff10_lookingglass";
const std::string FRAME_PATTERN = "time_%06d_view_%03d.png";

// Camera settings. This is deliberately in main, because this is the shot.
constexpr double CAMERA_ORBIT_DEGREES_START = 0.0 - 0 * 6 * 1.0;
constexpr double CAMERA_ORBIT_DEGREES_END   = 30.0 - 0 * 6 * 1.0;
constexpr double CAMERA_RADIUS_X = 220.0 * 0.5;
constexpr double CAMERA_RADIUS_Y = -320.0 * 0.5;
constexpr double CAMERA_Z = 180.0 * 0.5;
constexpr double CAMERA_FOV_Y_DEGREES = 90.0;

const Vec3 CAMERA_LOOK_AT(0.0, 0.0, 0.0);
const Vec3 CAMERA_UP(0.0, 0.0, 1.0);

double lerp(double a, double b, double t) {
    return (1.0 - t) * a + t * b;
}

double nominalOrbitDegrees(double frame01) {
    return lerp(CAMERA_ORBIT_DEGREES_START, CAMERA_ORBIT_DEGREES_END, frame01);
}

double lookingGlassViewOffsetDegrees(int viewIndex) {
    if (N_LK <= 1) return 0.0;
    const double view01 = static_cast<double>(viewIndex) / static_cast<double>(N_LK - 1);
    return (view01 - 0.5) * DTHETA_LK_DEGREES;
}

Vec3 rotatedCameraPosition(double orbitDegrees) {
    double theta = orbitDegrees * PI / 180.0;
    double c = std::cos(theta);
    double s = std::sin(theta);

    return Vec3(
        CAMERA_RADIUS_X * c + CAMERA_RADIUS_Y * s,
       -CAMERA_RADIUS_X * s + CAMERA_RADIUS_Y * c,
        CAMERA_Z
    );
}

Camera makeShotCameraAtOrbit(double orbitDegrees, const Params& params) {
    Vec3 cameraPos = rotatedCameraPosition(orbitDegrees);
    double aspect = static_cast<double>(params.width) / static_cast<double>(params.height);
    return Camera(cameraPos, CAMERA_LOOK_AT, CAMERA_UP, CAMERA_FOV_Y_DEGREES, aspect);
}

Camera makeLookingGlassCamera(double frame01, int viewIndex, const Params& params) {
    const double nominalOrbit = nominalOrbitDegrees(frame01);
    const double viewOffset = lookingGlassViewOffsetDegrees(viewIndex);
    return makeShotCameraAtOrbit(nominalOrbit + viewOffset, params);
}

std::string formatLookingGlassFrameFilename(const std::string& pattern, int timeIndex, int viewIndex) {
    std::vector<char> buf(1024);
    int n = std::snprintf(buf.data(), buf.size(), pattern.c_str(), timeIndex, viewIndex);
    if (n < 0) throw std::runtime_error("Failed to format Looking Glass frame filename pattern");
    if (static_cast<std::size_t>(n) >= buf.size()) {
        buf.resize(static_cast<std::size_t>(n) + 1);
        n = std::snprintf(buf.data(), buf.size(), pattern.c_str(), timeIndex, viewIndex);
        if (n < 0) throw std::runtime_error("Failed to format Looking Glass frame filename pattern");
    }
    return std::string(buf.data());
}

void printShotInfo() {
    std::cerr << "Looking Glass shot:\n";
    std::cerr << "  RENDER_ANIMATION=" << (RENDER_ANIMATION ? "true" : "false") << "\n";
    std::cerr << "  still time=" << STILL_TIME << "\n";
    std::cerr << "  animation: frames=" << NUM_FRAMES
              << " startTime=" << ANIM_START_TIME
              << " endTime=" << ANIM_END_TIME
              << " dir=" << FRAME_DIR
              << " pattern=" << FRAME_PATTERN << "\n";
    std::cerr << "  looking glass: N_LK=" << N_LK
              << " totalArcDegrees=" << DTHETA_LK_DEGREES
              << " halfArcDegrees=" << (0.5 * DTHETA_LK_DEGREES)
              << " dir=" << FRAME_DIR
              << " pattern=" << FRAME_PATTERN << "\n";
    std::cerr << "  camera: base=(" << CAMERA_RADIUS_X << ", " << CAMERA_RADIUS_Y << ", " << CAMERA_Z << ")"
              << " orbitStart=" << CAMERA_ORBIT_DEGREES_START
              << " orbitEnd=" << CAMERA_ORBIT_DEGREES_END
              << " lookAt=" << CAMERA_LOOK_AT
              << " up=" << CAMERA_UP
              << " fovY=" << CAMERA_FOV_Y_DEGREES << "\n";
}

void printCudaFrameTiming(const Params& frameParams, double seconds) {
    const double pixels = static_cast<double>(frameParams.width) * static_cast<double>(frameParams.height);

    int root = static_cast<int>(std::floor(std::sqrt(static_cast<double>(frameParams.samplesPerPixel))));
    if (root < 1) root = 1;

    const int actualSpp = (frameParams.samplesPerPixel <= 1) ? 1 : root * root;
    const double sampleRays = pixels * static_cast<double>(actualSpp);

    std::cerr << "CUDA view render time: " << seconds << " s"
              << " | mode=" << frameParams.renderMode
              << " | image=" << frameParams.width << "x" << frameParams.height
              << " | spp=" << actualSpp;

    if (seconds > 0.0) {
        std::cerr << " | " << (pixels / (1.0e6 * seconds)) << " Mpix/s"
                  << " | " << (sampleRays / (1.0e6 * seconds)) << " Mrays/s";
    }

    std::cerr << "\n";
}

void renderOneLookingGlassView(const Scene& scene,
                               const Params& frameParams,
                               double frame01,
                               int timeIndex,
                               int timeSerial,
                               int timeCount,
                               int viewIndex) {
    const double nominalOrbit = nominalOrbitDegrees(frame01);
    const double viewOffset = lookingGlassViewOffsetDegrees(viewIndex);
    const double viewOrbit = nominalOrbit + viewOffset;

    Camera camera = makeLookingGlassCamera(frame01, viewIndex, frameParams);

    std::cerr << "--- LK view " << (viewIndex + 1) << " / " << N_LK
              << " | timeSerial=" << timeSerial << "/" << timeCount
              << " | timeIndex=" << timeIndex
              << " | viewIndex=" << viewIndex
              << " | offsetDeg=" << viewOffset
              << " | orbitDeg=" << viewOrbit
              << " | out=" << frameParams.outputPath << " ---\n";

    const auto renderStart = std::chrono::steady_clock::now();
    Image image = renderCuda(camera, frameParams, scene);
    const auto renderEnd = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(renderEnd - renderStart).count();
    printCudaFrameTiming(frameParams, seconds);

    image.writePNG(frameParams.outputPath);
    std::cerr << "Wrote PNG: " << frameParams.outputPath << "\n";
}

} // namespace

int main() {
    try {
        Params baseParams;

        // CUDA renderer: fixed-step gwpv_peaks.  metricLensingEnabled is
        // controlled by Params.h: false gives the straight-ray CUDA path;
        // true gives MP-only fixed-RK4 lensing.
        baseParams.renderMode = "gwpv_peaks";
        baseParams.gwpvAdaptiveEnabled = false;
        baseParams.gwpvWavelengthStepScalingEnabled = false;
        baseParams.gwpvWavelengthCompEnabled = false;
        baseParams.gwpvWavelengthPeakWidthEnabled = false;
        baseParams.gwpvOpacityGainEnabled = false;

        printParams(baseParams);
        printShotInfo();

        Scene scene = buildScene(baseParams);

        fs::create_directories(FRAME_DIR);

        if (!RENDER_ANIMATION) {
            Params frameParams = baseParams;
            frameParams.time = STILL_TIME;

            const double frame01 = 0.0;
            const int timeIndex = 0;
            const int timeSerial = 1;
            const int timeCount = 1;

            std::cerr << "\n=== Still Looking Glass time"
                      << " | time=" << frameParams.time
                      << " | N_LK=" << N_LK << " ===\n";

            for (int view = 0; view < N_LK; ++view) {
                Params viewParams = frameParams;
                viewParams.outputPath = (fs::path(FRAME_DIR) /
                    formatLookingGlassFrameFilename(FRAME_PATTERN, timeIndex, view)).string();

                renderOneLookingGlassView(scene, viewParams, frame01, timeIndex, timeSerial, timeCount, view);
            }

            return 0;
        }

        const int timeCount = NUM_FRAMES;

        for (int frame = 0; frame < NUM_FRAMES; ++frame) {
            const int timeSerial = frame + 1;
            const double a = (NUM_FRAMES == 1)
                ? 0.0
                : static_cast<double>(frame) / static_cast<double>(NUM_FRAMES - 1);

            Params frameParams = baseParams;
            frameParams.time = lerp(ANIM_START_TIME, ANIM_END_TIME, a);

            const double nominalOrbit = nominalOrbitDegrees(a);

            std::cerr << "\n=== Looking Glass time step " << timeSerial << " / " << timeCount
                      << " | timeIndex=" << frame
                      << " | globalFrame=" << (frame + 1) << " / " << NUM_FRAMES
                      << " | a=" << a
                      << " | time=" << frameParams.time
                      << " | nominalOrbitDeg=" << nominalOrbit
                      << " | views=" << N_LK
                      << " | arcDeg=" << DTHETA_LK_DEGREES
                      << " ===\n";

            const auto timeStepStart = std::chrono::steady_clock::now();

            for (int view = 0; view < N_LK; ++view) {
                Params viewParams = frameParams;
                viewParams.outputPath = (fs::path(FRAME_DIR) /
                    formatLookingGlassFrameFilename(FRAME_PATTERN, frame, view)).string();

                renderOneLookingGlassView(scene, viewParams, a, frame, timeSerial, timeCount, view);
            }

            const auto timeStepEnd = std::chrono::steady_clock::now();
            const double timeStepSeconds = std::chrono::duration<double>(timeStepEnd - timeStepStart).count();

            std::cerr << "=== Finished Looking Glass time step " << timeSerial << " / " << timeCount
                      << " | timeIndex=" << frame
                      << " | views=" << N_LK
                      << " | totalWallTime=" << timeStepSeconds << " s"
                      << " | avgPerView=" << (N_LK > 0 ? timeStepSeconds / static_cast<double>(N_LK) : 0.0)
                      << " s ===\n";
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
