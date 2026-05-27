#include "BuildScene.h"
#include "Camera.h"
#include "Image.h"
#include "Params.h"
#include "Renderer.h"
#include "Vec3.h"

#include <cmath>
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
// Hardcoded shot/render control panel.
// Edit these values directly. No command-line parameters.
// -----------------------------------------------------------------------------

constexpr bool RENDER_ANIMATION = true;

// Still-frame settings, used only if RENDER_ANIMATION is false.
constexpr double STILL_TIME = 1415.8;
const std::string STILL_OUTPUT = "gravity_wave_volume.png";

// Animation settings.
constexpr int NUM_FRAMES = 600;
//constexpr int NUM_FRAMES = 1200;
constexpr double ANIM_START_TIME = 700-1*192-0*100;//1350.0-150.0; // 250 becomes 500
constexpr double ANIM_END_TIME = 700+600-1*192-0*100;//1600.0+100.0;
const std::string FRAME_DIR = "frames_r214_colorfalloff9";//"frames_othereye";
const std::string FRAME_PATTERN = "frame_%06d.png";

// Camera settings. This is deliberately in main, because this is the shot.
constexpr double CAMERA_ORBIT_DEGREES_START = 0.0-0*6*1.0;
constexpr double CAMERA_ORBIT_DEGREES_END = 30.0-0*6*1.0;  // Set nonzero, e.g. 30, to animate camera orbit.
constexpr double CAMERA_RADIUS_X = 220.0*0.5;
constexpr double CAMERA_RADIUS_Y = -320.0*0.5;
constexpr double CAMERA_Z = 180.0*0.5;
constexpr double CAMERA_FOV_Y_DEGREES = 90.0;

const Vec3 CAMERA_LOOK_AT(0.0, 0.0, 0.0);
const Vec3 CAMERA_UP(0.0, 0.0, 1.0);

double lerp(double a, double b, double t) {
    return (1.0 - t) * a + t * b;
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

Camera makeShotCamera(double frame01, const Params& params) {
    double orbitDegrees = lerp(CAMERA_ORBIT_DEGREES_START, CAMERA_ORBIT_DEGREES_END, frame01);
    Vec3 cameraPos = rotatedCameraPosition(orbitDegrees);
    double aspect = static_cast<double>(params.width) / static_cast<double>(params.height);
    return Camera(cameraPos, CAMERA_LOOK_AT, CAMERA_UP, CAMERA_FOV_Y_DEGREES, aspect);
}

std::string formatFrameFilename(const std::string& pattern, int frameIndex) {
    std::vector<char> buf(1024);
    int n = std::snprintf(buf.data(), buf.size(), pattern.c_str(), frameIndex);
    if (n < 0) throw std::runtime_error("Failed to format animation filename pattern");
    if (static_cast<std::size_t>(n) >= buf.size()) {
        buf.resize(static_cast<std::size_t>(n) + 1);
        n = std::snprintf(buf.data(), buf.size(), pattern.c_str(), frameIndex);
        if (n < 0) throw std::runtime_error("Failed to format animation filename pattern");
    }
    return std::string(buf.data());
}

void printShotInfo() {
    std::cerr << "Shot:\n";
    std::cerr << "  RENDER_ANIMATION=" << (RENDER_ANIMATION ? "true" : "false") << "\n";
    std::cerr << "  still time=" << STILL_TIME << " output=" << STILL_OUTPUT << "\n";
    std::cerr << "  animation: frames=" << NUM_FRAMES
              << " start=" << ANIM_START_TIME
              << " end=" << ANIM_END_TIME
              << " dir=" << FRAME_DIR
              << " pattern=" << FRAME_PATTERN << "\n";
    std::cerr << "  camera: base=(" << CAMERA_RADIUS_X << ", " << CAMERA_RADIUS_Y << ", " << CAMERA_Z << ")"
              << " orbitStart=" << CAMERA_ORBIT_DEGREES_START
              << " orbitEnd=" << CAMERA_ORBIT_DEGREES_END
              << " lookAt=" << CAMERA_LOOK_AT
              << " up=" << CAMERA_UP
              << " fovY=" << CAMERA_FOV_Y_DEGREES << "\n";
}

} // namespace

int main() {
    try {
        Params baseParams;
        printParams(baseParams);
        printShotInfo();

        Scene scene = buildScene(baseParams);

        if (!RENDER_ANIMATION) {
            Params frameParams = baseParams;
            frameParams.time = STILL_TIME;
            frameParams.outputPath = STILL_OUTPUT;

            Camera camera = makeShotCamera(0.0, frameParams);
            Renderer renderer(frameParams, scene);
            Image image = renderer.render(camera);
            image.writePNG(frameParams.outputPath);

            std::cerr << "Wrote PNG: " << frameParams.outputPath << "\n";
            return 0;
        }

        fs::create_directories(FRAME_DIR);

        for (int frame = 0; frame < NUM_FRAMES; ++frame) {
            double a = (NUM_FRAMES == 1) ? 0.0 : static_cast<double>(frame) / static_cast<double>(NUM_FRAMES - 1);

            Params frameParams = baseParams;
            frameParams.time = lerp(ANIM_START_TIME, ANIM_END_TIME, a);
            frameParams.outputPath = (fs::path(FRAME_DIR) / formatFrameFilename(FRAME_PATTERN, frame)).string();

            Camera camera = makeShotCamera(a, frameParams);

            std::cerr << "\n=== Frame " << (frame + 1) << " / " << NUM_FRAMES
                      << "   a=" << a
                      << "   time=" << frameParams.time
                      << "   out=" << frameParams.outputPath << " ===\n";

            Renderer renderer(frameParams, scene);
            Image image = renderer.render(camera);
            image.writePNG(frameParams.outputPath);
            std::cerr << "Wrote PNG: " << frameParams.outputPath << "\n";
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
