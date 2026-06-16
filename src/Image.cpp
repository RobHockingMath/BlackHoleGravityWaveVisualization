#include "Image.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

// stb is used here only: JPEG panorama load + PNG output.
// Do not also compile a separate stb implementation .cpp into this executable.
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace {
constexpr double PI = 3.141592653589793238462643383279502884;

double clamp01(double x) { return std::clamp(x, 0.0, 1.0); }

double wrap01(double x) {
    x = std::fmod(x, 1.0);
    if (x < 0.0) x += 1.0;
    return x;
}

} // namespace

Image::Image(int width, int height)
    : width_(width), height_(height), pixels_(static_cast<std::size_t>(width) * static_cast<std::size_t>(height)) {}

RGBf mixRGB(const RGBf& a, const RGBf& b, double t) {
    t = clamp01(t);
    return a * (1.0 - t) + b * t;
}

RGB8 makeRGB(double r, double g, double b, double outputGamma) {
    auto toByte = [&](double x) -> std::uint8_t {
        x = clamp01(x);
        if (outputGamma > 0.0 && outputGamma != 1.0) {
            x = std::pow(x, 1.0 / outputGamma);
        }
        return static_cast<std::uint8_t>(std::lround(255.0 * clamp01(x)));
    };
    return RGB8{toByte(r), toByte(g), toByte(b)};
}

RGB8 makeRGB(const RGBf& c, double outputGamma) {
    return makeRGB(c.r, c.g, c.b, outputGamma);
}

void Image::writePNG(const std::string& path) const {
    if (width_ <= 0 || height_ <= 0 || pixels_.empty()) {
        throw std::runtime_error("Cannot write empty image: " + path);
    }
    int strideBytes = width_ * static_cast<int>(sizeof(RGB8));
    int ok = stbi_write_png(path.c_str(), width_, height_, 3, pixels_.data(), strideBytes);
    if (!ok) throw std::runtime_error("Could not write PNG output image: " + path);
}

void Panorama::load(const std::string& path) {
    int w = 0, h = 0, n = 0;
    unsigned char* raw = stbi_load(path.c_str(), &w, &h, &n, 3);
    if (!raw) {
        std::string reason = stbi_failure_reason() ? stbi_failure_reason() : "unknown stb_image failure";
        throw std::runtime_error("Could not load panorama '" + path + "': " + reason);
    }

    width_ = w;
    height_ = h;
    path_ = path;
    pixels_.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            std::size_t i = static_cast<std::size_t>(y * w + x);
            pixels_[i] = RGBf{
                raw[3 * i + 0] / 255.0,
                raw[3 * i + 1] / 255.0,
                raw[3 * i + 2] / 255.0,
            };
        }
    }

    stbi_image_free(raw);
}

RGBf Panorama::atClamped(int x, int y) const {
    if (pixels_.empty()) return RGBf{0.0, 0.0, 0.0};
    x = std::clamp(x, 0, width_ - 1);
    y = std::clamp(y, 0, height_ - 1);
    return pixels_[static_cast<std::size_t>(y * width_ + x)];
}

RGBf Panorama::sampleDirection(const Vec3& dIn, double yawDegrees, double exposure) const {
    if (pixels_.empty() || width_ <= 0 || height_ <= 0) return RGBf{0.0, 0.0, 0.0};

    Vec3 d = normalize(dIn);
    if (length2(d) <= 0.0) return RGBf{0.0, 0.0, 0.0};

    // Standard equirectangular environment map.
    // u wraps around azimuth. v=0 is +z/north/top, v=1 is -z/south/bottom.
    double yaw = yawDegrees / 360.0;
    double u = wrap01(0.5 + std::atan2(d.y, d.x) / (2.0 * PI) + yaw);
    double v = std::acos(std::clamp(d.z, -1.0, 1.0)) / PI;

    double px = u * static_cast<double>(width_);
    double py = v * static_cast<double>(height_ - 1);

    int x0 = static_cast<int>(std::floor(px)) % width_;
    int x1 = (x0 + 1) % width_;
    int y0 = std::clamp(static_cast<int>(std::floor(py)), 0, height_ - 1);
    int y1 = std::clamp(y0 + 1, 0, height_ - 1);

    double tx = px - std::floor(px);
    double ty = py - std::floor(py);

    RGBf c00 = atClamped(x0, y0);
    RGBf c10 = atClamped(x1, y0);
    RGBf c01 = atClamped(x0, y1);
    RGBf c11 = atClamped(x1, y1);

    RGBf c0 = mixRGB(c00, c10, tx);
    RGBf c1 = mixRGB(c01, c11, tx);
    RGBf c = mixRGB(c0, c1, ty);
    return c * exposure;
}
