#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include "Vec3.h"

struct RGB8 {
    std::uint8_t r = 0;
    std::uint8_t g = 0;
    std::uint8_t b = 0;
};

struct RGBf {
    double r = 0.0;
    double g = 0.0;
    double b = 0.0;
};

inline RGBf operator+(const RGBf& a, const RGBf& b) { return RGBf{a.r + b.r, a.g + b.g, a.b + b.b}; }
inline RGBf operator-(const RGBf& a, const RGBf& b) { return RGBf{a.r - b.r, a.g - b.g, a.b - b.b}; }
inline RGBf operator*(const RGBf& a, double s) { return RGBf{a.r * s, a.g * s, a.b * s}; }
inline RGBf operator*(double s, const RGBf& a) { return a * s; }
inline RGBf operator*(const RGBf& a, const RGBf& b) { return RGBf{a.r * b.r, a.g * b.g, a.b * b.b}; }
inline RGBf& operator+=(RGBf& a, const RGBf& b) { a.r += b.r; a.g += b.g; a.b += b.b; return a; }

RGBf mixRGB(const RGBf& a, const RGBf& b, double t);
RGB8 makeRGB(double r, double g, double b, double outputGamma = 1.0);
RGB8 makeRGB(const RGBf& c, double outputGamma = 1.0);

class Image {
public:
    Image() = default;
    Image(int width, int height);

    int width() const { return width_; }
    int height() const { return height_; }

    RGB8& at(int x, int y) { return pixels_[y * width_ + x]; }
    const RGB8& at(int x, int y) const { return pixels_[y * width_ + x]; }

    void writePNG(const std::string& path) const;

private:
    int width_ = 0;
    int height_ = 0;
    std::vector<RGB8> pixels_;
};

class Panorama {
public:
    Panorama() = default;
    explicit Panorama(const std::string& path) { load(path); }

    void load(const std::string& path);
    bool empty() const { return pixels_.empty(); }
    RGBf sampleDirection(const Vec3& dir, double yawDegrees = 0.0, double exposure = 1.0) const;

    int width() const { return width_; }
    int height() const { return height_; }
    const std::string& path() const { return path_; }
    const std::vector<RGBf>& pixels() const { return pixels_; }

private:
    int width_ = 0;
    int height_ = 0;
    std::string path_;
    std::vector<RGBf> pixels_;

    RGBf atClamped(int x, int y) const;
};
