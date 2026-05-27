#include "Camera.h"

#include <cmath>
#include <stdexcept>

namespace {
constexpr double PI = 3.141592653589793238462643383279502884;
}

Camera::Camera(const Vec3& pos, const Vec3& lookAt, const Vec3& up, double fovYDegrees, double aspect)
    : origin_(pos), aspect_(aspect)
{
    forward_ = normalize(lookAt - pos);
    if (length2(forward_) <= 0.0) {
        throw std::runtime_error("Bad camera: lookAt equals cameraPos");
    }

    right_ = normalize(cross(forward_, up));
    if (length2(right_) <= 0.0) {
        throw std::runtime_error("Bad camera: up vector is parallel to viewing direction");
    }

    trueUp_ = normalize(cross(right_, forward_));
    tanHalfFovY_ = std::tan(0.5 * fovYDegrees * PI / 180.0);
}

Ray Camera::generateRay(double pixelX, double pixelY, int width, int height) const {
    double u = (pixelX + 0.5) / static_cast<double>(width);
    double v = (pixelY + 0.5) / static_cast<double>(height);

    double sx = (2.0 * u - 1.0) * aspect_ * tanHalfFovY_;
    double sy = (1.0 - 2.0 * v) * tanHalfFovY_;

    Vec3 dir = normalize(forward_ + sx * right_ + sy * trueUp_);
    return Ray{origin_, dir};
}
