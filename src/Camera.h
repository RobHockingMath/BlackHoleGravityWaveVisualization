#pragma once

#include "Vec3.h"

struct Ray {
    Vec3 origin;
    Vec3 dir;
};

class Camera {
public:
    Camera() = default;
    Camera(const Vec3& pos, const Vec3& lookAt, const Vec3& up, double fovYDegrees, double aspect);

    Ray generateRay(double pixelX, double pixelY, int width, int height) const;

private:
    Vec3 origin_;
    Vec3 forward_;
    Vec3 right_;
    Vec3 trueUp_;
    double tanHalfFovY_ = 1.0;
    double aspect_ = 1.0;
};
