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

    const Vec3& origin() const { return origin_; }
    const Vec3& forward() const { return forward_; }
    const Vec3& right() const { return right_; }
    const Vec3& trueUp() const { return trueUp_; }
    double tanHalfFovY() const { return tanHalfFovY_; }
    double aspect() const { return aspect_; }

private:
    Vec3 origin_;
    Vec3 forward_;
    Vec3 right_;
    Vec3 trueUp_;
    double tanHalfFovY_ = 1.0;
    double aspect_ = 1.0;
};
