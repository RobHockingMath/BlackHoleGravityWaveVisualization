#pragma once

#include "BuildScene.h"
#include "Camera.h"
#include "Image.h"
#include "Params.h"

class Renderer {
public:
    Renderer(const Params& params, const Scene& scene);
    Image render(const Camera& camera) const;

private:
    Params params_;
    const Scene& scene_;
    double frameAmplitudeMax_ = 1.0;
    double frameRealMax_ = 1.0;

    RGB8 shadePixel(const Camera& camera, int x, int y) const;
    RGBf shadeRay(const Ray& ray) const;
    double estimateFrameAmplitudeMax() const;
    double estimateFrameRealMax() const;
};
