#pragma once

#include "GravityWaveField.h"
#include "Image.h"
#include "Params.h"

struct Scene {
    GravityWaveField field;
    Panorama panorama;
};

Scene buildScene(const Params& params);
