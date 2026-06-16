#pragma once

#include "BuildScene.h"
#include "Camera.h"
#include "Image.h"
#include "Params.h"

Image renderCuda(const Camera& camera,
                 const Params& params,
                 const Scene& scene);
