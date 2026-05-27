#include "BuildScene.h"

#include <iostream>
#include <stdexcept>

Scene buildScene(const Params& params) {
    ModeDataSet modes = ModeDataSet::loadText(params.modeFile);
    modes.filterExtractionSpheres(params);
    modes.filterModes(params.activeModes);

    if (params.autoNormalizeModes) {
        modes.autoNormalizeByMaxAbs();
    }

    std::cerr << "Loaded mode file: " << params.modeFile << "\n";

    std::cerr << "  extraction radii:";
    for (double r : modes.radii()) std::cerr << " " << r;
    std::cerr << "\n";

    std::cerr << "  modes:";
    for (const auto& key : modes.modeKeys()) std::cerr << " " << modeName(key);
    std::cerr << "\n";

    std::cerr << "  common-ish time range: " << modes.timeMin() << " to " << modes.timeMax() << "\n";

    Scene scene;
    scene.field = GravityWaveField(std::move(modes), params);
    scene.panorama.load(params.panoramaPath);

    std::cerr << "Loaded panorama: " << scene.panorama.path()
              << " (" << scene.panorama.width() << " x " << scene.panorama.height() << ")\n";

    return scene;
}
