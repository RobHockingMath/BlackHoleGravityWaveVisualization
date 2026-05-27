#pragma once

#include <complex>
#include <string>
#include <vector>
#include "Params.h"

struct TimeSample {
    double t = 0.0;
    std::complex<double> y = {0.0, 0.0};
};

struct ModeSeries {
    ModeKey key;
    std::vector<TimeSample> samples;
};

struct RadiusShell {
    double radius = 0.0;
    std::vector<ModeSeries> modes;

    const ModeSeries* findMode(const ModeKey& key) const;
};

class ModeDataSet {
public:
    std::vector<RadiusShell> shells;

    static ModeDataSet loadText(const std::string& path);

    void filterModes(const std::vector<ModeKey>& activeModes);
    void filterExtractionSpheres(const Params& params);
    void autoNormalizeByMaxAbs();

    bool empty() const { return shells.empty(); }
    double timeMin() const;
    double timeMax() const;
    double suggestedPeakTime() const;

    std::vector<ModeKey> modeKeys() const;
    std::vector<double> radii() const;
};

struct ModeInterpolation {
    std::complex<double> value = {0.0, 0.0};
    std::complex<double> dt = {0.0, 0.0};  // derivative with respect to interpolation time t
};

// Legacy value-only linear interpolation. Kept so existing code still compiles.
std::complex<double> interpolateMode(const ModeSeries& mode, double t);

// Value + analytic derivative of the chosen interpolation model.
ModeInterpolation interpolateModeLinearWithDerivative(const ModeSeries& mode, double t);
ModeInterpolation interpolateModeCubicHermite(const ModeSeries& mode, double t);

std::string modeName(const ModeKey& key);
