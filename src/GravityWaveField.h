#pragma once

#include <complex>
#include "ModeData.h"
#include "Params.h"
#include "Vec3.h"

enum class ModeTimeInterpolation {
    Linear,
    CubicHermite
};

struct FieldRaySample {
    std::complex<double> value = {0.0, 0.0};
    std::complex<double> dValueDs = {0.0, 0.0};
};

struct CoefficientRadialSample {
    std::complex<double> value = {0.0, 0.0};
    std::complex<double> dValueDr = {0.0, 0.0};
};

class GravityWaveField {
public:
    GravityWaveField() = default;
    GravityWaveField(ModeDataSet modes, Params params,
                     ModeTimeInterpolation interpolation = ModeTimeInterpolation::CubicHermite);

    std::complex<double> eval(const Vec3& x, double tCoord) const;

    // Analytic derivative of the same reconstructed/interpolated field used by eval().
    // rayDir does not have to be normalized; dValueDs is with respect to x(s)=x+s*rayDir.
    FieldRaySample evalWithRayDerivative(const Vec3& x, const Vec3& rayDir, double tCoord) const;

    double suggestedTime() const { return modes_.suggestedPeakTime(); }
    const ModeDataSet& modes() const { return modes_; }
    ModeTimeInterpolation interpolation() const { return interpolation_; }
    const char* interpolationName() const;

private:
    ModeDataSet modes_;
    Params params_;
    ModeTimeInterpolation interpolation_ = ModeTimeInterpolation::CubicHermite;

    ModeInterpolation interpolateModeAtTime(const ModeSeries& mode, double t) const;

    std::complex<double> coefficientAtRadiusTime(const ModeKey& key, double r, double t) const;
    CoefficientRadialSample coefficientAtRadiusTimeWithRadialDerivative(const ModeKey& key, double r, double t) const;
};
