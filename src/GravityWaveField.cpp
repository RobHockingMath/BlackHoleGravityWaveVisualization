#include "GravityWaveField.h"
#include "SpinWeightedHarmonic.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

GravityWaveField::GravityWaveField(ModeDataSet modes, Params params, ModeTimeInterpolation interpolation)
    : modes_(std::move(modes)), params_(std::move(params)), interpolation_(interpolation) {}

const char* GravityWaveField::interpolationName() const {
    switch (interpolation_) {
        case ModeTimeInterpolation::Linear: return "linear";
        case ModeTimeInterpolation::CubicHermite: return "cubic Hermite";
    }
    return "unknown";
}

ModeInterpolation GravityWaveField::interpolateModeAtTime(const ModeSeries& mode, double t) const {
    switch (interpolation_) {
        case ModeTimeInterpolation::Linear:
            return interpolateModeLinearWithDerivative(mode, t);
        case ModeTimeInterpolation::CubicHermite:
            return interpolateModeCubicHermite(mode, t);
    }
    return interpolateModeCubicHermite(mode, t);
}

namespace {

double tortoiseRadius(double r, const Params& p) {
    if (!p.useTortoiseRetardedTime) return r;

    const double M = std::max(0.0, p.tortoiseMass);
    if (!(M > 0.0)) return r;

    const double horizon = 2.0 * M;
    const double floorFromHorizon = horizon * (1.0 + std::max(0.0, p.tortoiseSafetyEps));
    const double floorRadius = std::max({p.tortoiseRadiusFloor, p.innerWaveScaleRadius, floorFromHorizon});
    const double rr = std::max(r, floorRadius);

    // Schwarzschild tortoise radius: r* = r + 2M log(r/(2M)-1).
    return rr + 2.0 * M * std::log(rr / horizon - 1.0);
}

double tortoiseRadiusDerivative(double r, const Params& p) {
    if (!p.useTortoiseRetardedTime) return 1.0;

    const double M = std::max(0.0, p.tortoiseMass);
    if (!(M > 0.0)) return 1.0;

    const double horizon = 2.0 * M;
    const double floorFromHorizon = horizon * (1.0 + std::max(0.0, p.tortoiseSafetyEps));
    const double floorRadius = std::max({p.tortoiseRadiusFloor, p.innerWaveScaleRadius, floorFromHorizon});

    // We clamp r before applying r*, so below the floor the effective retarded
    // coordinate is constant and dr*/dr is zero. This avoids the horizon
    // singularity in the artificial inward continuation region.
    if (r <= floorRadius) return 0.0;

    return 1.0 / std::max(1.0e-12, 1.0 - horizon / r);
}

} // namespace

CoefficientRadialSample GravityWaveField::coefficientAtRadiusTimeWithRadialDerivative(
    const ModeKey& key, double r, double t) const
{
    const auto& shells = modes_.shells;
    if (shells.empty()) return {};

    // Outgoing-wave alignment in units c = 1.
    // Old flat version:
    //     u = t - r,        T_R = u + R
    // Optional tortoise version:
    //     u = t - r*(r),    T_R = u + r*(R)
    const double rStar = tortoiseRadius(r, params_);
    const double drStarDr = tortoiseRadiusDerivative(r, params_);
    const double u = t - rStar;
    const double rScale = std::max(r, params_.innerWaveScaleRadius);
    const double invRScale = (rScale > 0.0) ? 1.0 / rScale : 0.0;
    const double dInvRScaleDr = (r > params_.innerWaveScaleRadius && rScale > 0.0)
                              ? -1.0 / (rScale * rScale)
                              : 0.0;

    auto sampleShellW = [&](const RadiusShell& shell) {
        struct WSample {
            std::complex<double> w = {0.0, 0.0};
            std::complex<double> dwDr = {0.0, 0.0};
        } out;

        const ModeSeries* mode = shell.findMode(key);
        if (!mode) return out;

        const double shellTimeRadius = tortoiseRadius(shell.radius, params_);
        const double tShell = u + shellTimeRadius;
        const ModeInterpolation interp = interpolateModeAtTime(*mode, tShell);

        // w = R * C_R(t - r*(r) + r*(R)), so
        // dw/dr = -R * dC_R/dt * dr*/dr. With the flat setting,
        // dr*/dr=1 and r*(R)=R, reproducing the old formula exactly.
        out.w = shell.radius * interp.value;
        out.dwDr = -shell.radius * interp.dt * drStarDr;
        return out;
    };

    auto finish = [&](std::complex<double> w, std::complex<double> dwDr) {
        CoefficientRadialSample out;
        out.value = w * invRScale;
        out.dValueDr = dwDr * invRScale + w * dInvRScaleDr;
        return out;
    };

    if (shells.size() == 1) {
        const auto s = sampleShellW(shells.front());
        return finish(s.w, s.dwDr);
    }

    if (r <= shells.front().radius) {
        const auto s = sampleShellW(shells.front());
        return finish(s.w, s.dwDr);
    }

    if (r >= shells.back().radius) {
        const auto s = sampleShellW(shells.back());
        return finish(s.w, s.dwDr);
    }

    std::size_t hi = 1;
    while (hi < shells.size() && shells[hi].radius < r) ++hi;
    std::size_t lo = hi - 1;

    const RadiusShell& a = shells[lo];
    const RadiusShell& b = shells[hi];

    const double denom = b.radius - a.radius;
    double frac = (denom > 0.0) ? (r - a.radius) / denom : 0.0;
    frac = std::clamp(frac, 0.0, 1.0);
    const double dFracDr = (denom > 0.0) ? 1.0 / denom : 0.0;

    const auto sa = sampleShellW(a);
    const auto sb = sampleShellW(b);

    const std::complex<double> wLocal = (1.0 - frac) * sa.w + frac * sb.w;

    // d/dr[(1-f)wa + f wb] = (1-f)dwa/dr + f dwb/dr + f'(wb-wa)
    const std::complex<double> dwLocalDr =
        (1.0 - frac) * sa.dwDr + frac * sb.dwDr + dFracDr * (sb.w - sa.w);

    return finish(wLocal, dwLocalDr);
}

std::complex<double> GravityWaveField::coefficientAtRadiusTime(const ModeKey& key, double r, double t) const {
    return coefficientAtRadiusTimeWithRadialDerivative(key, r, t).value;
}

std::complex<double> GravityWaveField::eval(const Vec3& x, double tCoord) const {
    double r = length(x);
    if (r <= params_.rInner) return {0.0, 0.0};

    double theta = std::acos(std::clamp(x.z / std::max(r, 1e-300), -1.0, 1.0));
    double phi = std::atan2(x.y, x.x);

    std::complex<double> field(0.0, 0.0);

    for (const auto& key : modes_.modeKeys()) {
        std::complex<double> c = coefficientAtRadiusTime(key, r, tCoord);
        std::complex<double> ylm = spinWeightedY(key.l, key.m, -2, theta, phi);
        field += c * ylm;
    }

    return field;
}

FieldRaySample GravityWaveField::evalWithRayDerivative(const Vec3& x, const Vec3& rayDir, double tCoord) const {
    const double r = length(x);
    if (r <= params_.rInner) return {};

    const double rho2 = x.x * x.x + x.y * x.y;
    const double rho = std::sqrt(std::max(0.0, rho2));

    const double theta = std::acos(std::clamp(x.z / std::max(r, 1e-300), -1.0, 1.0));
    const double phi = std::atan2(x.y, x.x);

    const double drDs = dot(x, rayDir) / std::max(r, 1e-300);

    double dThetaDs = 0.0;
    double dPhiDs = 0.0;

    // Spherical-coordinate derivatives along x(s)=x+s*rayDir.
    // These are singular on the z-axis.  The renderer already suppresses the
    // visual polar seam with axisMaskFactor, so for the exact axis we use a
    // finite zero fallback rather than manufacturing infinities/NaNs.
    if (rho > 1.0e-12 * std::max(1.0, r)) {
        dThetaDs = (x.z * drDs - r * rayDir.z) / (r * rho);
        dPhiDs = (x.x * rayDir.y - x.y * rayDir.x) / rho2;
    }

    FieldRaySample out;

    for (const auto& key : modes_.modeKeys()) {
        CoefficientRadialSample c = coefficientAtRadiusTimeWithRadialDerivative(key, r, tCoord);
        SpinWeightedYDerivatives y = spinWeightedYWithDerivatives(key.l, key.m, -2, theta, phi);

        out.value += c.value * y.y;

        // c depends on the ray only through r(s), while Y depends on theta(s), phi(s).
        out.dValueDs += (c.dValueDr * drDs) * y.y
                      + c.value * (y.dtheta * dThetaDs + y.dphi * dPhiDs);
    }

    return out;
}
