#include "Renderer.h"
#include "AdaptiveRK.h"
#include "FrozenStrainMetric.h"
#include "BlackHoleSystem.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

#if __has_include("stb_image.h")
#define BLACK_HOLE_HAVE_STB_IMAGE 1
#ifndef STB_IMAGE_STATIC
#define STB_IMAGE_STATIC
#endif
#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#endif
#include "stb_image.h"
#else
#define BLACK_HOLE_HAVE_STB_IMAGE 0
#endif

namespace {
constexpr double PI = 3.141592653589793238462643383279502884;

double clamp01(double x) { return std::clamp(x, 0.0, 1.0); }

double smoothstep(double edge0, double edge1, double x) {
    if (edge1 <= edge0) return (x >= edge1) ? 1.0 : 0.0;
    double t = clamp01((x - edge0) / (edge1 - edge0));
    return t * t * (3.0 - 2.0 * t);
}

double metricGeodesicLocalMaxStep(const Vec3& x, const Params& params) {
    const double hInner = std::max(params.metricGeodesicMaxStepInner, 1.0e-12);
    const double hOuter = std::max(params.metricGeodesicMaxStepOuter, 1.0e-12);

    const double r = length(x);
    const double rInner = params.metricGeodesicMaxStepInnerRadius;
    const double rOuter = params.metricGeodesicMaxStepOuterRadius;

    double u = 0.0;
    if (rOuter > rInner) {
        u = smoothstep(rInner, rOuter, r);
    } else {
        u = (r >= rOuter) ? 1.0 : 0.0;
    }

    return (1.0 - u) * hInner + u * hOuter;
}

double axisMaskFactor(const Vec3& x, const Params& p) {
    if (!p.axisMaskEnabled) return 1.0;

    // Source-frame cylindrical radius around the z-axis.
    // This suppresses the visual contribution of the spin-frame/polar-axis seam
    // without changing the field evaluator itself.
    double rhoPerp = std::sqrt(x.x * x.x + x.y * x.y);
    return smoothstep(p.axisMaskInnerRadius, p.axisMaskOuterRadius, rhoPerp);
}

double hash01(int x, int y, int s, int channel) {
    // Tiny deterministic integer hash for anti-alias jitter.
    std::uint32_t v = static_cast<std::uint32_t>(x) * 1973u
                    ^ static_cast<std::uint32_t>(y) * 9277u
                    ^ static_cast<std::uint32_t>(s) * 26699u
                    ^ static_cast<std::uint32_t>(channel) * 31847u
                    ^ 0x9E3779B9u;
    v ^= v >> 16;
    v *= 0x7feb352du;
    v ^= v >> 15;
    v *= 0x846ca68bu;
    v ^= v >> 16;
    return static_cast<double>(v) / static_cast<double>(0xffffffffu);
}

bool intersectSphere(const Ray& ray, double radius, double& tEnter, double& tExit) {
    // Sphere centered at origin. ray.dir should be normalized.
    double b = dot(ray.origin, ray.dir);
    double c = dot(ray.origin, ray.origin) - radius * radius;
    double disc = b * b - c;
    if (disc < 0.0) return false;
    double s = std::sqrt(disc);
    tEnter = -b - s;
    tExit = -b + s;
    return tExit >= 0.0;
}

Vec3 normalizeVecOr(const Vec3& v, const Vec3& fallback) {
    const double L = std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
    if (!std::isfinite(L) || L <= 1.0e-300) return fallback;
    return (1.0 / L) * v;
}

struct BlackHoleSurfaceHit {
    bool hit = false;
    double t = std::numeric_limits<double>::infinity();
    double u = 0.0; // segment fraction when used
    Vec3 center = Vec3(0.0, 0.0, 0.0);
    double radius = 1.0;
};

bool intersectSphereAt(const Ray& ray,
                       const Vec3& center,
                       double radius,
                       double& tHit)
{
    if (!(radius > 0.0)) return false;

    const Vec3 oc = ray.origin - center;
    const double b = dot(oc, ray.dir);
    const double c = dot(oc, oc) - radius * radius;
    const double disc = b * b - c;
    if (disc < 0.0) return false;

    const double s = std::sqrt(disc);
    const double t0 = -b - s;
    const double t1 = -b + s;

    if (t1 < 0.0) return false;
    tHit = (t0 >= 0.0) ? t0 : t1;
    return std::isfinite(tHit);
}

bool segmentSphereHit(const Vec3& x0,
                      const Vec3& x1,
                      const Vec3& center,
                      double radius,
                      double& uHit)
{
    if (!(radius > 0.0)) return false;

    const Vec3 d = x1 - x0;
    const Vec3 oc = x0 - center;

    const double A = dot(d, d);
    if (!(A > 1.0e-300)) return false;

    const double C = dot(oc, oc) - radius * radius;
    if (C <= 0.0) {
        uHit = 0.0;
        return true;
    }

    const double B = 2.0 * dot(oc, d);
    const double disc = B * B - 4.0 * A * C;
    if (disc < 0.0) return false;

    const double s = std::sqrt(disc);
    const double u0 = (-B - s) / (2.0 * A);
    const double u1 = (-B + s) / (2.0 * A);

    double u = std::numeric_limits<double>::infinity();
    if (u0 >= 0.0 && u0 <= 1.0) u = u0;
    else if (u1 >= 0.0 && u1 <= 1.0) u = u1;

    if (!std::isfinite(u)) return false;
    uHit = std::clamp(u, 0.0, 1.0);
    return true;
}

BlackHoleSurfaceHit nearestBlackHoleRayHit(const Ray& ray,
                                           const BlackHoleSystem::State& bh,
                                           bool useRenderRadius)
{
    BlackHoleSurfaceHit best;
    if (!bh.valid) return best;

    const double plusRadius = useRenderRadius ? bh.plusRenderRadius : bh.plusCaptureRadius;
    const double minusRadius = useRenderRadius ? bh.minusRenderRadius : bh.minusCaptureRadius;

    double t = 0.0;
    if (intersectSphereAt(ray, bh.plusCenter, plusRadius, t) && t < best.t) {
        best.hit = true;
        best.t = t;
        best.center = bh.plusCenter;
        best.radius = plusRadius;
    }
    if (intersectSphereAt(ray, bh.minusCenter, minusRadius, t) && t < best.t) {
        best.hit = true;
        best.t = t;
        best.center = bh.minusCenter;
        best.radius = minusRadius;
    }

    return best;
}

BlackHoleSurfaceHit nearestBlackHoleSegmentHit(const Vec3& x0,
                                               const Vec3& x1,
                                               const BlackHoleSystem::State& bh,
                                               bool useRenderRadius)
{
    BlackHoleSurfaceHit best;
    if (!bh.valid) return best;

    const double plusRadius = useRenderRadius ? bh.plusRenderRadius : bh.plusCaptureRadius;
    const double minusRadius = useRenderRadius ? bh.minusRenderRadius : bh.minusCaptureRadius;

    double u = 0.0;
    if (segmentSphereHit(x0, x1, bh.plusCenter, plusRadius, u) && u < best.u + (!best.hit ? 2.0 : 0.0)) {
        if (!best.hit || u < best.u) {
            best.hit = true;
            best.u = u;
            best.center = bh.plusCenter;
            best.radius = plusRadius;
        }
    }
    if (segmentSphereHit(x0, x1, bh.minusCenter, minusRadius, u) && u < best.u + (!best.hit ? 2.0 : 0.0)) {
        if (!best.hit || u < best.u) {
            best.hit = true;
            best.u = u;
            best.center = bh.minusCenter;
            best.radius = minusRadius;
        }
    }

    return best;
}

struct BlackHoleTexture {
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<unsigned char> pixels;
    bool valid = false;
};

BlackHoleTexture loadBlackHoleTexture(const std::string& path) {
    BlackHoleTexture tex;

#if BLACK_HOLE_HAVE_STB_IMAGE
    int w = 0;
    int h = 0;
    int c = 0;
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &c, 3);
    if (!data) {
        std::cerr << "WARNING: Could not load black-hole texture '" << path
                  << "'. Falling back to pure black spheres.\n";
        return tex;
    }

    tex.width = w;
    tex.height = h;
    tex.channels = 3;
    tex.pixels.assign(data, data + static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 3u);
    tex.valid = (w > 0 && h > 0 && !tex.pixels.empty());
    stbi_image_free(data);

    if (tex.valid) {
        std::cerr << "Loaded black-hole texture '" << path << "' ("
                  << tex.width << " x " << tex.height << ").\n";
    }
#else
    (void)path;
#endif

    return tex;
}

const BlackHoleTexture& getBlackHoleTexture(const Params& params) {
    static std::once_flag once;
    static BlackHoleTexture tex;

    std::call_once(once, [&]() {
        tex = loadBlackHoleTexture(params.blackHoleTexturePath);
    });

    return tex;
}

RGBf sampleBlackHoleTexture(const Params& params, const Vec3& outwardNormal) {
    const BlackHoleTexture& tex = getBlackHoleTexture(params);
    if (!tex.valid) {
        return RGBf{0.0, 0.0, 0.0};
    }

    const Vec3 n = normalizeVecOr(outwardNormal, Vec3(0.0, 0.0, 1.0));

    double u = 0.5 + std::atan2(n.y, n.x) / (2.0 * PI);
    double v = 0.5 - std::asin(std::clamp(n.z, -1.0, 1.0)) / PI;

    u = u - std::floor(u);
    v = std::clamp(v, 0.0, 1.0);

    const int ix = std::clamp(static_cast<int>(u * static_cast<double>(tex.width)), 0, tex.width - 1);
    const int iy = std::clamp(static_cast<int>(v * static_cast<double>(tex.height)), 0, tex.height - 1);
    const std::size_t idx = (static_cast<std::size_t>(iy) * static_cast<std::size_t>(tex.width)
                           + static_cast<std::size_t>(ix)) * 3u;

    const double inv255 = 1.0 / 255.0;
    return RGBf{
        tex.pixels[idx + 0] * inv255,
        tex.pixels[idx + 1] * inv255,
        tex.pixels[idx + 2] * inv255
    };
}

RGBf shadeBlackHoleSurface(const Params& params,
                           const BlackHoleSurfaceHit& hit,
                           const Vec3& xHit)
{
    const Vec3 n = normalizeVecOr(xHit - hit.center, Vec3(0.0, 0.0, 1.0));
    return sampleBlackHoleTexture(params, n);
}


Vec3 safeNormalizeVec(const Vec3& v, const Vec3& fallback) {
    const double L = length(v);
    if (!std::isfinite(L) || L <= 1.0e-300) return fallback;
    return (1.0 / L) * v;
}

RGBf hsvToRgb(double h, double s, double v) {
    h = h - std::floor(h);
    s = clamp01(s);
    v = std::max(0.0, v);

    double x = h * 6.0;
    int i = static_cast<int>(std::floor(x));
    double f = x - i;
    double p = v * (1.0 - s);
    double q = v * (1.0 - s * f);
    double t = v * (1.0 - s * (1.0 - f));

    switch (i % 6) {
        case 0: return RGBf{v, t, p};
        case 1: return RGBf{q, v, p};
        case 2: return RGBf{p, v, t};
        case 3: return RGBf{p, q, v};
        case 4: return RGBf{t, p, v};
        default: return RGBf{v, p, q};
    }
}

RGBf localWaveColor(const std::complex<double>& psi, const Params& p) {
    double amp = std::abs(psi);
    if (p.colorMode == "white") {
        return RGBf{1.0, 0.92, 0.82} * p.waveBrightness;
    }

    if (p.colorMode == "phase") {
        double phase = std::atan2(std::imag(psi), std::real(psi));
        double hue = phase / (2.0 * PI) + 0.5;
        RGBf c = hsvToRgb(hue, 0.82 * p.colorSaturation, 1.0);
        return c * p.waveBrightness;
    }

    double component = (p.colorMode == "signed_imag") ? std::imag(psi) : std::real(psi);
    double s = (amp > 1e-14) ? std::clamp(component / amp, -1.0, 1.0) : 0.0;
    double k = std::pow(std::abs(s), std::max(0.0, p.colorSaturation));

    // Neutral is deliberately not pure white; it keeps weak phase regions from looking like flat fog.
    const RGBf neutral{0.78, 0.68, 0.92};
    const RGBf positive{1.00, 0.22, 0.55}; // pink/red
    const RGBf negative{0.36, 0.22, 1.00}; // violet/blue

    RGBf c = (s >= 0.0) ? mixRGB(neutral, positive, k) : mixRGB(neutral, negative, k);
    return c * p.waveBrightness;
}

double localDensityFromMaskedAmplitude(double maskedAmp, const Params& p, double frameMax) {
    double ampNorm = maskedAmp;

    if (p.normalizeAmplitudePerFrame && frameMax > 1e-15) {
        ampNorm /= frameMax;
    }

    ampNorm = std::max(0.0, ampNorm);

    // Simple normalized-amplitude cutoff:
    //   below ampCutoff: invisible
    //   above ampCutoff: linearly ramp to full normalized amplitude at 1
    double cutoff = std::clamp(p.ampCutoff, 0.0, 0.999999);

    if (ampNorm <= cutoff) {
        return 0.0;
    }

    double t = (ampNorm - cutoff) / (1.0 - cutoff);
    return p.densityScale * t;
}


RGBf paraviewPeakColor(double scalar, const Params& params) {
    const double a = params.paraviewPeaksFirstPosition;
    const double b = params.paraviewPeaksLastPosition;

    double u = 0.0;
    if (b > a) {
        u = std::clamp((scalar - a) / (b - a), 0.0, 1.0);
    }

    // Simple rainbow: blue/violet at low scalar, red at high scalar.
    double hue = (2.0 / 3.0) * (1.0 - u);
    return hsvToRgb(hue, 0.95, 1.0) * params.waveBrightness;
}

double paraviewEffectivePeakSigma(const Params& params) {
    const int n = std::max(1, params.paraviewPeaksNumPeaks);
    const double firstPos = params.paraviewPeaksFirstPosition;
    const double lastPos = params.paraviewPeaksLastPosition;

    double sigma = params.paraviewPeaksSigma;
    if (!(sigma > 0.0)) {
        const double spacing = (n > 1 && lastPos > firstPos)
            ? (lastPos - firstPos) / static_cast<double>(n - 1)
            : 0.01;
        sigma = 0.35 * spacing;
    }
    return std::max(1.0e-8, sigma);
}

bool paraviewUsesRPsi4Scalar(const Params& params) {
    return params.paraviewScalarRadialMode == "r_psi4";
}

bool paraviewUsesPsi4Scalar(const Params& params) {
    return params.paraviewScalarRadialMode == "psi4";
}

double paraviewDisplayRadius(double r, const Params& params) {
    // GravityWaveField returns a Psi4-like field with an inward 1/r cap based
    // on innerWaveScaleRadius.  Multiplying by this same display radius gives
    // a radius-rescaled scalar without reintroducing a center singularity.
    return std::max(r, params.innerWaveScaleRadius);
}

double paraviewRawDisplayScalar(const Vec3& x,
                                const std::complex<double>& psi,
                                const Params& params)
{
    const double re = std::real(psi);
    if (paraviewUsesRPsi4Scalar(params)) {
        const double r = length(x);
        return paraviewDisplayRadius(r, params) * re;
    }
    return re;
}

struct ParaviewScalarRaySample {
    double raw = 0.0;
    double dRawDs = 0.0;
};

ParaviewScalarRaySample paraviewRawDisplayScalarWithDerivative(
    const Vec3& x,
    const Vec3& rayDir,
    const FieldRaySample& sample,
    const Params& params)
{
    const double re = std::real(sample.value);
    const double dreDs = std::real(sample.dValueDs);

    if (paraviewUsesRPsi4Scalar(params)) {
        const double r = length(x);
        const double rDisplay = paraviewDisplayRadius(r, params);

        double drDisplayDs = 0.0;
        if (r > std::max(params.innerWaveScaleRadius, 1.0e-12)) {
            drDisplayDs = dot(x, rayDir) / r;
        }

        return ParaviewScalarRaySample{
            rDisplay * re,
            rDisplay * dreDs + drDisplayDs * re
        };
    }

    return ParaviewScalarRaySample{re, dreDs};
}

double paraviewOpacityRadialEnvelope(double r, const Params& params) {
    double envelope = 1.0;

    if (params.paraviewOpacityRadialFalloffEnabled) {
        const double ref = std::max(1.0e-9, params.paraviewOpacityReferenceRadius);
        const double rr = std::max(r, ref);
        const double power = std::max(0.0, params.paraviewOpacityFalloffPower);
        envelope *= std::pow(ref / rr, power);
    }

    if (params.paraviewOuterFadeWidth > 0.0) {
        const double R = std::max(1.0e-9, params.waveVolumeRadius);
        const double w = std::min(params.paraviewOuterFadeWidth, R);
        envelope *= 1.0 - smoothstep(R - w, R, r);
    }

    return clamp01(envelope);
}

double distanceToNearestParaviewPeakBand(double scalar,
                                         const Params& params,
                                         double bandSigmas)
{
    const int n = std::max(1, params.paraviewPeaksNumPeaks);
    const double firstPos = params.paraviewPeaksFirstPosition;
    const double lastPos = params.paraviewPeaksLastPosition;
    const double sigma = paraviewEffectivePeakSigma(params);
    const double halfWidth = std::max(0.0, bandSigmas) * sigma;

    double best = std::numeric_limits<double>::infinity();
    for (int i = 0; i < n; ++i) {
        double u = (n > 1) ? static_cast<double>(i) / static_cast<double>(n - 1) : 0.0;
        double pos = firstPos + (lastPos - firstPos) * u;
        double lo = pos - halfWidth;
        double hi = pos + halfWidth;

        if (scalar >= lo && scalar <= hi) return 0.0;

        double d = (scalar < lo) ? (lo - scalar) : (scalar - hi);
        best = std::min(best, d);
    }

    return best;
}

double paraviewAdaptiveSkipDistance(double scalar,
                                    double dScalarDs,
                                    const Params& params)
{
    // We only skip while S is outside a deliberately enlarged band around every
    // opacity peak.  Inside that enlarged band, the adaptive renderer falls back
    // to the exact same base step used by the known-good fixed renderer.
    double deltaS = distanceToNearestParaviewPeakBand(
        scalar,
        params,
        params.paraviewAdaptiveFineBandSigmas
    );

    if (!std::isfinite(deltaS) || deltaS <= 0.0) return params.stepSize;

    double derivative = std::abs(dScalarDs);
    derivative = std::max(derivative, std::max(1.0e-12, params.paraviewAdaptiveDerivativeFloor));

    double candidate = std::max(0.0, params.paraviewAdaptiveSafetyFactor) * deltaS / derivative;
    if (!std::isfinite(candidate)) candidate = params.paraviewAdaptiveMaxStep;

    const double baseStep = std::max(params.stepSize, 1.0e-6);
    const double maxStep = std::max(baseStep, params.paraviewAdaptiveMaxStep);
    return std::clamp(candidate, baseStep, maxStep);
}

double paraviewPeakOpacity(double scalar, const Params& params);
double alphaFromTransferOpacity(double opacityOverUnitDistance,
                                double stepDistance,
                                double scalarOpacityUnitDistance);

void compositeParaviewSegmentMidpoint(const Ray& ray,
                                      double s,
                                      double ds,
                                      const Params& params,
                                      const Scene& scene,
                                      double realScale,
                                      RGBf& accum,
                                      double& T)
{
    if (ds <= 0.0 || T <= params.transmittanceCutoff) return;

    const double smid = s + 0.5 * ds;
    Vec3 x = ray.origin + smid * ray.dir;
    std::complex<double> psi = scene.field.eval(x, params.time);

    const double r = length(x);
    double scalar = paraviewRawDisplayScalar(x, psi, params) / realScale;
    double opacity = paraviewPeakOpacity(scalar, params);

    if (opacity > 0.0) {
        opacity *= axisMaskFactor(x, params);
        opacity *= paraviewOpacityRadialEnvelope(r, params);

        if (opacity > 0.0) {
            RGBf c = paraviewPeakColor(scalar, params);

            double alpha = alphaFromTransferOpacity(
                opacity,
                ds,
                params.paraviewScalarOpacityUnitDistance
            );
            alpha = std::clamp(alpha, 0.0, params.maxStepAlpha);

            accum += T * alpha * c;
            T *= (1.0 - alpha);
        }
    }
}

double paraviewPeakOpacity(double scalar, const Params& params) {
    // This is an approximation of ParaView's "Peaks" opacity transfer function.
    // The returned value is an opacity over ScalarOpacityUnitDistance, not a
    // raw per-world-unit density. Converting it directly with
    // exp(-opacity * ds) makes the whole sphere turn into blue fog.
    if (scalar <= 0.0) return 0.0;

    const int n = std::max(1, params.paraviewPeaksNumPeaks);
    const double firstPos = params.paraviewPeaksFirstPosition;
    const double lastPos = params.paraviewPeaksLastPosition;
    const double firstOpacity = std::max(0.0, params.paraviewPeaksFirstOpacity);
    const double lastOpacity = std::max(0.0, params.paraviewPeaksLastOpacity);

    const double sigma = paraviewEffectivePeakSigma(params);

    const double invTwoSigma2 = 1.0 / (2.0 * sigma * sigma);
    constexpr double tailCutoffSigmas = 3.0;

    double opacity = 0.0;
    for (int i = 0; i < n; ++i) {
        double u = (n > 1) ? static_cast<double>(i) / static_cast<double>(n - 1) : 0.0;
        double pos = firstPos + (lastPos - firstPos) * u;
        double op = firstOpacity + (lastOpacity - firstOpacity) * u;
        double d = scalar - pos;

        // Hard-kill long Gaussian tails. Without this, tiny blue opacity spread
        // over hundreds of length units integrates into an opaque curtain.
        if (std::abs(d) > tailCutoffSigmas * sigma) continue;

        opacity += op * std::exp(-(d * d) * invTwoSigma2);
    }

    opacity *= std::max(0.0, params.paraviewPeaksStrength);
    return std::clamp(opacity, 0.0, 0.999999);
}

double alphaFromTransferOpacity(double opacityOverUnitDistance,
                                double stepDistance,
                                double scalarOpacityUnitDistance)
{
    // ParaView rescales the transfer-function opacity by sample distance using
    // ScalarOpacityUnitDistance. If opacity A is specified over distance U, then
    // opacity over distance ds is:
    //
    //     1 - (1 - A)^(ds/U)
    //
    // This preserves the visual opacity when the ray-march step size changes.
    double A = std::clamp(opacityOverUnitDistance, 0.0, 0.999999);
    if (A <= 0.0 || stepDistance <= 0.0) return 0.0;

    double U = std::max(1.0e-9, scalarOpacityUnitDistance);
    return 1.0 - std::exp(std::log1p(-A) * (stepDistance / U));
}


struct GwpvOpacityGainTable {
    std::vector<double> times;
    std::vector<double> gains;
    bool valid = false;
};

std::string trimCsvCell(std::string s) {
    while (!s.empty() && (s.back() == '\r' || s.back() == '\n' || s.back() == ' ' || s.back() == '\t')) {
        s.pop_back();
    }
    std::size_t first = 0;
    while (first < s.size() && (s[first] == ' ' || s[first] == '\t')) {
        ++first;
    }
    if (first > 0) s.erase(0, first);
    return s;
}

std::vector<std::string> splitCsvLine(const std::string& line) {
    std::vector<std::string> cells;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) {
        cells.push_back(trimCsvCell(cell));
    }
    return cells;
}

GwpvOpacityGainTable loadGwpvOpacityGainTable(const std::string& path) {
    GwpvOpacityGainTable table;

    std::ifstream in(path);
    if (!in) {
        std::cerr << "WARNING: Could not open gwpv opacity gain CSV '" << path
                  << "'. Opacity gain will be 1.0.\n";
        return table;
    }

    std::string headerLine;
    if (!std::getline(in, headerLine)) {
        std::cerr << "WARNING: Empty gwpv opacity gain CSV '" << path
                  << "'. Opacity gain will be 1.0.\n";
        return table;
    }

    std::vector<std::string> header = splitCsvLine(headerLine);
    int timeCol = -1;
    int gainCol = -1;
    for (int i = 0; i < static_cast<int>(header.size()); ++i) {
        if (header[i] == "time") timeCol = i;
        if (header[i] == "gain_smoothed" || header[i] == "gain_boost_only") gainCol = i;
    }

    if (timeCol < 0 || gainCol < 0) {
        std::cerr << "WARNING: gwpv opacity gain CSV '" << path
                  << "' must contain columns 'time' and 'gain_smoothed' or 'gain_boost_only'. "
                  << "Opacity gain will be 1.0.\n";
        return table;
    }

    std::string line;
    while (std::getline(in, line)) {
        if (trimCsvCell(line).empty()) continue;
        std::vector<std::string> cells = splitCsvLine(line);
        if (static_cast<int>(cells.size()) <= std::max(timeCol, gainCol)) continue;

        try {
            double t = std::stod(cells[timeCol]);
            double g = std::stod(cells[gainCol]);
            if (std::isfinite(t) && std::isfinite(g) && g > 0.0) {
                table.times.push_back(t);
                table.gains.push_back(g);
            }
        } catch (...) {
            // Skip malformed rows.
        }
    }

    if (table.times.size() < 2 || table.gains.size() != table.times.size()) {
        std::cerr << "WARNING: gwpv opacity gain CSV '" << path
                  << "' did not provide at least two valid rows. "
                  << "Opacity gain will be 1.0.\n";
        table.times.clear();
        table.gains.clear();
        return table;
    }

    std::vector<std::size_t> order(table.times.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return table.times[a] < table.times[b];
    });

    std::vector<double> sortedTimes;
    std::vector<double> sortedGains;
    sortedTimes.reserve(order.size());
    sortedGains.reserve(order.size());
    for (std::size_t idx : order) {
        sortedTimes.push_back(table.times[idx]);
        sortedGains.push_back(table.gains[idx]);
    }

    table.times = std::move(sortedTimes);
    table.gains = std::move(sortedGains);
    table.valid = true;

    std::cerr << "Loaded gwpv opacity gain table from '" << path
              << "' with " << table.times.size() << " rows, time range ["
              << table.times.front() << ", " << table.times.back() << "].\n";

    return table;
}

const GwpvOpacityGainTable& getGwpvOpacityGainTable(const Params& params) {
    static std::once_flag once;
    static GwpvOpacityGainTable table;

    std::call_once(once, [&]() {
        table = loadGwpvOpacityGainTable(params.gwpvOpacityGainCsvPath);
    });

    return table;
}

double gwpvOpacityGainAtTime(double time, const Params& params) {
    if (!params.gwpvOpacityGainEnabled) {
        return 1.0;
    }

    const GwpvOpacityGainTable& table = getGwpvOpacityGainTable(params);
    if (!table.valid || table.times.empty()) {
        return 1.0;
    }

    double gain = 1.0;
    if (time <= table.times.front()) {
        gain = table.gains.front();
    } else if (time >= table.times.back()) {
        gain = table.gains.back();
    } else {
        auto it = std::lower_bound(table.times.begin(), table.times.end(), time);
        std::size_t hi = static_cast<std::size_t>(it - table.times.begin());
        std::size_t lo = hi - 1;
        double t0 = table.times[lo];
        double t1 = table.times[hi];
        double g0 = table.gains[lo];
        double g1 = table.gains[hi];
        double u = (time - t0) / std::max(t1 - t0, 1.0e-30);
        gain = (1.0 - u) * g0 + u * g1;
    }

    gain *= std::max(0.0, params.gwpvOpacityGainMultiplier);
    if (!std::isfinite(gain) || gain <= 0.0) {
        gain = 1.0;
    }
    return gain;
}

struct GwpvWavelengthTable {
    std::vector<double> times;
    std::vector<double> csvGains;
    bool valid = false;
};

double interpolateWavelengthCsvGainClamped(const GwpvWavelengthTable& table, double time) {
    if (time <= table.times.front()) return table.csvGains.front();
    if (time >= table.times.back()) return table.csvGains.back();
    auto it = std::lower_bound(table.times.begin(), table.times.end(), time);
    std::size_t hi = static_cast<std::size_t>(it - table.times.begin());
    std::size_t lo = hi - 1;
    double t0 = table.times[lo];
    double t1 = table.times[hi];
    double g0 = table.csvGains[lo];
    double g1 = table.csvGains[hi];
    double u = (time - t0) / std::max(t1 - t0, 1.0e-30);
    return (1.0 - u) * g0 + u * g1;
}

GwpvWavelengthTable loadGwpvWavelengthTable(const std::string& path) {
    GwpvWavelengthTable table;

    std::ifstream in(path);
    if (!in) {
        std::cerr << "WARNING: Could not open gwpv wavelength CSV '" << path
                  << "'. Wavelength compensation will be disabled.\n";
        return table;
    }

    std::string headerLine;
    if (!std::getline(in, headerLine)) {
        std::cerr << "WARNING: Empty gwpv wavelength CSV '" << path
                  << "'. Wavelength compensation will be disabled.\n";
        return table;
    }

    std::vector<std::string> header = splitCsvLine(headerLine);
    int timeCol = -1;
    int gainCol = -1;
    for (int i = 0; i < static_cast<int>(header.size()); ++i) {
        if (header[i] == "time") timeCol = i;
        if (header[i] == "wavelength_gain") gainCol = i;
    }

    if (timeCol < 0 || gainCol < 0) {
        std::cerr << "WARNING: gwpv wavelength CSV '" << path
                  << "' must contain columns 'time' and 'wavelength_gain'. Wavelength compensation will be disabled.\n";
        return table;
    }

    std::string line;
    while (std::getline(in, line)) {
        if (trimCsvCell(line).empty()) continue;
        std::vector<std::string> cells = splitCsvLine(line);
        if (static_cast<int>(cells.size()) <= std::max(timeCol, gainCol)) continue;
        try {
            double t = std::stod(cells[timeCol]);
            double g = std::stod(cells[gainCol]);
            if (std::isfinite(t) && std::isfinite(g) && g > 0.0) {
                table.times.push_back(t);
                table.csvGains.push_back(g);
            }
        } catch (...) {
        }
    }

    if (table.times.size() < 2 || table.csvGains.size() != table.times.size()) {
        std::cerr << "WARNING: gwpv wavelength CSV '" << path
                  << "' did not provide at least two valid rows. Wavelength compensation will be disabled.\n";
        table.times.clear();
        table.csvGains.clear();
        return table;
    }

    std::vector<std::size_t> order(table.times.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return table.times[a] < table.times[b];
    });

    std::vector<double> sortedTimes;
    std::vector<double> sortedGains;
    sortedTimes.reserve(order.size());
    sortedGains.reserve(order.size());
    for (std::size_t idx : order) {
        sortedTimes.push_back(table.times[idx]);
        sortedGains.push_back(table.csvGains[idx]);
    }

    table.times = std::move(sortedTimes);
    table.csvGains = std::move(sortedGains);
    table.valid = true;

    std::cerr << "Loaded gwpv wavelength table from '" << path
              << "' with " << table.times.size() << " rows, time range ["
              << table.times.front() << ", " << table.times.back() << "], csvGain range ["
              << table.csvGains.front() << ", " << table.csvGains.back() << "].\n";

    return table;
}

const GwpvWavelengthTable& getGwpvWavelengthTable(const Params& params) {
    static std::once_flag once;
    static GwpvWavelengthTable table;

    std::call_once(once, [&]() {
        table = loadGwpvWavelengthTable(params.gwpvWavelengthCsvPath);
    });

    return table;
}

double gwpvWavelengthCsvGainAtRetardedTime(double tLookup, const Params& params) {
    if (!params.gwpvWavelengthCompEnabled && !params.gwpvWavelengthStepScalingEnabled) {
        return 1.0;
    }

    const GwpvWavelengthTable& table = getGwpvWavelengthTable(params);
    if (!table.valid || table.times.empty()) {
        return 1.0;
    }

    const double tStart = params.gwpvWavelengthSecondPlateauStartTime;
    const double tEnd = std::max(tStart, params.gwpvWavelengthSecondPlateauEndTime);

    if (!params.gwpvWavelengthSecondPlateauEnabled) {
        return interpolateWavelengthCsvGainClamped(table, std::min(tLookup, tStart));
    }

    if (tLookup <= tStart) {
        return interpolateWavelengthCsvGainClamped(table, tLookup);
    }

    const double gStart = interpolateWavelengthCsvGainClamped(table, tStart);
    const double gEnd = interpolateWavelengthCsvGainClamped(table, tEnd);
    if (tLookup >= tEnd) {
        return gEnd;
    }

    const double u = (tLookup - tStart) / std::max(tEnd - tStart, 1.0e-30);
    return (1.0 - u) * gStart + u * gEnd;
}

double gwpvWavelengthRatioAtPoint(const Vec3& x, double frameTime, const Params& params) {
    const double r = length(x);
    //const double tLookup = frameTime + params.gwpvWavelengthReferenceRadius - r;
    const double tLookup = frameTime;
    const double csvGain = gwpvWavelengthCsvGainAtRetardedTime(tLookup, params);
    const double csvBasePower = std::max(1.0e-12, params.gwpvWavelengthCsvBasePower);
    double ratio = std::pow(std::max(csvGain, 1.0e-12), 1.0 / csvBasePower);
    if (!std::isfinite(ratio) || ratio <= 0.0) ratio = 1.0;
    return ratio;
}

double gwpvLocalWavelengthOpacityGain(const Vec3& x, double frameTime, const Params& params) {
    if (!params.gwpvWavelengthCompEnabled) return 1.0;
    const double ratio = gwpvWavelengthRatioAtPoint(x, frameTime, params);
    double gain = std::pow(std::max(ratio, 1.0e-12), params.gwpvWavelengthOpacityPower);
    gain = std::clamp(gain, params.gwpvWavelengthOpacityGainMin, params.gwpvWavelengthOpacityGainMax);
    if (!std::isfinite(gain) || gain <= 0.0) gain = 1.0;
    return gain;
}

double gwpvLocalWavelengthColorGain(const Vec3& x, double frameTime, const Params& params) {
    if (!params.gwpvWavelengthCompEnabled) return 1.0;
    const double ratio = gwpvWavelengthRatioAtPoint(x, frameTime, params);
    double gain = std::pow(std::max(ratio, 1.0e-12), params.gwpvWavelengthColorPower);
    gain = std::clamp(gain, params.gwpvWavelengthColorGainMin, params.gwpvWavelengthColorGainMax);
    if (!std::isfinite(gain) || gain <= 0.0) gain = 1.0;
    return gain;
}

double gwpvLocalWavelengthPeakWidthGain(const Vec3& x, double frameTime, const Params& params) {
    if (!params.gwpvWavelengthPeakWidthEnabled) return 1.0;
    const double ratio = gwpvWavelengthRatioAtPoint(x, frameTime, params);
    double gain = std::pow(std::max(ratio, 1.0e-12), params.gwpvWavelengthPeakWidthPower);
    gain = std::clamp(gain, params.gwpvWavelengthPeakWidthGainMin, params.gwpvWavelengthPeakWidthGainMax);
    if (!std::isfinite(gain) || gain <= 0.0) gain = 1.0;
    return gain;
}

double gwpvLocalBaseStep(const Vec3& x, const Params& params) {
    const double baseStep = std::max(params.stepSize, 1.0e-6);
    if (!params.gwpvWavelengthStepScalingEnabled) return baseStep;
    const double ratio = gwpvWavelengthRatioAtPoint(x, params.time, params);
    double scale = std::pow(std::max(ratio, 1.0e-12), params.gwpvWavelengthStepPower);
    scale = std::max(scale, 1.0);
    double localStep = baseStep / scale;
    localStep = std::clamp(localStep, params.gwpvWavelengthMinStep, baseStep);
    if (!std::isfinite(localStep) || localStep <= 0.0) localStep = baseStep;
    return localStep;
}

RGBf gwpvPeakColor(double scalar, const Params& params) {
    const double a = params.gwpvPeaksFirstPosition;
    const double b = params.gwpvPeaksLastPosition;

    double u = 0.0;
    if (b > a) {
        u = std::clamp((scalar - a) / (b - a), 0.0, 1.0);
    }

    // Approximation of ParaView's "Rainbow Uniform" preset: blue/violet at
    // the first peak, red at the last peak.
    double hue = (2.0 / 3.0) * (1.0 - u);
    return hsvToRgb(hue, 0.95, 1.0) * params.waveBrightness;
}

double gwpvPeakOpacity(double scalar, double peakWidthGain, const Params& params) {
    // Match the GWPV/ParaView "Peaks" piecewise opacity function. This is not
    // a Gaussian. For each peak, ParaView creates points:
    //   peak - peakDecay/100 -> 0
    //   peak                 -> peakOpacity
    //   peak + peakDecay     -> 0
    // so the visible sheet has an extremely sharp low-side edge and a broader
    // high-side falloff.
    if (scalar <= 0.0) return 0.0;

    const int n = std::max(1, params.gwpvPeaksNumPeaks);
    const double firstPos = params.gwpvPeaksFirstPosition;
    const double lastPos = params.gwpvPeaksLastPosition;
    const double firstOpacity = std::max(0.0, params.gwpvPeaksFirstOpacity);
    const double lastOpacity = std::max(0.0, params.gwpvPeaksLastOpacity);

    const double spacing = (n > 1 && lastPos > firstPos)
        ? (lastPos - firstPos) / static_cast<double>(n - 1)
        : std::max(1.0e-6, std::abs(lastPos - firstPos));
    const double peakDecayBase = 0.5 * std::max(1.0e-12, spacing);
    const double peakDecay = peakDecayBase * std::max(1.0, peakWidthGain);

    double opacity = 0.0;
    for (int i = 0; i < n; ++i) {
        const double u = (n > 1) ? static_cast<double>(i) / static_cast<double>(n - 1) : 0.0;
        const double peak = firstPos + (lastPos - firstPos) * u;
        const double peakOpacity = firstOpacity + (lastOpacity - firstOpacity) * u;

        const double left = peak - peakDecay / 100.0;
        const double right = peak + peakDecay;

        double opHere = 0.0;
        if (scalar >= left && scalar <= peak) {
            opHere = peakOpacity * (scalar - left) / std::max(peak - left, 1.0e-30);
        } else if (scalar > peak && scalar <= right) {
            opHere = peakOpacity * (1.0 - (scalar - peak) / std::max(right - peak, 1.0e-30));
        }

        // vtkPiecewiseFunction is one opacity function, not a sum of Gaussian
        // lobes. max() is the safest approximation if peaks are ever close.
        opacity = std::max(opacity, opHere);
    }

    opacity *= std::max(0.0, params.gwpvPeaksStrength);
    return std::clamp(opacity, 0.0, 0.999999);
}


double distanceToNearestGwpvPeakBand(double scalar,
                                     const Params& params,
                                     double guardPeakDecays)
{
    // Conservative guard-band distance for GWPV's asymmetric piecewise peaks.
    // The true visible support for each peak is
    //     [peak - peakDecay/100, peak + peakDecay].
    // We expand this by guardPeakDecays*peakDecay on both sides for adaptive
    // skipping. Inside the expanded band we fall back to fixed stepping.
    const int n = std::max(1, params.gwpvPeaksNumPeaks);
    const double firstPos = params.gwpvPeaksFirstPosition;
    const double lastPos = params.gwpvPeaksLastPosition;

    if (scalar <= 0.0) {
        // The transfer function is positive-only. Returning a finite distance
        // lets the marcher skip through strongly negative / near-zero scalar
        // regions instead of crawling through the entire wave sphere.
    }

    const double spacing = (n > 1 && lastPos > firstPos)
        ? (lastPos - firstPos) / static_cast<double>(n - 1)
        : std::max(1.0e-6, std::abs(lastPos - firstPos));
    const double peakDecay = 0.5 * std::max(1.0e-12, spacing);
    const double guard = std::max(0.0, guardPeakDecays) * peakDecay;

    double best = std::numeric_limits<double>::infinity();
    for (int i = 0; i < n; ++i) {
        const double u = (n > 1) ? static_cast<double>(i) / static_cast<double>(n - 1) : 0.0;
        const double peak = firstPos + (lastPos - firstPos) * u;
        const double left = peak - peakDecay / 100.0 - guard;
        const double right = peak + peakDecay + guard;

        if (scalar >= left && scalar <= right) return 0.0;

        const double d = (scalar < left) ? (left - scalar) : (scalar - right);
        best = std::min(best, d);
    }

    return best;
}

double gwpvAdaptiveSkipDistance(double scalar,
                                double dScalarDs,
                                const Params& params)
{
    const double deltaS = distanceToNearestGwpvPeakBand(
        scalar,
        params,
        params.gwpvAdaptiveGuardPeakDecays
    );

    if (!std::isfinite(deltaS) || deltaS <= 0.0) return params.stepSize;

    double derivative = std::abs(dScalarDs);
    derivative = std::max(derivative, std::max(1.0e-12, params.paraviewAdaptiveDerivativeFloor));

    double candidate = std::max(0.0, params.paraviewAdaptiveSafetyFactor) * deltaS / derivative;
    if (!std::isfinite(candidate)) candidate = params.paraviewAdaptiveMaxStep;

    const double baseStep = std::max(params.stepSize, 1.0e-6);
    const double maxStep = std::max(baseStep, params.paraviewAdaptiveMaxStep);
    return std::clamp(candidate, baseStep, maxStep);
}


void compositeGwpvPoint(const Vec3& x,
                        double ds,
                        const Params& params,
                        const Scene& scene,
                        double frameOpacityGain,
                        RGBf& accum,
                        double& T)
{
    if (ds <= 0.0 || T <= params.transmittanceCutoff) return;

    std::complex<double> psi = scene.field.eval(x, params.time);

    const double r = length(x);
    const double scalar = paraviewRawDisplayScalar(x, psi, params);  // absolute; no frameRealMax division
    const double localPeakWidthGain = gwpvLocalWavelengthPeakWidthGain(x, params.time, params);
    double opacity = gwpvPeakOpacity(scalar, localPeakWidthGain, params);

    if (opacity > 0.0) {
        const double localOpacityGain = gwpvLocalWavelengthOpacityGain(x, params.time, params);
        const double localColorGain = gwpvLocalWavelengthColorGain(x, params.time, params);
        const double totalOpacityGain = frameOpacityGain * localOpacityGain;
        opacity *= totalOpacityGain;
        opacity = std::clamp(opacity, 0.0, 0.999999);

        if (params.gwpvUseAxisMask) {
            opacity *= axisMaskFactor(x, params);
        }
        if (params.gwpvUseOpacityRadialEnvelope) {
            opacity *= paraviewOpacityRadialEnvelope(r, params);
        }

        if (opacity > 0.0) {
            RGBf c = gwpvPeakColor(scalar, params);
            c = c * localColorGain;

            double alpha = alphaFromTransferOpacity(
                opacity,
                ds,
                params.gwpvScalarOpacityUnitDistance
            );
            alpha = std::clamp(alpha, 0.0, params.maxStepAlpha);

            accum += T * alpha * c;
            T *= (1.0 - alpha);
        }
    }
}

void compositeGwpvSegmentMidpoint(const Ray& ray,
                                  double s,
                                  double ds,
                                  const Params& params,
                                  const Scene& scene,
                                  double frameOpacityGain,
                                  RGBf& accum,
                                  double& T)
{
    if (ds <= 0.0 || T <= params.transmittanceCutoff) return;

    const double smid = s + 0.5 * ds;
    Vec3 x = ray.origin + smid * ray.dir;
    std::complex<double> psi = scene.field.eval(x, params.time);

    const double r = length(x);
    const double scalar = paraviewRawDisplayScalar(x, psi, params);  // absolute; no frameRealMax division
    const double localPeakWidthGain = gwpvLocalWavelengthPeakWidthGain(x, params.time, params);
    double opacity = gwpvPeakOpacity(scalar, localPeakWidthGain, params);

    if (opacity > 0.0) {
        const double localOpacityGain = gwpvLocalWavelengthOpacityGain(x, params.time, params);
        const double localColorGain = gwpvLocalWavelengthColorGain(x, params.time, params);
        const double totalOpacityGain = frameOpacityGain * localOpacityGain;
        opacity *= totalOpacityGain;
        opacity = std::clamp(opacity, 0.0, 0.999999);

        if (params.gwpvUseAxisMask) {
            opacity *= axisMaskFactor(x, params);
        }
        if (params.gwpvUseOpacityRadialEnvelope) {
            opacity *= paraviewOpacityRadialEnvelope(r, params);
        }

        if (opacity > 0.0) {
            RGBf c = gwpvPeakColor(scalar, params);
            c = c * localColorGain;

            double alpha = alphaFromTransferOpacity(
                opacity,
                ds,
                params.gwpvScalarOpacityUnitDistance
            );
            alpha = std::clamp(alpha, 0.0, params.maxStepAlpha);

            accum += T * alpha * c;
            T *= (1.0 - alpha);
        }
    }
}

Vec3 randomPointInWaveSphere(std::mt19937_64& rng, double rInner, double rOuter) {
    std::uniform_real_distribution<double> U(0.0, 1.0);

    double z = 2.0 * U(rng) - 1.0;
    double phi = 2.0 * PI * U(rng);
    double rho = std::sqrt(std::max(0.0, 1.0 - z * z));

    double r3Inner = rInner * rInner * rInner;
    double r3Outer = rOuter * rOuter * rOuter;
    double r = std::cbrt(r3Inner + (r3Outer - r3Inner) * U(rng));

    return Vec3(
        r * rho * std::cos(phi),
        r * rho * std::sin(phi),
        r * z
    );
}

} // namespace

Renderer::Renderer(const Params& params, const Scene& scene)
    : params_(params), scene_(scene)
{
    if (params_.renderMode != "fog" &&
        params_.renderMode != "paraview_peaks" &&
        params_.renderMode != "paraview_peaks_adaptive" &&
        params_.renderMode != "gwpv_peaks") {
        throw std::runtime_error("Unknown Params::renderMode: " + params_.renderMode +
                                 " (expected \"fog\", \"paraview_peaks\", \"paraview_peaks_adaptive\", or \"gwpv_peaks\")");
    }

    if (!paraviewUsesPsi4Scalar(params_) && !paraviewUsesRPsi4Scalar(params_)) {
        throw std::runtime_error("Unknown Params::paraviewScalarRadialMode: " +
                                 params_.paraviewScalarRadialMode +
                                 " (expected \"psi4\" or \"r_psi4\")");
    }

    if (params_.renderMode == "gwpv_peaks") {
        // This diagnostic mode uses absolute transfer-function peak positions.
        // Do not compute or apply a per-frame scalar normalization; that was the
        // source of the non-propagating/moving-level-set artifact.
        frameAmplitudeMax_ = 1.0;
        frameRealMax_ = 1.0;
        std::cerr << "Frame real max disabled for gwpv_peaks; using absolute scalar levels.\n";
        if (params_.gwpvOpacityGainEnabled) {
            double g = gwpvOpacityGainAtTime(params_.time, params_);
            std::cerr << "GWPV frame-global opacity gain at t=" << params_.time << " is " << g << "\n";
        }
        if (params_.gwpvWavelengthCompEnabled || params_.gwpvWavelengthStepScalingEnabled) {
            Vec3 xRef(params_.waveVolumeRadius, 0.0, 0.0);
            double ratio = gwpvWavelengthRatioAtPoint(xRef, params_.time, params_);
            double localOpacityGain = gwpvLocalWavelengthOpacityGain(xRef, params_.time, params_);
            double localColorGain = gwpvLocalWavelengthColorGain(xRef, params_.time, params_);
            double localPeakWidthGain = gwpvLocalWavelengthPeakWidthGain(xRef, params_.time, params_);
            double localStep = gwpvLocalBaseStep(xRef, params_);
            std::cerr << "GWPV local wavelength ratio at r=waveVolumeRadius is " << ratio
                      << "; local opacity gain=" << localOpacityGain
                      << "; local color gain=" << localColorGain
                      << "; local peak-width gain=" << localPeakWidthGain
                      << "; local step=" << localStep << "\n";
        }
    } else {
        frameAmplitudeMax_ = estimateFrameAmplitudeMax();
        frameRealMax_ = estimateFrameRealMax();

        if (params_.normalizeAmplitudePerFrame) {
            std::cerr << "Frame amplitude max = " << frameAmplitudeMax_ << "\n";
            if (params_.renderMode == "paraview_peaks" || params_.renderMode == "paraview_peaks_adaptive") {
                std::cerr << "Frame real max = " << frameRealMax_ << "\n";
            }
        }
    }
}

double Renderer::estimateFrameAmplitudeMax() const {
    if (!params_.normalizeAmplitudePerFrame) return 1.0;

    int n = std::max(1000, params_.frameAmplitudeSampleCount);
    double maxAmp = 0.0;

    std::mt19937_64 rng(123456789ULL);

    for (int i = 0; i < n; ++i) {
        Vec3 x = randomPointInWaveSphere(rng, std::max(0.0, params_.rInner), params_.waveVolumeRadius);
        std::complex<double> psi = scene_.field.eval(x, params_.time);
        double maskedAmp = std::abs(psi) * axisMaskFactor(x, params_);
        maxAmp = std::max(maxAmp, maskedAmp);
    }

    if (!std::isfinite(maxAmp) || maxAmp <= 1e-15) maxAmp = 1.0;
    return maxAmp;
}

double Renderer::estimateFrameRealMax() const {
    if (!params_.normalizeAmplitudePerFrame) return 1.0;

    int n = std::max(1000, params_.frameAmplitudeSampleCount);
    double maxRe = 0.0;

    std::mt19937_64 rng(987654321ULL);

    for (int i = 0; i < n; ++i) {
        Vec3 x = randomPointInWaveSphere(rng, std::max(0.0, params_.rInner), params_.waveVolumeRadius);
        std::complex<double> psi = scene_.field.eval(x, params_.time);

        // Match the rendered ParaView scalar: if r_psi4 mode is enabled,
        // normalize by max |rDisplay*Re(Psi4)| rather than max |Re(Psi4)|.
        // Keep the seam mask here so an unrendered axis artifact does not set
        // the transfer-function scale.
        double rawScalar = paraviewRawDisplayScalar(x, psi, params_);
        double maskedRe = std::abs(rawScalar) * axisMaskFactor(x, params_);
        maxRe = std::max(maxRe, maskedRe);
    }

    if (!std::isfinite(maxRe) || maxRe <= 1e-15) maxRe = 1.0;
    return maxRe;
}


RGBf shadeParaviewPeaksRay(const Ray& ray,
                           const Params& params,
                           const Scene& scene,
                           double frameRealMax)
{
    double tEnter = 0.0;
    double tExit = 0.0;
    bool hitWaveSphere = intersectSphere(ray, params.waveVolumeRadius, tEnter, tExit);

    RGBf accum{0.0, 0.0, 0.0};
    double T = 1.0;

    if (hitWaveSphere) {
        double s0 = std::max({tEnter, params.rayTMin, 0.0});
        double s1 = std::min(tExit, params.rayTMax);

        if (s1 > s0) {
            const double realScale = std::max(frameRealMax, 1.0e-15);
            const double step = std::max(params.stepSize, 1.0e-6);

            for (double s = s0; s < s1;) {
                const double ds = std::min(step, s1 - s);
                compositeParaviewSegmentMidpoint(ray, s, ds, params, scene, realScale, accum, T);
                if (T <= params.transmittanceCutoff) break;
                s += ds;
            }
        }
    }

    RGBf bg = scene.panorama.sampleDirection(ray.dir, params.panoramaYawDegrees, params.panoramaExposure);
    accum += T * bg;
    return accum;
}

RGBf shadeParaviewPeaksAdaptiveRay(const Ray& ray,
                                   const Params& params,
                                   const Scene& scene,
                                   double frameRealMax)
{
    double tEnter = 0.0;
    double tExit = 0.0;
    bool hitWaveSphere = intersectSphere(ray, params.waveVolumeRadius, tEnter, tExit);

    RGBf accum{0.0, 0.0, 0.0};
    double T = 1.0;

    if (hitWaveSphere) {
        double s0 = std::max({tEnter, params.rayTMin, 0.0});
        double s1 = std::min(tExit, params.rayTMax);

        if (s1 > s0) {
            const double realScale = std::max(frameRealMax, 1.0e-15);
            const double baseStep = std::max(params.stepSize, 1.0e-6);

            for (double s = s0; s < s1;) {
                const double remaining = s1 - s;
                const double baseDs = std::min(baseStep, remaining);

                // Use the analytic derivative of the reconstructed/interpolated
                // field to estimate how far S can move before it gets near a
                // visible transfer-function band.
                Vec3 x = ray.origin + s * ray.dir;
                FieldRaySample sample = scene.field.evalWithRayDerivative(x, ray.dir, params.time);
                ParaviewScalarRaySample scalarSample =
                    paraviewRawDisplayScalarWithDerivative(x, ray.dir, sample, params);
                const double scalar = scalarSample.raw / realScale;
                const double dScalarDs = scalarSample.dRawDs / realScale;

                const double skipDs = std::min(
                    paraviewAdaptiveSkipDistance(scalar, dScalarDs, params),
                    remaining
                );

                // If the derivative predictor says we are near a peak band, do
                // exactly the same midpoint compositing step as the fixed renderer.
                // This is the key rule that preserves the known-good look.
                if (skipDs <= 1.25 * baseStep) {
                    compositeParaviewSegmentMidpoint(ray, s, baseDs, params, scene, realScale, accum, T);
                    if (T <= params.transmittanceCutoff) break;
                    s += baseDs;
                } else {
                    // Empty-space skip. We intentionally do not composite here;
                    // paraviewAdaptiveSkipDistance only returns a large step when
                    // S is outside an enlarged guard band around every opacity peak.
                    s += skipDs;
                }
            }
        }
    }

    RGBf bg = scene.panorama.sampleDirection(ray.dir, params.panoramaYawDegrees, params.panoramaExposure);
    accum += T * bg;
    return accum;
}

RGBf shadeGwpvPeaksRay(const Ray& ray,
                       const Params& params,
                       const Scene& scene)
{
    double tEnter = 0.0;
    double tExit = 0.0;
    bool hitWaveSphere = intersectSphere(ray, params.waveVolumeRadius, tEnter, tExit);

    RGBf accum{0.0, 0.0, 0.0};
    double T = 1.0;

    const BlackHoleSystem::State bh = BlackHoleSystem::stateAtTime(params.time, params);
    BlackHoleSurfaceHit bhHit = nearestBlackHoleRayHit(ray, bh, true);

    const double tMinRay = std::max({params.rayTMin, 0.0});
    const double tMaxRay = params.rayTMax;
    const bool hasBHSurfaceHit =
        bhHit.hit &&
        bhHit.t >= tMinRay &&
        bhHit.t <= tMaxRay;

    if (hitWaveSphere) {
        double s0 = std::max({tEnter, params.rayTMin, 0.0});
        double s1 = std::min(tExit, params.rayTMax);
        if (hasBHSurfaceHit) {
            s1 = std::min(s1, bhHit.t);
        }

        if (s1 > s0) {
            const double baseFrameOpacityGain = gwpvOpacityGainAtTime(params.time, params);

            for (double s = s0; s < s1;) {
                const double remaining = s1 - s;
                Vec3 x = ray.origin + s * ray.dir;
                const double localBaseStep = std::min(gwpvLocalBaseStep(x, params), remaining);

                if (!params.gwpvAdaptiveEnabled) {
                    compositeGwpvSegmentMidpoint(ray, s, localBaseStep, params, scene, baseFrameOpacityGain, accum, T);
                    if (T <= params.transmittanceCutoff) break;
                    s += localBaseStep;
                    continue;
                }

                // Same derivative predictor as paraview_peaks_adaptive, but
                // using the GWPV asymmetric peak guard bands rather than
                // Gaussian sigma bands.
                FieldRaySample sample = scene.field.evalWithRayDerivative(x, ray.dir, params.time);
                ParaviewScalarRaySample scalarSample =
                    paraviewRawDisplayScalarWithDerivative(x, ray.dir, sample, params);

                const double scalar = scalarSample.raw;       // absolute; no frameRealMax division
                const double dScalarDs = scalarSample.dRawDs; // absolute; no frameRealMax division

                double skipDs = gwpvAdaptiveSkipDistance(scalar, dScalarDs, params);
                if (params.gwpvWavelengthStepScalingEnabled) {
                    skipDs = std::min(skipDs, params.gwpvWavelengthMaxSkipMultiplier * localBaseStep);
                }
                skipDs = std::min(skipDs, remaining);

                if (skipDs <= 1.25 * localBaseStep) {
                    compositeGwpvSegmentMidpoint(ray, s, localBaseStep, params, scene, baseFrameOpacityGain, accum, T);
                    if (T <= params.transmittanceCutoff) break;
                    s += localBaseStep;
                } else {
                    s += skipDs;
                }
            }
        }
    }

    if (hasBHSurfaceHit) {
        const Vec3 xHit = ray.origin + bhHit.t * ray.dir;
        accum += T * shadeBlackHoleSurface(params, bhHit, xHit);
        return accum;
    }

    RGBf bg = scene.panorama.sampleDirection(ray.dir, params.panoramaYawDegrees, params.panoramaExposure);
    accum += T * bg;
    return accum;
}


RGBf shadeGwpvPeaksGeodesicRay(const Ray& ray,
                               const Params& params,
                               const Scene& scene)
{
    double tEnter = 0.0;
    double tExit = 0.0;
    bool hitWaveSphere = intersectSphere(ray, params.waveVolumeRadius, tEnter, tExit);

    RGBf accum{0.0, 0.0, 0.0};
    double T = 1.0;
    Vec3 finalDir = ray.dir;

    const BlackHoleSystem::State bh = BlackHoleSystem::stateAtTime(params.time, params);

    if (hitWaveSphere) {
        double s0 = std::max({tEnter, params.rayTMin, 0.0});
        double s1 = std::min(tExit, params.rayTMax);

        if (s1 > s0) {
            const double baseFrameOpacityGain = gwpvOpacityGainAtTime(params.time, params);

            AdaptiveRK::State state;
            state.x = ray.origin + s0 * ray.dir;
            state.v = FrozenStrainMetric::normalizeNullSpeed(state.x, ray.dir, params.time, scene, params);
            finalDir = safeNormalizeVec(state.v, ray.dir);

            const double hMin = std::max(params.metricGeodesicMinStep, 1.0e-12);
            double hMaxLocal = std::max(metricGeodesicLocalMaxStep(state.x, params), hMin);
            double h = std::clamp(
                std::abs(params.metricGeodesicInitialStep),
                hMin,
                hMaxLocal
            );

            const double colorStep = std::max(params.metricColorStep, 1.0e-6);
            const int maxAccepted = std::max(1, params.metricMaxAcceptedSteps);

            for (int acceptedSteps = 0;
                 acceptedSteps < maxAccepted && T > params.transmittanceCutoff;
                 ++acceptedSteps)
            {
                const double rNow = length(state.x);
                if (rNow >= params.waveVolumeRadius && dot(state.x, state.v) > 0.0) {
                    break;
                }

                auto deriv = [&](const AdaptiveRK::State& y) {
                    return FrozenStrainMetric::geodesicDerivative(y, params.time, scene, params);
                };

                hMaxLocal = std::max(metricGeodesicLocalMaxStep(state.x, params), hMin);
                h = std::clamp(h, hMin, hMaxLocal);

                const AdaptiveRK::State yStart = state;
                AdaptiveRK::StepResult step = AdaptiveRK::adaptiveCashKarpStep(
                    state,
                    h,
                    hMin,
                    hMaxLocal,
                    params.metricGeodesicAbsTol,
                    params.metricGeodesicRelTol,
                    deriv
                );

                if (!step.accepted) {
                    // Defensive fallback: do not silently continue with a bad geodesic step.
                    break;
                }

                const AdaptiveRK::State yEndRaw = step.y1;
                Vec3 x0 = yStart.x;
                Vec3 x1 = yEndRaw.x;
                Vec3 dx = x1 - x0;
                const double segLen = length(dx);

                if (std::isfinite(segLen) && segLen > 0.0) {
                    const int nSub = std::max(1, static_cast<int>(std::ceil(segLen / colorStep)));
                    const double ds = segLen / static_cast<double>(nSub);

                    for (int i = 0; i < nSub; ++i) {
                        const double a0 = static_cast<double>(i) / static_cast<double>(nSub);
                        const double a1 = static_cast<double>(i + 1) / static_cast<double>(nSub);
                        const Vec3 xA = x0 + a0 * dx;
                        const Vec3 xB = x0 + a1 * dx;

                        BlackHoleSurfaceHit captureHit = nearestBlackHoleSegmentHit(xA, xB, bh, false);
                        if (captureHit.hit) {
                            const Vec3 subDx = xB - xA;
                            const Vec3 xHit = xA + captureHit.u * subDx;
                            const double dsBefore = length(xHit - xA);

                            if (std::isfinite(dsBefore) && dsBefore > 1.0e-9 && length(xA) <= params.waveVolumeRadius) {
                                const Vec3 xMid = xA + 0.5 * captureHit.u * subDx;
                                compositeGwpvPoint(xMid, dsBefore, params, scene, baseFrameOpacityGain, accum, T);
                            }

                            // In the geodesic/lensing path this is a capture/shadow event,
                            // not a visible textured surface. Do not sample black_sphere_texture.png here.
                            // Keep the already accumulated foreground wave opacity, then terminate
                            // without adding panorama/background light behind the hole.
                            accum += T * RGBf{0.0, 0.0, 0.0};
                            return accum;
                        }

                        const double aMid = 0.5 * (a0 + a1);
                        const Vec3 xMid = x0 + aMid * dx;
                        if (length(xMid) <= params.waveVolumeRadius) {
                            compositeGwpvPoint(xMid, ds, params, scene, baseFrameOpacityGain, accum, T);
                            if (T <= params.transmittanceCutoff) break;
                        }
                    }
                }

                state = yEndRaw;
                if (params.metricRenormalizeNullSpeed) {
                    state.v = FrozenStrainMetric::normalizeNullSpeed(state.x, state.v, params.time, scene, params);
                }
                finalDir = safeNormalizeVec(state.v, finalDir);
                h = step.hNext;
            }
        }
    }

    RGBf bg = scene.panorama.sampleDirection(finalDir, params.panoramaYawDegrees, params.panoramaExposure);
    accum += T * bg;
    return accum;
}


RGBf Renderer::shadeRay(const Ray& ray) const {
    if (params_.renderMode == "gwpv_peaks") {
        if (params_.metricLensingEnabled) {
            return shadeGwpvPeaksGeodesicRay(ray, params_, scene_);
        }
        return shadeGwpvPeaksRay(ray, params_, scene_);
    }
    if (params_.renderMode == "paraview_peaks") {
        return shadeParaviewPeaksRay(ray, params_, scene_, frameRealMax_);
    }
    if (params_.renderMode == "paraview_peaks_adaptive") {
        return shadeParaviewPeaksAdaptiveRay(ray, params_, scene_, frameRealMax_);
    }

    double tEnter = 0.0;
    double tExit = 0.0;
    bool hitWaveSphere = intersectSphere(ray, params_.waveVolumeRadius, tEnter, tExit);

    RGBf accum{0.0, 0.0, 0.0};
    double T = 1.0; // remaining transparency from camera to current sample.

    if (hitWaveSphere) {
        double s0 = std::max({tEnter, params_.rayTMin, 0.0});
        double s1 = std::min(tExit, params_.rayTMax);

        if (s1 > s0) {
            for (double s = s0; s <= s1; s += params_.stepSize) {
                Vec3 x = ray.origin + s * ray.dir;
                std::complex<double> psi = scene_.field.eval(x, params_.time);

                double maskedAmp = std::abs(psi) * axisMaskFactor(x, params_);
                double density = localDensityFromMaskedAmplitude(maskedAmp, params_, frameAmplitudeMax_);

                if (density > 0.0) {
                    double alpha = 1.0 - std::exp(-density * params_.stepSize);
                    alpha = std::clamp(alpha, 0.0, params_.maxStepAlpha);

                    RGBf c = localWaveColor(psi, params_);

                    // Front-to-back emission/absorption compositing.
                    accum += T * alpha * c;
                    T *= (1.0 - alpha);

                    if (T <= params_.transmittanceCutoff) break;
                }
            }
        }
    }

    RGBf bg = scene_.panorama.sampleDirection(ray.dir, params_.panoramaYawDegrees, params_.panoramaExposure);
    accum += T * bg;
    return accum;
}

RGB8 Renderer::shadePixel(const Camera& camera, int x, int y) const {
    if (params_.samplesPerPixel <= 1) {
        Ray ray = camera.generateRay(static_cast<double>(x), static_cast<double>(y), params_.width, params_.height);
        return makeRGB(shadeRay(ray), params_.outputGamma);
    }

    int root = static_cast<int>(std::floor(std::sqrt(static_cast<double>(params_.samplesPerPixel))));
    if (root < 1) root = 1;
    int spp = root * root;

    RGBf sum{0.0, 0.0, 0.0};
    int index = 0;
    for (int sy = 0; sy < root; ++sy) {
        for (int sx = 0; sx < root; ++sx) {
            double jx = hash01(x, y, index, 0);
            double jy = hash01(x, y, index, 1);
            double px = static_cast<double>(x) + (static_cast<double>(sx) + jx) / static_cast<double>(root) - 0.5;
            double py = static_cast<double>(y) + (static_cast<double>(sy) + jy) / static_cast<double>(root) - 0.5;
            Ray ray = camera.generateRay(px, py, params_.width, params_.height);
            sum += shadeRay(ray);
            ++index;
        }
    }

    sum = sum * (1.0 / static_cast<double>(spp));
    return makeRGB(sum, params_.outputGamma);
}

Image Renderer::render(const Camera& camera) const {
    const auto renderStart = std::chrono::steady_clock::now();

    Image img(params_.width, params_.height);

    int nThreads = params_.numThreads;
    if (nThreads <= 0) {
        nThreads = static_cast<int>(std::thread::hardware_concurrency());
        if (nThreads <= 0) nThreads = 1;
    }

    std::atomic<int> nextRow{0};
    std::atomic<int> rowsDone{0};

    auto worker = [&]() {
        while (true) {
            int y = nextRow.fetch_add(1);
            if (y >= params_.height) break;

            for (int x = 0; x < params_.width; ++x) {
                img.at(x, y) = shadePixel(camera, x, y);
            }

            int done = rowsDone.fetch_add(1) + 1;
            if (done % 20 == 0 || done == params_.height) {
                std::cerr << "Rendered row " << done << " / " << params_.height << "\r" << std::flush;
            }
        }
    };

    std::cerr << "Rendering with " << nThreads << " threads...\n";
    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(nThreads));
    for (int i = 0; i < nThreads; ++i) threads.emplace_back(worker);
    for (auto& t : threads) t.join();

    const auto renderEnd = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(renderEnd - renderStart).count();

    const double pixels = static_cast<double>(params_.width) * static_cast<double>(params_.height);
    int root = static_cast<int>(std::floor(std::sqrt(static_cast<double>(params_.samplesPerPixel))));
    if (root < 1) root = 1;
    const int actualSpp = (params_.samplesPerPixel <= 1) ? 1 : root * root;
    const double sampleRays = pixels * static_cast<double>(actualSpp);

    std::cerr << "\n";
    std::cerr << "Frame render time: " << seconds << " s"
              << " | mode=" << params_.renderMode
              << " | image=" << params_.width << "x" << params_.height
              << " | spp=" << actualSpp
              << " | threads=" << nThreads;

    if (seconds > 0.0) {
        std::cerr << " | " << (pixels / (1.0e6 * seconds)) << " Mpix/s"
                  << " | " << (sampleRays / (1.0e6 * seconds)) << " Mrays/s";
    }

    std::cerr << "\n";

    return img;
}
