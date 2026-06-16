#include "RendererCuda.h"
#include "BlackHoleSystem.h"
#include "NumericalMetricData.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using real = float;
#ifdef CUDA_GEOMETRY_DOUBLE
using geom_real = double;
constexpr geom_real GEOM_PI = 3.141592653589793238462643383279502884;
#else
using geom_real = float;
constexpr geom_real GEOM_PI = 3.141592653589793238462643383279502884f;
#endif

constexpr real PI_F = 3.141592653589793238462643383279502884f;

#define CUDA_CHECK(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_err) + \
                                 " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    } \
} while (0)

template <class T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (ptr_) cudaFree(ptr_);
    }

    void upload(const std::vector<T>& host) {
        count_ = host.size();
        if (count_ == 0) {
            ptr_ = nullptr;
            return;
        }
        CUDA_CHECK(cudaMalloc(&ptr_, count_ * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(ptr_, host.data(), count_ * sizeof(T), cudaMemcpyHostToDevice));
    }

    void allocate(std::size_t count) {
        count_ = count;
        if (count_ == 0) {
            ptr_ = nullptr;
            return;
        }
        CUDA_CHECK(cudaMalloc(&ptr_, count_ * sizeof(T)));
    }

    T* get() const { return ptr_; }
    std::size_t size() const { return count_; }

private:
    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

struct GpuVec3 {
    real x, y, z;
};

struct GpuVec3d {
    double x, y, z;
};

struct GpuGeoVec3 {
    geom_real x, y, z;
};

struct GpuRGBf {
    real r, g, b;
};

struct GpuRGB8 {
    unsigned char r, g, b;
};

struct GpuComplex {
    real re, im;
};

struct GpuModeKey {
    int l;
    int m;
};

struct GpuSample {
    real t;
    real re;
    real im;
    real slopeRe;
    real slopeIm;
};

struct GpuSeries {
    int sampleOffset;
    int sampleCount;
};

struct GpuShell {
    real radius;
};

struct GpuNumericalMetricGrid {
    int familyId;
    int layer;
    int nx, ny, nz;
    real centerX, centerY, centerZ;
    real halfWidth;
    real dx;
    long long points;
    long long dataOffsetFloats;
    long long pointOffset;
    int gridIndex;
};

struct GpuCamera {
    GpuVec3 origin;
    GpuVec3 forward;
    GpuVec3 right;
    GpuVec3 trueUp;
    real tanHalfFovY;
    real aspect;
};

struct GpuParams {
    int width;
    int height;
    int samplesPerPixel;

    real stepSize;
    real rayTMin;
    real rayTMax;
    real waveVolumeRadius;
    real transmittanceCutoff;
    real time;

    real rInner;
    real innerWaveScaleRadius;

    int useTortoiseRetardedTime;
    real tortoiseMass;
    real tortoiseRadiusFloor;
    real tortoiseSafetyEps;

    real panoramaYawDegrees;
    real panoramaExposure;

    int paraviewScalarUsesR;
    int paraviewOpacityRadialFalloffEnabled;
    real paraviewOpacityReferenceRadius;
    real paraviewOpacityFalloffPower;
    real paraviewOuterFadeWidth;

    int gwpvPeaksNumPeaks;
    real gwpvPeaksFirstPosition;
    real gwpvPeaksLastPosition;
    real gwpvPeaksFirstOpacity;
    real gwpvPeaksLastOpacity;
    real gwpvPeaksStrength;
    real gwpvScalarOpacityUnitDistance;

    int gwpvUseAxisMask;
    int axisMaskEnabled;
    real axisMaskInnerRadius;
    real axisMaskOuterRadius;

    int gwpvUseOpacityRadialEnvelope;

    real waveBrightness;
    real maxStepAlpha;
    real outputGamma;

    int numShells;
    int numModes;
    int panoramaWidth;
    int panoramaHeight;

    int blackHolesEnabled;
    int blackHoleStateValid;
    GpuVec3 bhPlusCenter;
    GpuVec3 bhMinusCenter;
    real bhPlusRenderRadius;
    real bhMinusRenderRadius;
    real bhPlusCaptureRadius;
    real bhMinusCaptureRadius;
    real bhPlusMetricMass;
    real bhMinusMetricMass;

    int metricLensingEnabled;
    int metricUseMajumdarPapapetrou;
    real metricMPSoftening;
    real cudaMetricStep;
    real cudaMetricColorStep;
    real cudaDoublePixelRadiusFraction;
    int cudaMetricMaxSteps;

    int numericalMetricEnabled;
    int numericalMetricGridCount;
    int numericalMetricMaxLayer;
    int numericalMetricUseTimeDerivatives;
    real numericalMetricBoundaryBufferCells;
    real numericalMetricMetricFailAbs;
    real numericalMetricGammaFailAbs;
    real numericalMetricAccelFailAbs;
    int numericalMetricFailFast;
    int numericalMetricDebugMask;
    long long numericalMetricTotalPoints;
    int numericalAh1Valid;
    int numericalAh2Valid;
    GpuVec3 numericalAh1Center;
    GpuVec3 numericalAh2Center;
    real numericalAh1CaptureRadius;
    real numericalAh2CaptureRadius;
};

struct GpuSceneData {
    const GpuShell* shells;
    const GpuModeKey* modeKeys;
    const GpuSeries* series;   // indexed by shell * numModes + mode
    const GpuSample* samples;
    const GpuRGBf* panorama;

    const GpuNumericalMetricGrid* numericalMetricGrids;
    // Precomputed Christoffel payload, field-major per grid:
    //   fields 0..9   = metric components g00..g33
    //   fields 10..49 = Gamma0_00..Gamma3_33
    const real* numericalMetricFields;
    const real* numericalMetricChristoffel; // legacy/unused; kept to avoid ABI churn inside this file
};

__host__ __device__ inline GpuVec3 makeVec3(real x, real y, real z) {
    return GpuVec3{x, y, z};
}

__host__ __device__ inline GpuVec3 operator+(const GpuVec3& a, const GpuVec3& b) {
    return makeVec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline GpuVec3 operator-(const GpuVec3& a, const GpuVec3& b) {
    return makeVec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline GpuVec3 operator*(real s, const GpuVec3& v) {
    return makeVec3(s * v.x, s * v.y, s * v.z);
}

__host__ __device__ inline GpuVec3 operator*(const GpuVec3& v, real s) {
    return s * v;
}

__host__ __device__ inline real dot3(const GpuVec3& a, const GpuVec3& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__host__ __device__ inline real length2_3(const GpuVec3& v) {
    return dot3(v, v);
}

__host__ __device__ inline real length3(const GpuVec3& v) {
    return sqrtf(length2_3(v));
}

__host__ __device__ inline GpuVec3 normalize3(const GpuVec3& v) {
    real L = length3(v);
    if (!(L > real(0))) return makeVec3(0, 0, 0);
    return (real(1) / L) * v;
}

__host__ __device__ inline real clampReal(real x, real a, real b) {
    return fminf(fmaxf(x, a), b);
}

__host__ __device__ inline GpuGeoVec3 makeGeoVec3(geom_real x, geom_real y, geom_real z) {
    return GpuGeoVec3{x, y, z};
}

__host__ __device__ inline GpuGeoVec3 toGeoVec3(const GpuVec3& v) {
    return makeGeoVec3(static_cast<geom_real>(v.x), static_cast<geom_real>(v.y), static_cast<geom_real>(v.z));
}

__host__ __device__ inline GpuVec3 toGpuVec3f(const GpuGeoVec3& v) {
    return makeVec3(static_cast<real>(v.x), static_cast<real>(v.y), static_cast<real>(v.z));
}

__host__ __device__ inline GpuGeoVec3 operator+(const GpuGeoVec3& a, const GpuGeoVec3& b) {
    return makeGeoVec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline GpuGeoVec3 operator-(const GpuGeoVec3& a, const GpuGeoVec3& b) {
    return makeGeoVec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline GpuGeoVec3 operator*(geom_real s, const GpuGeoVec3& v) {
    return makeGeoVec3(s * v.x, s * v.y, s * v.z);
}

__host__ __device__ inline GpuGeoVec3 operator*(const GpuGeoVec3& v, geom_real s) {
    return s * v;
}

__host__ __device__ inline GpuGeoVec3 operator/(const GpuGeoVec3& v, geom_real s) {
    return makeGeoVec3(v.x / s, v.y / s, v.z / s);
}

__host__ __device__ inline geom_real dot3g(const GpuGeoVec3& a, const GpuGeoVec3& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__host__ __device__ inline geom_real length2_3g(const GpuGeoVec3& v) {
    return dot3g(v, v);
}

__host__ __device__ inline geom_real length3g(const GpuGeoVec3& v) {
#ifdef CUDA_GEOMETRY_DOUBLE
    return sqrt(length2_3g(v));
#else
    return sqrtf(length2_3g(v));
#endif
}

__host__ __device__ inline GpuGeoVec3 normalize3g(const GpuGeoVec3& v) {
    const geom_real L = length3g(v);
    if (!(L > geom_real(0))) return makeGeoVec3(0, 0, 0);
    return (geom_real(1) / L) * v;
}

__host__ __device__ inline geom_real clampGeom(geom_real x, geom_real a, geom_real b) {
#ifdef CUDA_GEOMETRY_DOUBLE
    return fmin(fmax(x, a), b);
#else
    return fminf(fmaxf(x, a), b);
#endif
}

__host__ __device__ inline bool finiteGeom(geom_real x) {
    return isfinite(x);
}


__host__ __device__ inline GpuVec3d makeVec3d(double x, double y, double z) {
    return GpuVec3d{x, y, z};
}

__host__ __device__ inline GpuVec3d toVec3dGpu(const GpuVec3& v) {
    return makeVec3d(static_cast<double>(v.x), static_cast<double>(v.y), static_cast<double>(v.z));
}

__host__ __device__ inline GpuVec3 toVec3fGpu(const GpuVec3d& v) {
    return makeVec3(static_cast<real>(v.x), static_cast<real>(v.y), static_cast<real>(v.z));
}

__host__ __device__ inline GpuVec3d operator+(const GpuVec3d& a, const GpuVec3d& b) {
    return makeVec3d(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline GpuVec3d operator-(const GpuVec3d& a, const GpuVec3d& b) {
    return makeVec3d(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline GpuVec3d operator*(double s, const GpuVec3d& v) {
    return makeVec3d(s * v.x, s * v.y, s * v.z);
}

__host__ __device__ inline GpuVec3d operator*(const GpuVec3d& v, double s) {
    return s * v;
}

__host__ __device__ inline double dot3d(const GpuVec3d& a, const GpuVec3d& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__host__ __device__ inline double length2_3d(const GpuVec3d& v) {
    return dot3d(v, v);
}

__host__ __device__ inline double length3d(const GpuVec3d& v) {
    return sqrt(length2_3d(v));
}

__host__ __device__ inline GpuVec3d normalize3d(const GpuVec3d& v) {
    const double L = length3d(v);
    if (!(L > 0.0)) return makeVec3d(0.0, 0.0, 0.0);
    return (1.0 / L) * v;
}

__host__ __device__ inline real clamp01f(real x) {
    return clampReal(x, real(0), real(1));
}

__host__ __device__ inline real smoothstepGpu(real edge0, real edge1, real x) {
    if (edge1 <= edge0) return (x >= edge1) ? real(1) : real(0);
    real t = clamp01f((x - edge0) / (edge1 - edge0));
    return t * t * (real(3) - real(2) * t);
}

__host__ __device__ inline GpuRGBf rgbAdd(const GpuRGBf& a, const GpuRGBf& b) {
    return GpuRGBf{a.r + b.r, a.g + b.g, a.b + b.b};
}

__host__ __device__ inline GpuRGBf rgbMul(const GpuRGBf& a, real s) {
    return GpuRGBf{a.r * s, a.g * s, a.b * s};
}

__host__ __device__ inline GpuRGBf rgbMix(const GpuRGBf& a, const GpuRGBf& b, real t) {
    t = clamp01f(t);
    return rgbAdd(rgbMul(a, real(1) - t), rgbMul(b, t));
}

__host__ __device__ inline GpuComplex cx(real re, real im) { return GpuComplex{re, im}; }
__host__ __device__ inline GpuComplex cxAdd(GpuComplex a, GpuComplex b) { return cx(a.re + b.re, a.im + b.im); }
__host__ __device__ inline GpuComplex cxSub(GpuComplex a, GpuComplex b) { return cx(a.re - b.re, a.im - b.im); }
__host__ __device__ inline GpuComplex cxMul(GpuComplex a, GpuComplex b) { return cx(a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re); }
__host__ __device__ inline GpuComplex cxMulReal(GpuComplex a, real s) { return cx(a.re*s, a.im*s); }
__host__ __device__ inline GpuComplex cxAddMul(GpuComplex a, GpuComplex b, real s) { return cx(a.re + b.re*s, a.im + b.im*s); }

__device__ real factorialSmall(int n) {
    if (n < 0) return real(0);
    real out = real(1);
    for (int i = 2; i <= n; ++i) out *= static_cast<real>(i);
    return out;
}

__device__ real powIntPositive(real x, int n) {
    real out = real(1);
    for (int i = 0; i < n; ++i) out *= x;
    return out;
}

__device__ real wignerSmallD(int l, int mp, int m, real theta) {
    const real pref = sqrtf(
        factorialSmall(l + m) * factorialSmall(l - m) *
        factorialSmall(l + mp) * factorialSmall(l - mp)
    );

    const real ct = cosf(real(0.5) * theta);
    const real st = sinf(real(0.5) * theta);

    real total = real(0);
    const int kmin = (m - mp > 0) ? (m - mp) : 0;
    const int kmax = (l + m < l - mp) ? (l + m) : (l - mp);

    for (int k = kmin; k <= kmax; ++k) {
        const real denom =
            factorialSmall(l + m - k) *
            factorialSmall(k) *
            factorialSmall(mp - m + k) *
            factorialSmall(l - mp - k);

        const int expC = 2 * l + m - mp - 2 * k;
        const int expS = mp - m + 2 * k;
        const real sign = ((k - m + mp) & 1) ? real(-1) : real(1);
        const real coeff = sign * pref / denom;
        total += coeff * powIntPositive(ct, expC) * powIntPositive(st, expS);
    }

    return total;
}

__device__ GpuComplex spinWeightedYDevice(int l, int m, int s, real theta, real phi) {
    if ((s < 0 ? -s : s) > l || (m < 0 ? -m : m) > l) return cx(0, 0);

    const real sign = (s & 1) ? real(-1) : real(1);
    const real amp = sign * sqrtf((real(2) * l + real(1)) / (real(4) * PI_F));
    const real d = wignerSmallD(l, m, -s, theta);
    const real phase = static_cast<real>(m) * phi;
    return cx(amp * d * cosf(phase), amp * d * sinf(phase));
}

__device__ real tortoiseRadiusGpu(real r, const GpuParams& p) {
    if (!p.useTortoiseRetardedTime) return r;
    const real M = fmaxf(real(0), p.tortoiseMass);
    if (!(M > real(0))) return r;

    const real horizon = real(2) * M;
    const real floorFromHorizon = horizon * (real(1) + fmaxf(real(0), p.tortoiseSafetyEps));
    const real floorRadius = fmaxf(fmaxf(p.tortoiseRadiusFloor, p.innerWaveScaleRadius), floorFromHorizon);
    const real rr = fmaxf(r, floorRadius);
    return rr + real(2) * M * logf(rr / horizon - real(1));
}

__device__ GpuComplex interpolateModeCubicGpu(const GpuSample* samples, int offset, int count, real t) {
    if (count <= 0) return cx(0, 0);
    const GpuSample* s = samples + offset;

    if (t < s[0].t || t > s[count - 1].t) return cx(0, 0);
    if (count == 1 || t <= s[0].t) return cx(s[0].re, s[0].im);
    if (t >= s[count - 1].t) return cx(s[count - 1].re, s[count - 1].im);

    int lo = 0;
    int hi = count - 1;
    while (lo + 1 < hi) {
        int mid = (lo + hi) >> 1;
        if (s[mid].t < t) lo = mid;
        else hi = mid;
    }

    const GpuSample a = s[lo];
    const GpuSample b = s[hi];
    const real h = b.t - a.t;
    if (!(h > real(0))) return cx(a.re, a.im);

    real u = (t - a.t) / h;
    u = clamp01f(u);
    const real u2 = u * u;
    const real u3 = u2 * u;

    const real h00 =  real(2) * u3 - real(3) * u2 + real(1);
    const real h10 =          u3 - real(2) * u2 + u;
    const real h01 = -real(2) * u3 + real(3) * u2;
    const real h11 =          u3 -          u2;

    const real re = h00 * a.re + h10 * h * a.slopeRe + h01 * b.re + h11 * h * b.slopeRe;
    const real im = h00 * a.im + h10 * h * a.slopeIm + h01 * b.im + h11 * h * b.slopeIm;
    return cx(re, im);
}

__device__ GpuComplex sampleShellW(int shellIndex,
                                   int modeIndex,
                                   real r,
                                   real t,
                                   const GpuParams& p,
                                   const GpuSceneData& scene) {
    const GpuSeries ser = scene.series[shellIndex * p.numModes + modeIndex];
    if (ser.sampleCount <= 0) return cx(0, 0);

    const real rStar = tortoiseRadiusGpu(r, p);
    const real u = t - rStar;
    const real shellR = scene.shells[shellIndex].radius;
    const real shellTimeRadius = tortoiseRadiusGpu(shellR, p);
    const real tShell = u + shellTimeRadius;

    const GpuComplex interp = interpolateModeCubicGpu(scene.samples, ser.sampleOffset, ser.sampleCount, tShell);
    return cxMulReal(interp, shellR);
}

__device__ GpuComplex coefficientAtRadiusTimeGpu(int modeIndex,
                                                 real r,
                                                 real t,
                                                 const GpuParams& p,
                                                 const GpuSceneData& scene) {
    if (p.numShells <= 0 || p.numModes <= 0) return cx(0, 0);

    const real rScale = fmaxf(r, p.innerWaveScaleRadius);
    const real invRScale = (rScale > real(0)) ? real(1) / rScale : real(0);

    if (p.numShells == 1) {
        return cxMulReal(sampleShellW(0, modeIndex, r, t, p, scene), invRScale);
    }

    if (r <= scene.shells[0].radius) {
        return cxMulReal(sampleShellW(0, modeIndex, r, t, p, scene), invRScale);
    }

    if (r >= scene.shells[p.numShells - 1].radius) {
        return cxMulReal(sampleShellW(p.numShells - 1, modeIndex, r, t, p, scene), invRScale);
    }

    int hi = 1;
    while (hi < p.numShells && scene.shells[hi].radius < r) ++hi;
    const int lo = hi - 1;

    const real ra = scene.shells[lo].radius;
    const real rb = scene.shells[hi].radius;
    const real denom = rb - ra;
    real frac = (denom > real(0)) ? (r - ra) / denom : real(0);
    frac = clamp01f(frac);

    const GpuComplex wa = sampleShellW(lo, modeIndex, r, t, p, scene);
    const GpuComplex wb = sampleShellW(hi, modeIndex, r, t, p, scene);
    const GpuComplex w = cxAdd(cxMulReal(wa, real(1) - frac), cxMulReal(wb, frac));
    return cxMulReal(w, invRScale);
}

__device__ GpuComplex evalFieldGpu(const GpuVec3& x,
                                   real t,
                                   const GpuParams& p,
                                   const GpuSceneData& scene) {
    const real r = length3(x);
    if (r <= p.rInner) return cx(0, 0);

    const real zr = clampReal(x.z / fmaxf(r, real(1.0e-30)), real(-1), real(1));
    const real theta = acosf(zr);
    const real phi = atan2f(x.y, x.x);

    GpuComplex field = cx(0, 0);
    for (int i = 0; i < p.numModes; ++i) {
        const GpuModeKey key = scene.modeKeys[i];
        const GpuComplex c = coefficientAtRadiusTimeGpu(i, r, t, p, scene);
        const GpuComplex ylm = spinWeightedYDevice(key.l, key.m, -2, theta, phi);
        field = cxAdd(field, cxMul(c, ylm));
    }
    return field;
}

__device__ bool intersectSphereOriginGpu(const GpuVec3& origin,
                                         const GpuVec3& dir,
                                         real radius,
                                         real& tEnter,
                                         real& tExit) {
    const real b = dot3(origin, dir);
    const real c = dot3(origin, origin) - radius * radius;
    const real disc = b * b - c;
    if (disc < real(0)) return false;
    const real s = sqrtf(disc);
    tEnter = -b - s;
    tExit = -b + s;
    return tExit >= real(0);
}

__device__ bool intersectSphereAtGpu(const GpuVec3& origin,
                                     const GpuVec3& dir,
                                     const GpuVec3& center,
                                     real radius,
                                     real& tHit) {
    if (!(radius > real(0))) return false;
    const GpuVec3 oc = origin - center;
    const real b = dot3(oc, dir);
    const real c = dot3(oc, oc) - radius * radius;
    const real disc = b * b - c;
    if (disc < real(0)) return false;
    const real s = sqrtf(disc);
    const real t0 = -b - s;
    const real t1 = -b + s;
    if (t1 < real(0)) return false;
    tHit = (t0 >= real(0)) ? t0 : t1;
    return isfinite(tHit);
}

__device__ bool nearestBlackHoleHitGpu(const GpuVec3& origin,
                                       const GpuVec3& dir,
                                       const GpuParams& p,
                                       real& tHit) {
    if (!p.blackHolesEnabled || !p.blackHoleStateValid) return false;

    bool hit = false;
    tHit = real(3.4e38);
    real t = real(0);
    if (intersectSphereAtGpu(origin, dir, p.bhPlusCenter, p.bhPlusRenderRadius, t) && t < tHit) {
        hit = true;
        tHit = t;
    }
    if (intersectSphereAtGpu(origin, dir, p.bhMinusCenter, p.bhMinusRenderRadius, t) && t < tHit) {
        hit = true;
        tHit = t;
    }
    return hit;
}

__device__ bool segmentSphereHitGpu(const GpuVec3& x0,
                                    const GpuVec3& x1,
                                    const GpuVec3& center,
                                    real radius,
                                    real& uHit) {
    if (!(radius > real(0))) return false;

    const GpuVec3 d = x1 - x0;
    const GpuVec3 oc = x0 - center;
    const real A = dot3(d, d);
    if (!(A > real(1.0e-30))) return false;

    const real C = dot3(oc, oc) - radius * radius;
    if (C <= real(0)) {
        uHit = real(0);
        return true;
    }

    const real B = real(2) * dot3(oc, d);
    const real disc = B * B - real(4) * A * C;
    if (disc < real(0)) return false;

    const real s = sqrtf(disc);
    const real inv2A = real(1) / (real(2) * A);
    const real u0 = (-B - s) * inv2A;
    const real u1 = (-B + s) * inv2A;

    real u = real(3.4e38);
    if (u0 >= real(0) && u0 <= real(1)) u = u0;
    else if (u1 >= real(0) && u1 <= real(1)) u = u1;

    if (!isfinite(u)) return false;
    uHit = clampReal(u, real(0), real(1));
    return true;
}

__device__ bool nearestBlackHoleCaptureSegmentHitGpu(const GpuVec3& x0,
                                                     const GpuVec3& x1,
                                                     const GpuParams& p,
                                                     real& uHit) {
    if (!p.blackHolesEnabled || !p.blackHoleStateValid) return false;

    bool hit = false;
    uHit = real(3.4e38);
    real u = real(0);

    if (segmentSphereHitGpu(x0, x1, p.bhPlusCenter, p.bhPlusCaptureRadius, u) && u < uHit) {
        hit = true;
        uHit = u;
    }
    if (segmentSphereHitGpu(x0, x1, p.bhMinusCenter, p.bhMinusCaptureRadius, u) && u < uHit) {
        hit = true;
        uHit = u;
    }

    return hit;
}

struct GpuMPPotential {
    geom_real U;
    GpuGeoVec3 gradU;
};

__device__ void addMPSourceGpu(const GpuGeoVec3& x,
                               const GpuGeoVec3& center,
                               geom_real mass,
                               geom_real softening,
                               GpuMPPotential& out) {
    if (!(mass > geom_real(0))) return;

    const GpuGeoVec3 d = x - center;
#ifdef CUDA_GEOMETRY_DOUBLE
    const geom_real eps = fmax(geom_real(0), softening);
#else
    const geom_real eps = fmaxf(geom_real(0), softening);
#endif
    const geom_real r2 = dot3g(d, d) + eps * eps;
#ifdef CUDA_GEOMETRY_DOUBLE
    const geom_real r = sqrt(fmax(r2, geom_real(1.0e-30)));
#else
    const geom_real r = sqrtf(fmaxf(r2, geom_real(1.0e-30)));
#endif

    out.U += mass / r;

    const geom_real invR3 = geom_real(1) / ((r2 * r > geom_real(1.0e-30)) ? (r2 * r) : geom_real(1.0e-30));
    out.gradU = out.gradU + (-mass * invR3) * d;
}

__device__ GpuMPPotential mpPotentialGpu(const GpuGeoVec3& x, const GpuParams& p) {
    GpuMPPotential out;
    out.U = geom_real(1);
    out.gradU = makeGeoVec3(geom_real(0), geom_real(0), geom_real(0));

    if (!p.metricUseMajumdarPapapetrou || !p.blackHolesEnabled || !p.blackHoleStateValid) {
        return out;
    }

    addMPSourceGpu(x, toGeoVec3(p.bhPlusCenter),  static_cast<geom_real>(p.bhPlusMetricMass),
                   static_cast<geom_real>(p.metricMPSoftening), out);
    addMPSourceGpu(x, toGeoVec3(p.bhMinusCenter), static_cast<geom_real>(p.bhMinusMetricMass),
                   static_cast<geom_real>(p.metricMPSoftening), out);

    if (!finiteGeom(out.U) || out.U < geom_real(1.0e-12)) {
        out.U = geom_real(1);
        out.gradU = makeGeoVec3(geom_real(0), geom_real(0), geom_real(0));
    }
    return out;
}

__device__ GpuGeoVec3 normalizeOpticalVelocityGpu(const GpuGeoVec3& x,
                                                  const GpuGeoVec3& v,
                                                  const GpuParams& p) {
    GpuGeoVec3 dir = normalize3g(v);
    if (length2_3g(dir) <= geom_real(0)) dir = makeGeoVec3(geom_real(1), geom_real(0), geom_real(0));

    const GpuMPPotential mp = mpPotentialGpu(x, p);
    const geom_real invU2 = geom_real(1) / ((mp.U * mp.U > geom_real(1.0e-12)) ? (mp.U * mp.U) : geom_real(1.0e-12));
    return invU2 * dir;
}

struct GpuGeoState {
    GpuGeoVec3 x;
    GpuGeoVec3 v;
};

struct GpuGeoDeriv {
    GpuGeoVec3 dx;
    GpuGeoVec3 dv;
};

__device__ GpuGeoVec3 mpAccelerationGpu(const GpuGeoVec3& x,
                                        const GpuGeoVec3& v,
                                        const GpuParams& p) {
    const GpuMPPotential mp = mpPotentialGpu(x, p);
    if (!(mp.U > geom_real(1.0e-12))) return makeGeoVec3(geom_real(0), geom_real(0), geom_real(0));

    const GpuGeoVec3 gradLogU = (geom_real(1) / mp.U) * mp.gradU;
    const geom_real v2 = dot3g(v, v);
    const geom_real vg = dot3g(v, gradLogU);

    return (geom_real(2) * v2) * gradLogU - (geom_real(4) * vg) * v;
}

__device__ GpuGeoDeriv geodesicDerivMPGpu(const GpuGeoState& s,
                                          const GpuParams& p) {
    GpuGeoDeriv d;
    d.dx = s.v;
    d.dv = mpAccelerationGpu(s.x, s.v, p);
    return d;
}

__device__ GpuGeoState addGeoScaled(const GpuGeoState& y,
                                    const GpuGeoDeriv& k,
                                    geom_real h) {
    GpuGeoState out;
    out.x = y.x + h * k.dx;
    out.v = y.v + h * k.dv;
    return out;
}

__device__ GpuGeoState rk4StepMPGpu(const GpuGeoState& y,
                                    geom_real h,
                                    const GpuParams& p) {
    const GpuGeoDeriv k1 = geodesicDerivMPGpu(y, p);
    const GpuGeoDeriv k2 = geodesicDerivMPGpu(addGeoScaled(y, k1, geom_real(0.5) * h), p);
    const GpuGeoDeriv k3 = geodesicDerivMPGpu(addGeoScaled(y, k2, geom_real(0.5) * h), p);
    const GpuGeoDeriv k4 = geodesicDerivMPGpu(addGeoScaled(y, k3, h), p);

    GpuGeoState out;
    out.x = y.x + (h / geom_real(6)) * (k1.dx + geom_real(2) * k2.dx + geom_real(2) * k3.dx + k4.dx);
    out.v = y.v + (h / geom_real(6)) * (k1.dv + geom_real(2) * k2.dv + geom_real(2) * k3.dv + k4.dv);
    out.v = normalizeOpticalVelocityGpu(out.x, out.v, p);
    return out;
}

__device__ bool segmentSphereHitGeoGpu(const GpuGeoVec3& x0,
                                       const GpuGeoVec3& x1,
                                       const GpuGeoVec3& center,
                                       geom_real radius,
                                       geom_real& uHit) {
    if (!(radius > geom_real(0))) return false;

    const GpuGeoVec3 d = x1 - x0;
    const GpuGeoVec3 oc = x0 - center;
    const geom_real a = dot3g(d, d);
    const geom_real c = dot3g(oc, oc) - radius * radius;

    // Important for capture: if a previous finite step landed inside the
    // inflated AH sphere, terminate immediately instead of waiting for an
    // exit crossing.  The old geo version missed this case and could run to
    // cudaMetricMaxSteps for swallowed rays.
    if (c <= geom_real(0)) {
        uHit = geom_real(0);
        return true;
    }

    if (!(a > geom_real(0))) return false;
    const geom_real b = geom_real(2) * dot3g(oc, d);
    const geom_real disc = b*b - geom_real(4)*a*c;
    if (!(disc >= geom_real(0))) return false;
#ifdef CUDA_GEOMETRY_DOUBLE
    const geom_real sqrtDisc = sqrt(disc);
#else
    const geom_real sqrtDisc = sqrtf(disc);
#endif
    const geom_real inv2a = geom_real(0.5) / a;
    const geom_real t0 = (-b - sqrtDisc) * inv2a;
    const geom_real t1 = (-b + sqrtDisc) * inv2a;
    bool hit = false;
    uHit = geom_real(1.0e300);
    if (t0 >= geom_real(0) && t0 <= geom_real(1)) { uHit = t0; hit = true; }
    if (t1 >= geom_real(0) && t1 <= geom_real(1) && t1 < uHit) { uHit = t1; hit = true; }
    return hit;
}

__device__ bool nearestBlackHoleCaptureSegmentHitGeoGpu(const GpuGeoVec3& x0,
                                                        const GpuGeoVec3& x1,
                                                        const GpuParams& p,
                                                        geom_real& uHit) {
    if (!p.blackHolesEnabled || !p.blackHoleStateValid) return false;
    bool hit = false;
    uHit = geom_real(1.0e300);
    geom_real u = geom_real(0);
    if (segmentSphereHitGeoGpu(x0, x1, toGeoVec3(p.bhPlusCenter), static_cast<geom_real>(p.bhPlusCaptureRadius), u) && u < uHit) {
        hit = true; uHit = u;
    }
    if (segmentSphereHitGeoGpu(x0, x1, toGeoVec3(p.bhMinusCenter), static_cast<geom_real>(p.bhMinusCaptureRadius), u) && u < uHit) {
        hit = true; uHit = u;
    }
    return hit;
}

struct GpuMPPotentialD {
    double U;
    GpuVec3d gradU;
};

__device__ void addMPSourceGpuD(const GpuVec3d& x,
                                const GpuVec3d& center,
                                double mass,
                                double softening,
                                GpuMPPotentialD& out) {
    if (!(mass > 0.0)) return;

    const GpuVec3d d = x - center;
    const double eps = fmax(0.0, softening);
    const double r2 = dot3d(d, d) + eps * eps;
    const double r = sqrt(fmax(r2, 1.0e-30));

    out.U += mass / r;

    const double invR3 = 1.0 / fmax(r2 * r, 1.0e-30);
    out.gradU = out.gradU + (-mass * invR3) * d;
}

__device__ GpuMPPotentialD mpPotentialGpuD(const GpuVec3d& x, const GpuParams& p) {
    GpuMPPotentialD out;
    out.U = 1.0;
    out.gradU = makeVec3d(0.0, 0.0, 0.0);

    if (!p.metricUseMajumdarPapapetrou || !p.blackHolesEnabled || !p.blackHoleStateValid) {
        return out;
    }

    addMPSourceGpuD(x, toVec3dGpu(p.bhPlusCenter),
                    static_cast<double>(p.bhPlusMetricMass),
                    static_cast<double>(p.metricMPSoftening),
                    out);
    addMPSourceGpuD(x, toVec3dGpu(p.bhMinusCenter),
                    static_cast<double>(p.bhMinusMetricMass),
                    static_cast<double>(p.metricMPSoftening),
                    out);

    if (!isfinite(out.U) || out.U < 1.0e-12) {
        out.U = 1.0;
        out.gradU = makeVec3d(0.0, 0.0, 0.0);
    }

    return out;
}

__device__ GpuVec3d normalizeOpticalVelocityGpuD(const GpuVec3d& x,
                                                 const GpuVec3d& v,
                                                 const GpuParams& p) {
    GpuVec3d dir = normalize3d(v);
    if (length2_3d(dir) <= 0.0) dir = makeVec3d(1.0, 0.0, 0.0);

    const GpuMPPotentialD mp = mpPotentialGpuD(x, p);
    const double invU2 = 1.0 / fmax(mp.U * mp.U, 1.0e-12);
    return invU2 * dir;
}

struct GpuGeoStateD {
    GpuVec3d x;
    GpuVec3d v;
};

struct GpuGeoDerivD {
    GpuVec3d dx;
    GpuVec3d dv;
};

__device__ GpuVec3d mpAccelerationGpuD(const GpuVec3d& x,
                                       const GpuVec3d& v,
                                       const GpuParams& p) {
    const GpuMPPotentialD mp = mpPotentialGpuD(x, p);
    if (!(mp.U > 1.0e-12)) return makeVec3d(0.0, 0.0, 0.0);

    const GpuVec3d gradLogU = (1.0 / mp.U) * mp.gradU;
    const double v2 = dot3d(v, v);
    const double vg = dot3d(v, gradLogU);

    return (2.0 * v2) * gradLogU - (4.0 * vg) * v;
}

__device__ GpuGeoDerivD geodesicDerivMPGpuD(const GpuGeoStateD& s,
                                            const GpuParams& p) {
    GpuGeoDerivD d;
    d.dx = s.v;
    d.dv = mpAccelerationGpuD(s.x, s.v, p);
    return d;
}

__device__ GpuGeoStateD addGeoScaledD(const GpuGeoStateD& y,
                                      const GpuGeoDerivD& k,
                                      double h) {
    GpuGeoStateD out;
    out.x = y.x + h * k.dx;
    out.v = y.v + h * k.dv;
    return out;
}

__device__ GpuGeoStateD rk4StepMPGpuD(const GpuGeoStateD& y,
                                      double h,
                                      const GpuParams& p) {
    const GpuGeoDerivD k1 = geodesicDerivMPGpuD(y, p);
    const GpuGeoDerivD k2 = geodesicDerivMPGpuD(addGeoScaledD(y, k1, 0.5 * h), p);
    const GpuGeoDerivD k3 = geodesicDerivMPGpuD(addGeoScaledD(y, k2, 0.5 * h), p);
    const GpuGeoDerivD k4 = geodesicDerivMPGpuD(addGeoScaledD(y, k3, h), p);

    GpuGeoStateD out;
    out.x = y.x + (h / 6.0) * (k1.dx + 2.0 * k2.dx + 2.0 * k3.dx + k4.dx);
    out.v = y.v + (h / 6.0) * (k1.dv + 2.0 * k2.dv + 2.0 * k3.dv + k4.dv);
    out.v = normalizeOpticalVelocityGpuD(out.x, out.v, p);
    return out;
}

__device__ bool segmentSphereHitGpuD(const GpuVec3d& x0,
                                     const GpuVec3d& x1,
                                     const GpuVec3d& center,
                                     double radius,
                                     double& uHit) {
    if (!(radius > 0.0)) return false;

    const GpuVec3d d = x1 - x0;
    const GpuVec3d oc = x0 - center;
    const double A = dot3d(d, d);
    if (!(A > 1.0e-30)) return false;

    const double C = dot3d(oc, oc) - radius * radius;
    if (C <= 0.0) {
        uHit = 0.0;
        return true;
    }

    const double B = 2.0 * dot3d(oc, d);
    double disc = B * B - 4.0 * A * C;
    if (disc < 0.0) {
        const double scale = fabs(B * B) + fabs(4.0 * A * C) + 1.0;
        if (disc > -1.0e-12 * scale) disc = 0.0;
        else return false;
    }

    const double s = sqrt(disc);
    const double inv2A = 1.0 / (2.0 * A);
    const double u0 = (-B - s) * inv2A;
    const double u1 = (-B + s) * inv2A;

    double u = 1.0e300;
    if (u0 >= 0.0 && u0 <= 1.0) u = u0;
    else if (u1 >= 0.0 && u1 <= 1.0) u = u1;

    if (!isfinite(u) || u > 2.0) return false;
    uHit = fmin(fmax(u, 0.0), 1.0);
    return true;
}

__device__ bool nearestBlackHoleCaptureSegmentHitGpuD(const GpuVec3d& x0,
                                                      const GpuVec3d& x1,
                                                      const GpuParams& p,
                                                      double& uHit) {
    if (!p.blackHolesEnabled || !p.blackHoleStateValid) return false;

    bool hit = false;
    uHit = 1.0e300;
    double u = 0.0;

    if (segmentSphereHitGpuD(x0, x1, toVec3dGpu(p.bhPlusCenter),
                             static_cast<double>(p.bhPlusCaptureRadius), u) && u < uHit) {
        hit = true;
        uHit = u;
    }
    if (segmentSphereHitGpuD(x0, x1, toVec3dGpu(p.bhMinusCenter),
                             static_cast<double>(p.bhMinusCaptureRadius), u) && u < uHit) {
        hit = true;
        uHit = u;
    }

    return hit;
}


__device__ GpuRGBf hsvToRgbGpu(real h, real s, real v) {
    h = h - floorf(h);
    s = clamp01f(s);
    v = fmaxf(real(0), v);

    const real x = h * real(6);
    const int i = static_cast<int>(floorf(x));
    const real f = x - static_cast<real>(i);
    const real p = v * (real(1) - s);
    const real q = v * (real(1) - s * f);
    const real t = v * (real(1) - s * (real(1) - f));

    switch (i % 6) {
        case 0: return GpuRGBf{v, t, p};
        case 1: return GpuRGBf{q, v, p};
        case 2: return GpuRGBf{p, v, t};
        case 3: return GpuRGBf{p, q, v};
        case 4: return GpuRGBf{t, p, v};
        default: return GpuRGBf{v, p, q};
    }
}

__device__ GpuRGBf gwpvPeakColorGpu(real scalar, const GpuParams& p) {
    const real a = p.gwpvPeaksFirstPosition;
    const real b = p.gwpvPeaksLastPosition;
    real u = real(0);
    if (b > a) u = clamp01f((scalar - a) / (b - a));
    const real hue = (real(2) / real(3)) * (real(1) - u);
    return rgbMul(hsvToRgbGpu(hue, real(0.95), real(1)), p.waveBrightness);
}

__device__ real gwpvPeakOpacityGpu(real scalar, const GpuParams& p) {
    if (scalar <= real(0)) return real(0);

    const int n = (p.gwpvPeaksNumPeaks > 1) ? p.gwpvPeaksNumPeaks : 1;
    const real firstPos = p.gwpvPeaksFirstPosition;
    const real lastPos = p.gwpvPeaksLastPosition;
    const real firstOpacity = fmaxf(real(0), p.gwpvPeaksFirstOpacity);
    const real lastOpacity = fmaxf(real(0), p.gwpvPeaksLastOpacity);

    const real spacing = (n > 1 && lastPos > firstPos)
        ? (lastPos - firstPos) / static_cast<real>(n - 1)
        : fmaxf(real(1.0e-6), fabsf(lastPos - firstPos));
    const real peakDecay = real(0.5) * fmaxf(real(1.0e-12), spacing);

    real opacity = real(0);
    for (int i = 0; i < n; ++i) {
        const real u = (n > 1) ? static_cast<real>(i) / static_cast<real>(n - 1) : real(0);
        const real peak = firstPos + (lastPos - firstPos) * u;
        const real peakOpacity = firstOpacity + (lastOpacity - firstOpacity) * u;

        const real left = peak - peakDecay / real(100);
        const real right = peak + peakDecay;

        real opHere = real(0);
        if (scalar >= left && scalar <= peak) {
            opHere = peakOpacity * (scalar - left) / fmaxf(peak - left, real(1.0e-30));
        } else if (scalar > peak && scalar <= right) {
            opHere = peakOpacity * (real(1) - (scalar - peak) / fmaxf(right - peak, real(1.0e-30)));
        }
        opacity = fmaxf(opacity, opHere);
    }

    opacity *= fmaxf(real(0), p.gwpvPeaksStrength);
    return clampReal(opacity, real(0), real(0.999999));
}

__device__ real alphaFromTransferOpacityGpu(real opacityOverUnitDistance,
                                            real stepDistance,
                                            real scalarOpacityUnitDistance) {
    const real A = clampReal(opacityOverUnitDistance, real(0), real(0.999999));
    if (A <= real(0) || stepDistance <= real(0)) return real(0);
    const real U = fmaxf(real(1.0e-9), scalarOpacityUnitDistance);
    return real(1) - expf(log1pf(-A) * (stepDistance / U));
}

__device__ real axisMaskFactorGpu(const GpuVec3& x, const GpuParams& p) {
    if (!p.axisMaskEnabled) return real(1);
    const real rhoPerp = sqrtf(x.x*x.x + x.y*x.y);
    return smoothstepGpu(p.axisMaskInnerRadius, p.axisMaskOuterRadius, rhoPerp);
}

__device__ real opacityRadialEnvelopeGpu(real r, const GpuParams& p) {
    real envelope = real(1);

    if (p.paraviewOpacityRadialFalloffEnabled) {
        const real ref = fmaxf(real(1.0e-9), p.paraviewOpacityReferenceRadius);
        const real rr = fmaxf(r, ref);
        const real power = fmaxf(real(0), p.paraviewOpacityFalloffPower);
        envelope *= powf(ref / rr, power);
    }

    if (p.paraviewOuterFadeWidth > real(0)) {
        const real R = fmaxf(real(1.0e-9), p.waveVolumeRadius);
        const real w = fminf(p.paraviewOuterFadeWidth, R);
        envelope *= real(1) - smoothstepGpu(R - w, R, r);
    }

    return clamp01f(envelope);
}

__device__ void compositeGwpvPointGpu(const GpuVec3& x,
                                      real ds,
                                      const GpuParams& p,
                                      const GpuSceneData& scene,
                                      GpuRGBf& accum,
                                      real& T) {
    if (ds <= real(0) || T <= p.transmittanceCutoff) return;

    const GpuComplex psi = evalFieldGpu(x, p.time, p, scene);
    const real r = length3(x);
    const real scalar = p.paraviewScalarUsesR
        ? fmaxf(r, p.innerWaveScaleRadius) * psi.re
        : psi.re;

    real opacity = gwpvPeakOpacityGpu(scalar, p);
    if (opacity <= real(0)) return;

    if (p.gwpvUseAxisMask) opacity *= axisMaskFactorGpu(x, p);
    if (p.gwpvUseOpacityRadialEnvelope) opacity *= opacityRadialEnvelopeGpu(r, p);
    if (opacity <= real(0)) return;

    GpuRGBf c = gwpvPeakColorGpu(scalar, p);
    real alpha = alphaFromTransferOpacityGpu(opacity, ds, p.gwpvScalarOpacityUnitDistance);
    alpha = clampReal(alpha, real(0), p.maxStepAlpha);

    accum = rgbAdd(accum, rgbMul(c, T * alpha));
    T *= (real(1) - alpha);
}

__device__ real wrap01Gpu(real x) {
    x = x - floorf(x);
    return x;
}

__device__ GpuRGBf samplePanoramaGpu(const GpuVec3& dIn,
                                     const GpuParams& p,
                                     const GpuSceneData& scene) {
    if (p.panoramaWidth <= 0 || p.panoramaHeight <= 0 || !scene.panorama) {
        return GpuRGBf{0, 0, 0};
    }

    const GpuVec3 d = normalize3(dIn);
    if (length2_3(d) <= real(0)) return GpuRGBf{0, 0, 0};

    const real yaw = p.panoramaYawDegrees / real(360);
    const real u = wrap01Gpu(real(0.5) + atan2f(d.y, d.x) / (real(2) * PI_F) + yaw);
    const real v = acosf(clampReal(d.z, real(-1), real(1))) / PI_F;

    const real px = u * static_cast<real>(p.panoramaWidth);
    const real py = v * static_cast<real>(p.panoramaHeight - 1);

    const int x0 = static_cast<int>(floorf(px)) % p.panoramaWidth;
    const int x1 = (x0 + 1) % p.panoramaWidth;
    int y0 = static_cast<int>(floorf(py));
    if (y0 < 0) y0 = 0;
    if (y0 > p.panoramaHeight - 1) y0 = p.panoramaHeight - 1;
    int y1 = y0 + 1;
    if (y1 > p.panoramaHeight - 1) y1 = p.panoramaHeight - 1;

    const real tx = px - floorf(px);
    const real ty = py - floorf(py);

    const GpuRGBf c00 = scene.panorama[y0 * p.panoramaWidth + x0];
    const GpuRGBf c10 = scene.panorama[y0 * p.panoramaWidth + x1];
    const GpuRGBf c01 = scene.panorama[y1 * p.panoramaWidth + x0];
    const GpuRGBf c11 = scene.panorama[y1 * p.panoramaWidth + x1];

    const GpuRGBf c0 = rgbMix(c00, c10, tx);
    const GpuRGBf c1 = rgbMix(c01, c11, tx);
    return rgbMul(rgbMix(c0, c1, ty), p.panoramaExposure);
}

__device__ unsigned char toByteGpu(real x, real outputGamma) {
    x = clamp01f(x);
    if (outputGamma > real(0) && outputGamma != real(1)) {
        x = powf(x, real(1) / outputGamma);
    }
    return static_cast<unsigned char>(floorf(real(255) * clamp01f(x) + real(0.5)));
}

__device__ GpuRGB8 makeRGB8Gpu(const GpuRGBf& c, real outputGamma) {
    return GpuRGB8{toByteGpu(c.r, outputGamma), toByteGpu(c.g, outputGamma), toByteGpu(c.b, outputGamma)};
}

__device__ unsigned int hashInt(unsigned int x, unsigned int y, unsigned int s, unsigned int channel) {
    unsigned int v = x * 1973u ^ y * 9277u ^ s * 26699u ^ channel * 31847u ^ 0x9E3779B9u;
    v ^= v >> 16;
    v *= 0x7feb352du;
    v ^= v >> 15;
    v *= 0x846ca68bu;
    v ^= v >> 16;
    return v;
}

__device__ real hash01Gpu(int x, int y, int s, int channel) {
    return static_cast<real>(hashInt(static_cast<unsigned int>(x), static_cast<unsigned int>(y),
                                     static_cast<unsigned int>(s), static_cast<unsigned int>(channel))) /
           static_cast<real>(0xffffffffu);
}

__device__ void generateRayGpu(const GpuCamera& cam,
                               real pixelX,
                               real pixelY,
                               int width,
                               int height,
                               GpuVec3& origin,
                               GpuVec3& dir) {
    const real u = (pixelX + real(0.5)) / static_cast<real>(width);
    const real v = (pixelY + real(0.5)) / static_cast<real>(height);

    const real sx = (real(2) * u - real(1)) * cam.aspect * cam.tanHalfFovY;
    const real sy = (real(1) - real(2) * v) * cam.tanHalfFovY;

    origin = cam.origin;
    dir = normalize3(cam.forward + sx * cam.right + sy * cam.trueUp);
}

__device__ GpuRGBf shadeRayFixedGpu(const GpuVec3& rayOrigin,
                                    const GpuVec3& rayDir,
                                    const GpuParams& p,
                                    const GpuSceneData& scene) {
    real tEnter = real(0);
    real tExit = real(0);
    const bool hitWaveSphere = intersectSphereOriginGpu(rayOrigin, rayDir, p.waveVolumeRadius, tEnter, tExit);

    GpuRGBf accum{0, 0, 0};
    real T = real(1);

    real bhT = real(0);
    const bool hasBHSurfaceHit = nearestBlackHoleHitGpu(rayOrigin, rayDir, p, bhT) &&
                                 bhT >= fmaxf(p.rayTMin, real(0)) && bhT <= p.rayTMax;

    if (hitWaveSphere) {
        real s0 = fmaxf(fmaxf(tEnter, p.rayTMin), real(0));
        real s1 = fminf(tExit, p.rayTMax);
        if (hasBHSurfaceHit) s1 = fminf(s1, bhT);

        if (s1 > s0) {
            const real step = fmaxf(p.stepSize, real(1.0e-6));
            for (real s = s0; s < s1;) {
                const real ds = fminf(step, s1 - s);
                const real smid = s + real(0.5) * ds;
                const GpuVec3 x = rayOrigin + smid * rayDir;
                compositeGwpvPointGpu(x, ds, p, scene, accum, T);
                if (T <= p.transmittanceCutoff) break;
                s += ds;
            }
        }
    }

    if (hasBHSurfaceHit) {
        // Explicit non-lensed black-hole spheres are rendered as black in CUDA v0.
        // Keep already accumulated foreground volume and terminate without background.
        return accum;
    }

    const GpuRGBf bg = samplePanoramaGpu(rayDir, p, scene);
    accum = rgbAdd(accum, rgbMul(bg, T));
    return accum;
}


__device__ GpuRGBf shadeRayMPGpu(const GpuVec3& rayOrigin,
                                 const GpuVec3& rayDir,
                                 const GpuParams& p,
                                 const GpuSceneData& scene) {
    real tEnter = real(0);
    real tExit = real(0);
    const bool hitWaveSphere = intersectSphereOriginGpu(rayOrigin, rayDir, p.waveVolumeRadius, tEnter, tExit);

    GpuRGBf accum{0, 0, 0};
    real T = real(1);
    GpuVec3 finalDir = rayDir;

    if (!hitWaveSphere) {
        const GpuRGBf bg = samplePanoramaGpu(rayDir, p, scene);
        return rgbMul(bg, T);
    }

    const real s0 = fmaxf(fmaxf(tEnter, p.rayTMin), real(0));
    const geom_real h = (static_cast<geom_real>(p.cudaMetricStep) > geom_real(1.0e-6)) ? static_cast<geom_real>(p.cudaMetricStep) : geom_real(1.0e-6);
    const geom_real colorStep = (static_cast<geom_real>(p.cudaMetricColorStep) > geom_real(1.0e-6)) ? static_cast<geom_real>(p.cudaMetricColorStep) : geom_real(1.0e-6);
    const int maxSteps = (p.cudaMetricMaxSteps > 1) ? p.cudaMetricMaxSteps : 1;

    GpuGeoState state;
    state.x = toGeoVec3(rayOrigin + s0 * rayDir);
    state.v = normalizeOpticalVelocityGpu(state.x, toGeoVec3(rayDir), p);
    finalDir = normalize3(toGpuVec3f(state.v));

    for (int stepIndex = 0; stepIndex < maxSteps && T > p.transmittanceCutoff; ++stepIndex) {
        const geom_real rNow = length3g(state.x);
        if (rNow >= static_cast<geom_real>(p.waveVolumeRadius) && dot3g(state.x, state.v) > geom_real(0)) {
            break;
        }

        const GpuGeoState yStart = state;
        GpuGeoState yEnd = rk4StepMPGpu(state, h, p);

        const GpuGeoVec3 x0 = yStart.x;
        const GpuGeoVec3 x1 = yEnd.x;
        const GpuGeoVec3 dx = x1 - x0;
        const geom_real segLen = length3g(dx);

        if (finiteGeom(segLen) && segLen > geom_real(0)) {
#ifdef CUDA_GEOMETRY_DOUBLE
            const int nSubRaw = static_cast<int>(ceil(segLen / colorStep));
#else
            const int nSubRaw = static_cast<int>(ceilf(segLen / colorStep));
#endif
            const int nSub = (nSubRaw > 1) ? nSubRaw : 1;
            const geom_real ds = segLen / static_cast<geom_real>(nSub);

            for (int i = 0; i < nSub; ++i) {
                const geom_real a0 = static_cast<geom_real>(i) / static_cast<geom_real>(nSub);
                const geom_real a1 = static_cast<geom_real>(i + 1) / static_cast<geom_real>(nSub);
                const GpuGeoVec3 xA = x0 + a0 * dx;
                const GpuGeoVec3 xB = x0 + a1 * dx;

                geom_real uHit = geom_real(0);
                if (nearestBlackHoleCaptureSegmentHitGeoGpu(xA, xB, p, uHit)) {
                    const GpuGeoVec3 subDx = xB - xA;
                    const GpuGeoVec3 xHit = xA + uHit * subDx;
                    const geom_real dsBefore = length3g(xHit - xA);

                    if (finiteGeom(dsBefore) && dsBefore > geom_real(1.0e-6) && length3g(xA) <= static_cast<geom_real>(p.waveVolumeRadius)) {
                        const GpuGeoVec3 xMid = xA + (geom_real(0.5) * uHit) * subDx;
                        compositeGwpvPointGpu(toGpuVec3f(xMid), static_cast<real>(dsBefore), p, scene, accum, T);
                    }

                    return accum;
                }

                const geom_real aMid = geom_real(0.5) * (a0 + a1);
                const GpuGeoVec3 xMid = x0 + aMid * dx;
                if (length3g(xMid) <= static_cast<geom_real>(p.waveVolumeRadius)) {
                    compositeGwpvPointGpu(toGpuVec3f(xMid), static_cast<real>(ds), p, scene, accum, T);
                    if (T <= p.transmittanceCutoff) break;
                }
            }
        }

        state = yEnd;
        finalDir = normalize3(toGpuVec3f(state.v));
    }

    if (!isfinite(finalDir.x) || !isfinite(finalDir.y) || !isfinite(finalDir.z) || length2_3(finalDir) <= real(1.0e-20)) {
        finalDir = rayDir;
    }
    if (!isfinite(finalDir.x) || !isfinite(finalDir.y) || !isfinite(finalDir.z) || length2_3(finalDir) <= real(1.0e-20)) {
        finalDir = rayDir;
    }
    const GpuRGBf bg = samplePanoramaGpu(finalDir, p, scene);
    accum = rgbAdd(accum, rgbMul(bg, T));
    return accum;
}


__device__ GpuRGBf shadeRayMPDoubleGpu(const GpuVec3& rayOrigin,
                                       const GpuVec3& rayDir,
                                       const GpuParams& p,
                                       const GpuSceneData& scene) {
    real tEnter = real(0);
    real tExit = real(0);
    const bool hitWaveSphere = intersectSphereOriginGpu(rayOrigin, rayDir, p.waveVolumeRadius, tEnter, tExit);

    GpuRGBf accum{0, 0, 0};
    real T = real(1);
    GpuVec3 finalDir = rayDir;

    if (!hitWaveSphere) {
        const GpuRGBf bg = samplePanoramaGpu(rayDir, p, scene);
        return rgbMul(bg, T);
    }

    const double s0 = static_cast<double>(fmaxf(fmaxf(tEnter, p.rayTMin), real(0)));
    const double h = fmax(static_cast<double>(p.cudaMetricStep), 1.0e-6);
    const double colorStep = fmax(static_cast<double>(p.cudaMetricColorStep), 1.0e-6);
    const int maxSteps = (p.cudaMetricMaxSteps > 1) ? p.cudaMetricMaxSteps : 1;

    GpuGeoStateD state;
    state.x = toVec3dGpu(rayOrigin) + s0 * toVec3dGpu(rayDir);
    state.v = normalizeOpticalVelocityGpuD(state.x, toVec3dGpu(rayDir), p);
    finalDir = normalize3(toVec3fGpu(state.v));

    for (int stepIndex = 0; stepIndex < maxSteps && T > p.transmittanceCutoff; ++stepIndex) {
        const double rNow = length3d(state.x);
        if (rNow >= static_cast<double>(p.waveVolumeRadius) && dot3d(state.x, state.v) > 0.0) {
            break;
        }

        const GpuGeoStateD yStart = state;
        const GpuGeoStateD yEnd = rk4StepMPGpuD(state, h, p);

        const GpuVec3d x0 = yStart.x;
        const GpuVec3d x1 = yEnd.x;
        const GpuVec3d dx = x1 - x0;
        const double segLen = length3d(dx);

        if (isfinite(segLen) && segLen > 0.0) {
            const int nSubRaw = static_cast<int>(ceil(segLen / colorStep));
            const int nSub = (nSubRaw > 1) ? nSubRaw : 1;
            const double ds = segLen / static_cast<double>(nSub);

            for (int i = 0; i < nSub; ++i) {
                const double a0 = static_cast<double>(i) / static_cast<double>(nSub);
                const double a1 = static_cast<double>(i + 1) / static_cast<double>(nSub);
                const GpuVec3d xA = x0 + a0 * dx;
                const GpuVec3d xB = x0 + a1 * dx;

                double uHit = 0.0;
                if (nearestBlackHoleCaptureSegmentHitGpuD(xA, xB, p, uHit)) {
                    const GpuVec3d subDx = xB - xA;
                    const GpuVec3d xHit = xA + uHit * subDx;
                    const double dsBefore = length3d(xHit - xA);

                    if (isfinite(dsBefore) && dsBefore > 1.0e-6 && length3d(xA) <= static_cast<double>(p.waveVolumeRadius)) {
                        const GpuVec3d xMid = xA + (0.5 * uHit) * subDx;
                        compositeGwpvPointGpu(toVec3fGpu(xMid), static_cast<real>(dsBefore), p, scene, accum, T);
                    }
                    return accum;
                }

                const double aMid = 0.5 * (a0 + a1);
                const GpuVec3d xMid = x0 + aMid * dx;
                if (length3d(xMid) <= static_cast<double>(p.waveVolumeRadius)) {
                    compositeGwpvPointGpu(toVec3fGpu(xMid), static_cast<real>(ds), p, scene, accum, T);
                    if (T <= p.transmittanceCutoff) break;
                }
            }
        }

        state = yEnd;
        finalDir = normalize3(toVec3fGpu(state.v));
    }

    if (!isfinite(finalDir.x) || !isfinite(finalDir.y) || !isfinite(finalDir.z) || length2_3(finalDir) <= real(1.0e-20)) {
        finalDir = rayDir;
    }
    const GpuRGBf bg = samplePanoramaGpu(finalDir, p, scene);
    accum = rgbAdd(accum, rgbMul(bg, T));
    return accum;
}


__device__ int metricPairIndexGpu(int a, int b) {
    if (a > b) { int t = a; a = b; b = t; }
    if (a == 0) return b;                  // 00,01,02,03 -> 0..3
    if (a == 1) return 4 + (b - 1);        // 11,12,13 -> 4..6
    if (a == 2) return 7 + (b - 2);        // 22,23 -> 7..8
    return 9;                              // 33
}

__device__ bool invert4x4Gpu(const real g[16], real inv[16]) {
    real a[4][8];
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) a[r][c] = g[4*r + c];
        for (int c = 0; c < 4; ++c) a[r][4+c] = (r == c) ? real(1) : real(0);
    }

    for (int col = 0; col < 4; ++col) {
        int pivot = col;
        real best = fabsf(a[col][col]);
        for (int r = col + 1; r < 4; ++r) {
            const real v = fabsf(a[r][col]);
            if (v > best) { best = v; pivot = r; }
        }
        if (!(best > real(1.0e-20)) || !isfinite(best)) return false;

        if (pivot != col) {
            for (int c = 0; c < 8; ++c) {
                const real tmp = a[col][c];
                a[col][c] = a[pivot][c];
                a[pivot][c] = tmp;
            }
        }

        const real invPivot = real(1) / a[col][col];
        for (int c = 0; c < 8; ++c) a[col][c] *= invPivot;

        for (int r = 0; r < 4; ++r) {
            if (r == col) continue;
            const real f = a[r][col];
            if (f == real(0)) continue;
            for (int c = 0; c < 8; ++c) a[r][c] -= f * a[col][c];
        }
    }

    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) inv[4*r + c] = a[r][4+c];
    }
    return true;
}

__global__ void precomputeNumericalChristoffelKernel(const GpuNumericalMetricGrid* grids,
                                                     int nGrids,
                                                     long long totalPoints,
                                                     const real* fields,
                                                     real* gammaOut,
                                                     int useTimeDerivatives) {
    const long long gid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= totalPoints) return;

    int gi = -1;
    long long local = 0;
    for (int g = 0; g < nGrids; ++g) {
        const long long b = grids[g].pointOffset;
        const long long e = b + grids[g].points;
        if (gid >= b && gid < e) {
            gi = g;
            local = gid - b;
            break;
        }
    }
    if (gi < 0) return;

    const GpuNumericalMetricGrid grid = grids[gi];

    real g10[10];
    real dg[4][10];
    for (int pidx = 0; pidx < 10; ++pidx) {
        g10[pidx] = fields[grid.dataOffsetFloats + static_cast<long long>(pidx) * grid.points + local];
        dg[0][pidx] = useTimeDerivatives ? fields[grid.dataOffsetFloats + static_cast<long long>(10 + pidx) * grid.points + local] : real(0);
        dg[1][pidx] = fields[grid.dataOffsetFloats + static_cast<long long>(20 + pidx) * grid.points + local];
        dg[2][pidx] = fields[grid.dataOffsetFloats + static_cast<long long>(30 + pidx) * grid.points + local];
        dg[3][pidx] = fields[grid.dataOffsetFloats + static_cast<long long>(40 + pidx) * grid.points + local];
    }

    real gmat[16];
    for (int a = 0; a < 4; ++a) {
        for (int b = 0; b < 4; ++b) {
            gmat[4*a + b] = g10[metricPairIndexGpu(a, b)];
        }
    }

    real ginv[16];
    if (!invert4x4Gpu(gmat, ginv)) {
        for (int c = 0; c < 40; ++c) gammaOut[gid * 40 + c] = real(0);
        return;
    }

    for (int mu = 0; mu < 4; ++mu) {
        for (int a = 0; a < 4; ++a) {
            for (int b = a; b < 4; ++b) {
                real sum = real(0);
                for (int nu = 0; nu < 4; ++nu) {
                    const int bnu = metricPairIndexGpu(b, nu);
                    const int anu = metricPairIndexGpu(a, nu);
                    const int ab  = metricPairIndexGpu(a, b);
                    sum += ginv[4*mu + nu] * (dg[a][bnu] + dg[b][anu] - dg[nu][ab]);
                }
                gammaOut[gid * 40 + mu * 10 + metricPairIndexGpu(a, b)] = real(0.5) * sum;
            }
        }
    }
}

__device__ geom_real numericalMetricBoundaryCellsGpu(const GpuNumericalMetricGrid& g,
                                                          const GpuGeoVec3& x) {
    const geom_real xmin = static_cast<geom_real>(g.centerX) - static_cast<geom_real>(g.halfWidth);
    const geom_real xmax = static_cast<geom_real>(g.centerX) + static_cast<geom_real>(g.halfWidth);
    const geom_real ymin = static_cast<geom_real>(g.centerY) - static_cast<geom_real>(g.halfWidth);
    const geom_real ymax = static_cast<geom_real>(g.centerY) + static_cast<geom_real>(g.halfWidth);
    const geom_real zmin = static_cast<geom_real>(g.centerZ) - static_cast<geom_real>(g.halfWidth);
    const geom_real zmax = static_cast<geom_real>(g.centerZ) + static_cast<geom_real>(g.halfWidth);

    const geom_real d0 = x.x - xmin;
    const geom_real d1 = xmax - x.x;
    const geom_real d2 = x.y - ymin;
    const geom_real d3 = ymax - x.y;
    const geom_real d4 = x.z - zmin;
    const geom_real d5 = zmax - x.z;

#ifdef CUDA_GEOMETRY_DOUBLE
    geom_real d = fmin(fmin(fmin(d0, d1), fmin(d2, d3)), fmin(d4, d5));
#else
    geom_real d = fminf(fminf(fminf(d0, d1), fminf(d2, d3)), fminf(d4, d5));
#endif
    return d / static_cast<geom_real>(g.dx);
}

__device__ bool numericalMetricGridContainsPointGpu(const GpuNumericalMetricGrid& g,
                                                    const GpuGeoVec3& x) {
    const geom_real dx = static_cast<geom_real>(x.x) - static_cast<geom_real>(g.centerX);
    const geom_real dy = static_cast<geom_real>(x.y) - static_cast<geom_real>(g.centerY);
    const geom_real dz = static_cast<geom_real>(x.z) - static_cast<geom_real>(g.centerZ);
    const geom_real h = static_cast<geom_real>(g.halfWidth);
    return dx >= -h && dx <= h && dy >= -h && dy <= h && dz >= -h && dz <= h;
}

__device__ const GpuNumericalMetricGrid* selectNumericalMetricGridGpu(const GpuGeoVec3& x,
                                                                      const GpuParams& p,
                                                                      const GpuSceneData& scene,
                                                                      geom_real* boundaryCellsOut = nullptr) {
    if (boundaryCellsOut) *boundaryCellsOut = geom_real(-1);
    if (!p.numericalMetricEnabled || p.numericalMetricGridCount <= 0 || !scene.numericalMetricGrids) return nullptr;

    const GpuNumericalMetricGrid* best = nullptr;
    int bestLayer = -999999;
    geom_real bestBoundaryCells = geom_real(-1);
    const geom_real B = static_cast<geom_real>(p.numericalMetricBoundaryBufferCells);

    for (int i = 0; i < p.numericalMetricGridCount; ++i) {
        const GpuNumericalMetricGrid* g = scene.numericalMetricGrids + i;
        if (g->layer > p.numericalMetricMaxLayer) continue;
        if (g->layer < bestLayer) continue;
        if (!numericalMetricGridContainsPointGpu(*g, x)) continue;

        const geom_real boundaryCells = numericalMetricBoundaryCellsGpu(*g, x);
        if (B > geom_real(0) && !(boundaryCells > B)) continue;

        best = g;
        bestLayer = g->layer;
        bestBoundaryCells = boundaryCells;
    }

    if (boundaryCellsOut) *boundaryCellsOut = bestBoundaryCells;
    return best;
}

__device__ long long numericalMetricGridLinearIndexGpu(const GpuNumericalMetricGrid& g, int ix, int iy, int iz) {
    return static_cast<long long>(ix) + static_cast<long long>(g.nx) * (static_cast<long long>(iy) + static_cast<long long>(g.ny) * static_cast<long long>(iz));
}

__device__ bool numericalMetricGridCellGpu(const GpuNumericalMetricGrid& g,
                                           const GpuGeoVec3& x,
                                           int& ix0, int& iy0, int& iz0,
                                           geom_real& tx, geom_real& ty, geom_real& tz) {
    const geom_real x0 = static_cast<geom_real>(g.centerX) - static_cast<geom_real>(g.halfWidth);
    const geom_real y0 = static_cast<geom_real>(g.centerY) - static_cast<geom_real>(g.halfWidth);
    const geom_real z0 = static_cast<geom_real>(g.centerZ) - static_cast<geom_real>(g.halfWidth);
    const geom_real invDx = geom_real(1) / static_cast<geom_real>(g.dx);

    const geom_real rx = (x.x - x0) * invDx;
    const geom_real ry = (x.y - y0) * invDx;
    const geom_real rz = (x.z - z0) * invDx;

    if (rx < geom_real(0) || ry < geom_real(0) || rz < geom_real(0) ||
        rx > static_cast<geom_real>(g.nx - 1) ||
        ry > static_cast<geom_real>(g.ny - 1) ||
        rz > static_cast<geom_real>(g.nz - 1)) {
        return false;
    }

#ifdef CUDA_GEOMETRY_DOUBLE
    ix0 = static_cast<int>(floor(rx));
    iy0 = static_cast<int>(floor(ry));
    iz0 = static_cast<int>(floor(rz));
#else
    ix0 = static_cast<int>(floorf(rx));
    iy0 = static_cast<int>(floorf(ry));
    iz0 = static_cast<int>(floorf(rz));
#endif
    if (ix0 >= g.nx - 1) ix0 = g.nx - 2;
    if (iy0 >= g.ny - 1) iy0 = g.ny - 2;
    if (iz0 >= g.nz - 1) iz0 = g.nz - 2;
    if (ix0 < 0 || iy0 < 0 || iz0 < 0) return false;

    tx = clampGeom(rx - static_cast<geom_real>(ix0), geom_real(0), geom_real(1));
    ty = clampGeom(ry - static_cast<geom_real>(iy0), geom_real(0), geom_real(1));
    tz = clampGeom(rz - static_cast<geom_real>(iz0), geom_real(0), geom_real(1));
    return true;
}


enum NumericalMetricFailureReason {
    NUM_FAIL_NONE = 0,
    NUM_FAIL_NONFINITE_METRIC = 2,
    NUM_FAIL_HUGE_METRIC = 3,
    NUM_FAIL_NONFINITE_GAMMA = 4,
    NUM_FAIL_HUGE_GAMMA = 5,
    NUM_FAIL_NONFINITE_ACCEL = 6,
    NUM_FAIL_HUGE_ACCEL = 7,
    NUM_FAIL_NONFINITE_RAY_STATE = 8
};

struct GpuNumericalFailure {
    int failed;
    int reason;
    int pixelX;
    int pixelY;
    int sampleIndex;
    int rkStep;
    int gridIndex;
    int familyId;
    int layer;
    int fieldIndex;
    int component;
    real value;
    real threshold;
    real x;
    real y;
    real z;
    real boundaryCells;
};

__device__ void recordNumericalFailureGpu(GpuNumericalFailure* failure,
                                          const GpuParams& p,
                                          int reason,
                                          int pixelX,
                                          int pixelY,
                                          int sampleIndex,
                                          int rkStep,
                                          const GpuNumericalMetricGrid* grid,
                                          geom_real boundaryCells,
                                          const GpuGeoVec3& x,
                                          int fieldIndex,
                                          int component,
                                          geom_real value,
                                          geom_real threshold) {
    if (!failure || !p.numericalMetricFailFast) return;
    if (atomicCAS(&failure->failed, 0, 1) != 0) return;

    failure->reason = reason;
    failure->pixelX = pixelX;
    failure->pixelY = pixelY;
    failure->sampleIndex = sampleIndex;
    failure->rkStep = rkStep;
    failure->gridIndex = grid ? grid->gridIndex : -1;
    failure->familyId = grid ? grid->familyId : -1;
    failure->layer = grid ? grid->layer : -1;
    failure->fieldIndex = fieldIndex;
    failure->component = component;
    failure->value = static_cast<real>(value);
    failure->threshold = static_cast<real>(threshold);
    failure->x = static_cast<real>(x.x);
    failure->y = static_cast<real>(x.y);
    failure->z = static_cast<real>(x.z);
    failure->boundaryCells = static_cast<real>(boundaryCells);
}

__host__ const char* numericalFailureReasonName(int reason) {
    switch (reason) {
        case NUM_FAIL_NONFINITE_METRIC: return "nonfinite metric interpolation";
        case NUM_FAIL_HUGE_METRIC: return "huge metric interpolation";
        case NUM_FAIL_NONFINITE_GAMMA: return "nonfinite Christoffel interpolation";
        case NUM_FAIL_HUGE_GAMMA: return "huge Christoffel interpolation";
        case NUM_FAIL_NONFINITE_ACCEL: return "nonfinite geodesic acceleration";
        case NUM_FAIL_HUGE_ACCEL: return "huge geodesic acceleration";
        case NUM_FAIL_NONFINITE_RAY_STATE: return "nonfinite ray state";
        default: return "unknown numerical failure";
    }
}

__device__ bool sampleNumericalMetricGpu(const GpuGeoVec3& x,
                                         const GpuParams& p,
                                         const GpuSceneData& scene,
                                         real g10[10],
                                         GpuNumericalFailure* failure = nullptr,
                                         int pixelX = -1,
                                         int pixelY = -1,
                                         int sampleIndex = -1,
                                         int rkStep = -1) {
    geom_real boundaryCells = geom_real(-1);
    const GpuNumericalMetricGrid* gpGrid = selectNumericalMetricGridGpu(x, p, scene, &boundaryCells);
    if (!gpGrid || !scene.numericalMetricFields) return false;
    const GpuNumericalMetricGrid g = *gpGrid;

    int ix0=0, iy0=0, iz0=0;
    geom_real tx=0, ty=0, tz=0;
    if (!numericalMetricGridCellGpu(g, x, ix0, iy0, iz0, tx, ty, tz)) return false;

    for (int c = 0; c < 10; ++c) g10[c] = real(0);

    for (int dz = 0; dz <= 1; ++dz) {
        const geom_real wz = dz ? tz : (geom_real(1) - tz);
        for (int dy = 0; dy <= 1; ++dy) {
            const geom_real wy = dy ? ty : (geom_real(1) - ty);
            for (int dx = 0; dx <= 1; ++dx) {
                const geom_real wx = dx ? tx : (geom_real(1) - tx);
                const real w = static_cast<real>(wx * wy * wz);
                const long long local = numericalMetricGridLinearIndexGpu(g, ix0 + dx, iy0 + dy, iz0 + dz);
                for (int c = 0; c < 10; ++c) {
                    g10[c] += w * scene.numericalMetricFields[g.dataOffsetFloats + static_cast<long long>(c) * g.points + local];
                }
            }
        }
    }

    const geom_real metricLimit = static_cast<geom_real>(p.numericalMetricMetricFailAbs);
    for (int c = 0; c < 10; ++c) {
        const geom_real v = static_cast<geom_real>(g10[c]);
        if (!finiteGeom(v)) {
            recordNumericalFailureGpu(failure, p, NUM_FAIL_NONFINITE_METRIC,
                                      pixelX, pixelY, sampleIndex, rkStep,
                                      &g, boundaryCells, x, c, c, v, metricLimit);
            return false;
        }
        if (metricLimit > geom_real(0) && fabs(static_cast<double>(v)) > static_cast<double>(metricLimit)) {
            recordNumericalFailureGpu(failure, p, NUM_FAIL_HUGE_METRIC,
                                      pixelX, pixelY, sampleIndex, rkStep,
                                      &g, boundaryCells, x, c, c, v, metricLimit);
            return false;
        }
    }
    return true;
}

__device__ bool sampleNumericalChristoffelGpu(const GpuGeoVec3& x,
                                              const GpuParams& p,
                                              const GpuSceneData& scene,
                                              geom_real gamma[40],
                                              GpuNumericalFailure* failure = nullptr,
                                              int pixelX = -1,
                                              int pixelY = -1,
                                              int sampleIndex = -1,
                                              int rkStep = -1) {
    geom_real boundaryCells = geom_real(-1);
    const GpuNumericalMetricGrid* gpGrid = selectNumericalMetricGridGpu(x, p, scene, &boundaryCells);
    if (!gpGrid || !scene.numericalMetricFields) return false;
    const GpuNumericalMetricGrid g = *gpGrid;

    int ix0=0, iy0=0, iz0=0;
    geom_real tx=0, ty=0, tz=0;
    if (!numericalMetricGridCellGpu(g, x, ix0, iy0, iz0, tx, ty, tz)) return false;

    for (int c = 0; c < 40; ++c) gamma[c] = geom_real(0);

    // The renderer now expects metric_4d_plus_christoffel_v1 files:
    //   field 0..9   = metric components
    //   field 10..49 = precomputed Christoffel components
    // This deliberately avoids constructing Gamma inside renderCuda.
    for (int dz = 0; dz <= 1; ++dz) {
        const geom_real wz = dz ? tz : (geom_real(1) - tz);
        for (int dy = 0; dy <= 1; ++dy) {
            const geom_real wy = dy ? ty : (geom_real(1) - ty);
            for (int dx = 0; dx <= 1; ++dx) {
                const geom_real wx = dx ? tx : (geom_real(1) - tx);
                const geom_real w = wx * wy * wz;
                const long long local = numericalMetricGridLinearIndexGpu(g, ix0 + dx, iy0 + dy, iz0 + dz);
                for (int c = 0; c < 40; ++c) {
                    gamma[c] += w * static_cast<geom_real>(
                        scene.numericalMetricFields[g.dataOffsetFloats + static_cast<long long>(10 + c) * g.points + local]);
                }
            }
        }
    }

    const geom_real gammaLimit = static_cast<geom_real>(p.numericalMetricGammaFailAbs);
    for (int c = 0; c < 40; ++c) {
        const geom_real v = gamma[c];
        if (!finiteGeom(v)) {
            recordNumericalFailureGpu(failure, p, NUM_FAIL_NONFINITE_GAMMA,
                                      pixelX, pixelY, sampleIndex, rkStep,
                                      &g, boundaryCells, x, 10 + c, c, v, gammaLimit);
            return false;
        }
        if (gammaLimit > geom_real(0) && fabs(static_cast<double>(v)) > static_cast<double>(gammaLimit)) {
            recordNumericalFailureGpu(failure, p, NUM_FAIL_HUGE_GAMMA,
                                      pixelX, pixelY, sampleIndex, rkStep,
                                      &g, boundaryCells, x, 10 + c, c, v, gammaLimit);
            return false;
        }
    }
    return true;
}

struct GpuNumState {
    geom_real x[4];
    geom_real k[4];
};

struct GpuNumDeriv {
    geom_real dx[4];
    geom_real dk[4];
};

__device__ void renormalizeNumericalNullGpu(GpuNumState& s,
                                            const GpuParams& p,
                                            const GpuSceneData& scene,
                                            GpuNumericalFailure* failure = nullptr,
                                            int pixelX = -1,
                                            int pixelY = -1,
                                            int sampleIndex = -1) {
    real g10[10];
    const bool hasMetric = sampleNumericalMetricGpu(makeGeoVec3(s.x[1], s.x[2], s.x[3]), p, scene, g10,
                                                    failure, pixelX, pixelY, sampleIndex, -1);

    if (!hasMetric) {
        const geom_real v2 = s.k[1]*s.k[1] + s.k[2]*s.k[2] + s.k[3]*s.k[3];
#ifdef CUDA_GEOMETRY_DOUBLE
        s.k[0] = sqrt(fmax(v2, geom_real(1.0e-30)));
#else
        s.k[0] = sqrtf(fmaxf(v2, geom_real(1.0e-30)));
#endif
        return;
    }

    const geom_real k1 = s.k[1], k2 = s.k[2], k3 = s.k[3];
    const geom_real A = static_cast<geom_real>(g10[0]);
    const geom_real B = geom_real(2) * (static_cast<geom_real>(g10[1])*k1 + static_cast<geom_real>(g10[2])*k2 + static_cast<geom_real>(g10[3])*k3);
    const geom_real C = static_cast<geom_real>(g10[4])*k1*k1 + geom_real(2)*static_cast<geom_real>(g10[5])*k1*k2
                      + geom_real(2)*static_cast<geom_real>(g10[6])*k1*k3 + static_cast<geom_real>(g10[7])*k2*k2
                      + geom_real(2)*static_cast<geom_real>(g10[8])*k2*k3 + static_cast<geom_real>(g10[9])*k3*k3;
    const geom_real D = B*B - geom_real(4)*A*C;
    if (!(D >= geom_real(0)) || !(fabsf(static_cast<float>(A)) > 1.0e-20f)) {
        return;
    }
#ifdef CUDA_GEOMETRY_DOUBLE
    const geom_real sqrtD = sqrt(D);
#else
    const geom_real sqrtD = sqrtf(D);
#endif
    const geom_real r0 = (-B - sqrtD) / (geom_real(2)*A);
    const geom_real r1 = (-B + sqrtD) / (geom_real(2)*A);
    if (r0 > geom_real(0) && finiteGeom(r0)) s.k[0] = r0;
    else if (r1 > geom_real(0) && finiteGeom(r1)) s.k[0] = r1;
}

// Forward declaration: numericalGeodesicDerivGpu is used by RK substages before
// the horizon-capture helpers are defined below.
__device__ bool pointInsideNumericalHorizonCaptureGpu(const GpuGeoVec3& x,
                                                      const GpuParams& p);

__device__ GpuNumDeriv numericalGeodesicDerivGpu(const GpuNumState& s,
                                                 const GpuParams& p,
                                                 const GpuSceneData& scene,
                                                 GpuNumericalFailure* failure = nullptr,
                                                 int pixelX = -1,
                                                 int pixelY = -1,
                                                 int sampleIndex = -1,
                                                 int rkStep = -1) {
    GpuNumDeriv d;
    for (int mu = 0; mu < 4; ++mu) {
        d.dx[mu] = s.k[mu];
        d.dk[mu] = geom_real(0);
    }

    geom_real gamma[40];
    const GpuGeoVec3 pos = makeGeoVec3(s.x[1], s.x[2], s.x[3]);

    // RK4 evaluates derivatives at intermediate substages.  A substage can
    // already be inside the numerical AH capture sphere even though the outer
    // step has not yet reached the segment-capture test below.  Do not sample
    // Christoffels or run acceleration fail-fast inside the capture region;
    // the enclosing step will be terminated by the existing segment hit test.
    if (pointInsideNumericalHorizonCaptureGpu(pos, p)) return d;

    const bool hasGamma = sampleNumericalChristoffelGpu(pos, p, scene, gamma,
                                                        failure, pixelX, pixelY, sampleIndex, rkStep);
    if (!hasGamma) return d;

    for (int mu = 0; mu < 4; ++mu) {
        geom_real acc = geom_real(0);
        for (int a = 0; a < 4; ++a) {
            for (int b = 0; b < 4; ++b) {
                const int pair = metricPairIndexGpu(a, b);
                acc += gamma[mu * 10 + pair] * s.k[a] * s.k[b];
            }
        }
        d.dk[mu] = -acc;
        const geom_real accelLimit = static_cast<geom_real>(p.numericalMetricAccelFailAbs);
        if (!finiteGeom(d.dk[mu])) {
            recordNumericalFailureGpu(failure, p, NUM_FAIL_NONFINITE_ACCEL,
                                      pixelX, pixelY, sampleIndex, rkStep,
                                      nullptr, geom_real(-1), pos, -1, mu, d.dk[mu], accelLimit);
            d.dk[mu] = geom_real(0);
        } else if (accelLimit > geom_real(0) && fabs(static_cast<double>(d.dk[mu])) > static_cast<double>(accelLimit)) {
            recordNumericalFailureGpu(failure, p, NUM_FAIL_HUGE_ACCEL,
                                      pixelX, pixelY, sampleIndex, rkStep,
                                      nullptr, geom_real(-1), pos, -1, mu, d.dk[mu], accelLimit);
            d.dk[mu] = geom_real(0);
        }
    }
    return d;
}

__device__ GpuNumState addNumScaledGpu(const GpuNumState& y, const GpuNumDeriv& k, geom_real h) {
    GpuNumState out;
    for (int i = 0; i < 4; ++i) {
        out.x[i] = y.x[i] + h * k.dx[i];
        out.k[i] = y.k[i] + h * k.dk[i];
    }
    return out;
}

__device__ GpuNumState rk4StepNumericalGpu(const GpuNumState& y,
                                           geom_real h,
                                           const GpuParams& p,
                                           const GpuSceneData& scene,
                                           GpuNumericalFailure* failure = nullptr,
                                           int pixelX = -1,
                                           int pixelY = -1,
                                           int sampleIndex = -1,
                                           int rkStep = -1) {
    const GpuNumDeriv k1 = numericalGeodesicDerivGpu(y, p, scene, failure, pixelX, pixelY, sampleIndex, rkStep);
    const GpuNumDeriv k2 = numericalGeodesicDerivGpu(addNumScaledGpu(y, k1, geom_real(0.5)*h), p, scene, failure, pixelX, pixelY, sampleIndex, rkStep);
    const GpuNumDeriv k3 = numericalGeodesicDerivGpu(addNumScaledGpu(y, k2, geom_real(0.5)*h), p, scene, failure, pixelX, pixelY, sampleIndex, rkStep);
    const GpuNumDeriv k4 = numericalGeodesicDerivGpu(addNumScaledGpu(y, k3, h), p, scene, failure, pixelX, pixelY, sampleIndex, rkStep);

    GpuNumState out;
    for (int i = 0; i < 4; ++i) {
        out.x[i] = y.x[i] + (h / geom_real(6)) * (k1.dx[i] + geom_real(2)*k2.dx[i] + geom_real(2)*k3.dx[i] + k4.dx[i]);
        out.k[i] = y.k[i] + (h / geom_real(6)) * (k1.dk[i] + geom_real(2)*k2.dk[i] + geom_real(2)*k3.dk[i] + k4.dk[i]);
    }
    renormalizeNumericalNullGpu(out, p, scene, failure, pixelX, pixelY, sampleIndex);
    return out;
}

__device__ bool nearestNumericalHorizonCaptureSegmentHitGpu(const GpuGeoVec3& x0,
                                                            const GpuGeoVec3& x1,
                                                            const GpuParams& p,
                                                            geom_real& uHit,
                                                            int& ahId) {
    bool hit = false;
    ahId = 0;
    uHit = geom_real(1.0e300);
    geom_real u = geom_real(0);

    if (p.numericalAh1Valid && segmentSphereHitGeoGpu(x0, x1, toGeoVec3(p.numericalAh1Center), static_cast<geom_real>(p.numericalAh1CaptureRadius), u) && u < uHit) {
        hit = true;
        uHit = u;
        ahId = 1;
    }
    if (p.numericalAh2Valid && segmentSphereHitGeoGpu(x0, x1, toGeoVec3(p.numericalAh2Center), static_cast<geom_real>(p.numericalAh2CaptureRadius), u) && u < uHit) {
        hit = true;
        uHit = u;
        ahId = 2;
    }
    return hit;
}

__device__ int pointInsideNumericalHorizonCaptureIdGpu(const GpuGeoVec3& x,
                                                       const GpuParams& p) {
    bool inside1 = false;
    bool inside2 = false;
    if (p.numericalAh1Valid) {
        const GpuGeoVec3 d = x - toGeoVec3(p.numericalAh1Center);
        const geom_real r = static_cast<geom_real>(p.numericalAh1CaptureRadius);
        inside1 = (r > geom_real(0) && dot3g(d, d) <= r * r);
    }
    if (p.numericalAh2Valid) {
        const GpuGeoVec3 d = x - toGeoVec3(p.numericalAh2Center);
        const geom_real r = static_cast<geom_real>(p.numericalAh2CaptureRadius);
        inside2 = (r > geom_real(0) && dot3g(d, d) <= r * r);
    }
    if (inside1 && inside2) return 3;
    if (inside1) return 1;
    if (inside2) return 2;
    return 0;
}

__device__ bool pointInsideNumericalHorizonCaptureGpu(const GpuGeoVec3& x,
                                                      const GpuParams& p) {
    return pointInsideNumericalHorizonCaptureIdGpu(x, p) != 0;
}


// Fast localization helpers for the numerical metric.  The numerical metric is
// only defined on small BH-following boxes.  Rays that miss the outer layer-0
// boxes must use the old straight-ray renderer and should not pay geodesic or
// Christoffel interpolation cost.
__device__ bool pointInsideNumericalActivationBoxGpu(const GpuGeoVec3& x,
                                                     const GpuParams& p,
                                                     const GpuSceneData& scene) {
    if (!p.numericalMetricEnabled || p.numericalMetricGridCount <= 0 || !scene.numericalMetricGrids) return false;

    for (int i = 0; i < p.numericalMetricGridCount; ++i) {
        const GpuNumericalMetricGrid* g = scene.numericalMetricGrids + i;
        if (g->layer != 0) continue; // activation region is the outer BH box only
        if (g->layer > p.numericalMetricMaxLayer) continue;

        const geom_real dx = x.x - static_cast<geom_real>(g->centerX);
        const geom_real dy = x.y - static_cast<geom_real>(g->centerY);
        const geom_real dz = x.z - static_cast<geom_real>(g->centerZ);
        const geom_real h = static_cast<geom_real>(g->halfWidth);
        if (dx >= -h && dx <= h && dy >= -h && dy <= h && dz >= -h && dz <= h) {
            return true;
        }
    }
    return false;
}

__device__ bool intersectNumericalBoxGpu(const GpuGeoVec3& ro,
                                         const GpuGeoVec3& rd,
                                         const GpuNumericalMetricGrid& g,
                                         geom_real& t0,
                                         geom_real& t1) {
    const geom_real minx = static_cast<geom_real>(g.centerX) - static_cast<geom_real>(g.halfWidth);
    const geom_real maxx = static_cast<geom_real>(g.centerX) + static_cast<geom_real>(g.halfWidth);
    const geom_real miny = static_cast<geom_real>(g.centerY) - static_cast<geom_real>(g.halfWidth);
    const geom_real maxy = static_cast<geom_real>(g.centerY) + static_cast<geom_real>(g.halfWidth);
    const geom_real minz = static_cast<geom_real>(g.centerZ) - static_cast<geom_real>(g.halfWidth);
    const geom_real maxz = static_cast<geom_real>(g.centerZ) + static_cast<geom_real>(g.halfWidth);

    t0 = geom_real(-1.0e30);
    t1 = geom_real( 1.0e30);
    const geom_real eps = geom_real(1.0e-30);

#define NUM_METRIC_SLAB(o, d, mn, mx) \
    do { \
        if (fabsf(static_cast<float>(d)) <= static_cast<float>(eps)) { \
            if ((o) < (mn) || (o) > (mx)) return false; \
        } else { \
            geom_real a = ((mn) - (o)) / (d); \
            geom_real b = ((mx) - (o)) / (d); \
            if (a > b) { geom_real tmp = a; a = b; b = tmp; } \
            if (a > t0) t0 = a; \
            if (b < t1) t1 = b; \
            if (t1 < t0) return false; \
        } \
    } while (0)

    NUM_METRIC_SLAB(ro.x, rd.x, minx, maxx);
    NUM_METRIC_SLAB(ro.y, rd.y, miny, maxy);
    NUM_METRIC_SLAB(ro.z, rd.z, minz, maxz);

#undef NUM_METRIC_SLAB
    return true;
}

__device__ bool nearestNumericalActivationBoxHitGpu(const GpuVec3& rayOrigin,
                                                    const GpuVec3& rayDir,
                                                    const GpuParams& p,
                                                    const GpuSceneData& scene,
                                                    real sMin,
                                                    real sMax,
                                                    real& hitEnter,
                                                    real& hitExit) {
    if (!p.numericalMetricEnabled || p.numericalMetricGridCount <= 0 || !scene.numericalMetricGrids) return false;

    const GpuGeoVec3 ro = toGeoVec3(rayOrigin);
    const GpuGeoVec3 rd = toGeoVec3(rayDir);
    geom_real bestEnter = geom_real(1.0e30);
    geom_real bestExit = geom_real(1.0e30);
    bool hit = false;

    const geom_real s0 = static_cast<geom_real>(sMin);
    const geom_real s1 = static_cast<geom_real>(sMax);

    for (int i = 0; i < p.numericalMetricGridCount; ++i) {
        const GpuNumericalMetricGrid* g = scene.numericalMetricGrids + i;
        if (g->layer != 0) continue; // only test outer boxes for activation
        if (g->layer > p.numericalMetricMaxLayer) continue;

        geom_real a = geom_real(0), b = geom_real(0);
        if (!intersectNumericalBoxGpu(ro, rd, *g, a, b)) continue;
        if (b < s0 || a > s1) continue;
        if (a < s0) a = s0;
        if (b > s1) b = s1;
        if (b <= a) continue;
        if (a < bestEnter) {
            bestEnter = a;
            bestExit = b;
            hit = true;
        }
    }

    if (!hit) return false;
    hitEnter = static_cast<real>(bestEnter);
    hitExit = static_cast<real>(bestExit);
    return true;
}


__device__ bool straightRayHitsNumericalActivationGpu(const GpuVec3& rayOrigin,
                                                       const GpuVec3& rayDir,
                                                       const GpuParams& p,
                                                       const GpuSceneData& scene) {
    if (!p.numericalMetricEnabled || p.numericalMetricGridCount <= 0 || !scene.numericalMetricGrids) return false;

    real tEnter = real(0);
    real tExit = real(0);
    if (!intersectSphereOriginGpu(rayOrigin, rayDir, p.waveVolumeRadius, tEnter, tExit)) return false;

    const real s0 = fmaxf(fmaxf(tEnter, p.rayTMin), real(0));
    const real s1 = fminf(tExit, p.rayTMax);
    if (!(s1 > s0)) return false;

    real boxEnter = real(0);
    real boxExit = real(0);
    return nearestNumericalActivationBoxHitGpu(rayOrigin, rayDir, p, scene, s0, s1, boxEnter, boxExit);
}

__global__ void classifyNumericalPixelMaskKernel(unsigned char* mask,
                                                 int* hitPixelList,
                                                 unsigned int* hitCounter,
                                                 GpuParams p,
                                                 GpuCamera cam,
                                                 GpuSceneData scene) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.width || y >= p.height) return;

    bool hit = false;

    // Pixel-center ray. This is the important cheap classifier.
    GpuVec3 ro;
    GpuVec3 rd;
    generateRayGpu(cam, static_cast<real>(x), static_cast<real>(y), p.width, p.height, ro, rd);
    hit = straightRayHitsNumericalActivationGpu(ro, rd, p, scene);

    // For AA renders, also test pixel-corner rays so a pixel whose jittered samples
    // graze a tiny BH box is not incorrectly routed to the fully straight path.
    // This is still cheap: five straight ray/AABB tests per pixel, once per frame.
    if (!hit && p.samplesPerPixel > 1) {
        const real px = static_cast<real>(x);
        const real py = static_cast<real>(y);
        const real off = real(0.5);
        generateRayGpu(cam, px - off, py - off, p.width, p.height, ro, rd);
        hit = straightRayHitsNumericalActivationGpu(ro, rd, p, scene);
        if (!hit) {
            generateRayGpu(cam, px + off, py - off, p.width, p.height, ro, rd);
            hit = straightRayHitsNumericalActivationGpu(ro, rd, p, scene);
        }
        if (!hit) {
            generateRayGpu(cam, px - off, py + off, p.width, p.height, ro, rd);
            hit = straightRayHitsNumericalActivationGpu(ro, rd, p, scene);
        }
        if (!hit) {
            generateRayGpu(cam, px + off, py + off, p.width, p.height, ro, rd);
            hit = straightRayHitsNumericalActivationGpu(ro, rd, p, scene);
        }
    }

    const int pixelIndex = y * p.width + x;
    const unsigned char m = hit ? static_cast<unsigned char>(1) : static_cast<unsigned char>(0);
    mask[pixelIndex] = m;
    if (hit && hitCounter) {
        const unsigned int dst = atomicAdd(hitCounter, 1u);
        if (hitPixelList) hitPixelList[dst] = pixelIndex;
    }
}

struct GpuNumericalRayStats {
    unsigned int rkSteps = 0;
    int enteredNumerical = 0;
    int hitMaxRkSteps = 0;
    int capturedAh = 0;
    int capturedAhId = 0; // 1=AH1, 2=AH2, 3=ambiguous/unknown
    int exitedNumericalBox = 0;
};

struct GpuNumericalDiagnostics {
    unsigned long long numericalSamples;
    unsigned long long enteredNumericalSamples;
    unsigned long long totalRkSteps;
    unsigned long long samplesHitMaxRkSteps;
    unsigned long long samplesCapturedAh;
    unsigned long long samplesCapturedAh1;
    unsigned long long samplesCapturedAh2;
    unsigned long long samplesCapturedAhUnknown;
    unsigned long long samplesExitedNumericalBox;
    unsigned int maxRkSteps;
};

__device__ void accumulateNumericalDiagnosticsGpu(GpuNumericalDiagnostics* diag,
                                                  const GpuNumericalRayStats& stats) {
    if (!diag) return;
    atomicAdd(&diag->numericalSamples, 1ull);
    if (stats.enteredNumerical) atomicAdd(&diag->enteredNumericalSamples, 1ull);
    atomicAdd(&diag->totalRkSteps, static_cast<unsigned long long>(stats.rkSteps));
    atomicMax(&diag->maxRkSteps, stats.rkSteps);
    if (stats.hitMaxRkSteps) atomicAdd(&diag->samplesHitMaxRkSteps, 1ull);
    if (stats.capturedAh) {
        atomicAdd(&diag->samplesCapturedAh, 1ull);
        if (stats.capturedAhId == 1) atomicAdd(&diag->samplesCapturedAh1, 1ull);
        else if (stats.capturedAhId == 2) atomicAdd(&diag->samplesCapturedAh2, 1ull);
        else atomicAdd(&diag->samplesCapturedAhUnknown, 1ull);
    }
    if (stats.exitedNumericalBox) atomicAdd(&diag->samplesExitedNumericalBox, 1ull);
}

__device__ GpuRGBf numericalDebugMaskColorGpu(const GpuNumericalRayStats& stats) {
    // Bright diagnostic colors, intentionally not physically shaded:
    //   red          captured by AH1
    //   green        captured by AH2
    //   magenta      captured by an unknown/ambiguous AH path
    //   orange       hit RK step cap before termination
    //   cyan         entered at least one safe numerical grid, but did not capture
    //   dark blue    never entered a safe numerical grid
    if (stats.capturedAh) {
        if (stats.capturedAhId == 1) return GpuRGBf{real(1.0), real(0.0), real(0.15)};
        if (stats.capturedAhId == 2) return GpuRGBf{real(0.0), real(1.0), real(0.15)};
        return GpuRGBf{real(1.0), real(0.0), real(1.0)};
    }
    if (stats.hitMaxRkSteps) {
        return GpuRGBf{real(1.0), real(0.55), real(0.0)};
    }
    if (stats.enteredNumerical) {
        return GpuRGBf{real(0.0), real(1.0), real(1.0)};
    }
    return GpuRGBf{real(0.05), real(0.08), real(0.35)};
}

__device__ GpuRGBf numericalDebugFailureColorGpu() {
    return GpuRGBf{real(1.0), real(1.0), real(1.0)};
}

__device__ void compositeGwpvStraightSegmentGpu(const GpuGeoVec3& origin,
                                                const GpuGeoVec3& dir,
                                                geom_real s0,
                                                geom_real s1,
                                                const GpuParams& p,
                                                const GpuSceneData& scene,
                                                GpuRGBf& accum,
                                                real& T) {
    if (!(s1 > s0) || T <= p.transmittanceCutoff) return;

    const geom_real step = static_cast<geom_real>(fmaxf(p.stepSize, real(1.0e-6)));
    geom_real s = s0;
    while (s < s1 && T > p.transmittanceCutoff) {
        geom_real ds = step;
        if (s + ds > s1) ds = s1 - s;
        const GpuGeoVec3 xMid = origin + (s + geom_real(0.5) * ds) * dir;
        compositeGwpvPointGpu(toGpuVec3f(xMid), static_cast<real>(ds), p, scene, accum, T);
        s += ds;
    }
}

__device__ GpuRGBf shadeRayNumericalMetricGpu(const GpuVec3& rayOrigin,
                                              const GpuVec3& rayDir,
                                              const GpuParams& p,
                                              const GpuSceneData& scene,
                                              GpuNumericalRayStats* stats,
                                              GpuNumericalFailure* failure = nullptr,
                                              int pixelX = -1,
                                              int pixelY = -1,
                                              int sampleIndex = -1) {
    if (stats) *stats = GpuNumericalRayStats{};
    real tEnter = real(0);
    real tExit = real(0);
    const bool hitWaveSphere = intersectSphereOriginGpu(rayOrigin, rayDir, p.waveVolumeRadius, tEnter, tExit);

    if (!hitWaveSphere) {
        if (p.numericalMetricDebugMask) {
            return GpuRGBf{real(0.05), real(0.08), real(0.35)};
        }
        return samplePanoramaGpu(rayDir, p, scene);
    }

    const real sWave0 = fmaxf(fmaxf(tEnter, p.rayTMin), real(0));
    const real sWave1 = fminf(tExit, p.rayTMax);
    if (!(sWave1 > sWave0)) {
        if (p.numericalMetricDebugMask) {
            return GpuRGBf{real(0.05), real(0.08), real(0.35)};
        }
        return samplePanoramaGpu(rayDir, p, scene);
    }

    // Full-ray numerical mode: every pixel follows one continuous RK ray from
    // wave-sphere entry to exit.  The numerical Christoffel sampler returns false
    // outside safe sampled boxes, and numericalGeodesicDerivGpu() then leaves the
    // acceleration at zero.  This removes the old straight-pass/classify/overwrite
    // compositing model; there is now one ray path, one capture decision, and one
    // final background direction for each sample.

    GpuRGBf accum{0, 0, 0};
    real T = real(1);

    const GpuGeoVec3 rayOriginG = toGeoVec3(rayOrigin);
    const GpuGeoVec3 rayDirG = toGeoVec3(rayDir);

    const geom_real h = static_cast<geom_real>((p.cudaMetricStep > real(1.0e-6)) ? p.cudaMetricStep : real(1.0e-6));
    const geom_real colorStep = static_cast<geom_real>((p.cudaMetricColorStep > real(1.0e-6)) ? p.cudaMetricColorStep : real(1.0e-6));
    const int maxSteps = (p.cudaMetricMaxSteps > 1) ? p.cudaMetricMaxSteps : 1;
    const geom_real waveRadius = static_cast<geom_real>(p.waveVolumeRadius);

    GpuNumState state;
    state.x[0] = static_cast<geom_real>(p.time);
    const GpuGeoVec3 xStart = rayOriginG + static_cast<geom_real>(sWave0) * rayDirG;
    state.x[1] = xStart.x;
    state.x[2] = xStart.y;
    state.x[3] = xStart.z;
    state.k[1] = static_cast<geom_real>(rayDir.x);
    state.k[2] = static_cast<geom_real>(rayDir.y);
    state.k[3] = static_cast<geom_real>(rayDir.z);
    state.k[0] = geom_real(1);
    renormalizeNumericalNullGpu(state, p, scene, failure, pixelX, pixelY, sampleIndex);

    GpuGeoVec3 lastX = xStart;
    GpuVec3 finalDir = rayDir;

    for (int stepIndex = 0; stepIndex < maxSteps && T > p.transmittanceCutoff; ++stepIndex) {
        const GpuGeoVec3 xNow = makeGeoVec3(state.x[1], state.x[2], state.x[3]);
        const GpuGeoVec3 kSpatialNow = makeGeoVec3(state.k[1], state.k[2], state.k[3]);

        if (stats && !stats->enteredNumerical) {
            geom_real dbgBoundaryCells = geom_real(-1);
            if (selectNumericalMetricGridGpu(xNow, p, scene, &dbgBoundaryCells) != nullptr) {
                stats->enteredNumerical = 1;
            }
        }

        // If we have propagated outside the wave volume and are moving outward,
        // the volume integration is complete.  This replaces the old
        // numerical-box-exit termination; box exits are now just ordinary RK
        // points with zero Christoffel acceleration until a later box re-entry.
        if (stepIndex > 0 && length3g(xNow) > waveRadius && dot3g(xNow, kSpatialNow) > geom_real(0)) {
            break;
        }

        // Capture must be checked at the start of every RK step.  This is the
        // only physical termination inside the numerical boxes.
        const int insideAhId = pointInsideNumericalHorizonCaptureIdGpu(xNow, p);
        if (insideAhId != 0) {
            if (stats) {
                stats->capturedAh = 1;
                stats->capturedAhId = insideAhId;
            }
            if (p.numericalMetricDebugMask) {
                return stats ? numericalDebugMaskColorGpu(*stats) : GpuRGBf{real(1.0), real(0.0), real(1.0)};
            }
            return accum;
        }

        const GpuNumState yStart = state;
        const GpuNumState yEnd = rk4StepNumericalGpu(state, h, p, scene, failure, pixelX, pixelY, sampleIndex, stepIndex);
        if (stats) ++stats->rkSteps;

        for (int c = 0; c < 4; ++c) {
            if (!finiteGeom(yEnd.x[c])) {
                recordNumericalFailureGpu(failure, p, NUM_FAIL_NONFINITE_RAY_STATE,
                                          pixelX, pixelY, sampleIndex, stepIndex,
                                          nullptr, geom_real(-1), xNow, -1, c, yEnd.x[c], geom_real(0));
                return p.numericalMetricDebugMask ? numericalDebugFailureColorGpu() : accum;
            }
            if (!finiteGeom(yEnd.k[c])) {
                recordNumericalFailureGpu(failure, p, NUM_FAIL_NONFINITE_RAY_STATE,
                                          pixelX, pixelY, sampleIndex, stepIndex,
                                          nullptr, geom_real(-1), xNow, -1, 4 + c, yEnd.k[c], geom_real(0));
                return p.numericalMetricDebugMask ? numericalDebugFailureColorGpu() : accum;
            }
        }

        const GpuGeoVec3 x0 = makeGeoVec3(yStart.x[1], yStart.x[2], yStart.x[3]);
        const GpuGeoVec3 x1 = makeGeoVec3(yEnd.x[1], yEnd.x[2], yEnd.x[3]);
        const GpuGeoVec3 dx = x1 - x0;
        const geom_real segLen = length3g(dx);

        bool crossedWaveExit = false;
        if (finiteGeom(segLen) && segLen > geom_real(0)) {
#ifdef CUDA_GEOMETRY_DOUBLE
            int nSubRaw = static_cast<int>(ceil(segLen / colorStep));
#else
            int nSubRaw = static_cast<int>(ceilf(segLen / colorStep));
#endif
            const int nSub = (nSubRaw > 1) ? nSubRaw : 1;
            const geom_real ds = segLen / static_cast<geom_real>(nSub);

            for (int i = 0; i < nSub; ++i) {
                const geom_real a0 = static_cast<geom_real>(i) / static_cast<geom_real>(nSub);
                const geom_real a1 = static_cast<geom_real>(i + 1) / static_cast<geom_real>(nSub);
                const GpuGeoVec3 xA = x0 + a0 * dx;
                const GpuGeoVec3 xB = x0 + a1 * dx;

                geom_real uHit = geom_real(0);
                int hitAhId = 0;
                if (nearestNumericalHorizonCaptureSegmentHitGpu(xA, xB, p, uHit, hitAhId)) {
                    const GpuGeoVec3 subDx = xB - xA;
                    const GpuGeoVec3 xHit = xA + uHit * subDx;
                    const geom_real dsBefore = length3g(xHit - xA);
                    if (finiteGeom(dsBefore) && dsBefore > geom_real(1.0e-6) && length3g(xA) <= waveRadius) {
                        const GpuGeoVec3 xMid = xA + (geom_real(0.5) * uHit) * subDx;
                        compositeGwpvPointGpu(toGpuVec3f(xMid), static_cast<real>(dsBefore), p, scene, accum, T);
                    }
                    if (stats) {
                        stats->capturedAh = 1;
                        stats->capturedAhId = hitAhId;
                    }
                    if (p.numericalMetricDebugMask) {
                        return stats ? numericalDebugMaskColorGpu(*stats) : GpuRGBf{real(1.0), real(0.0), real(1.0)};
                    }
                    return accum;
                }

                const geom_real aMid = geom_real(0.5) * (a0 + a1);
                const GpuGeoVec3 xMid = x0 + aMid * dx;
                if (length3g(xMid) <= waveRadius) {
                    compositeGwpvPointGpu(toGpuVec3f(xMid), static_cast<real>(ds), p, scene, accum, T);
                    if (T <= p.transmittanceCutoff) break;
                }
            }

            const bool x0Inside = length3g(x0) <= waveRadius;
            const bool x1Outside = length3g(x1) > waveRadius;
            const GpuGeoVec3 kSpatialEnd = makeGeoVec3(yEnd.k[1], yEnd.k[2], yEnd.k[3]);
            crossedWaveExit = x0Inside && x1Outside && dot3g(x1, kSpatialEnd) > geom_real(0);
        }

        state = yEnd;
        lastX = makeGeoVec3(state.x[1], state.x[2], state.x[3]);
        finalDir = normalize3(makeVec3(static_cast<real>(state.k[1]), static_cast<real>(state.k[2]), static_cast<real>(state.k[3])));
        if (!isfinite(finalDir.x) || !isfinite(finalDir.y) || !isfinite(finalDir.z) || length2_3(finalDir) <= real(1.0e-20)) {
            finalDir = rayDir;
        }

        if (crossedWaveExit) {
            break;
        }
    }

    if (stats && stats->enteredNumerical && !stats->capturedAh &&
        stats->rkSteps >= static_cast<unsigned int>(maxSteps) && T > p.transmittanceCutoff) {
        stats->hitMaxRkSteps = 1;
    }

    if (p.numericalMetricDebugMask) {
        return stats ? numericalDebugMaskColorGpu(*stats) : GpuRGBf{real(0.05), real(0.08), real(0.35)};
    }

    const GpuRGBf bg = samplePanoramaGpu(finalDir, p, scene);
    accum = rgbAdd(accum, rgbMul(bg, T));
    return accum;
}

__global__ void renderStraightKernel(GpuRGB8* out,
                                   GpuParams p,
                                   GpuCamera cam,
                                   GpuSceneData scene) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.width || y >= p.height) return;

    int root = 1;
    if (p.samplesPerPixel > 1) {
        root = static_cast<int>(floorf(sqrtf(static_cast<real>(p.samplesPerPixel))));
        if (root < 1) root = 1;
    }
    const int spp = (p.samplesPerPixel <= 1) ? 1 : root * root;

    GpuRGBf sum{0, 0, 0};
    int index = 0;
    for (int sy = 0; sy < root; ++sy) {
        for (int sx = 0; sx < root; ++sx) {
            real px = static_cast<real>(x);
            real py = static_cast<real>(y);
            if (p.samplesPerPixel > 1) {
                const real jx = hash01Gpu(x, y, index, 0);
                const real jy = hash01Gpu(x, y, index, 1);
                px = static_cast<real>(x) + (static_cast<real>(sx) + jx) / static_cast<real>(root) - real(0.5);
                py = static_cast<real>(y) + (static_cast<real>(sy) + jy) / static_cast<real>(root) - real(0.5);
            }

            GpuVec3 rayOrigin;
            GpuVec3 rayDir;
            generateRayGpu(cam, px, py, p.width, p.height, rayOrigin, rayDir);
            sum = rgbAdd(sum, shadeRayFixedGpu(rayOrigin, rayDir, p, scene));
            ++index;
        }
    }

    sum = rgbMul(sum, real(1) / static_cast<real>(spp));
    out[y * p.width + x] = makeRGB8Gpu(sum, p.outputGamma);
}

__global__ void numericalCorrectionKernel(GpuRGB8* out,
                                          const int* hitPixelList,
                                          unsigned int hitPixelCount,
                                          GpuParams p,
                                          GpuCamera cam,
                                          GpuSceneData scene,
                                          GpuNumericalDiagnostics* diagnostics,
                                          GpuNumericalFailure* failure) {
    const unsigned int listIndex = static_cast<unsigned int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (listIndex >= hitPixelCount) return;

    const int pixelIndex = hitPixelList[listIndex];
    if (pixelIndex < 0 || pixelIndex >= p.width * p.height) return;

    const int x = pixelIndex % p.width;
    const int y = pixelIndex / p.width;

    int root = 1;
    if (p.samplesPerPixel > 1) {
        root = static_cast<int>(floorf(sqrtf(static_cast<real>(p.samplesPerPixel))));
        if (root < 1) root = 1;
    }
    const int spp = (p.samplesPerPixel <= 1) ? 1 : root * root;

    GpuRGBf sum{0, 0, 0};
    int index = 0;
    for (int sy = 0; sy < root; ++sy) {
        for (int sx = 0; sx < root; ++sx) {
            real px = static_cast<real>(x);
            real py = static_cast<real>(y);
            if (p.samplesPerPixel > 1) {
                const real jx = hash01Gpu(x, y, index, 0);
                const real jy = hash01Gpu(x, y, index, 1);
                px = static_cast<real>(x) + (static_cast<real>(sx) + jx) / static_cast<real>(root) - real(0.5);
                py = static_cast<real>(y) + (static_cast<real>(sy) + jy) / static_cast<real>(root) - real(0.5);
            }

            GpuVec3 rayOrigin;
            GpuVec3 rayDir;
            generateRayGpu(cam, px, py, p.width, p.height, rayOrigin, rayDir);
            GpuNumericalRayStats stats{};
            const GpuRGBf sampleColor = shadeRayNumericalMetricGpu(rayOrigin, rayDir, p, scene, &stats, failure, x, y, index);
            accumulateNumericalDiagnosticsGpu(diagnostics, stats);
            sum = rgbAdd(sum, sampleColor);
            ++index;
        }
    }

    sum = rgbMul(sum, real(1) / static_cast<real>(spp));
    out[pixelIndex] = makeRGB8Gpu(sum, p.outputGamma);
}

__global__ void renderNumericalMetricKernel(GpuRGB8* out,
                                          GpuParams p,
                                          GpuCamera cam,
                                          GpuSceneData scene,
                                          GpuNumericalDiagnostics* diagnostics,
                                          GpuNumericalFailure* failure) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.width || y >= p.height) return;

    int root = 1;
    if (p.samplesPerPixel > 1) {
        root = static_cast<int>(floorf(sqrtf(static_cast<real>(p.samplesPerPixel))));
        if (root < 1) root = 1;
    }
    const int spp = (p.samplesPerPixel <= 1) ? 1 : root * root;

    GpuRGBf sum{0, 0, 0};
    int index = 0;
    for (int sy = 0; sy < root; ++sy) {
        for (int sx = 0; sx < root; ++sx) {
            real px = static_cast<real>(x);
            real py = static_cast<real>(y);
            if (p.samplesPerPixel > 1) {
                const real jx = hash01Gpu(x, y, index, 0);
                const real jy = hash01Gpu(x, y, index, 1);
                px = static_cast<real>(x) + (static_cast<real>(sx) + jx) / static_cast<real>(root) - real(0.5);
                py = static_cast<real>(y) + (static_cast<real>(sy) + jy) / static_cast<real>(root) - real(0.5);
            }

            GpuVec3 rayOrigin;
            GpuVec3 rayDir;
            generateRayGpu(cam, px, py, p.width, p.height, rayOrigin, rayDir);
            GpuNumericalRayStats stats{};
            const GpuRGBf sampleColor = shadeRayNumericalMetricGpu(rayOrigin, rayDir, p, scene, &stats, failure, x, y, index);
            accumulateNumericalDiagnosticsGpu(diagnostics, stats);
            sum = rgbAdd(sum, sampleColor);
            ++index;
        }
    }

    sum = rgbMul(sum, real(1) / static_cast<real>(spp));
    out[y * p.width + x] = makeRGB8Gpu(sum, p.outputGamma);
}

__global__ void renderKernel(GpuRGB8* out,
                             GpuParams p,
                             GpuCamera cam,
                             GpuSceneData scene,
                             const unsigned char* numericalPixelMask) {
    (void)numericalPixelMask;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.width || y >= p.height) return;

    int root = 1;
    if (p.samplesPerPixel > 1) {
        root = static_cast<int>(floorf(sqrtf(static_cast<real>(p.samplesPerPixel))));
        if (root < 1) root = 1;
    }
    const int spp = (p.samplesPerPixel <= 1) ? 1 : root * root;

    const real pixelCx = static_cast<real>(x) + real(0.5) - real(0.5) * static_cast<real>(p.width);
    const real pixelCy = static_cast<real>(y) + real(0.5) - real(0.5) * static_cast<real>(p.height);
    const real pixelR2 = pixelCx * pixelCx + pixelCy * pixelCy;
    const real doubleRadius = fmaxf(real(0), p.cudaDoublePixelRadiusFraction) * static_cast<real>(p.width);
    const bool useDoubleGeometry = p.metricLensingEnabled && !p.numericalMetricEnabled &&
                                   doubleRadius > real(0) && pixelR2 <= doubleRadius * doubleRadius;

    GpuRGBf sum{0, 0, 0};
    int index = 0;
    for (int sy = 0; sy < root; ++sy) {
        for (int sx = 0; sx < root; ++sx) {
            real px = static_cast<real>(x);
            real py = static_cast<real>(y);
            if (p.samplesPerPixel > 1) {
                const real jx = hash01Gpu(x, y, index, 0);
                const real jy = hash01Gpu(x, y, index, 1);
                px = static_cast<real>(x) + (static_cast<real>(sx) + jx) / static_cast<real>(root) - real(0.5);
                py = static_cast<real>(y) + (static_cast<real>(sy) + jy) / static_cast<real>(root) - real(0.5);
            }

            GpuVec3 rayOrigin;
            GpuVec3 rayDir;
            generateRayGpu(cam, px, py, p.width, p.height, rayOrigin, rayDir);
            GpuRGBf sampleColor;
            if (p.metricLensingEnabled && !p.numericalMetricEnabled) {
                sampleColor = useDoubleGeometry
                    ? shadeRayMPDoubleGpu(rayOrigin, rayDir, p, scene)
                    : shadeRayMPGpu(rayOrigin, rayDir, p, scene);
            } else {
                sampleColor = shadeRayFixedGpu(rayOrigin, rayDir, p, scene);
            }
            sum = rgbAdd(sum, sampleColor);
            ++index;
        }
    }

    sum = rgbMul(sum, real(1) / static_cast<real>(spp));
    out[y * p.width + x] = makeRGB8Gpu(sum, p.outputGamma);
}

GpuVec3 toGpuVec3(const Vec3& v) {
    return GpuVec3{static_cast<real>(v.x), static_cast<real>(v.y), static_cast<real>(v.z)};
}

GpuRGBf toGpuRGBf(const RGBf& c) {
    return GpuRGBf{static_cast<real>(c.r), static_cast<real>(c.g), static_cast<real>(c.b)};
}

std::complex<double> sampleSlope(const std::vector<TimeSample>& s, std::size_t i) {
    if (s.size() < 2) return {0.0, 0.0};
    if (i == 0) {
        const double dt = s[1].t - s[0].t;
        if (dt <= 0.0) return {0.0, 0.0};
        return (s[1].y - s[0].y) / dt;
    }
    if (i + 1 >= s.size()) {
        const double dt = s[i].t - s[i - 1].t;
        if (dt <= 0.0) return {0.0, 0.0};
        return (s[i].y - s[i - 1].y) / dt;
    }
    const double dt = s[i + 1].t - s[i - 1].t;
    if (dt <= 0.0) return {0.0, 0.0};
    return (s[i + 1].y - s[i - 1].y) / dt;
}

double secondsSince(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

void validatePrecomputedChristoffelSnapshot(const NumericalMetricSnapshot& snap) {
    if (snap.nfields != 50 || snap.fieldOrder.size() != 50) {
        throw std::runtime_error("CUDA numerical metric now expects a 50-field precomputed Christoffel snapshot");
    }
    if (snap.fieldOrder[0] != "g00" || snap.fieldOrder[9] != "g33" ||
        snap.fieldOrder[10] != "Gamma0_00" || snap.fieldOrder[49] != "Gamma3_33") {
        throw std::runtime_error(
            "CUDA numerical metric expected field_order g00..g33 then Gamma0_00..Gamma3_33. "
            "You probably passed the old nearfield_metric_* stem instead of nearfield_christoffel_*."
        );
    }
}

void validateCudaParams(const Params& params) {
    if (params.renderMode != "gwpv_peaks") {
        throw std::runtime_error("CUDA v0 only supports renderMode=\"gwpv_peaks\"");
    }
    if (params.metricLensingEnabled) {
        if (params.numericalMetricEnabled) {
            if (params.numericalMetricSnapshotStem.empty()) {
                throw std::runtime_error("CUDA numerical metric requested, but numericalMetricSnapshotStem is empty");
            }
            if (params.numericalMetricHorizonPath.empty()) {
                throw std::runtime_error("CUDA numerical metric requested, but numericalMetricHorizonPath is empty");
            }
        } else {
            if (!params.blackHolesEnabled) {
                throw std::runtime_error("CUDA MP lensing requires blackHolesEnabled=true");
            }
            if (!params.metricUseMajumdarPapapetrou) {
                throw std::runtime_error("CUDA MP lensing requires metricUseMajumdarPapapetrou=true");
            }
            if (!(params.metricMPMassScale > 0.0)) {
                throw std::runtime_error("CUDA MP lensing requires metricMPMassScale > 0");
            }
            if (std::abs(params.metricPerturbationScale) > 1.0e-12) {
                static bool warnedStrainIgnored = false;
                if (!warnedStrainIgnored) {
                    std::cerr << "WARNING: CUDA MP lensing ignores frozen-strain metric perturbation scale="
                              << params.metricPerturbationScale << ".\n";
                    warnedStrainIgnored = true;
                }
            }
        }
    }
    if (params.gwpvAdaptiveEnabled) {
        throw std::runtime_error("CUDA v0 requires gwpvAdaptiveEnabled=false for constant-step rendering");
    }
    if (params.gwpvWavelengthStepScalingEnabled) {
        throw std::runtime_error("CUDA v0 requires gwpvWavelengthStepScalingEnabled=false for constant-step rendering");
    }
    if (params.gwpvWavelengthCompEnabled || params.gwpvWavelengthPeakWidthEnabled) {
        throw std::runtime_error("CUDA v0 does not support GWPV wavelength compensation/peak-width modulation yet");
    }
    if (params.gwpvOpacityGainEnabled) {
        throw std::runtime_error("CUDA v0 does not support frame-global opacity gain CSV yet");
    }
    if (params.colorMode != "signed_real") {
        // CUDA v0 uses the gwpv_peaks color map, not fog colorMode, but this catches surprises.
        std::cerr << "WARNING: CUDA v0 ignores colorMode for gwpv_peaks; using GWPV rainbow color map.\n";
    }
    if (params.paraviewScalarRadialMode != "r_psi4" && params.paraviewScalarRadialMode != "psi4") {
        throw std::runtime_error("CUDA v0 only supports paraviewScalarRadialMode=\"r_psi4\" or \"psi4\"");
    }
}

} // namespace

Image renderCuda(const Camera& camera,
                 const Params& params,
                 const Scene& scene) {
    validateCudaParams(params);

    double christoffelLoadSeconds = 0.0;
    double gpuUploadSeconds = 0.0;
    double straightRenderSeconds = 0.0;
    double classificationSeconds = 0.0;
    double numericalCorrectionSeconds = 0.0;

    const ModeDataSet& modes = scene.field.modes();
    const std::vector<ModeKey> keys = modes.modeKeys();
    if (modes.shells.empty() || keys.empty()) {
        throw std::runtime_error("CUDA render: empty mode data");
    }
    if (scene.panorama.empty()) {
        throw std::runtime_error("CUDA render: panorama is empty");
    }

    std::vector<GpuShell> hShells;
    hShells.reserve(modes.shells.size());
    for (const auto& shell : modes.shells) {
        hShells.push_back(GpuShell{static_cast<real>(shell.radius)});
    }

    std::vector<GpuModeKey> hKeys;
    hKeys.reserve(keys.size());
    for (const auto& key : keys) {
        hKeys.push_back(GpuModeKey{key.l, key.m});
    }

    std::vector<GpuSeries> hSeries(modes.shells.size() * keys.size(), GpuSeries{-1, 0});
    std::vector<GpuSample> hSamples;

    for (std::size_t si = 0; si < modes.shells.size(); ++si) {
        const RadiusShell& shell = modes.shells[si];
        for (std::size_t mi = 0; mi < keys.size(); ++mi) {
            const ModeSeries* series = shell.findMode(keys[mi]);
            if (!series || series->samples.empty()) continue;

            const int offset = static_cast<int>(hSamples.size());
            const int count = static_cast<int>(series->samples.size());
            hSeries[si * keys.size() + mi] = GpuSeries{offset, count};

            for (std::size_t i = 0; i < series->samples.size(); ++i) {
                const TimeSample& ts = series->samples[i];
                const std::complex<double> sl = sampleSlope(series->samples, i);
                hSamples.push_back(GpuSample{
                    static_cast<real>(ts.t),
                    static_cast<real>(std::real(ts.y)),
                    static_cast<real>(std::imag(ts.y)),
                    static_cast<real>(std::real(sl)),
                    static_cast<real>(std::imag(sl))
                });
            }
        }
    }

    if (hSamples.empty()) {
        throw std::runtime_error("CUDA render: no mode samples after flattening");
    }

    std::vector<GpuRGBf> hPanorama;
    hPanorama.reserve(scene.panorama.pixels().size());
    for (const RGBf& c : scene.panorama.pixels()) {
        hPanorama.push_back(toGpuRGBf(c));
    }

    DeviceBuffer<GpuShell> dShells;
    DeviceBuffer<GpuModeKey> dKeys;
    DeviceBuffer<GpuSeries> dSeries;
    DeviceBuffer<GpuSample> dSamples;
    DeviceBuffer<GpuRGBf> dPanorama;
    DeviceBuffer<GpuRGB8> dOutput;
    DeviceBuffer<unsigned char> dNumericalPixelMask;
    DeviceBuffer<int> dNumericalPixelList;
    DeviceBuffer<unsigned int> dNumericalPixelHitCounter;
    DeviceBuffer<GpuNumericalDiagnostics> dNumericalDiagnostics;
    DeviceBuffer<GpuNumericalFailure> dNumericalFailure;
    unsigned int numericalPixelHits = 0;
    GpuNumericalDiagnostics hNumericalDiagnostics{};
    GpuNumericalFailure hNumericalFailure{};

    DeviceBuffer<GpuNumericalMetricGrid> dNumMetricGrids;
    DeviceBuffer<real> dNumMetricFields;
    NumericalMetricBundle numericalBundle;
    bool numericalMetricLoaded = false;
    std::vector<GpuNumericalMetricGrid> hNumMetricGrids;

    {
        const auto uploadStart = std::chrono::steady_clock::now();
        dShells.upload(hShells);
        dKeys.upload(hKeys);
        dSeries.upload(hSeries);
        dSamples.upload(hSamples);
        dPanorama.upload(hPanorama);
        dOutput.allocate(static_cast<std::size_t>(params.width) * static_cast<std::size_t>(params.height));
        gpuUploadSeconds += secondsSince(uploadStart);
    }

    if (params.metricLensingEnabled && params.numericalMetricEnabled) {
        {
            const auto loadStart = std::chrono::steady_clock::now();
            numericalBundle = loadNumericalMetricBundle(params.numericalMetricSnapshotStem,
                                                        params.numericalMetricHorizonPath);
            validatePrecomputedChristoffelSnapshot(numericalBundle.metric);
            christoffelLoadSeconds = secondsSince(loadStart);
            std::cerr << "CUDA setup: Christoffel load complete: "
                      << christoffelLoadSeconds << " s\n" << std::flush;
        }
        numericalMetricLoaded = true;

        hNumMetricGrids.reserve(numericalBundle.metric.grids.size());
        for (std::size_t gi = 0; gi < numericalBundle.metric.grids.size(); ++gi) {
            const auto& g = numericalBundle.metric.grids[gi];
            hNumMetricGrids.push_back(GpuNumericalMetricGrid{
                g.familyId,
                g.layer,
                g.nx, g.ny, g.nz,
                static_cast<real>(g.center.x),
                static_cast<real>(g.center.y),
                static_cast<real>(g.center.z),
                static_cast<real>(g.halfWidth),
                static_cast<real>(g.actualDx),
                g.points,
                g.dataOffsetFloats,
                g.pointOffset,
                static_cast<int>(gi)
            });
        }

        {
            const auto uploadStart = std::chrono::steady_clock::now();
            dNumMetricGrids.upload(hNumMetricGrids);
            dNumMetricFields.upload(numericalBundle.metric.data);
            const double numericalMetricUploadSeconds = secondsSince(uploadStart);
            gpuUploadSeconds += numericalMetricUploadSeconds;
            std::cerr << "CUDA setup: numerical metric GPU upload complete: "
                      << numericalMetricUploadSeconds << " s\n" << std::flush;
        }

        std::cerr << "CUDA numerical metric: using precomputed Christoffel fields from "
                  << params.numericalMetricSnapshotStem
                  << " points=" << numericalBundle.metric.totalGridPoints
                  << " grids=" << hNumMetricGrids.size() << "\n";
    }

    const BlackHoleSystem::State bh = BlackHoleSystem::stateAtTime(params.time, params);
    if (params.metricLensingEnabled && !params.numericalMetricEnabled && !bh.valid) {
        throw std::runtime_error("CUDA MP lensing requested, but black-hole trajectory/state is invalid");
    }

    GpuParams gp{};
    gp.width = params.width;
    gp.height = params.height;
    gp.samplesPerPixel = params.samplesPerPixel;
    gp.stepSize = static_cast<real>(params.stepSize);
    gp.rayTMin = static_cast<real>(params.rayTMin);
    gp.rayTMax = std::isinf(params.rayTMax) ? real(3.4e38) : static_cast<real>(params.rayTMax);
    gp.waveVolumeRadius = static_cast<real>(params.waveVolumeRadius);
    gp.transmittanceCutoff = static_cast<real>(params.transmittanceCutoff);
    gp.time = static_cast<real>(params.time);
    gp.rInner = static_cast<real>(params.rInner);
    gp.innerWaveScaleRadius = static_cast<real>(params.innerWaveScaleRadius);
    gp.useTortoiseRetardedTime = params.useTortoiseRetardedTime ? 1 : 0;
    gp.tortoiseMass = static_cast<real>(params.tortoiseMass);
    gp.tortoiseRadiusFloor = static_cast<real>(params.tortoiseRadiusFloor);
    gp.tortoiseSafetyEps = static_cast<real>(params.tortoiseSafetyEps);
    gp.panoramaYawDegrees = static_cast<real>(params.panoramaYawDegrees);
    gp.panoramaExposure = static_cast<real>(params.panoramaExposure);
    gp.paraviewScalarUsesR = (params.paraviewScalarRadialMode == "r_psi4") ? 1 : 0;
    gp.paraviewOpacityRadialFalloffEnabled = params.paraviewOpacityRadialFalloffEnabled ? 1 : 0;
    gp.paraviewOpacityReferenceRadius = static_cast<real>(params.paraviewOpacityReferenceRadius);
    gp.paraviewOpacityFalloffPower = static_cast<real>(params.paraviewOpacityFalloffPower);
    gp.paraviewOuterFadeWidth = static_cast<real>(params.paraviewOuterFadeWidth);
    gp.gwpvPeaksNumPeaks = params.gwpvPeaksNumPeaks;
    gp.gwpvPeaksFirstPosition = static_cast<real>(params.gwpvPeaksFirstPosition);
    gp.gwpvPeaksLastPosition = static_cast<real>(params.gwpvPeaksLastPosition);
    gp.gwpvPeaksFirstOpacity = static_cast<real>(params.gwpvPeaksFirstOpacity);
    gp.gwpvPeaksLastOpacity = static_cast<real>(params.gwpvPeaksLastOpacity);
    gp.gwpvPeaksStrength = static_cast<real>(params.gwpvPeaksStrength);
    gp.gwpvScalarOpacityUnitDistance = static_cast<real>(params.gwpvScalarOpacityUnitDistance);
    gp.gwpvUseAxisMask = params.gwpvUseAxisMask ? 1 : 0;
    gp.axisMaskEnabled = params.axisMaskEnabled ? 1 : 0;
    gp.axisMaskInnerRadius = static_cast<real>(params.axisMaskInnerRadius);
    gp.axisMaskOuterRadius = static_cast<real>(params.axisMaskOuterRadius);
    gp.gwpvUseOpacityRadialEnvelope = params.gwpvUseOpacityRadialEnvelope ? 1 : 0;
    gp.waveBrightness = static_cast<real>(params.waveBrightness);
    gp.maxStepAlpha = static_cast<real>(params.maxStepAlpha);
    gp.outputGamma = static_cast<real>(params.outputGamma);
    gp.numShells = static_cast<int>(hShells.size());
    gp.numModes = static_cast<int>(hKeys.size());
    gp.panoramaWidth = scene.panorama.width();
    gp.panoramaHeight = scene.panorama.height();
    gp.blackHolesEnabled = params.blackHolesEnabled ? 1 : 0;
    gp.blackHoleStateValid = bh.valid ? 1 : 0;
    gp.bhPlusCenter = toGpuVec3(bh.plusCenter);
    gp.bhMinusCenter = toGpuVec3(bh.minusCenter);
    gp.bhPlusRenderRadius = static_cast<real>(bh.plusRenderRadius);
    gp.bhMinusRenderRadius = static_cast<real>(bh.minusRenderRadius);
    gp.bhPlusCaptureRadius = static_cast<real>(bh.plusCaptureRadius);
    gp.bhMinusCaptureRadius = static_cast<real>(bh.minusCaptureRadius);
    gp.bhPlusMetricMass = static_cast<real>(params.metricMPMassScale * bh.plusMass);
    gp.bhMinusMetricMass = static_cast<real>(params.metricMPMassScale * bh.minusMass);
    gp.metricLensingEnabled = params.metricLensingEnabled ? 1 : 0;
    gp.metricUseMajumdarPapapetrou = params.metricUseMajumdarPapapetrou ? 1 : 0;
    gp.metricMPSoftening = static_cast<real>(params.metricMPSoftening);
    gp.cudaMetricStep = static_cast<real>(params.cudaMetricStep);
    gp.cudaMetricColorStep = static_cast<real>(params.cudaMetricColorStep);
    gp.cudaDoublePixelRadiusFraction = static_cast<real>(params.cudaDoublePixelRadiusFraction);
    gp.cudaMetricMaxSteps = params.cudaMetricMaxSteps;

    gp.numericalMetricEnabled = (params.metricLensingEnabled && params.numericalMetricEnabled && numericalMetricLoaded) ? 1 : 0;
    gp.numericalMetricGridCount = numericalMetricLoaded ? static_cast<int>(hNumMetricGrids.size()) : 0;
    gp.numericalMetricMaxLayer = params.numericalMetricMaxLayer;
    gp.numericalMetricUseTimeDerivatives = params.numericalMetricUseTimeDerivatives ? 1 : 0;
    gp.numericalMetricBoundaryBufferCells = static_cast<real>(params.numericalMetricBoundaryBufferCells);
    gp.numericalMetricMetricFailAbs = static_cast<real>(params.numericalMetricMetricFailAbs);
    gp.numericalMetricGammaFailAbs = static_cast<real>(params.numericalMetricGammaFailAbs);
    gp.numericalMetricAccelFailAbs = static_cast<real>(params.numericalMetricAccelFailAbs);
    gp.numericalMetricFailFast = params.numericalMetricFailFast ? 1 : 0;
    gp.numericalMetricDebugMask = params.numericalMetricDebugMask ? 1 : 0;
    gp.numericalMetricTotalPoints = numericalMetricLoaded ? numericalBundle.metric.totalGridPoints : 0;
    gp.numericalAh1Valid = (numericalMetricLoaded && numericalBundle.horizonValid && numericalBundle.horizon.ah1Valid) ? 1 : 0;
    gp.numericalAh2Valid = (numericalMetricLoaded && numericalBundle.horizonValid && numericalBundle.horizon.ah2Valid) ? 1 : 0;
    gp.numericalAh1Center = numericalMetricLoaded ? toGpuVec3(numericalBundle.horizon.ah1Center) : GpuVec3{0,0,0};
    gp.numericalAh2Center = numericalMetricLoaded ? toGpuVec3(numericalBundle.horizon.ah2Center) : GpuVec3{0,0,0};
    gp.numericalAh1CaptureRadius = numericalMetricLoaded ? static_cast<real>(params.numericalMetricHorizonSafetyFactor * numericalBundle.horizon.ah1RMax) : real(0);
    gp.numericalAh2CaptureRadius = numericalMetricLoaded ? static_cast<real>(params.numericalMetricHorizonSafetyFactor * numericalBundle.horizon.ah2RMax) : real(0);

    GpuCamera gc{};
    gc.origin = toGpuVec3(camera.origin());
    gc.forward = toGpuVec3(camera.forward());
    gc.right = toGpuVec3(camera.right());
    gc.trueUp = toGpuVec3(camera.trueUp());
    gc.tanHalfFovY = static_cast<real>(camera.tanHalfFovY());
    gc.aspect = static_cast<real>(camera.aspect());

    GpuSceneData gd{};
    gd.shells = dShells.get();
    gd.modeKeys = dKeys.get();
    gd.series = dSeries.get();
    gd.samples = dSamples.get();
    gd.panorama = dPanorama.get();
    gd.numericalMetricGrids = numericalMetricLoaded ? dNumMetricGrids.get() : nullptr;
    gd.numericalMetricFields = numericalMetricLoaded ? dNumMetricFields.get() : nullptr;
    gd.numericalMetricChristoffel = nullptr;

    const dim3 block(16, 16);
    const dim3 grid((params.width + block.x - 1) / block.x,
                    (params.height + block.y - 1) / block.y);

    std::cerr << "CUDA render: fixed-step "
              << (params.metricLensingEnabled
                    ? (params.numericalMetricEnabled ? "numerical-metric full-ray" : "MP-lensed")
                    : "non-lensed")
              << " gwpv_peaks, image="
              << params.width << "x" << params.height
              << " spp=" << params.samplesPerPixel
              << " step=" << params.stepSize;
    if (params.metricLensingEnabled) {
        std::cerr << " rkStep=" << params.cudaMetricStep
                  << " colorStep=" << params.cudaMetricColorStep
                  << " doublePixelRadiusFrac=" << params.cudaDoublePixelRadiusFraction
                  << " maxRKSteps=" << params.cudaMetricMaxSteps;
        if (params.numericalMetricEnabled && numericalMetricLoaded) {
            std::cerr << " numericalSnapshot=" << numericalBundle.metric.snapshotIndex
                      << " numericalTime=" << numericalBundle.metric.time
                      << " grids=" << numericalBundle.metric.grids.size()
                      << " maxLayer=" << params.numericalMetricMaxLayer
                      << " boundaryBufferCells=" << params.numericalMetricBoundaryBufferCells
                      << " gammaFailAbs=" << params.numericalMetricGammaFailAbs
                      << " failFast=" << (params.numericalMetricFailFast ? "true" : "false")
                      << " debugMask=" << (params.numericalMetricDebugMask ? "true" : "false")
                      << " AHradii=(" << gp.numericalAh1CaptureRadius
                      << "," << gp.numericalAh2CaptureRadius << ")";
        }
    }
    std::cerr << " shells=" << hShells.size()
              << " modes=" << hKeys.size()
              << " samples=" << hSamples.size()
              << " block=" << block.x << "x" << block.y << "\n";

    if (gp.numericalMetricEnabled) {
        // Full-image numerical render.  Every pixel follows the same one-path
        // piecewise geodesic: numerical Christoffels inside safe sampled boxes,
        // zero acceleration outside them.  No straight-ray classification pass
        // and no numerical overwrite/compositing pass are used.
        dNumericalDiagnostics.allocate(1);
        CUDA_CHECK(cudaMemset(dNumericalDiagnostics.get(), 0, sizeof(GpuNumericalDiagnostics)));
        dNumericalFailure.allocate(1);
        CUDA_CHECK(cudaMemset(dNumericalFailure.get(), 0, sizeof(GpuNumericalFailure)));

        const auto passStart = std::chrono::steady_clock::now();
        renderNumericalMetricKernel<<<grid, block>>>(
            dOutput.get(),
            gp,
            gc,
            gd,
            dNumericalDiagnostics.get(),
            dNumericalFailure.get());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        numericalCorrectionSeconds = secondsSince(passStart);
        numericalPixelHits = static_cast<unsigned int>(params.width * params.height);
        std::cerr << "CUDA full-image numerical render complete: "
                  << numericalCorrectionSeconds << " s, pixels="
                  << numericalPixelHits << "/"
                  << (params.width * params.height) << "\n" << std::flush;

        CUDA_CHECK(cudaMemcpy(&hNumericalDiagnostics,
                              dNumericalDiagnostics.get(),
                              sizeof(GpuNumericalDiagnostics),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&hNumericalFailure,
                              dNumericalFailure.get(),
                              sizeof(GpuNumericalFailure),
                              cudaMemcpyDeviceToHost));
        if (hNumericalFailure.failed) {
            std::cerr << "\nCUDA NUMERICAL METRIC FAIL-FAST\n"
                      << "  reason=" << hNumericalFailure.reason
                      << " (" << numericalFailureReasonName(hNumericalFailure.reason) << ")\n"
                      << "  pixel=(" << hNumericalFailure.pixelX
                      << "," << hNumericalFailure.pixelY << ")"
                      << " sample=" << hNumericalFailure.sampleIndex
                      << " rkStep=" << hNumericalFailure.rkStep << "\n"
                      << "  gridIndex=" << hNumericalFailure.gridIndex
                      << " familyId=" << hNumericalFailure.familyId
                      << " layer=" << hNumericalFailure.layer
                      << " boundaryCells=" << hNumericalFailure.boundaryCells << "\n"
                      << "  fieldIndex=" << hNumericalFailure.fieldIndex
                      << " component=" << hNumericalFailure.component
                      << " value=" << hNumericalFailure.value
                      << " threshold=" << hNumericalFailure.threshold << "\n"
                      << "  x,y,z=(" << hNumericalFailure.x
                      << ", " << hNumericalFailure.y
                      << ", " << hNumericalFailure.z << ")\n"
                      << std::flush;
            throw std::runtime_error("CUDA numerical metric fail-fast triggered");
        }
    } else {
        const auto passStart = std::chrono::steady_clock::now();
        renderKernel<<<grid, block>>>(dOutput.get(), gp, gc, gd, nullptr);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        straightRenderSeconds = secondsSince(passStart);
        std::cerr << "CUDA render pass complete: full-image render="
                  << straightRenderSeconds << " s\n" << std::flush;
    }


    if (gp.numericalMetricEnabled) {
        std::cerr << "CUDA numerical diagnostics: totalPixels=" << (params.width * params.height)
                  << " numericalPixels=" << numericalPixelHits
                  << "/" << (params.width * params.height)
                  << " numericalAASamples=" << hNumericalDiagnostics.numericalSamples
                  << " enteredNumericalSamples=" << hNumericalDiagnostics.enteredNumericalSamples
                  << " totalRKSteps=" << hNumericalDiagnostics.totalRkSteps
                  << " maxRKStepsUsed=" << hNumericalDiagnostics.maxRkSteps
                  << " samplesHitMaxRKSteps=" << hNumericalDiagnostics.samplesHitMaxRkSteps
                  << " samplesCapturedAH=" << hNumericalDiagnostics.samplesCapturedAh
                  << " samplesCapturedAH1=" << hNumericalDiagnostics.samplesCapturedAh1
                  << " samplesCapturedAH2=" << hNumericalDiagnostics.samplesCapturedAh2
                  << " samplesCapturedAHUnknown=" << hNumericalDiagnostics.samplesCapturedAhUnknown
                  << " samplesExitedNumericalBox=" << hNumericalDiagnostics.samplesExitedNumericalBox
                  << "\n";
    }

    std::cerr << "CUDA pass timing: christoffelLoad=" << christoffelLoadSeconds
              << " s gpuUpload=" << gpuUploadSeconds
              << " s straightRender=" << straightRenderSeconds
              << " s classification=" << classificationSeconds
              << " s numericalCorrection=" << numericalCorrectionSeconds
              << " s\n";

    std::vector<GpuRGB8> hOutput(static_cast<std::size_t>(params.width) * static_cast<std::size_t>(params.height));
    CUDA_CHECK(cudaMemcpy(hOutput.data(), dOutput.get(), hOutput.size() * sizeof(GpuRGB8), cudaMemcpyDeviceToHost));

    Image image(params.width, params.height);
    for (int y = 0; y < params.height; ++y) {
        for (int x = 0; x < params.width; ++x) {
            const GpuRGB8 c = hOutput[static_cast<std::size_t>(y) * static_cast<std::size_t>(params.width) + static_cast<std::size_t>(x)];
            image.at(x, y) = RGB8{c.r, c.g, c.b};
        }
    }

    return image;
}
