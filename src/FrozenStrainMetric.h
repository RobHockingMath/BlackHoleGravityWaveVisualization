#pragma once

#include "AdaptiveRK.h"
#include "BuildScene.h"
#include "Params.h"
#include "Vec3.h"
#include "BlackHoleSystem.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>

namespace FrozenStrainMetric {

struct Mat3 {
    double m[3][3] = {{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}};
};

struct MetricDerivatives {
    // d[k].m[i][j] = partial gamma_ij / partial x^k
    Mat3 d[3];
};

inline double component(const Vec3& v, int i) {
    return (i == 0) ? v.x : ((i == 1) ? v.y : v.z);
}

inline Vec3 vecFromComponents(double x, double y, double z) {
    return Vec3(x, y, z);
}

inline Vec3 basisVector(int i) {
    if (i == 0) return Vec3(1.0, 0.0, 0.0);
    if (i == 1) return Vec3(0.0, 1.0, 0.0);
    return Vec3(0.0, 0.0, 1.0);
}

inline Mat3 identity() {
    Mat3 A;
    A.m[0][0] = 1.0;
    A.m[1][1] = 1.0;
    A.m[2][2] = 1.0;
    return A;
}

inline double quadraticForm(const Mat3& A, const Vec3& v) {
    const double vv[3] = {v.x, v.y, v.z};
    double s = 0.0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            s += A.m[i][j] * vv[i] * vv[j];
        }
    }
    return s;
}

inline Vec3 matVec(const Mat3& A, const Vec3& v) {
    return Vec3(
        A.m[0][0]*v.x + A.m[0][1]*v.y + A.m[0][2]*v.z,
        A.m[1][0]*v.x + A.m[1][1]*v.y + A.m[1][2]*v.z,
        A.m[2][0]*v.x + A.m[2][1]*v.y + A.m[2][2]*v.z
    );
}

inline double det3(const Mat3& A) {
    return A.m[0][0] * (A.m[1][1]*A.m[2][2] - A.m[1][2]*A.m[2][1])
         - A.m[0][1] * (A.m[1][0]*A.m[2][2] - A.m[1][2]*A.m[2][0])
         + A.m[0][2] * (A.m[1][0]*A.m[2][1] - A.m[1][1]*A.m[2][0]);
}

inline bool inverse3(const Mat3& A, Mat3& invOut) {
    const double det = det3(A);
    if (!std::isfinite(det) || std::abs(det) < 1.0e-12) return false;

    invOut.m[0][0] =  (A.m[1][1]*A.m[2][2] - A.m[1][2]*A.m[2][1]) / det;
    invOut.m[0][1] = -(A.m[0][1]*A.m[2][2] - A.m[0][2]*A.m[2][1]) / det;
    invOut.m[0][2] =  (A.m[0][1]*A.m[1][2] - A.m[0][2]*A.m[1][1]) / det;

    invOut.m[1][0] = -(A.m[1][0]*A.m[2][2] - A.m[1][2]*A.m[2][0]) / det;
    invOut.m[1][1] =  (A.m[0][0]*A.m[2][2] - A.m[0][2]*A.m[2][0]) / det;
    invOut.m[1][2] = -(A.m[0][0]*A.m[1][2] - A.m[0][2]*A.m[1][0]) / det;

    invOut.m[2][0] =  (A.m[1][0]*A.m[2][1] - A.m[1][1]*A.m[2][0]) / det;
    invOut.m[2][1] = -(A.m[0][0]*A.m[2][1] - A.m[0][1]*A.m[2][0]) / det;
    invOut.m[2][2] =  (A.m[0][0]*A.m[1][1] - A.m[0][1]*A.m[1][0]) / det;
    return true;
}

inline void localSphericalFrame(const Vec3& x, Vec3& n, Vec3& eTheta, Vec3& ePhi) {
    const double r = length(x);
    if (!(r > 1.0e-12)) {
        n = Vec3(0.0, 0.0, 1.0);
        eTheta = Vec3(1.0, 0.0, 0.0);
        ePhi = Vec3(0.0, 1.0, 0.0);
        return;
    }

    n = (1.0 / r) * x;

    const double rho = std::sqrt(std::max(0.0, x.x*x.x + x.y*x.y));
    if (rho > 1.0e-12 * std::max(1.0, r)) {
        const double cosTheta = x.z / r;
        const double sinTheta = rho / r;
        const double cosPhi = x.x / rho;
        const double sinPhi = x.y / rho;
        eTheta = Vec3(cosTheta * cosPhi, cosTheta * sinPhi, -sinTheta);
        ePhi = Vec3(-sinPhi, cosPhi, 0.0);
    } else {
        // At the polar axis the azimuthal frame is arbitrary. Pick a stable one.
        eTheta = Vec3(1.0, 0.0, 0.0);
        ePhi = Vec3(0.0, (x.z >= 0.0 ? 1.0 : -1.0), 0.0);
    }
}


inline bool localSphericalFrameWithDerivatives(const Vec3& x,
                                               Vec3& n,
                                               Vec3& eTheta,
                                               Vec3& ePhi,
                                               Vec3 deTheta[3],
                                               Vec3 dePhi[3]) {
    localSphericalFrame(x, n, eTheta, ePhi);

    for (int k = 0; k < 3; ++k) {
        deTheta[k] = Vec3(0.0, 0.0, 0.0);
        dePhi[k] = Vec3(0.0, 0.0, 0.0);
    }

    const double X = x.x;
    const double Y = x.y;
    const double Z = x.z;
    const double r = length(x);
    const double rho2 = X*X + Y*Y;
    const double rho = std::sqrt(std::max(0.0, rho2));

    // The usual spherical polarization frame is singular on the z-axis.  The
    // renderer already masks the visual seam; for metric derivatives we fall
    // back to finite differences there rather than pretending the analytic
    // basis derivatives are well-defined.
    if (!(r > 1.0e-12) || rho <= 1.0e-12 * std::max(1.0, r)) {
        return false;
    }

    const double invR = 1.0 / r;
    const double invRho = 1.0 / rho;
    const double invR2 = invR * invR;
    const double invRho2 = invRho * invRho;

    const Vec3 eRho(X * invRho, Y * invRho, 0.0);
    const double a = Z * invR;      // cos(theta)

    for (int k = 0; k < 3; ++k) {
        const double dxk = (k == 0) ? 1.0 : 0.0;
        const double dyk = (k == 1) ? 1.0 : 0.0;
        const double dzk = (k == 2) ? 1.0 : 0.0;
        const double xk = component(x, k);

        const double dr = xk * invR;
        const double drho = (k == 0) ? (X * invRho) : ((k == 1) ? (Y * invRho) : 0.0);

        const Vec3 deRho(
            dxk * invRho - X * drho * invRho2,
            dyk * invRho - Y * drho * invRho2,
            0.0
        );

        const double da = dzk * invR - Z * dr * invR2;
        const double db = drho * invR - rho * dr * invR2; // derivative of rho/r

        // e_theta = (z/r) e_rho - (rho/r) e_z
        deTheta[k] = da * eRho + a * deRho - db * Vec3(0.0, 0.0, 1.0);

        // e_phi = (-y/rho, x/rho, 0)
        dePhi[k] = Vec3(
            -dyk * invRho + Y * drho * invRho2,
             dxk * invRho - X * drho * invRho2,
             0.0
        );
    }

    return true;
}

inline Mat3 makeStrainSpatialMetric(const Vec3& x,
                                    double frameTime,
                                    const Scene& scene,
                                    const Params& params)
{
    Mat3 gamma = identity();
    if (!(params.metricPerturbationScale != 0.0)) return gamma;

    std::complex<double> H = scene.field.eval(x, frameTime);

    if (params.metricUseRadiusScaledStrain) {
        const double rDisplay = std::max(length(x), params.innerWaveScaleRadius);
        H *= rDisplay;
    }

    // Convention: H = h_plus - i h_cross.  This is easy to flip later if needed.
    const double hPlus = std::real(H);
    const double hCross = -std::imag(H);

    Vec3 n, eTheta, ePhi;
    localSphericalFrame(x, n, eTheta, ePhi);

    const double et[3] = {eTheta.x, eTheta.y, eTheta.z};
    const double ep[3] = {ePhi.x,   ePhi.y,   ePhi.z};

    const double A = params.metricPerturbationScale;

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            const double plusTensor = et[i]*et[j] - ep[i]*ep[j];
            const double crossTensor = et[i]*ep[j] + ep[i]*et[j];
            gamma.m[i][j] += A * (hPlus * plusTensor + hCross * crossTensor);
        }
    }

    return gamma;
}


inline Mat3 applyMajumdarPapapetrouFactor(const Mat3& baseGamma,
                                          const Vec3& x,
                                          double frameTime,
                                          const Params& params)
{
    if (!params.blackHolesEnabled || !params.metricUseMajumdarPapapetrou) {
        return baseGamma;
    }

    const BlackHoleSystem::MPPotential pot =
        BlackHoleSystem::majumdarPapapetrouPotential(x, frameTime, params);

    const double U = std::max(pot.U, 1.0e-12);
    const double U2 = U * U;
    const double U4 = U2 * U2;

    Mat3 gamma = baseGamma;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            gamma.m[i][j] *= U4;
        }
    }
    return gamma;
}

inline Mat3 makeSpatialMetric(const Vec3& x,
                              double frameTime,
                              const Scene& scene,
                              const Params& params)
{
    const Mat3 baseGamma = makeStrainSpatialMetric(x, frameTime, scene, params);
    return applyMajumdarPapapetrouFactor(baseGamma, x, frameTime, params);
}

inline MetricDerivatives finiteDifferenceMetricDerivatives(const Vec3& x,
                                                           double frameTime,
                                                           const Scene& scene,
                                                           const Params& params)
{
    MetricDerivatives out;
    const double eps = std::max(params.metricDerivativeEps, 1.0e-6);

    for (int k = 0; k < 3; ++k) {
        const Vec3 e = basisVector(k);
        const Mat3 gp = makeSpatialMetric(x + eps * e, frameTime, scene, params);
        const Mat3 gm = makeSpatialMetric(x - eps * e, frameTime, scene, params);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                out.d[k].m[i][j] = (gp.m[i][j] - gm.m[i][j]) / (2.0 * eps);
            }
        }
    }

    return out;
}


inline MetricDerivatives analyticMetricDerivatives(const Vec3& x,
                                                   double frameTime,
                                                   const Scene& scene,
                                                   const Params& params)
{
    MetricDerivatives out;

    const bool useStrainPerturbation = (params.metricPerturbationScale != 0.0);

    if (useStrainPerturbation) {
        Vec3 n, eTheta, ePhi;
        Vec3 deTheta[3];
        Vec3 dePhi[3];
        if (!localSphericalFrameWithDerivatives(x, n, eTheta, ePhi, deTheta, dePhi)) {
            // Near the spin/polar-frame seam the analytic spherical basis derivatives
            // are singular. Fall back to finite differences of the full metric there.
            return finiteDifferenceMetricDerivatives(x, frameTime, scene, params);
        }

        const std::complex<double> H0 = scene.field.eval(x, frameTime);
        std::complex<double> dH[3];
        for (int k = 0; k < 3; ++k) {
            const FieldRaySample sample = scene.field.evalWithRayDerivative(x, basisVector(k), frameTime);
            dH[k] = sample.dValueDs;
        }

        std::complex<double> H = H0;
        std::complex<double> dHused[3] = {dH[0], dH[1], dH[2]};

        if (params.metricUseRadiusScaledStrain) {
            const double r = length(x);
            const double rDisplay = std::max(r, params.innerWaveScaleRadius);
            H = rDisplay * H0;

            for (int k = 0; k < 3; ++k) {
                double dR = 0.0;
                if (r > std::max(params.innerWaveScaleRadius, 1.0e-12)) {
                    dR = component(x, k) / r;
                }
                dHused[k] = rDisplay * dH[k] + dR * H0;
            }
        }

        const double hPlus = std::real(H);
        const double hCross = -std::imag(H);

        const double et[3] = {eTheta.x, eTheta.y, eTheta.z};
        const double ep[3] = {ePhi.x,   ePhi.y,   ePhi.z};
        const double A = params.metricPerturbationScale;

        for (int k = 0; k < 3; ++k) {
            const double dhPlus = std::real(dHused[k]);
            const double dhCross = -std::imag(dHused[k]);

            const double det[3] = {deTheta[k].x, deTheta[k].y, deTheta[k].z};
            const double dep[3] = {dePhi[k].x,   dePhi[k].y,   dePhi[k].z};

            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    const double plusTensor = et[i]*et[j] - ep[i]*ep[j];
                    const double crossTensor = et[i]*ep[j] + ep[i]*et[j];

                    const double dPlusTensor =
                        det[i]*et[j] + et[i]*det[j]
                      - dep[i]*ep[j] - ep[i]*dep[j];

                    const double dCrossTensor =
                        det[i]*ep[j] + et[i]*dep[j]
                      + dep[i]*et[j] + ep[i]*det[j];

                    out.d[k].m[i][j] = A * (
                        dhPlus  * plusTensor  + hPlus  * dPlusTensor
                      + dhCross * crossTensor + hCross * dCrossTensor
                    );
                }
            }
        }
    }

    if (params.blackHolesEnabled && params.metricUseMajumdarPapapetrou) {
        const Mat3 baseGamma = makeStrainSpatialMetric(x, frameTime, scene, params);
        const BlackHoleSystem::MPPotential pot =
            BlackHoleSystem::majumdarPapapetrouPotential(x, frameTime, params);

        const double U = std::max(pot.U, 1.0e-12);
        const double U2 = U * U;
        const double U3 = U2 * U;
        const double U4 = U2 * U2;
        const double dU[3] = {pot.gradU.x, pot.gradU.y, pot.gradU.z};

        for (int k = 0; k < 3; ++k) {
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    out.d[k].m[i][j] =
                        U4 * out.d[k].m[i][j]
                      + 4.0 * U3 * dU[k] * baseGamma.m[i][j];
                }
            }
        }
    }

    return out;
}


inline MetricDerivatives metricDerivatives(const Vec3& x,
                                           double frameTime,
                                           const Scene& scene,
                                           const Params& params)
{
    if (params.metricUseAnalyticDerivatives) {
        return analyticMetricDerivatives(x, frameTime, scene, params);
    }
    return finiteDifferenceMetricDerivatives(x, frameTime, scene, params);
}

inline Vec3 acceleration(const Vec3& x,
                         const Vec3& v,
                         double frameTime,
                         const Scene& scene,
                         const Params& params)
{
    const Mat3 gamma = makeSpatialMetric(x, frameTime, scene, params);
    const MetricDerivatives dg = metricDerivatives(x, frameTime, scene, params);

    const double vv[3] = {v.x, v.y, v.z};
    double rhs[3] = {0.0, 0.0, 0.0};

    for (int i = 0; i < 3; ++i) {
        double term1 = 0.0;
        double term2 = 0.0;
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                term1 += 0.5 * dg.d[i].m[j][k] * vv[j] * vv[k];
                term2 += dg.d[k].m[i][j] * vv[k] * vv[j];
            }
        }
        rhs[i] = term1 - term2;
    }

    Mat3 invGamma;
    if (!inverse3(gamma, invGamma)) {
        return Vec3(0.0, 0.0, 0.0);
    }

    const Vec3 rhsVec(rhs[0], rhs[1], rhs[2]);
    Vec3 a = matVec(invGamma, rhsVec);

    if (!std::isfinite(a.x) || !std::isfinite(a.y) || !std::isfinite(a.z)) {
        return Vec3(0.0, 0.0, 0.0);
    }
    return a;
}

inline Vec3 normalizeNullSpeed(const Vec3& x,
                               const Vec3& v,
                               double frameTime,
                               const Scene& scene,
                               const Params& params)
{
    const Mat3 gamma = makeSpatialMetric(x, frameTime, scene, params);
    const double q = quadraticForm(gamma, v);
    if (!(q > 1.0e-30) || !std::isfinite(q)) return v;
    return (1.0 / std::sqrt(q)) * v;
}

inline AdaptiveRK::Derivative geodesicDerivative(const AdaptiveRK::State& y,
                                                 double frameTime,
                                                 const Scene& scene,
                                                 const Params& params)
{
    AdaptiveRK::Derivative d;
    d.dx = y.v;
    d.dv = acceleration(y.x, y.v, frameTime, scene, params);
    return d;
}

} // namespace FrozenStrainMetric
