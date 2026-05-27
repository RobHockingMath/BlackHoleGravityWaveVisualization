#pragma once

#include "Vec3.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace AdaptiveRK {

struct State {
    Vec3 x;
    Vec3 v;
};

struct Derivative {
    Vec3 dx;
    Vec3 dv;
};

struct StepResult {
    bool accepted = false;
    State y0;
    State y1;
    double hUsed = 0.0;
    double hNext = 0.0;
    double error = 0.0;
    int rejectedSteps = 0;
};

inline State addScaled(const State& y, const Derivative& k, double h, double a) {
    State out;
    out.x = y.x + (h * a) * k.dx;
    out.v = y.v + (h * a) * k.dv;
    return out;
}

inline State addCombination(const State& y,
                            double h,
                            const Derivative& k1, double a1,
                            const Derivative& k2, double a2,
                            const Derivative& k3, double a3,
                            const Derivative& k4, double a4,
                            const Derivative& k5, double a5,
                            const Derivative& k6, double a6)
{
    State out;
    out.x = y.x + h * (a1*k1.dx + a2*k2.dx + a3*k3.dx + a4*k4.dx + a5*k5.dx + a6*k6.dx);
    out.v = y.v + h * (a1*k1.dv + a2*k2.dv + a3*k3.dv + a4*k4.dv + a5*k5.dv + a6*k6.dv);
    return out;
}

inline double stateErrorNorm(const State& y4,
                             const State& y5,
                             const State& yRef,
                             double absTol,
                             double relTol)
{
    const Vec3 dx = y5.x - y4.x;
    const Vec3 dv = y5.v - y4.v;

    const double sx = absTol + relTol * std::max(1.0, length(yRef.x));
    const double sv = absTol + relTol * std::max(1.0, length(yRef.v));

    const double ex = length(dx) / std::max(sx, 1.0e-300);
    const double ev = length(dv) / std::max(sv, 1.0e-300);
    return std::max(ex, ev);
}

// One adaptive Cash-Karp RK45 step. The derivative functor must have signature:
//     Derivative f(const State& y)
// and should return dx/dt, dv/dt.
template <class DerivFunc>
StepResult adaptiveCashKarpStep(const State& y,
                                double hInitial,
                                double hMin,
                                double hMax,
                                double absTol,
                                double relTol,
                                DerivFunc&& f)
{
    StepResult result;
    result.y0 = y;

    double h = std::clamp(std::abs(hInitial), std::max(hMin, 1.0e-12), std::max(hMax, hMin));
    hMin = std::max(hMin, 1.0e-12);
    hMax = std::max(hMax, hMin);

    constexpr int maxRejects = 32;
    //constexpr int maxRejects = 8;

    for (int attempt = 0; attempt <= maxRejects; ++attempt) {
        const Derivative k1 = f(y);
        const Derivative k2 = f(addScaled(y, k1, h, 1.0/5.0));
        const Derivative k3 = f(addCombination(y, h, k1, 3.0/40.0, k2, 9.0/40.0, k1, 0.0, k1, 0.0, k1, 0.0, k1, 0.0));
        const Derivative k4 = f(addCombination(y, h, k1, 3.0/10.0, k2, -9.0/10.0, k3, 6.0/5.0, k1, 0.0, k1, 0.0, k1, 0.0));
        const Derivative k5 = f(addCombination(y, h, k1, -11.0/54.0, k2, 5.0/2.0, k3, -70.0/27.0, k4, 35.0/27.0, k1, 0.0, k1, 0.0));
        const Derivative k6 = f(addCombination(y, h, k1, 1631.0/55296.0, k2, 175.0/512.0, k3, 575.0/13824.0, k4, 44275.0/110592.0, k5, 253.0/4096.0, k1, 0.0));

        // 5th-order Cash-Karp solution.
        const State y5 = addCombination(y, h,
            k1, 37.0/378.0,
            k2, 0.0,
            k3, 250.0/621.0,
            k4, 125.0/594.0,
            k5, 0.0,
            k6, 512.0/1771.0);

        // Embedded 4th-order solution.
        const State y4 = addCombination(y, h,
            k1, 2825.0/27648.0,
            k2, 0.0,
            k3, 18575.0/48384.0,
            k4, 13525.0/55296.0,
            k5, 277.0/14336.0,
            k6, 1.0/4.0);

        const double err = stateErrorNorm(y4, y5, y, absTol, relTol);
        const bool forceAccept = (h <= hMin * (1.0 + 1.0e-12));

        if (err <= 1.0 || forceAccept) {
            const double safety = 0.90;
            const double exponent = 0.20; // 1/(order+1) for error-controlled growth, conservative.
            double factor = 2.0;
            if (err > 1.0e-14) {
                factor = safety * std::pow(1.0 / err, exponent);
                factor = std::clamp(factor, 0.2, 4.0);
            }

            result.accepted = true;
            result.y1 = y5;
            result.hUsed = h;
            result.hNext = std::clamp(h * factor, hMin, hMax);
            result.error = err;
            result.rejectedSteps = attempt;
            return result;
        }

        const double safety = 0.90;
        double factor = safety * std::pow(1.0 / std::max(err, 1.0e-14), 0.25);
        factor = std::clamp(factor, 0.1, 0.5);
        h = std::max(hMin, h * factor);
    }

    // Extremely defensive fallback: if everything failed, take the minimum step using Euler-like
    // advancement through the derivative. This should almost never happen; callers can also abort
    // on accepted=false if they prefer.
    const Derivative k = f(y);
    result.accepted = false;
    result.y1 = addScaled(y, k, hMin, 1.0);
    result.hUsed = hMin;
    result.hNext = hMin;
    result.error = std::numeric_limits<double>::infinity();
    result.rejectedSteps = maxRejects;
    return result;
}

} // namespace AdaptiveRK
