#include "SpinWeightedHarmonic.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>

namespace {
constexpr double PI = 3.141592653589793238462643383279502884;

long double factorialLD(int n) {
    if (n < 0) throw std::runtime_error("Negative factorial");
    return std::tgammal(static_cast<long double>(n) + 1.0L);
}

long double powIntLD(long double x, int n) {
    if (n < 0) throw std::runtime_error("Negative integer exponent");
    if (n == 0) return 1.0L;
    return std::pow(x, n);
}

struct WignerDValueDeriv {
    long double value = 0.0L;
    long double dtheta = 0.0L;
};

WignerDValueDeriv wignerSmallDWithDerivative(int l, int mp, int m, double theta) {
    long double pref = std::sqrt(
        factorialLD(l + m) * factorialLD(l - m) *
        factorialLD(l + mp) * factorialLD(l - mp)
    );

    long double ct = std::cos(0.5L * static_cast<long double>(theta));
    long double st = std::sin(0.5L * static_cast<long double>(theta));

    long double total = 0.0L;
    long double dtotal = 0.0L;

    int kmin = std::max(0, m - mp);
    int kmax = std::min(l + m, l - mp);

    for (int k = kmin; k <= kmax; ++k) {
        long double denom =
            factorialLD(l + m - k) *
            factorialLD(k) *
            factorialLD(mp - m + k) *
            factorialLD(l - mp - k);

        int expC = 2 * l + m - mp - 2 * k;
        int expS = mp - m + 2 * k;
        long double sign = ((k - m + mp) % 2) ? -1.0L : 1.0L;
        long double coeff = sign * pref / denom;

        long double cPow = powIntLD(ct, expC);
        long double sPow = powIntLD(st, expS);
        total += coeff * cPow * sPow;

        long double dTerm = 0.0L;
        if (expC > 0) {
            dTerm += -0.5L * static_cast<long double>(expC) * st * powIntLD(ct, expC - 1) * sPow;
        }
        if (expS > 0) {
            dTerm +=  0.5L * static_cast<long double>(expS) * ct * cPow * powIntLD(st, expS - 1);
        }
        dtotal += coeff * dTerm;
    }

    return WignerDValueDeriv{total, dtotal};
}

} // namespace

SpinWeightedYDerivatives spinWeightedYWithDerivatives(int l, int m, int s, double theta, double phi) {
    if (std::abs(s) > l || std::abs(m) > l) return {};

    // Same convention as the earlier implementation:
    // _sY_lm = (-1)^s sqrt((2l+1)/(4pi)) d^l_{m,-s}(theta) exp(i m phi)
    double sign = (s % 2) ? -1.0 : 1.0;
    double amp = sign * std::sqrt((2.0 * l + 1.0) / (4.0 * PI));

    WignerDValueDeriv d = wignerSmallDWithDerivative(l, m, -s, theta);
    std::complex<double> phase = std::polar(1.0, static_cast<double>(m) * phi);

    std::complex<double> y = amp * static_cast<double>(d.value) * phase;
    std::complex<double> dtheta = amp * static_cast<double>(d.dtheta) * phase;
    std::complex<double> dphi = std::complex<double>(0.0, static_cast<double>(m)) * y;

    return SpinWeightedYDerivatives{y, dtheta, dphi};
}

std::complex<double> spinWeightedY(int l, int m, int s, double theta, double phi) {
    return spinWeightedYWithDerivatives(l, m, s, theta, phi).y;
}
