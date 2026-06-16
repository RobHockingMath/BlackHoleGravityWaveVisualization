#pragma once

#include <complex>

struct SpinWeightedYDerivatives {
    std::complex<double> y = {0.0, 0.0};
    std::complex<double> dtheta = {0.0, 0.0};
    std::complex<double> dphi = {0.0, 0.0};
};

std::complex<double> spinWeightedY(int l, int m, int s, double theta, double phi);
SpinWeightedYDerivatives spinWeightedYWithDerivatives(int l, int m, int s, double theta, double phi);
