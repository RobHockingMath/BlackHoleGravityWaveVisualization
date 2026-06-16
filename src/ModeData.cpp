#include "ModeData.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>

std::string modeName(const ModeKey& key) {
    std::ostringstream os;
    os << "(" << key.l << "," << key.m << ")";
    return os.str();
}

const ModeSeries* RadiusShell::findMode(const ModeKey& key) const {
    for (const auto& mode : modes) {
        if (mode.key == key) return &mode;
    }
    return nullptr;
}

namespace {

std::string stripComment(const std::string& line) {
    auto pos = line.find('#');
    std::string s = (pos == std::string::npos) ? line : line.substr(0, pos);

    auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";

    auto last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

bool useRadiusFromParams(double radius, const Params& p) {
    int r = static_cast<int>(std::lround(radius));
    switch (r) {
        case 100: return p.use_extraction_sphere_100;
        case 115: return p.use_extraction_sphere_115;
        case 136: return p.use_extraction_sphere_136;
        case 167: return p.use_extraction_sphere_167;
        case 214: return p.use_extraction_sphere_214;
        case 300: return p.use_extraction_sphere_300;
        case 500: return p.use_extraction_sphere_500;
        default:
            // Unknown radii are kept by default. This makes the loader future-proof.
            return true;
    }
}

std::complex<double> modeSampleSlope(const std::vector<TimeSample>& s, std::size_t i) {
    if (s.size() < 2) return {0.0, 0.0};

    if (i == 0) {
        double dt = s[1].t - s[0].t;
        if (dt <= 0.0) return {0.0, 0.0};
        return (s[1].y - s[0].y) / dt;
    }

    if (i + 1 >= s.size()) {
        double dt = s[i].t - s[i - 1].t;
        if (dt <= 0.0) return {0.0, 0.0};
        return (s[i].y - s[i - 1].y) / dt;
    }

    double dt = s[i + 1].t - s[i - 1].t;
    if (dt <= 0.0) return {0.0, 0.0};
    return (s[i + 1].y - s[i - 1].y) / dt;
}

} // namespace

ModeDataSet ModeDataSet::loadText(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Could not open mode file: " + path);

    ModeDataSet data;
    RadiusShell* currentShell = nullptr;

    std::string raw;
    int lineNo = 0;

    while (std::getline(in, raw)) {
        ++lineNo;
        std::string line = stripComment(raw);
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string tok;
        ss >> tok;

        if (tok == "GW_MODE_MULTI_RADIUS_TEXT_V1" || tok == "GW_MODE_TEXT_V1") {
            continue;
        }

        if (tok == "radius") {
            double radius = 0.0;
            ss >> radius;
            if (!ss) throw std::runtime_error("Bad radius line in " + path + " at line " + std::to_string(lineNo));

            data.shells.push_back(RadiusShell{});
            currentShell = &data.shells.back();
            currentShell->radius = radius;
            continue;
        }

        if (tok == "endradius") {
            currentShell = nullptr;
            continue;
        }

        if (tok == "mode") {
            if (!currentShell) {
                throw std::runtime_error("Mode before radius in " + path + " at line " + std::to_string(lineNo));
            }

            ModeSeries ms;
            int n = 0;
            ss >> ms.key.l >> ms.key.m >> n;
            if (!ss || n <= 0) {
                throw std::runtime_error("Bad mode header in " + path + " at line " + std::to_string(lineNo));
            }

            ms.samples.reserve(static_cast<std::size_t>(n));

            while (static_cast<int>(ms.samples.size()) < n) {
                if (!std::getline(in, raw)) {
                    throw std::runtime_error("Unexpected EOF while reading mode " + modeName(ms.key));
                }
                ++lineNo;
                std::string sampleLine = stripComment(raw);
                if (sampleLine.empty()) continue;

                std::stringstream ps(sampleLine);
                std::string maybeEnd;
                ps >> maybeEnd;
                if (maybeEnd == "endmode") {
                    throw std::runtime_error("Unexpected endmode before all samples for mode " + modeName(ms.key));
                }

                double t = std::stod(maybeEnd);
                double re = 0.0;
                double im = 0.0;
                ps >> re >> im;
                if (!ps) {
                    throw std::runtime_error("Bad sample line for mode " + modeName(ms.key) +
                                             " at line " + std::to_string(lineNo));
                }

                ms.samples.push_back(TimeSample{t, std::complex<double>(re, im)});
            }

            // Consume optional endmode. If the next non-empty line is not endmode, put it back is hard;
            // our exporter always writes endmode, so require it when present in the stream.
            std::streampos before;
            while (true) {
                before = in.tellg();
                if (!std::getline(in, raw)) break;
                ++lineNo;
                std::string maybe = stripComment(raw);
                if (maybe.empty()) continue;
                if (maybe == "endmode") break;

                // We read one line too far. This should not happen for our exporter, but fail loudly
                // rather than silently skipping a token.
                throw std::runtime_error("Expected endmode after mode " + modeName(ms.key) +
                                         " in " + path + " at line " + std::to_string(lineNo) +
                                         ", got: " + maybe);
            }

            std::sort(ms.samples.begin(), ms.samples.end(),
                      [](const TimeSample& a, const TimeSample& b) { return a.t < b.t; });

            currentShell->modes.push_back(std::move(ms));
            continue;
        }

        if (tok == "endmode") {
            continue;
        }

        throw std::runtime_error("Unknown token in " + path + " at line " + std::to_string(lineNo) + ": " + tok);
    }

    if (data.shells.empty()) {
        throw std::runtime_error("No extraction radii found in " + path);
    }

    std::sort(data.shells.begin(), data.shells.end(),
              [](const RadiusShell& a, const RadiusShell& b) { return a.radius < b.radius; });

    return data;
}

void ModeDataSet::filterModes(const std::vector<ModeKey>& activeModes) {
    if (activeModes.empty()) return;

    for (auto& shell : shells) {
        std::vector<ModeSeries> kept;
        for (auto& mode : shell.modes) {
            auto it = std::find(activeModes.begin(), activeModes.end(), mode.key);
            if (it != activeModes.end()) kept.push_back(std::move(mode));
        }
        shell.modes = std::move(kept);
    }

    shells.erase(std::remove_if(shells.begin(), shells.end(),
                                [](const RadiusShell& shell) { return shell.modes.empty(); }),
                 shells.end());

    if (shells.empty()) {
        std::ostringstream os;
        os << "Harmonic filter removed all modes. Requested:";
        for (const auto& lm : activeModes) os << " " << modeName(lm);
        throw std::runtime_error(os.str());
    }
}

void ModeDataSet::filterExtractionSpheres(const Params& params) {
    shells.erase(std::remove_if(shells.begin(), shells.end(),
                                [&](const RadiusShell& shell) { return !useRadiusFromParams(shell.radius, params); }),
                 shells.end());

    if (shells.empty()) {
        throw std::runtime_error("Extraction-sphere filter removed all radii");
    }
}

void ModeDataSet::autoNormalizeByMaxAbs() {
    double maxAbs = 0.0;
    for (const auto& shell : shells) {
        for (const auto& mode : shell.modes) {
            for (const auto& s : mode.samples) maxAbs = std::max(maxAbs, std::abs(s.y));
        }
    }

    if (!std::isfinite(maxAbs) || maxAbs <= 0.0) return;

    for (auto& shell : shells) {
        for (auto& mode : shell.modes) {
            for (auto& s : mode.samples) s.y /= maxAbs;
        }
    }

    std::cerr << "Display-normalized all mode coefficients by max |coeff| = " << maxAbs << "\n";
}

double ModeDataSet::timeMin() const {
    double v = std::numeric_limits<double>::infinity();
    for (const auto& shell : shells) {
        for (const auto& mode : shell.modes) {
            if (!mode.samples.empty()) v = std::min(v, mode.samples.front().t);
        }
    }
    return v;
}

double ModeDataSet::timeMax() const {
    double v = -std::numeric_limits<double>::infinity();
    for (const auto& shell : shells) {
        for (const auto& mode : shell.modes) {
            if (!mode.samples.empty()) v = std::max(v, mode.samples.back().t);
        }
    }
    return v;
}

double ModeDataSet::suggestedPeakTime() const {
    const ModeSeries* preferred = nullptr;

    // Prefer the outermost available (2,2) mode.
    for (auto it = shells.rbegin(); it != shells.rend(); ++it) {
        preferred = it->findMode(ModeKey{2, 2});
        if (preferred) break;
    }

    if (!preferred) {
        for (auto it = shells.rbegin(); it != shells.rend(); ++it) {
            if (!it->modes.empty()) {
                preferred = &it->modes.front();
                break;
            }
        }
    }

    if (!preferred || preferred->samples.empty()) return 0.0;

    std::size_t best = 0;
    double bestAbs = -1.0;
    for (std::size_t i = 0; i < preferred->samples.size(); ++i) {
        double a = std::abs(preferred->samples[i].y);
        if (a > bestAbs) { bestAbs = a; best = i; }
    }
    return preferred->samples[best].t;
}

std::vector<ModeKey> ModeDataSet::modeKeys() const {
    std::set<std::pair<int, int>> seen;
    for (const auto& shell : shells) {
        for (const auto& mode : shell.modes) {
            seen.insert({mode.key.l, mode.key.m});
        }
    }

    std::vector<ModeKey> keys;
    for (const auto& p : seen) keys.push_back(ModeKey{p.first, p.second});
    return keys;
}

std::vector<double> ModeDataSet::radii() const {
    std::vector<double> out;
    out.reserve(shells.size());
    for (const auto& shell : shells) out.push_back(shell.radius);
    return out;
}

ModeInterpolation interpolateModeLinearWithDerivative(const ModeSeries& mode, double t) {
    const auto& s = mode.samples;
    if (s.empty()) return {};
    if (t < s.front().t || t > s.back().t) return {};
    if (s.size() == 1) return ModeInterpolation{s.front().y, {0.0, 0.0}};

    auto it = std::lower_bound(s.begin(), s.end(), t, [](const TimeSample& sample, double value) {
        return sample.t < value;
    });

    if (it == s.begin()) {
        return ModeInterpolation{it->y, modeSampleSlope(s, 0)};
    }
    if (it == s.end()) {
        return ModeInterpolation{s.back().y, modeSampleSlope(s, s.size() - 1)};
    }

    const TimeSample& b = *it;
    const TimeSample& a = *(it - 1);
    double denom = b.t - a.t;
    if (denom <= 0.0) return ModeInterpolation{a.y, {0.0, 0.0}};

    double w = (t - a.t) / denom;
    std::complex<double> value = (1.0 - w) * a.y + w * b.y;
    std::complex<double> deriv = (b.y - a.y) / denom;
    return ModeInterpolation{value, deriv};
}

std::complex<double> interpolateMode(const ModeSeries& mode, double t) {
    return interpolateModeLinearWithDerivative(mode, t).value;
}

ModeInterpolation interpolateModeCubicHermite(const ModeSeries& mode, double t) {
    const auto& s = mode.samples;
    if (s.empty()) return {};
    if (t < s.front().t || t > s.back().t) return {};
    if (s.size() == 1) return ModeInterpolation{s.front().y, {0.0, 0.0}};

    auto it = std::lower_bound(s.begin(), s.end(), t, [](const TimeSample& sample, double value) {
        return sample.t < value;
    });

    if (it == s.begin()) {
        return ModeInterpolation{it->y, modeSampleSlope(s, 0)};
    }
    if (it == s.end()) {
        return ModeInterpolation{s.back().y, modeSampleSlope(s, s.size() - 1)};
    }

    std::size_t i1 = static_cast<std::size_t>(it - s.begin());
    std::size_t i0 = i1 - 1;

    const TimeSample& a = s[i0];
    const TimeSample& b = s[i1];
    double h = b.t - a.t;
    if (h <= 0.0) return ModeInterpolation{a.y, {0.0, 0.0}};

    double u = (t - a.t) / h;
    u = std::clamp(u, 0.0, 1.0);

    // Cubic Hermite interpolation.  The endpoint slopes are estimated from the
    // neighboring samples.  This gives a C1 interpolation in time without using
    // finite differences at render time.
    std::complex<double> m0 = modeSampleSlope(s, i0);
    std::complex<double> m1 = modeSampleSlope(s, i1);

    double u2 = u * u;
    double u3 = u2 * u;

    double h00 =  2.0 * u3 - 3.0 * u2 + 1.0;
    double h10 =        u3 - 2.0 * u2 + u;
    double h01 = -2.0 * u3 + 3.0 * u2;
    double h11 =        u3 -       u2;

    std::complex<double> value = h00 * a.y + h10 * h * m0 + h01 * b.y + h11 * h * m1;

    double dh00 =  6.0 * u2 - 6.0 * u;
    double dh10 =  3.0 * u2 - 4.0 * u + 1.0;
    double dh01 = -6.0 * u2 + 6.0 * u;
    double dh11 =  3.0 * u2 - 2.0 * u;

    std::complex<double> deriv = (dh00 * a.y + dh10 * h * m0 + dh01 * b.y + dh11 * h * m1) / h;

    return ModeInterpolation{value, deriv};
}
