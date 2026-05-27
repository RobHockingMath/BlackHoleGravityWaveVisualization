#pragma once

#include "Params.h"
#include "Vec3.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace BlackHoleSystem {

struct TrajectorySample {
    double time = 0.0;
    Vec3 rel = Vec3(0.0, 0.0, 0.0); // plus minus minus, in source/render coordinates
};

struct TrajectoryTable {
    std::vector<TrajectorySample> samples;
    bool valid = false;
};

struct State {
    Vec3 plusCenter = Vec3(0.0, 0.0, 0.0);
    Vec3 minusCenter = Vec3(0.0, 0.0, 0.0);

    double plusMass = 36.0 / 65.0;
    double minusMass = 29.0 / 65.0;

    double plusRenderRadius = 2.0 * (36.0 / 65.0);
    double minusRenderRadius = 2.0 * (29.0 / 65.0);

    double plusCaptureRadius = 2.0 * (36.0 / 65.0);
    double minusCaptureRadius = 2.0 * (29.0 / 65.0);

    bool valid = false;
};

inline std::string trimCell(std::string s) {
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

inline std::vector<std::string> splitCsvLine(const std::string& line) {
    std::vector<std::string> cells;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) {
        cells.push_back(trimCell(cell));
    }
    return cells;
}

inline bool parseDoubleCell(const std::string& s, double& out) {
    try {
        std::size_t pos = 0;
        out = std::stod(s, &pos);
        return pos > 0 && std::isfinite(out);
    } catch (...) {
        return false;
    }
}

inline TrajectoryTable loadTrajectoryTable(const std::string& path) {
    TrajectoryTable table;

    std::ifstream in(path);
    if (!in) {
        std::cerr << "WARNING: Could not open black-hole trajectory CSV '" << path
                  << "'. Black holes will be disabled for this frame.\n";
        return table;
    }

    std::string line;
    bool sawAnyData = false;
    while (std::getline(in, line)) {
        if (trimCell(line).empty()) continue;

        const std::vector<std::string> cells = splitCsvLine(line);
        if (cells.size() < 3) continue;

        double c0 = 0.0;
        double c1 = 0.0;
        double c2 = 0.0;
        double c3 = 0.0;

        // Supported formats:
        //   time,x_rel,y_rel,z_rel,separation
        //   time,x_rel,y_rel
        // Header lines are skipped because parseDoubleCell fails.
        if (!parseDoubleCell(cells[0], c0) ||
            !parseDoubleCell(cells[1], c1) ||
            !parseDoubleCell(cells[2], c2)) {
            continue;
        }

        if (cells.size() >= 4 && parseDoubleCell(cells[3], c3)) {
            // four or more numeric columns: time,x,y,z,...
        } else {
            c3 = 0.0;
        }

        TrajectorySample sample;
        sample.time = c0;
        sample.rel = Vec3(c1, c2, c3);
        table.samples.push_back(sample);
        sawAnyData = true;
    }

    if (!sawAnyData || table.samples.size() < 2) {
        std::cerr << "WARNING: black-hole trajectory CSV '" << path
                  << "' did not contain at least two usable samples.\n";
        table.samples.clear();
        return table;
    }

    std::sort(table.samples.begin(), table.samples.end(),
              [](const TrajectorySample& a, const TrajectorySample& b) {
                  return a.time < b.time;
              });

    // Drop exact duplicate times, keeping the first.
    std::vector<TrajectorySample> unique;
    unique.reserve(table.samples.size());
    for (const auto& s : table.samples) {
        if (unique.empty() || std::abs(s.time - unique.back().time) > 1.0e-12) {
            unique.push_back(s);
        }
    }
    table.samples = std::move(unique);

    if (table.samples.size() < 2) {
        std::cerr << "WARNING: black-hole trajectory CSV '" << path
                  << "' collapsed to fewer than two unique-time samples.\n";
        table.samples.clear();
        return table;
    }

    table.valid = true;

    std::cerr << "Loaded black-hole trajectory table from '" << path
              << "' with " << table.samples.size() << " rows, time range ["
              << table.samples.front().time << ", " << table.samples.back().time << "].\n";

    return table;
}

inline const TrajectoryTable& getTrajectoryTable(const Params& params) {
    // One cache per translation unit. That is fine for this header-only prototype.
    static std::once_flag once;
    static TrajectoryTable table;

    std::call_once(once, [&]() {
        table = loadTrajectoryTable(params.blackHoleTrajectoryCsvPath);
    });

    return table;
}

inline Vec3 lerpVec(const Vec3& a, const Vec3& b, double u) {
    return (1.0 - u) * a + u * b;
}

inline bool relativePositionAtTime(double time, const Params& params, Vec3& relOut) {
    const TrajectoryTable& table = getTrajectoryTable(params);
    if (!table.valid || table.samples.empty()) {
        relOut = Vec3(0.0, 0.0, 0.0);
        return false;
    }

    if (time <= table.samples.front().time) {
        relOut = table.samples.front().rel;
        return true;
    }
    if (time >= table.samples.back().time) {
        relOut = table.samples.back().rel;
        return true;
    }

    auto it = std::lower_bound(
        table.samples.begin(),
        table.samples.end(),
        time,
        [](const TrajectorySample& s, double t) {
            return s.time < t;
        }
    );

    if (it == table.samples.begin()) {
        relOut = it->rel;
        return true;
    }

    const auto& hi = *it;
    const auto& lo = *(it - 1);
    const double denom = std::max(hi.time - lo.time, 1.0e-30);
    const double u = std::clamp((time - lo.time) / denom, 0.0, 1.0);
    relOut = lerpVec(lo.rel, hi.rel, u);
    return true;
}

inline State stateAtTime(double time, const Params& params) {
    State s;
    s.plusMass = params.blackHolePlusMass;
    s.minusMass = params.blackHoleMinusMass;

    s.plusRenderRadius = params.blackHoleRenderRadiusScale * s.plusMass;
    s.minusRenderRadius = params.blackHoleRenderRadiusScale * s.minusMass;

    s.plusCaptureRadius = params.blackHoleCaptureRadiusScale * s.plusMass;
    s.minusCaptureRadius = params.blackHoleCaptureRadiusScale * s.minusMass;

    if (!params.blackHolesEnabled) {
        s.valid = false;
        return s;
    }

    Vec3 rel;
    if (!relativePositionAtTime(time, params, rel)) {
        s.valid = false;
        return s;
    }

    const double totalMass = std::max(s.plusMass + s.minusMass, 1.0e-30);

    // reftrajectories_with_time.csv stores rel = x_plus - x_minus.
    // Keep the center of mass at the origin:
    //   m_plus*x_plus + m_minus*x_minus = 0.
    s.plusCenter = (s.minusMass / totalMass) * rel;
    s.minusCenter = -(s.plusMass / totalMass) * rel;

    s.valid = true;
    return s;
}

inline double squaredDistance(const Vec3& a, const Vec3& b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    const double dz = a.z - b.z;
    return dx*dx + dy*dy + dz*dz;
}

struct MPPotential {
    double U = 1.0;
    Vec3 gradU = Vec3(0.0, 0.0, 0.0);
};

inline void addMPSource(const Vec3& x,
                        const Vec3& center,
                        double mass,
                        double softening,
                        MPPotential& out)
{
    const Vec3 d = x - center;
    const double eps2 = std::max(0.0, softening) * std::max(0.0, softening);
    const double r2 = d.x*d.x + d.y*d.y + d.z*d.z + eps2;
    const double r = std::sqrt(std::max(r2, 1.0e-300));

    out.U += mass / r;

    const double invR3 = 1.0 / std::max(r2 * r, 1.0e-300);
    out.gradU = out.gradU + (-mass * invR3) * d;
}

inline MPPotential majumdarPapapetrouPotential(const Vec3& x,
                                               double frameTime,
                                               const Params& params)
{
    MPPotential out;

    if (!params.blackHolesEnabled || !params.metricUseMajumdarPapapetrou) {
        return out;
    }

    const State bh = stateAtTime(frameTime, params);
    if (!bh.valid) {
        return out;
    }

    const double massScale = params.metricMPMassScale;
    const double softening = params.metricMPSoftening;

    addMPSource(x, bh.plusCenter,  massScale * bh.plusMass,  softening, out);
    addMPSource(x, bh.minusCenter, massScale * bh.minusMass, softening, out);

    if (!std::isfinite(out.U) || out.U < 1.0e-12) {
        out.U = 1.0;
        out.gradU = Vec3(0.0, 0.0, 0.0);
    }

    return out;
}

} // namespace BlackHoleSystem
