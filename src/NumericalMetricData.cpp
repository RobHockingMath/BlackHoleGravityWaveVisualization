#include "NumericalMetricData.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {

std::vector<std::string> splitWords(const std::string& line) {
    std::istringstream ss(line);
    std::vector<std::string> out;
    std::string w;
    while (ss >> w) out.push_back(w);
    return out;
}

std::string requireValue(const std::vector<std::string>& tok, const std::string& key) {
    for (std::size_t i = 0; i + 1 < tok.size(); ++i) {
        if (tok[i] == key) return tok[i + 1];
    }
    throw std::runtime_error("Missing key '" + key + "' in metadata line");
}

int toInt(const std::string& s) {
    return std::stoi(s);
}

long long toLL(const std::string& s) {
    return std::stoll(s);
}

double toDouble(const std::string& s) {
    return std::stod(s);
}

int familyIdFromName(const std::string& family) {
    if (family == "BH1") return 1;
    if (family == "BH2") return 2;
    return 0;
}

std::string rankFileName(const std::string& stem, int rank, int nprocs) {
    std::ostringstream ss;
    ss << stem
       << ".rank" << std::setw(4) << std::setfill('0') << rank
       << "of" << std::setw(4) << std::setfill('0') << nprocs
       << ".f32";
    return ss.str();
}

void readExactly(std::ifstream& in, char* dst, std::size_t bytes, const std::string& what) {
    in.read(dst, static_cast<std::streamsize>(bytes));
    if (static_cast<std::size_t>(in.gcount()) != bytes) {
        throw std::runtime_error("Unexpected EOF while reading " + what);
    }
}

} // namespace

NumericalMetricSnapshot loadNumericalMetricSnapshot(const std::string& snapshotStem) {
    NumericalMetricSnapshot snap;
    snap.stem = snapshotStem;

    const std::string metaPath = snapshotStem + ".meta.txt";
    std::ifstream meta(metaPath);
    if (!meta) {
        throw std::runtime_error("Could not open numerical metric metadata: " + metaPath);
    }

    std::string line;
    while (std::getline(meta, line)) {
        const std::vector<std::string> tok = splitWords(line);
        if (tok.empty()) continue;

        if (tok[0] == "snapshot_index" && tok.size() >= 2) {
            snap.snapshotIndex = toInt(tok[1]);
        } else if (tok[0] == "iteration" && tok.size() >= 2) {
            snap.iteration = toInt(tok[1]);
        } else if (tok[0] == "time" && tok.size() >= 2) {
            snap.time = toDouble(tok[1]);
        } else if (tok[0] == "nprocs" && tok.size() >= 2) {
            snap.nprocs = toInt(tok[1]);
        } else if (tok[0] == "nfields" && tok.size() >= 2) {
            snap.nfields = toInt(tok[1]);
        } else if (tok[0] == "field_order") {
            snap.fieldOrder.assign(tok.begin() + 1, tok.end());
        } else if (tok[0] == "ngrids") {
            // informational only; grid lines are authoritative
        } else if (tok[0] == "total_grid_points" && tok.size() >= 2) {
            snap.totalGridPoints = toLL(tok[1]);
        } else if (tok[0] == "expected_global_f32_bytes" && tok.size() >= 2) {
            snap.expectedGlobalF32Bytes = toLL(tok[1]);
        } else if (tok[0] == "grid") {
            if (tok.size() < 2) throw std::runtime_error("Malformed grid metadata line");
            const int gridIndex = toInt(tok[1]);
            if (gridIndex < 0) throw std::runtime_error("Negative grid index in metadata");
            if (static_cast<int>(snap.grids.size()) <= gridIndex) {
                snap.grids.resize(static_cast<std::size_t>(gridIndex + 1));
            }

            NumericalMetricGridMeta g;
            g.family = requireValue(tok, "family");
            g.familyId = familyIdFromName(g.family);
            g.layer = toInt(requireValue(tok, "layer"));
            g.center = Vec3(toDouble(requireValue(tok, "center_x")),
                            toDouble(requireValue(tok, "center_y")),
                            toDouble(requireValue(tok, "center_z")));
            g.halfWidth = toDouble(requireValue(tok, "half_width"));
            g.requestedDx = toDouble(requireValue(tok, "requested_dx"));
            g.nx = toInt(requireValue(tok, "nx"));
            g.ny = toInt(requireValue(tok, "ny"));
            g.nz = toInt(requireValue(tok, "nz"));
            g.actualDx = toDouble(requireValue(tok, "actual_dx"));
            g.points = toLL(requireValue(tok, "points"));

            snap.grids[static_cast<std::size_t>(gridIndex)] = std::move(g);
        } else if (tok[0] == "rank_partition") {
            const int rank = toInt(requireValue(tok, "rank"));
            const int grid = toInt(requireValue(tok, "grid"));
            const long long begin = toLL(requireValue(tok, "begin"));
            const long long end = toLL(requireValue(tok, "end"));

            if (grid < 0 || static_cast<int>(snap.grids.size()) <= grid) {
                throw std::runtime_error("rank_partition references missing grid");
            }
            if (rank < 0) throw std::runtime_error("Negative rank in rank_partition");
            auto& g = snap.grids[static_cast<std::size_t>(grid)];
            if (static_cast<int>(g.begin.size()) <= rank) {
                g.begin.resize(static_cast<std::size_t>(rank + 1), 0);
                g.end.resize(static_cast<std::size_t>(rank + 1), 0);
            }
            g.begin[static_cast<std::size_t>(rank)] = begin;
            g.end[static_cast<std::size_t>(rank)] = end;
        }
    }

    if (snap.snapshotIndex < 0 || snap.iteration < 0 || snap.nprocs <= 0 || snap.nfields <= 0) {
        throw std::runtime_error("Incomplete numerical metric metadata in " + metaPath);
    }
    if (snap.fieldOrder.size() != static_cast<std::size_t>(snap.nfields)) {
        throw std::runtime_error("field_order count does not match nfields in " + metaPath);
    }
    if (snap.grids.empty()) {
        throw std::runtime_error("No grids found in " + metaPath);
    }

    long long totalPoints = 0;
    long long dataOffset = 0;
    for (auto& g : snap.grids) {
        if (g.points <= 0 || g.nx <= 0 || g.ny <= 0 || g.nz <= 0) {
            throw std::runtime_error("Invalid grid dimensions in " + metaPath);
        }
        if (static_cast<int>(g.begin.size()) != snap.nprocs || static_cast<int>(g.end.size()) != snap.nprocs) {
            throw std::runtime_error("Missing rank_partition entries in " + metaPath);
        }
        g.pointOffset = totalPoints;
        g.dataOffsetFloats = dataOffset;
        totalPoints += g.points;
        dataOffset += g.points * static_cast<long long>(snap.nfields);
    }

    if (snap.totalGridPoints == 0) snap.totalGridPoints = totalPoints;
    if (snap.totalGridPoints != totalPoints) {
        throw std::runtime_error("total_grid_points mismatch in " + metaPath);
    }

    const long long totalFloatsLL = totalPoints * static_cast<long long>(snap.nfields);
    if (totalFloatsLL <= 0 || totalFloatsLL > static_cast<long long>(std::numeric_limits<std::size_t>::max() / sizeof(float))) {
        throw std::runtime_error("Numerical metric snapshot is too large for this platform");
    }

    snap.data.assign(static_cast<std::size_t>(totalFloatsLL), 0.0f);

    std::vector<float> tmp;
    for (int rank = 0; rank < snap.nprocs; ++rank) {
        const std::string path = rankFileName(snapshotStem, rank, snap.nprocs);
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            throw std::runtime_error("Could not open numerical metric rank file: " + path);
        }

        for (std::size_t gi = 0; gi < snap.grids.size(); ++gi) {
            const auto& g = snap.grids[gi];
            const long long begin = g.begin[static_cast<std::size_t>(rank)];
            const long long end = g.end[static_cast<std::size_t>(rank)];
            const long long count = end - begin;
            if (count < 0 || begin < 0 || end > g.points) {
                throw std::runtime_error("Bad rank chunk in " + metaPath);
            }

            tmp.resize(static_cast<std::size_t>(count));
            for (int f = 0; f < snap.nfields; ++f) {
                if (count > 0) {
                    readExactly(in, reinterpret_cast<char*>(tmp.data()),
                                static_cast<std::size_t>(count) * sizeof(float), path);
                    const long long dstBase = g.dataOffsetFloats + static_cast<long long>(f) * g.points + begin;
                    std::copy(tmp.begin(), tmp.end(), snap.data.begin() + static_cast<std::ptrdiff_t>(dstBase));
                }
            }
        }

        // A strict EOF check can be painful if future versions append metadata;
        // leave extra bytes alone but report obvious short reads via readExactly.
    }

    std::cerr << "Loaded numerical metric snapshot '" << snapshotStem << "'"
              << " index=" << snap.snapshotIndex
              << " time=" << snap.time
              << " grids=" << snap.grids.size()
              << " points=" << snap.totalGridPoints
              << " fields=" << snap.nfields
              << " bytes=" << (static_cast<double>(snap.data.size()) * sizeof(float) / 1.0e9)
              << " GB\n";

    return snap;
}

NumericalHorizonSnapshot loadNumericalHorizonForSnapshot(const std::string& horizonPath,
                                                         int snapshotIndex) {
    std::ifstream in(horizonPath);
    if (!in) {
        throw std::runtime_error("Could not open horizon companion file: " + horizonPath);
    }

    std::string line;
    while (std::getline(in, line)) {
        const std::vector<std::string> tok = splitWords(line);
        if (tok.empty() || tok[0].empty() || tok[0][0] == '#') continue;
        if (tok.size() < 18) continue;

        const int idx = toInt(tok[0]);
        if (idx != snapshotIndex) continue;

        NumericalHorizonSnapshot h;
        h.snapshotIndex = idx;
        h.metricIteration = toInt(tok[1]);
        h.metricTime = toDouble(tok[2]);

        h.ah1Valid = (toInt(tok[3]) != 0);
        h.ah1Center = Vec3(toDouble(tok[4]), toDouble(tok[5]), toDouble(tok[6]));
        h.ah1RMin = toDouble(tok[7]);
        h.ah1RMax = toDouble(tok[8]);
        h.ah1RMean = toDouble(tok[9]);

        h.ah2Valid = (toInt(tok[10]) != 0);
        h.ah2Center = Vec3(toDouble(tok[11]), toDouble(tok[12]), toDouble(tok[13]));
        h.ah2RMin = toDouble(tok[14]);
        h.ah2RMax = toDouble(tok[15]);
        h.ah2RMean = toDouble(tok[16]);

        h.commonValid = (toInt(tok[17]) != 0);

        std::cerr << "Loaded horizon companion row for snapshot " << snapshotIndex
                  << ": AH1 rmax=" << h.ah1RMax
                  << " AH2 rmax=" << h.ah2RMax << "\n";
        return h;
    }

    throw std::runtime_error("No horizon companion row for snapshot_index " +
                             std::to_string(snapshotIndex) + " in " + horizonPath);
}

NumericalMetricBundle loadNumericalMetricBundle(const std::string& snapshotStem,
                                                const std::string& horizonPath) {
    NumericalMetricBundle b;
    b.metric = loadNumericalMetricSnapshot(snapshotStem);
    if (!horizonPath.empty()) {
        b.horizon = loadNumericalHorizonForSnapshot(horizonPath, b.metric.snapshotIndex);
        b.horizonValid = true;
    }
    return b;
}
