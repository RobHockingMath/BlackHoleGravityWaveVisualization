// precompute_christoffels.cpp
//
// Standalone converter for NearFieldExtract v7 metric+derivative snapshots.
//
// Input:
//   nearfield_metric_t100_000099.meta.txt
//   nearfield_metric_t100_000099.rank0000of0020.f32
//   ...
//
// Output, default:
//   nearfield_christoffel_t100_000099.meta.txt
//   nearfield_christoffel_t100_000099.rank0000of0020.f32
//   ...
//
// Default output payload is 50 float32 fields:
//   10 metric components + 40 Christoffel components.
//
// Gamma component order:
//   for mu = 0..3:
//     Gamma{mu}_00 Gamma{mu}_01 Gamma{mu}_02 Gamma{mu}_03
//     Gamma{mu}_11 Gamma{mu}_12 Gamma{mu}_13
//     Gamma{mu}_22 Gamma{mu}_23 Gamma{mu}_33
//
// Build:
//   g++ -O3 -std=c++17 -fopenmp precompute_christoffels.cpp -o precompute_christoffels
//
// If OpenMP is unavailable:
//   g++ -O3 -std=c++17 precompute_christoffels.cpp -o precompute_christoffels
//
// Usage:
//   ./precompute_christoffels INPUT_STEM OUTPUT_STEM
//
// Example:
//   ./precompute_christoffels \
//     /mnt/nvme0/home/sasuke/EinsteinToolkit/Cactus/NearField_v7_t100_metric/nearfield_metric_t100_000099 \
//     /mnt/nvme0/home/sasuke/EinsteinToolkit/Cactus/NearField_v7_t100_metric/nearfield_christoffel_t100_000099
//
// Optional flags:
//   --gamma-only   write only 40 Christoffel fields, not the metric
//   --zero-dt      set ∂t g = 0 before computing Christoffels

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr int IN_FIELDS = 50;
constexpr int METRIC_FIELDS = 10;
constexpr int DG_DIRECTIONS = 4;
constexpr int GAMMA_FIELDS = 40;

const std::array<std::pair<int,int>, 10> SYM_PAIRS = {{
    {0,0}, {0,1}, {0,2}, {0,3},
    {1,1}, {1,2}, {1,3},
    {2,2}, {2,3},
    {3,3}
}};

int symIndex(int a, int b) {
    if (a > b) std::swap(a, b);
    if (a == 0 && b == 0) return 0;
    if (a == 0 && b == 1) return 1;
    if (a == 0 && b == 2) return 2;
    if (a == 0 && b == 3) return 3;
    if (a == 1 && b == 1) return 4;
    if (a == 1 && b == 2) return 5;
    if (a == 1 && b == 3) return 6;
    if (a == 2 && b == 2) return 7;
    if (a == 2 && b == 3) return 8;
    if (a == 3 && b == 3) return 9;
    throw std::runtime_error("bad symmetric index");
}

int gammaIndex(int mu, int a, int b) {
    return 10 * mu + symIndex(a, b);
}

std::vector<std::string> split(const std::string& s) {
    std::istringstream in(s);
    std::vector<std::string> out;
    std::string tok;
    while (in >> tok) out.push_back(tok);
    return out;
}

std::string rankFileName(const std::string& stem, int rank, int nprocs) {
    std::ostringstream os;
    os << stem << ".rank"
       << std::setw(4) << std::setfill('0') << rank
       << "of"
       << std::setw(4) << std::setfill('0') << nprocs
       << ".f32";
    return os.str();
}

std::string statsFileName(const std::string& stem, int rank, int nprocs) {
    std::ostringstream os;
    os << stem << ".rank"
       << std::setw(4) << std::setfill('0') << rank
       << "of"
       << std::setw(4) << std::setfill('0') << nprocs
       << ".stats.txt";
    return os.str();
}

template <class T>
T valueAfterKey(const std::vector<std::string>& toks, const std::string& key) {
    for (std::size_t i = 0; i + 1 < toks.size(); ++i) {
        if (toks[i] == key) {
            std::istringstream ss(toks[i + 1]);
            T v{};
            ss >> v;
            if (!ss) throw std::runtime_error("could not parse value after key " + key);
            return v;
        }
    }
    throw std::runtime_error("missing key " + key);
}

struct Partition {
    long long begin = 0;
    long long end = 0;
    long long count = 0;
};

struct GridMeta {
    int id = -1;
    std::string line;
    long long points = 0;
    std::vector<Partition> parts;
};

struct SnapshotMeta {
    int snapshot_index = -1;
    long long iteration = -1;
    double time = 0.0;
    int nprocs = 0;
    int nfields = 0;
    int ngrids = 0;
    long long total_grid_points = 0;
    std::string center_mode;
    std::string payload;
    std::vector<GridMeta> grids;
};

SnapshotMeta parseMeta(const std::string& metaPath) {
    std::ifstream in(metaPath);
    if (!in) throw std::runtime_error("could not open meta file: " + metaPath);

    SnapshotMeta m;
    std::string line;

    while (std::getline(in, line)) {
        auto toks = split(line);
        if (toks.empty()) continue;

        if (toks[0] == "payload" && toks.size() >= 2) {
            m.payload = toks[1];
        } else if (toks[0] == "snapshot_index" && toks.size() >= 2) {
            m.snapshot_index = std::stoi(toks[1]);
        } else if (toks[0] == "iteration" && toks.size() >= 2) {
            m.iteration = std::stoll(toks[1]);
        } else if (toks[0] == "time" && toks.size() >= 2) {
            m.time = std::stod(toks[1]);
        } else if (toks[0] == "nprocs" && toks.size() >= 2) {
            m.nprocs = std::stoi(toks[1]);
        } else if (toks[0] == "center_mode" && toks.size() >= 2) {
            m.center_mode = toks[1];
        } else if (toks[0] == "nfields" && toks.size() >= 2) {
            m.nfields = std::stoi(toks[1]);
        } else if (toks[0] == "ngrids" && toks.size() >= 2) {
            m.ngrids = std::stoi(toks[1]);
            m.grids.resize(static_cast<std::size_t>(m.ngrids));
        } else if (toks[0] == "total_grid_points" && toks.size() >= 2) {
            m.total_grid_points = std::stoll(toks[1]);
        } else if (toks[0] == "grid" && toks.size() >= 3) {
            const int gid = std::stoi(toks[1]);
            if (gid < 0) throw std::runtime_error("negative grid id in meta");
            if (gid >= static_cast<int>(m.grids.size())) m.grids.resize(gid + 1);

            GridMeta& g = m.grids[gid];
            g.id = gid;
            g.line = line;
            g.points = valueAfterKey<long long>(toks, "points");
        } else if (toks[0] == "rank_partition") {
            const int rank = valueAfterKey<int>(toks, "rank");
            const int gid = valueAfterKey<int>(toks, "grid");
            if (gid < 0) throw std::runtime_error("bad grid in rank_partition");
            if (gid >= static_cast<int>(m.grids.size())) m.grids.resize(gid + 1);
            if (rank < 0) throw std::runtime_error("bad rank in rank_partition");

            GridMeta& g = m.grids[gid];
            if (m.nprocs <= 0) {
                // Usually nprocs appears before partition lines. If not, grow dynamically;
                // we validate after parsing.
                if (rank >= static_cast<int>(g.parts.size())) g.parts.resize(rank + 1);
            } else if (g.parts.empty()) {
                g.parts.resize(m.nprocs);
            }

            if (rank >= static_cast<int>(g.parts.size())) g.parts.resize(rank + 1);
            Partition& p = g.parts[rank];
            p.begin = valueAfterKey<long long>(toks, "begin");
            p.end = valueAfterKey<long long>(toks, "end");
            p.count = valueAfterKey<long long>(toks, "count");
        }
    }

    if (m.snapshot_index < 0) throw std::runtime_error("meta missing snapshot_index");
    if (m.nprocs <= 0) throw std::runtime_error("meta missing/invalid nprocs");
    if (m.nfields != IN_FIELDS) {
        std::ostringstream os;
        os << "expected input nfields=" << IN_FIELDS << " but meta says " << m.nfields;
        throw std::runtime_error(os.str());
    }
    if (m.grids.empty()) throw std::runtime_error("meta has no grids");

    for (auto& g : m.grids) {
        if (g.id < 0) throw std::runtime_error("missing grid line in meta");
        if (g.points <= 0) throw std::runtime_error("bad grid points in meta");
        if (static_cast<int>(g.parts.size()) != m.nprocs) {
            std::ostringstream os;
            os << "grid " << g.id << " has " << g.parts.size()
               << " rank partitions; expected " << m.nprocs;
            throw std::runtime_error(os.str());
        }
        long long sum = 0;
        for (const auto& p : g.parts) sum += p.count;
        if (sum != g.points) {
            std::ostringstream os;
            os << "grid " << g.id << " partition counts sum to " << sum
               << ", expected " << g.points;
            throw std::runtime_error(os.str());
        }
    }

    long long total = 0;
    for (const auto& g : m.grids) total += g.points;
    if (m.total_grid_points > 0 && total != m.total_grid_points) {
        std::cerr << "WARNING: parsed total grid points " << total
                  << " differs from meta total_grid_points " << m.total_grid_points << "\n";
    }
    m.total_grid_points = total;
    m.ngrids = static_cast<int>(m.grids.size());

    return m;
}

bool invert4x4(const double g[4][4], double inv[4][4]) {
    double a[4][8];

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) a[i][j] = g[i][j];
        for (int j = 0; j < 4; ++j) a[i][4 + j] = (i == j) ? 1.0 : 0.0;
    }

    for (int col = 0; col < 4; ++col) {
        int pivot = col;
        double best = std::abs(a[col][col]);
        for (int r = col + 1; r < 4; ++r) {
            double v = std::abs(a[r][col]);
            if (v > best) {
                best = v;
                pivot = r;
            }
        }

        if (!std::isfinite(best) || best < 1.0e-30) return false;

        if (pivot != col) {
            for (int j = 0; j < 8; ++j) std::swap(a[pivot][j], a[col][j]);
        }

        const double piv = a[col][col];
        for (int j = 0; j < 8; ++j) a[col][j] /= piv;

        for (int r = 0; r < 4; ++r) {
            if (r == col) continue;
            const double f = a[r][col];
            if (f == 0.0) continue;
            for (int j = 0; j < 8; ++j) a[r][j] -= f * a[col][j];
        }
    }

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            inv[i][j] = a[i][4 + j];
            if (!std::isfinite(inv[i][j])) return false;
        }
    }

    return true;
}

void computeGammaForPoint(const float* inFields,
                          long long q,
                          long long count,
                          bool zeroDt,
                          float* gammaOut40,
                          bool& badInverse)
{
    double g[4][4] = {};
    double dg[4][4][4] = {}; // dg[a][mu][nu] = partial_a g_munu

    for (int c = 0; c < 10; ++c) {
        const auto [mu, nu] = SYM_PAIRS[c];
        const double gv = static_cast<double>(inFields[c * count + q]);
        g[mu][nu] = gv;
        g[nu][mu] = gv;
    }

    for (int a = 0; a < 4; ++a) {
        for (int c = 0; c < 10; ++c) {
            const auto [mu, nu] = SYM_PAIRS[c];
            const int inputIndex = 10 + a * 10 + c; // dt, dx, dy, dz
            double v = static_cast<double>(inFields[inputIndex * count + q]);
            if (zeroDt && a == 0) v = 0.0;
            dg[a][mu][nu] = v;
            dg[a][nu][mu] = v;
        }
    }

    double gInv[4][4] = {};
    if (!invert4x4(g, gInv)) {
        badInverse = true;
        for (int i = 0; i < 40; ++i) gammaOut40[i] = 0.0f;
        return;
    }

    badInverse = false;

    for (int mu = 0; mu < 4; ++mu) {
        for (int pair = 0; pair < 10; ++pair) {
            const int a = SYM_PAIRS[pair].first;
            const int b = SYM_PAIRS[pair].second;

            double sum = 0.0;
            for (int nu = 0; nu < 4; ++nu) {
                sum += gInv[mu][nu] * (dg[a][b][nu] + dg[b][a][nu] - dg[nu][a][b]);
            }

            const double gamma = 0.5 * sum;
            gammaOut40[10 * mu + pair] = std::isfinite(gamma) ? static_cast<float>(gamma) : 0.0f;
        }
    }
}

struct FieldStats {
    long long finite = 0;
    long long nonfinite = 0;
    double minv = std::numeric_limits<double>::infinity();
    double maxv = -std::numeric_limits<double>::infinity();
    long double sumsq = 0.0;

    void add(float x) {
        if (std::isfinite(x)) {
            ++finite;
            minv = std::min(minv, static_cast<double>(x));
            maxv = std::max(maxv, static_cast<double>(x));
            sumsq += static_cast<long double>(x) * static_cast<long double>(x);
        } else {
            ++nonfinite;
        }
    }

    double rms() const {
        if (finite <= 0) return std::numeric_limits<double>::quiet_NaN();
        return std::sqrt(static_cast<double>(sumsq / static_cast<long double>(finite)));
    }
};

std::vector<std::string> outputFieldNames(bool gammaOnly) {
    std::vector<std::string> names;

    if (!gammaOnly) {
        const char* gnames[10] = {"g00","g01","g02","g03","g11","g12","g13","g22","g23","g33"};
        for (auto s : gnames) names.push_back(s);
    }

    const char* pairNames[10] = {"00","01","02","03","11","12","13","22","23","33"};
    for (int mu = 0; mu < 4; ++mu) {
        for (int p = 0; p < 10; ++p) {
            std::ostringstream os;
            os << "Gamma" << mu << "_" << pairNames[p];
            names.push_back(os.str());
        }
    }

    return names;
}

void writeOutputMeta(const SnapshotMeta& m,
                     const std::string& outStem,
                     bool gammaOnly,
                     bool zeroDt)
{
    const auto names = outputFieldNames(gammaOnly);
    const int nOut = static_cast<int>(names.size());
    const long long bytes = m.total_grid_points * static_cast<long long>(nOut) * 4LL;

    std::ofstream out(outStem + ".meta.txt");
    if (!out) throw std::runtime_error("could not write output meta: " + outStem + ".meta.txt");

    out << "format NearFieldExtract_christoffel_f32_ranked_v1\n";
    out << "payload " << (gammaOnly ? "christoffel_4d_v1" : "metric_4d_plus_christoffel_v1") << "\n";
    out << "source_payload " << m.payload << "\n";
    out << "source_zero_dt " << (zeroDt ? "yes" : "no") << "\n";
    out << "snapshot_index " << m.snapshot_index << "\n";
    out << "iteration " << m.iteration << "\n";
    out << std::setprecision(17) << "time " << m.time << "\n";
    out << "nprocs " << m.nprocs << "\n";
    if (!m.center_mode.empty()) out << "center_mode " << m.center_mode << "\n";
    out << "nfields " << nOut << "\n";
    out << "field_order";
    for (const auto& name : names) out << " " << name;
    out << "\n";
    out << "ngrids " << m.ngrids << "\n";
    out << "total_grid_points " << m.total_grid_points << "\n";
    out << "expected_global_f32_bytes " << bytes << "\n";

    for (const auto& g : m.grids) {
        out << g.line << "\n";
        for (int rank = 0; rank < m.nprocs; ++rank) {
            const Partition& p = g.parts[rank];
            out << "rank_partition rank " << rank
                << " grid " << g.id
                << " begin " << p.begin
                << " end " << p.end
                << " count " << p.count << "\n";
        }
    }
}

void readExact(std::ifstream& in, char* dst, std::size_t bytes, const std::string& what) {
    in.read(dst, static_cast<std::streamsize>(bytes));
    if (static_cast<std::size_t>(in.gcount()) != bytes) {
        throw std::runtime_error("short read while reading " + what);
    }
}

void writeExact(std::ofstream& out, const char* src, std::size_t bytes, const std::string& what) {
    out.write(src, static_cast<std::streamsize>(bytes));
    if (!out) throw std::runtime_error("write failed while writing " + what);
}

void processRank(const SnapshotMeta& meta,
                 const std::string& inStem,
                 const std::string& outStem,
                 int rank,
                 bool gammaOnly,
                 bool zeroDt)
{
    const int nOut = gammaOnly ? GAMMA_FIELDS : (METRIC_FIELDS + GAMMA_FIELDS);

    const std::string inPath = rankFileName(inStem, rank, meta.nprocs);
    const std::string outPath = rankFileName(outStem, rank, meta.nprocs);
    const std::string statsPath = statsFileName(outStem, rank, meta.nprocs);

    std::ifstream in(inPath, std::ios::binary);
    if (!in) throw std::runtime_error("could not open input rank file: " + inPath);

    std::ofstream out(outPath, std::ios::binary);
    if (!out) throw std::runtime_error("could not open output rank file: " + outPath);

    std::ofstream stats(statsPath);
    if (!stats) throw std::runtime_error("could not open output stats file: " + statsPath);

    stats << "# stats for " << outPath << "\n";
    stats << "# fields: ";
    auto names = outputFieldNames(gammaOnly);
    for (const auto& name : names) stats << name << " ";
    stats << "\n";

    long long totalBadInverses = 0;
    long long totalPoints = 0;

    for (const auto& grid : meta.grids) {
        const long long countLL = grid.parts[rank].count;
        if (countLL < 0) throw std::runtime_error("negative rank count");
        const std::size_t count = static_cast<std::size_t>(countLL);
        totalPoints += countLL;

        std::vector<float> inBuf(static_cast<std::size_t>(IN_FIELDS) * count);
        std::vector<float> outBuf(static_cast<std::size_t>(nOut) * count);

        const std::size_t inBytes = inBuf.size() * sizeof(float);
        readExact(in, reinterpret_cast<char*>(inBuf.data()), inBytes, "grid input buffer");

        long long badLocal = 0;

        #pragma omp parallel for reduction(+:badLocal) schedule(static)
        for (long long qLL = 0; qLL < static_cast<long long>(count); ++qLL) {
            const std::size_t q = static_cast<std::size_t>(qLL);

            if (!gammaOnly) {
                for (int c = 0; c < METRIC_FIELDS; ++c) {
                    outBuf[static_cast<std::size_t>(c) * count + q] =
                        inBuf[static_cast<std::size_t>(c) * count + q];
                }
            }

            float gamma[40];
            bool bad = false;
            computeGammaForPoint(inBuf.data(), static_cast<long long>(q), static_cast<long long>(count),
                                 zeroDt, gamma, bad);
            if (bad) ++badLocal;

            const int gammaOffset = gammaOnly ? 0 : METRIC_FIELDS;
            for (int c = 0; c < GAMMA_FIELDS; ++c) {
                outBuf[static_cast<std::size_t>(gammaOffset + c) * count + q] = gamma[c];
            }
        }

        totalBadInverses += badLocal;

        const std::size_t outBytes = outBuf.size() * sizeof(float);
        writeExact(out, reinterpret_cast<const char*>(outBuf.data()), outBytes, "grid output buffer");

        stats << "grid " << grid.id
              << " points " << count
              << " bad_inverses " << badLocal << "\n";

        // Stats are serial and deliberately simple; this is not the hot path.
        for (int f = 0; f < nOut; ++f) {
            FieldStats st;
            const float* field = outBuf.data() + static_cast<std::size_t>(f) * count;
            for (std::size_t q = 0; q < count; ++q) st.add(field[q]);
            stats << "  " << names[f]
                  << " finite=" << st.finite
                  << " nonfinite=" << st.nonfinite
                  << " min=" << st.minv
                  << " max=" << st.maxv
                  << " rms=" << st.rms() << "\n";
        }
    }

    // Check for unexpected trailing bytes.
    in.peek();
    if (!in.eof()) {
        std::cerr << "WARNING: input rank file may have trailing data: " << inPath << "\n";
    }

    stats << "rank " << rank
          << " total_points " << totalPoints
          << " total_bad_inverses " << totalBadInverses << "\n";

    std::cerr << "rank " << rank
              << ": wrote " << outPath
              << " bad_inverses=" << totalBadInverses << "\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            std::cerr
                << "Usage:\n"
                << "  " << argv[0] << " INPUT_STEM OUTPUT_STEM [--gamma-only] [--zero-dt]\n";
            return 2;
        }

        const std::string inStem = argv[1];
        const std::string outStem = argv[2];

        bool gammaOnly = false;
        bool zeroDt = false;

        for (int i = 3; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--gamma-only") gammaOnly = true;
            else if (arg == "--zero-dt") zeroDt = true;
            else throw std::runtime_error("unknown argument: " + arg);
        }

        const auto t0 = std::chrono::steady_clock::now();

        const std::string metaPath = inStem + ".meta.txt";
        SnapshotMeta meta = parseMeta(metaPath);

        std::cerr << "Input stem:  " << inStem << "\n";
        std::cerr << "Output stem: " << outStem << "\n";
        std::cerr << "snapshot=" << meta.snapshot_index
                  << " time=" << std::setprecision(17) << meta.time
                  << " nprocs=" << meta.nprocs
                  << " ngrids=" << meta.ngrids
                  << " points=" << meta.total_grid_points
                  << " inputFields=" << meta.nfields
                  << " outputFields=" << (gammaOnly ? GAMMA_FIELDS : METRIC_FIELDS + GAMMA_FIELDS)
                  << " zeroDt=" << (zeroDt ? "yes" : "no")
                  << "\n";

#ifdef _OPENMP
        std::cerr << "OpenMP enabled, max threads=" << omp_get_max_threads() << "\n";
#else
        std::cerr << "OpenMP not enabled\n";
#endif

        writeOutputMeta(meta, outStem, gammaOnly, zeroDt);

        for (int rank = 0; rank < meta.nprocs; ++rank) {
            const auto tr0 = std::chrono::steady_clock::now();
            processRank(meta, inStem, outStem, rank, gammaOnly, zeroDt);
            const auto tr1 = std::chrono::steady_clock::now();
            const double sec = std::chrono::duration<double>(tr1 - tr0).count();
            std::cerr << "rank " << rank << " elapsed " << sec << " s\n";
        }

        const auto t1 = std::chrono::steady_clock::now();
        const double totalSec = std::chrono::duration<double>(t1 - t0).count();

        const int nOut = gammaOnly ? GAMMA_FIELDS : METRIC_FIELDS + GAMMA_FIELDS;
        const double outGB = static_cast<double>(meta.total_grid_points)
                           * static_cast<double>(nOut)
                           * 4.0 / 1.0e9;

        std::cerr << "Done.\n";
        std::cerr << "Output expected size: " << outGB << " GB\n";
        std::cerr << "Total elapsed: " << totalSec << " s\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
