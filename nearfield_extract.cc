#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>
#include <cctk_Interp.h>
#include <util_Table.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

extern "C" void NearFieldExtract_Output(CCTK_ARGUMENTS);

namespace {

constexpr int MAX_LAYERS = 16;
constexpr int RAW_FIELDS = 50;
constexpr int METRIC_FIELDS = 50;
constexpr int PSI_FIELDS = 10;
constexpr double PI = 3.141592653589793238462643383279502884;

const std::array<const char*, METRIC_FIELDS> METRIC_FIELD_NAMES = {{"g00", "g01", "g02", "g03", "g11", "g12", "g13", "g22", "g23", "g33", "dt_g00", "dt_g01", "dt_g02", "dt_g03", "dt_g11", "dt_g12", "dt_g13", "dt_g22", "dt_g23", "dt_g33", "dx_g00", "dx_g01", "dx_g02", "dx_g03", "dx_g11", "dx_g12", "dx_g13", "dx_g22", "dx_g23", "dx_g33", "dy_g00", "dy_g01", "dy_g02", "dy_g03", "dy_g11", "dy_g12", "dy_g13", "dy_g22", "dy_g23", "dy_g33", "dz_g00", "dz_g01", "dz_g02", "dz_g03", "dz_g11", "dz_g12", "dz_g13", "dz_g22", "dz_g23", "dz_g33"}};
const std::array<const char*, RAW_FIELDS> RAW_VAR_NAMES = {{"ADMBase::alp", "ADMBase::betax", "ADMBase::betay", "ADMBase::betaz", "ADMBase::gxx", "ADMBase::gxy", "ADMBase::gxz", "ADMBase::gyy", "ADMBase::gyz", "ADMBase::gzz", "ADMBase::dtalp", "ADMBase::dtbetax", "ADMBase::dtbetay", "ADMBase::dtbetaz", "ADMDerivatives::gxx_dt", "ADMDerivatives::gxy_dt", "ADMDerivatives::gxz_dt", "ADMDerivatives::gyy_dt", "ADMDerivatives::gyz_dt", "ADMDerivatives::gzz_dt", "ADMDerivatives::alp_dx", "ADMDerivatives::betax_dx", "ADMDerivatives::betay_dx", "ADMDerivatives::betaz_dx", "ADMDerivatives::gxx_dx", "ADMDerivatives::gxy_dx", "ADMDerivatives::gxz_dx", "ADMDerivatives::gyy_dx", "ADMDerivatives::gyz_dx", "ADMDerivatives::gzz_dx", "ADMDerivatives::alp_dy", "ADMDerivatives::betax_dy", "ADMDerivatives::betay_dy", "ADMDerivatives::betaz_dy", "ADMDerivatives::gxx_dy", "ADMDerivatives::gxy_dy", "ADMDerivatives::gxz_dy", "ADMDerivatives::gyy_dy", "ADMDerivatives::gyz_dy", "ADMDerivatives::gzz_dy", "ADMDerivatives::alp_dz", "ADMDerivatives::betax_dz", "ADMDerivatives::betay_dz", "ADMDerivatives::betaz_dz", "ADMDerivatives::gxx_dz", "ADMDerivatives::gxy_dz", "ADMDerivatives::gxz_dz", "ADMDerivatives::gyy_dz", "ADMDerivatives::gyz_dz", "ADMDerivatives::gzz_dz"}};
const std::array<const char*, PSI_FIELDS> PSI_FIELD_NAMES = {{"Psi0r", "Psi0i", "Psi1r", "Psi1i", "Psi2r", "Psi2i", "Psi3r", "Psi3i", "Psi4r", "Psi4i"}};
const std::array<const char*, PSI_FIELDS> PSI_VAR_NAMES = {{"WeylScal4::Psi0r", "WeylScal4::Psi0i", "WeylScal4::Psi1r", "WeylScal4::Psi1i", "WeylScal4::Psi2r", "WeylScal4::Psi2i", "WeylScal4::Psi3r", "WeylScal4::Psi3i", "WeylScal4::Psi4r", "WeylScal4::Psi4i"}};

struct Vec3 { double x, y, z; };
struct CenterInfo { Vec3 bh1, bh2; const char *bh1_source, *bh2_source, *center_mode; double separation; };
struct GridSpec { const char* family; int layer; Vec3 center; double half_width, requested_dx; int nx, ny, nz; double actual_dx; long long points; };
struct SphereSpec { const char* family; int layer; Vec3 center; double radius; long long points; };

struct DebugPoint { std::string label; Vec3 x; };

const std::array<std::pair<int,int>, 10> SYM_PAIRS_4D = {{
    {0,0}, {0,1}, {0,2}, {0,3},
    {1,1}, {1,2}, {1,3},
    {2,2}, {2,3},
    {3,3}
}};

int sym4_index(int a, int b) {
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
    return -1;
}

const char* gamma_pair_name(int pair) {
    static const char* names[10] = {"00","01","02","03","11","12","13","22","23","33"};
    return (pair >= 0 && pair < 10) ? names[pair] : "??";
}

struct FieldStats {
    double min = std::numeric_limits<double>::infinity();
    double max = -std::numeric_limits<double>::infinity();
    double sum = 0.0, sumsq = 0.0;
    long long finite_count = 0, nonfinite_count = 0;
    void add(double v) {
        if (std::isfinite(v)) { min = std::min(min, v); max = std::max(max, v); sum += v; sumsq += v*v; ++finite_count; }
        else { ++nonfinite_count; }
    }
    double mean() const { return finite_count ? sum / double(finite_count) : std::numeric_limits<double>::quiet_NaN(); }
    double rms() const { return finite_count ? std::sqrt(sumsq / double(finite_count)) : std::numeric_limits<double>::quiet_NaN(); }
};

Vec3 sub(Vec3 a, Vec3 b) { return Vec3{a.x-b.x, a.y-b.y, a.z-b.z}; }
double norm(Vec3 a) { return std::sqrt(a.x*a.x + a.y*a.y + a.z*a.z); }
bool finite_vec(Vec3 v) { return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z); }

// Experimental renderer test layout matching the N=6 outside-AH moving Carpet
// refinement hierarchy for the short performance/render test.
//
// The simulation has two CarpetRegrid2 centers.  The larger/"plus" stack is assigned
// to the current +x center, and the smaller/"minus" stack is assigned to the current
// -x center.  The radii follow
//
//     R_n - R_AH = (R_0 - R_AH) / 2^n
//
// with conservative AH design radii approximately R_AH(+x)=0.49 and
// R_AH(-x)=0.365, based on the t≈20M AH diagnostics.  Layer numbering is 0..6.  The renderer should use
// numericalMetricMaxLayer = 6 for this experiment.
double origin_default_half_width(int i) { const double v[1] = {50.0}; return (i>=0 && i<1) ? v[i] : 0.0; }
double origin_default_dx(int i) { const double v[1] = {1.0}; return (i>=0 && i<1) ? v[i] : 0.0; }
double bh_default_half_width(int i) { const double v[4] = {4.0, 2.0, 1.0, 0.5}; return (i>=0 && i<4) ? v[i] : 0.0; }
double bh_default_dx(int i) { const double v[4] = {0.1000, 0.0500, 0.0250, 0.0125}; return (i>=0 && i<4) ? v[i] : 0.0; }
double sim_plus_half_width(int i) { const double v[7] = {21.267692, 10.878846, 5.684423, 3.0872115, 1.78860575, 1.139302875, 0.8146514375}; return (i >= 0 && i < 7) ? v[i] : 0.0; }
double sim_minus_half_width(int i) { const double v[7] = {17.132308, 8.748654, 4.556827, 2.4609135, 1.41295675, 0.888978375, 0.6269891875}; return (i >= 0 && i < 7) ? v[i] : 0.0; }
double sim_refinement_dx(int i) {
    constexpr double h0 = 1.2237362637362637;
    return (i>=0 && i<7) ? h0 / double(1 << (i + 1)) : 0.0;
}
double sphere_default_radius(int i) { const double v[2] = {50.0, 100.0}; return (i>=0 && i<2) ? v[i] : 0.0; }

int grid_n(double half_width, double dx) {
    if (!std::isfinite(half_width) || !std::isfinite(dx) || half_width <= 0.0 || dx <= 0.0) return 0;
    return std::max(2, int(std::llround(2.0 * half_width / dx)) + 1);
}

GridSpec make_grid(const char* family, int layer, Vec3 center, double half_width, double dx) {
    GridSpec g{family, layer, center, half_width, dx, 0,0,0,0.0,0};
    g.nx = grid_n(half_width, dx); g.ny = g.nx; g.nz = g.nx;
    g.actual_dx = g.nx > 1 ? (2.0 * half_width / double(g.nx - 1)) : 0.0;
    g.points = static_cast<long long>(g.nx) * static_cast<long long>(g.ny) * static_cast<long long>(g.nz);
    return g;
}

long long chunk_begin(long long n, int rank, int nprocs) { return (n * static_cast<long long>(rank)) / static_cast<long long>(nprocs); }
long long chunk_end(long long n, int rank, int nprocs) { return (n * static_cast<long long>(rank + 1)) / static_cast<long long>(nprocs); }

bool nearfield_match_simulation_refinement_layout() {
    return true;
}

std::vector<GridSpec> make_grids(const CenterInfo& c) {
    DECLARE_CCTK_PARAMETERS;
    std::vector<GridSpec> grids;
    Vec3 origin{0,0,0};

    if (nearfield_match_simulation_refinement_layout()) {
        // Match the intended N=6 outside-AH CarpetRegrid2 moving-box hierarchy:
        //   +x / larger stack: seven positive radii, layers 0..6
        //   -x / smaller stack: seven positive radii, layers 0..6
        // Assign these by current x-order so the experiment remains correct even if
        // NearFieldExtract's BH1/BH2 labels are opposite to the +x/-x names.
        const bool bh1_is_plus = (c.bh1.x >= c.bh2.x);
        const Vec3 plus_center  = bh1_is_plus ? c.bh1 : c.bh2;
        const Vec3 minus_center = bh1_is_plus ? c.bh2 : c.bh1;
        const char* plus_family  = bh1_is_plus ? "BH1" : "BH2";
        const char* minus_family = bh1_is_plus ? "BH2" : "BH1";

        for (int i = 0; i < 7; ++i) {
            const int layer = i;
            const double dx = sim_refinement_dx(i);
            grids.push_back(make_grid(plus_family, layer, plus_center,
                                      sim_plus_half_width(i), dx));
            grids.push_back(make_grid(minus_family, layer, minus_center,
                                      sim_minus_half_width(i), dx));
        }
        return grids;
    }

    int no = std::max(0, std::min(MAX_LAYERS, int(nearfield_origin_grid_count)));
    for (int i=0; i<no; ++i) {
        double h = double(nearfield_origin_grid_half_width[i]); if (h <= 0.0) h = origin_default_half_width(i);
        double dx = double(nearfield_origin_grid_dx[i]); if (dx <= 0.0) dx = origin_default_dx(i);
        if (h > 0.0 && dx > 0.0) grids.push_back(make_grid("origin", i, origin, h, dx));
    }
    int nb = std::max(0, std::min(MAX_LAYERS, int(nearfield_bh_grid_layer_count)));
    for (int i=0; i<nb; ++i) {
        double h = double(nearfield_bh_grid_half_width[i]); if (h <= 0.0) h = bh_default_half_width(i);
        double dx = double(nearfield_bh_grid_dx[i]); if (dx <= 0.0) dx = bh_default_dx(i);
        if (h > 0.0 && dx > 0.0) { grids.push_back(make_grid("BH1", i, c.bh1, h, dx)); grids.push_back(make_grid("BH2", i, c.bh2, h, dx)); }
    }
    return grids;
}

std::vector<SphereSpec> make_spheres() {
    DECLARE_CCTK_PARAMETERS;
    std::vector<SphereSpec> spheres; Vec3 origin{0,0,0};
    int ns = std::max(0, std::min(MAX_LAYERS, int(nearfield_origin_sphere_count)));
    long long ntheta = std::max(2, int(nearfield_sphere_ntheta));
    long long nphi = std::max(4, int(nearfield_sphere_nphi));
    for (int i=0; i<ns; ++i) {
        double r = double(nearfield_origin_sphere_radius[i]); if (r <= 0.0) r = sphere_default_radius(i);
        if (r > 0.0) spheres.push_back(SphereSpec{"origin", i, origin, r, ntheta*nphi});
    }
    return spheres;
}

void fill_grid_coords_chunk(const GridSpec& g, long long global_begin, long long count, std::vector<CCTK_REAL>& xs, std::vector<CCTK_REAL>& ys, std::vector<CCTK_REAL>& zs) {
    xs.resize(static_cast<std::size_t>(count)); ys.resize(static_cast<std::size_t>(count)); zs.resize(static_cast<std::size_t>(count));
    const double x0 = g.center.x - g.half_width, y0 = g.center.y - g.half_width, z0 = g.center.z - g.half_width;
    for (long long q=0; q<count; ++q) {
        long long p = global_begin + q;
        int i = int(p % g.nx); long long tmp = p / g.nx; int j = int(tmp % g.ny); int k = int(tmp / g.ny);
        xs[static_cast<std::size_t>(q)] = CCTK_REAL(x0 + double(i) * g.actual_dx);
        ys[static_cast<std::size_t>(q)] = CCTK_REAL(y0 + double(j) * g.actual_dx);
        zs[static_cast<std::size_t>(q)] = CCTK_REAL(z0 + double(k) * g.actual_dx);
    }
}

void fill_sphere_coords_chunk(const SphereSpec& s, long long global_begin, long long count, std::vector<CCTK_REAL>& xs, std::vector<CCTK_REAL>& ys, std::vector<CCTK_REAL>& zs) {
    DECLARE_CCTK_PARAMETERS;
    int ntheta = std::max(2, int(nearfield_sphere_ntheta));
    int nphi = std::max(4, int(nearfield_sphere_nphi));
    xs.resize(static_cast<std::size_t>(count)); ys.resize(static_cast<std::size_t>(count)); zs.resize(static_cast<std::size_t>(count));
    for (long long q=0; q<count; ++q) {
        long long p = global_begin + q;
        int it = int(p / nphi);
        int ip = int(p % nphi);
        double theta = PI * (double(it) + 0.5) / double(ntheta);
        double phi = 2.0 * PI * double(ip) / double(nphi);
        double st = std::sin(theta), ct = std::cos(theta);
        xs[static_cast<std::size_t>(q)] = CCTK_REAL(s.center.x + s.radius * st * std::cos(phi));
        ys[static_cast<std::size_t>(q)] = CCTK_REAL(s.center.y + s.radius * st * std::sin(phi));
        zs[static_cast<std::size_t>(q)] = CCTK_REAL(s.center.z + s.radius * ct);
    }
}

std::string metric_snapshot_stem(int snapshot_index) {
    DECLARE_CCTK_PARAMETERS;
    std::ostringstream ss; ss << std::string(nearfield_grid_output_prefix) << "_" << std::setw(6) << std::setfill('0') << snapshot_index; return ss.str();
}
std::string sphere_snapshot_stem(int snapshot_index) {
    DECLARE_CCTK_PARAMETERS;
    std::ostringstream ss; ss << std::string(nearfield_sphere_output_prefix) << "_" << std::setw(6) << std::setfill('0') << snapshot_index; return ss.str();
}
std::string rank_suffix(int rank, int nprocs) {
    std::ostringstream ss; ss << ".rank" << std::setw(4) << std::setfill('0') << rank << "of" << std::setw(4) << std::setfill('0') << nprocs; return ss.str();
}

std::string output_path_for(const std::string& name) {
    DECLARE_CCTK_PARAMETERS;
    if (name.empty()) return std::string(nearfield_output_dir);
    if (!name.empty() && name[0] == '/') return name;
    return std::string(nearfield_output_dir) + "/" + name;
}

std::string csv_escape(const std::string& s) {
    bool needs = false;
    for (char c : s) if (c == ',' || c == '"' || c == '\n' || c == '\r') needs = true;
    if (!needs) return s;
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') out += "\"\"";
        else out += c;
    }
    out += "\"";
    return out;
}

std::vector<int> parse_debug_orders(const std::string& s) {
    std::string clean = s;
    for (char& c : clean) if (c == ',' || c == ';') c = ' ';
    std::istringstream in(clean);
    std::vector<int> orders;
    int v = 0;
    while (in >> v) if (v >= 0) orders.push_back(v);
    if (orders.empty()) orders.push_back(4);
    return orders;
}

std::vector<DebugPoint> builtin_debug_points() {
    return std::vector<DebugPoint>{
        {"step22_start_sane", Vec3{-5.89448,     3.40780,     0.000150299}},
        {"step22_stage2_bad", Vec3{-5.84389,     3.41649,     0.000153711}},
        {"step22_stage3_bad", Vec3{-5.84374,     3.41593,     0.000153666}},
        {"bad_L1_corner_A",   Vec3{-5.80861678,  3.38656154,  0.05}},
        {"bad_L1_corner_B",   Vec3{-5.80861678,  3.43656154,  0.05}},
        {"bad_L0_corner_A",   Vec3{-5.80861678,  3.38656154,  0.10}},
        {"bad_L0_corner_B",   Vec3{-5.80861678,  3.48656154,  0.10}}
    };
}

std::vector<DebugPoint> load_debug_points(int rank) {
    DECLARE_CCTK_PARAMETERS;
    std::vector<DebugPoint> points;
    const std::string configured = std::string(nearfield_debug_interp_points_file);
    std::vector<std::string> candidates;
    if (!configured.empty()) {
        candidates.push_back(configured);
        if (configured[0] != '/') candidates.push_back(std::string(nearfield_output_dir) + "/" + configured);
    }

    for (const auto& path : candidates) {
        std::ifstream in(path.c_str());
        if (!in) continue;
        std::string line;
        while (std::getline(in, line)) {
            std::string trimmed = line;
            std::size_t hash = trimmed.find('#');
            if (hash != std::string::npos) trimmed = trimmed.substr(0, hash);
            std::istringstream ss(trimmed);
            DebugPoint p;
            if (ss >> p.label >> p.x.x >> p.x.y >> p.x.z) points.push_back(p);
        }
        if (!points.empty()) {
            if (rank == 0) CCTK_VInfo(CCTK_THORNSTRING, "NearFieldExtract debug interpolation loaded %d points from %s", int(points.size()), path.c_str());
            return points;
        }
    }

    points = builtin_debug_points();
    if (rank == 0) CCTK_VInfo(CCTK_THORNSTRING,
               "NearFieldExtract debug interpolation could not load '%s'; using %d built-in bad-point probes",
               configured.c_str(), int(points.size()));
    return points;
}

void write_grid_corners(std::ofstream& out, const GridSpec& g) {
    const double h = g.half_width;
    const int s[8][3] = {{-1,-1,-1},{1,-1,-1},{-1,1,-1},{1,1,-1},{-1,-1,1},{1,-1,1},{-1,1,1},{1,1,1}};
    for (int q=0; q<8; ++q) out << "    corner " << q << " x " << g.center.x+s[q][0]*h << " y " << g.center.y+s[q][1]*h << " z " << g.center.z+s[q][2]*h << "\n";
}
void write_sphere_axis_samples(std::ofstream& out, const SphereSpec& s) {
    const char* label[6] = {"+x","-x","+y","-y","+z","-z"};
    const int d[6][3] = {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
    for (int q=0; q<6; ++q) out << "    sample " << label[q] << " x " << s.center.x+d[q][0]*s.radius << " y " << s.center.y+d[q][1]*s.radius << " z " << s.center.z+d[q][2]*s.radius << "\n";
}

bool var_indices(const std::array<const char*, RAW_FIELDS>& names, std::array<CCTK_INT, RAW_FIELDS>& vi) {
    for (int f=0; f<RAW_FIELDS; ++f) {
        vi[f] = CCTK_VarIndex(names[f]);
        if (vi[f] < 0) { CCTK_VWarn(1, __LINE__, __FILE__, CCTK_THORNSTRING, "Could not find grid function '%s'", names[f]); return false; }
    }
    return true;
}

bool psi_var_indices(std::array<CCTK_INT, PSI_FIELDS>& vi) {
    for (int f=0; f<PSI_FIELDS; ++f) {
        vi[f] = CCTK_VarIndex(PSI_VAR_NAMES[f]);
        if (vi[f] < 0) { CCTK_VWarn(1, __LINE__, __FILE__, CCTK_THORNSTRING, "Could not find grid function '%s'", PSI_VAR_NAMES[f]); return false; }
    }
    return true;
}

void sym3_from6(double out[3][3], double xx, double xy, double xz, double yy, double yz, double zz) {
    out[0][0]=xx; out[0][1]=xy; out[0][2]=xz;
    out[1][0]=xy; out[1][1]=yy; out[1][2]=yz;
    out[2][0]=xz; out[2][1]=yz; out[2][2]=zz;
}

void pack_metric10(const double gamma[3][3], const double beta[3], double alpha, double out[10]) {
    double g0i[3] = {0.0, 0.0, 0.0};
    double bb = 0.0;
    for (int i=0; i<3; ++i) {
        for (int j=0; j<3; ++j) {
            g0i[i] += gamma[i][j] * beta[j];
            bb += gamma[i][j] * beta[i] * beta[j];
        }
    }
    out[0] = -alpha*alpha + bb;
    out[1] = g0i[0]; out[2] = g0i[1]; out[3] = g0i[2];
    out[4] = gamma[0][0]; out[5] = gamma[0][1]; out[6] = gamma[0][2];
    out[7] = gamma[1][1]; out[8] = gamma[1][2]; out[9] = gamma[2][2];
}

void pack_metric_deriv10(const double gamma[3][3], const double dgamma[3][3], const double beta[3], const double dbeta[3], double alpha, double dalpha, double out[10]) {
    double dg00 = -2.0 * alpha * dalpha;
    double dg0i[3] = {0.0, 0.0, 0.0};
    for (int i=0; i<3; ++i) {
        for (int j=0; j<3; ++j) {
            dg00 += dgamma[i][j] * beta[i] * beta[j] + 2.0 * gamma[i][j] * beta[i] * dbeta[j];
            dg0i[i] += dgamma[i][j] * beta[j] + gamma[i][j] * dbeta[j];
        }
    }
    out[0] = dg00;
    out[1] = dg0i[0]; out[2] = dg0i[1]; out[3] = dg0i[2];
    out[4] = dgamma[0][0]; out[5] = dgamma[0][1]; out[6] = dgamma[0][2];
    out[7] = dgamma[1][1]; out[8] = dgamma[1][2]; out[9] = dgamma[2][2];
}

void compute_metric_fields_at_point(const std::vector<std::vector<CCTK_REAL>>& raw, std::size_t q, double out[METRIC_FIELDS]) {
    double alpha = double(raw[0][q]);
    double beta[3] = { double(raw[1][q]), double(raw[2][q]), double(raw[3][q]) };
    double gamma[3][3];
    sym3_from6(gamma, double(raw[4][q]), double(raw[5][q]), double(raw[6][q]), double(raw[7][q]), double(raw[8][q]), double(raw[9][q]));

    pack_metric10(gamma, beta, alpha, out);

    double dalpha_t = double(raw[10][q]);
    double dbeta_t[3] = { double(raw[11][q]), double(raw[12][q]), double(raw[13][q]) };
    double dgamma_t[3][3];
    sym3_from6(dgamma_t, double(raw[14][q]), double(raw[15][q]), double(raw[16][q]), double(raw[17][q]), double(raw[18][q]), double(raw[19][q]));
    pack_metric_deriv10(gamma, dgamma_t, beta, dbeta_t, alpha, dalpha_t, out + 10);

    for (int dir=0; dir<3; ++dir) {
        int base = 20 + 10*dir;
        double dalpha = double(raw[base+0][q]);
        double dbeta[3] = { double(raw[base+1][q]), double(raw[base+2][q]), double(raw[base+3][q]) };
        double dgamma[3][3];
        sym3_from6(dgamma, double(raw[base+4][q]), double(raw[base+5][q]), double(raw[base+6][q]), double(raw[base+7][q]), double(raw[base+8][q]), double(raw[base+9][q]));
        pack_metric_deriv10(gamma, dgamma, beta, dbeta, alpha, dalpha, out + 20 + 10*dir);
    }
}


bool invert4x4_debug(const double g[4][4], double inv[4][4]) {
    double a[4][8];
    for (int i=0; i<4; ++i) {
        for (int j=0; j<4; ++j) a[i][j] = g[i][j];
        for (int j=0; j<4; ++j) a[i][4+j] = (i == j) ? 1.0 : 0.0;
    }
    for (int col=0; col<4; ++col) {
        int pivot = col;
        double best = std::abs(a[col][col]);
        for (int r=col+1; r<4; ++r) {
            double v = std::abs(a[r][col]);
            if (v > best) { best = v; pivot = r; }
        }
        if (!std::isfinite(best) || best < 1.0e-30) return false;
        if (pivot != col) for (int j=0; j<8; ++j) std::swap(a[pivot][j], a[col][j]);
        const double piv = a[col][col];
        for (int j=0; j<8; ++j) a[col][j] /= piv;
        for (int r=0; r<4; ++r) {
            if (r == col) continue;
            const double f = a[r][col];
            if (f == 0.0) continue;
            for (int j=0; j<8; ++j) a[r][j] -= f * a[col][j];
        }
    }
    for (int i=0; i<4; ++i) {
        for (int j=0; j<4; ++j) {
            inv[i][j] = a[i][4+j];
            if (!std::isfinite(inv[i][j])) return false;
        }
    }
    return true;
}

struct GammaDebugSummary {
    bool ok = false;
    double gamma0_12 = std::numeric_limits<double>::quiet_NaN();
    double max_abs_gamma = 0.0;
    int max_mu = -1;
    int max_pair = -1;
    double max_abs_g = 0.0;
    double max_abs_ginv = 0.0;
};

GammaDebugSummary compute_gamma_debug_summary(const double fields[METRIC_FIELDS]) {
    GammaDebugSummary s;
    double g[4][4] = {};
    double dg[4][4][4] = {};

    for (int c=0; c<10; ++c) {
        const auto ij = SYM_PAIRS_4D[c];
        const int i = ij.first, j = ij.second;
        const double v = fields[c];
        g[i][j] = v;
        g[j][i] = v;
        if (std::isfinite(v)) s.max_abs_g = std::max(s.max_abs_g, std::abs(v));
    }
    for (int a=0; a<4; ++a) {
        for (int c=0; c<10; ++c) {
            const auto ij = SYM_PAIRS_4D[c];
            const int i = ij.first, j = ij.second;
            const double v = fields[10 + 10*a + c];
            dg[a][i][j] = v;
            dg[a][j][i] = v;
        }
    }

    double gi[4][4] = {};
    if (!invert4x4_debug(g, gi)) return s;
    s.ok = true;
    for (int i=0; i<4; ++i) for (int j=0; j<4; ++j) {
        if (std::isfinite(gi[i][j])) s.max_abs_ginv = std::max(s.max_abs_ginv, std::abs(gi[i][j]));
    }

    for (int mu=0; mu<4; ++mu) {
        for (int pair=0; pair<10; ++pair) {
            const int a = SYM_PAIRS_4D[pair].first;
            const int b = SYM_PAIRS_4D[pair].second;
            double sum = 0.0;
            for (int nu=0; nu<4; ++nu) {
                sum += gi[mu][nu] * (dg[a][b][nu] + dg[b][a][nu] - dg[nu][a][b]);
            }
            const double gamma = 0.5 * sum;
            if (mu == 0 && a == 1 && b == 2) s.gamma0_12 = gamma;
            if (std::isfinite(gamma) && std::abs(gamma) > s.max_abs_gamma) {
                s.max_abs_gamma = std::abs(gamma);
                s.max_mu = mu;
                s.max_pair = pair;
            }
        }
    }
    return s;
}

bool write_metric_global_meta(int snap, int iter, double time, int nprocs, const CenterInfo& centers, const std::vector<GridSpec>& grids) {
    DECLARE_CCTK_PARAMETERS;
    std::string stem = metric_snapshot_stem(snap);
    std::ofstream meta((std::string(nearfield_output_dir)+"/"+stem+".meta.txt").c_str());
    if (!meta) return false;
    long long total = 0; for (const auto& g: grids) total += g.points;
    meta << std::setprecision(17);
    meta << "format NearFieldExtract_metric_derivs_f32_ranked_v1\n";
    meta << "payload metric_4d_with_derivatives_v1\n";
    meta << "snapshot_index " << snap << "\niteration " << iter << "\ntime " << time << "\nnprocs " << nprocs << "\n";
    meta << "center_mode " << centers.center_mode << "\n";
    meta << "bh1_x " << centers.bh1.x << "\nbh1_y " << centers.bh1.y << "\nbh1_z " << centers.bh1.z << "\nbh1_source " << centers.bh1_source << "\n";
    meta << "bh2_x " << centers.bh2.x << "\nbh2_y " << centers.bh2.y << "\nbh2_z " << centers.bh2.z << "\nbh2_source " << centers.bh2_source << "\n";
    meta << "separation " << centers.separation << "\nnfields " << METRIC_FIELDS << "\nfield_order";
    for (auto* n: METRIC_FIELD_NAMES) meta << " " << n;
    meta << "\nngrids " << grids.size() << "\ntotal_grid_points " << total << "\nexpected_global_f32_bytes " << total*METRIC_FIELDS*4LL << "\n";
    for (std::size_t gi=0; gi<grids.size(); ++gi) {
        const GridSpec& g = grids[gi];
        meta << "grid " << gi << " family " << g.family << " layer " << g.layer << " center_x " << g.center.x << " center_y " << g.center.y << " center_z " << g.center.z
             << " half_width " << g.half_width << " requested_dx " << g.requested_dx << " nx " << g.nx << " ny " << g.ny << " nz " << g.nz << " actual_dx " << g.actual_dx << " points " << g.points << "\n";
        for (int r=0; r<nprocs; ++r) {
            long long b = chunk_begin(g.points, r, nprocs), e = chunk_end(g.points, r, nprocs);
            meta << "rank_partition rank " << r << " grid " << gi << " begin " << b << " end " << e << " count " << (e-b) << "\n";
        }
    }
    return true;
}

bool write_sphere_global_meta(int snap, int iter, double time, int nprocs, const std::vector<SphereSpec>& spheres) {
    DECLARE_CCTK_PARAMETERS;
    std::string stem = sphere_snapshot_stem(snap);
    std::ofstream meta((std::string(nearfield_output_dir)+"/"+stem+".meta.txt").c_str());
    if (!meta) return false;
    long long total = 0; for (const auto& s: spheres) total += s.points;
    int ntheta = std::max(2, int(nearfield_sphere_ntheta));
    int nphi = std::max(4, int(nearfield_sphere_nphi));
    meta << std::setprecision(17);
    meta << "format NearFieldExtract_psi_sphere_f32_ranked_v1\n";
    meta << "payload psi0_to_psi4_sphere_samples_v1\n";
    meta << "snapshot_index " << snap << "\niteration " << iter << "\ntime " << time << "\nnprocs " << nprocs << "\n";
    meta << "nfields " << PSI_FIELDS << "\nfield_order";
    for (auto* n: PSI_FIELD_NAMES) meta << " " << n;
    meta << "\nntheta " << ntheta << "\nnphi " << nphi << "\nnspheres " << spheres.size() << "\ntotal_sphere_points " << total << "\nexpected_global_f32_bytes " << total*PSI_FIELDS*4LL << "\n";
    for (std::size_t si=0; si<spheres.size(); ++si) {
        const SphereSpec& s = spheres[si];
        meta << "sphere " << si << " family " << s.family << " layer " << s.layer << " center_x " << s.center.x << " center_y " << s.center.y << " center_z " << s.center.z << " radius " << s.radius << " points " << s.points << "\n";
        for (int r=0; r<nprocs; ++r) {
            long long b = chunk_begin(s.points, r, nprocs), e = chunk_end(s.points, r, nprocs);
            meta << "rank_sphere_partition rank " << r << " sphere " << si << " begin " << b << " end " << e << " count " << (e-b) << "\n";
        }
    }
    return true;
}

bool interpolate_fields_with_order(const cGH* cctkGH, int npoints, const std::vector<CCTK_REAL>& xs, const std::vector<CCTK_REAL>& ys, const std::vector<CCTK_REAL>& zs, int nfields, const CCTK_INT* var_indices, int interp_order, std::vector<std::vector<CCTK_REAL>>& out) {
    DECLARE_CCTK_PARAMETERS;
    if (npoints <= 0) return true;
    int ih = CCTK_InterpHandle(nearfield_interpolator_name);
    if (ih < 0) { CCTK_VWarn(1, __LINE__, __FILE__, CCTK_THORNSTRING, "Could not get interpolation handle for '%s'", nearfield_interpolator_name); return false; }
    int ch = CCTK_CoordSystemHandle("cart3d");
    if (ch < 0) { CCTK_WARN(1, "Could not get Cactus coordinate system handle for cart3d"); return false; }
    std::vector<void*> op(nfields);
    std::vector<CCTK_INT> ot(nfields, CCTK_VARIABLE_REAL);
    for (int f=0; f<nfields; ++f) { out[f].resize(static_cast<std::size_t>(npoints)); op[f] = out[f].data(); }
    const void* cp[3] = { xs.data(), ys.data(), zs.data() };
    int table = Util_TableCreate(UTIL_TABLE_FLAGS_CASE_INSENSITIVE);
    Util_TableSetInt(table, int(interp_order), "order");
    int ierr = CCTK_InterpGridArrays(cctkGH, 3, ih, table, ch, npoints, CCTK_VARIABLE_REAL, cp, nfields, var_indices, nfields, ot.data(), op.data());
    Util_TableDestroy(table);
    if (ierr < 0) { CCTK_VWarn(1, __LINE__, __FILE__, CCTK_THORNSTRING, "CCTK_InterpGridArrays failed with order=%d ierr=%d", interp_order, ierr); return false; }
    return true;
}

bool interpolate_fields(const cGH* cctkGH, int npoints, const std::vector<CCTK_REAL>& xs, const std::vector<CCTK_REAL>& ys, const std::vector<CCTK_REAL>& zs, int nfields, const CCTK_INT* var_indices, std::vector<std::vector<CCTK_REAL>>& out) {
    DECLARE_CCTK_PARAMETERS;
    return interpolate_fields_with_order(cctkGH, npoints, xs, ys, zs, nfields, var_indices, int(nearfield_interpolation_order), out);
}

bool write_debug_interp_snapshot(const cGH* cctkGH, int iter, double time, int rank, int nprocs) {
    DECLARE_CCTK_PARAMETERS;
    std::array<CCTK_INT, RAW_FIELDS> vi;
    if (!var_indices(RAW_VAR_NAMES, vi)) return false;

    const std::vector<DebugPoint> points = load_debug_points(rank);
    if (points.empty()) return false;
    const std::vector<int> orders = parse_debug_orders(std::string(nearfield_debug_interp_orders));

    std::vector<CCTK_REAL> xs(points.size()), ys(points.size()), zs(points.size());
    for (std::size_t i=0; i<points.size(); ++i) {
        xs[i] = CCTK_REAL(points[i].x.x);
        ys[i] = CCTK_REAL(points[i].x.y);
        zs[i] = CCTK_REAL(points[i].x.z);
    }

    const std::string outpath = output_path_for(std::string(nearfield_debug_interp_output_file));
    const bool write_header = (rank == 0) && !std::ifstream(outpath.c_str()).good();
    std::ofstream out;
    if (rank == 0) {
        out.open(outpath.c_str(), std::ios::app);
        if (!out) return false;
        out << std::setprecision(17);
        if (write_header) {
            out << "iteration,time,rank,nprocs,point_index,point_label,x,y,z,interp_order,interp_ok";
            out << ",max_abs_raw,max_abs_raw_field,max_abs_metric,max_abs_metric_field";
            out << ",gamma_ok,gamma0_12,max_abs_gamma,max_abs_gamma_name,max_abs_g,max_abs_ginv";
            for (int f=0; f<RAW_FIELDS; ++f) out << ",raw_" << RAW_VAR_NAMES[f];
            for (int f=0; f<METRIC_FIELDS; ++f) out << ",metric_" << METRIC_FIELD_NAMES[f];
            out << "\n";
        }
    }

    for (int order : orders) {
        std::vector<std::vector<CCTK_REAL>> raw(RAW_FIELDS);
        bool ok = interpolate_fields_with_order(cctkGH, int(points.size()), xs, ys, zs, RAW_FIELDS, vi.data(), order, raw);

        if (rank != 0) continue;

        for (std::size_t q=0; q<points.size(); ++q) {
            double metric[METRIC_FIELDS] = {};
            if (ok) compute_metric_fields_at_point(raw, q, metric);

            double max_abs_raw = 0.0;
            int max_raw_field = -1;
            for (int f=0; ok && f<RAW_FIELDS; ++f) {
                const double v = double(raw[f][q]);
                if (std::isfinite(v) && std::abs(v) > max_abs_raw) { max_abs_raw = std::abs(v); max_raw_field = f; }
            }
            double max_abs_metric = 0.0;
            int max_metric_field = -1;
            for (int f=0; ok && f<METRIC_FIELDS; ++f) {
                const double v = metric[f];
                if (std::isfinite(v) && std::abs(v) > max_abs_metric) { max_abs_metric = std::abs(v); max_metric_field = f; }
            }
            const GammaDebugSummary gs = ok ? compute_gamma_debug_summary(metric) : GammaDebugSummary{};
            std::ostringstream gamma_name;
            if (gs.max_mu >= 0 && gs.max_pair >= 0) gamma_name << "Gamma" << gs.max_mu << "_" << gamma_pair_name(gs.max_pair);
            else gamma_name << "none";

            out << iter << "," << time << "," << rank << "," << nprocs
                << "," << q << "," << csv_escape(points[q].label)
                << "," << points[q].x.x << "," << points[q].x.y << "," << points[q].x.z
                << "," << order << "," << (ok ? 1 : 0)
                << "," << max_abs_raw << "," << (max_raw_field >= 0 ? RAW_VAR_NAMES[max_raw_field] : "none")
                << "," << max_abs_metric << "," << (max_metric_field >= 0 ? METRIC_FIELD_NAMES[max_metric_field] : "none")
                << "," << (gs.ok ? 1 : 0) << "," << gs.gamma0_12
                << "," << gs.max_abs_gamma << "," << gamma_name.str()
                << "," << gs.max_abs_g << "," << gs.max_abs_ginv;
            for (int f=0; f<RAW_FIELDS; ++f) out << "," << (ok ? double(raw[f][q]) : std::numeric_limits<double>::quiet_NaN());
            for (int f=0; f<METRIC_FIELDS; ++f) out << "," << (ok ? metric[f] : std::numeric_limits<double>::quiet_NaN());
            out << "\n";
        }
    }

    if (rank == 0 && nearfield_verbose) {
        CCTK_VInfo(CCTK_THORNSTRING, "NearFieldExtract debug interpolation wrote %d points x %d orders to %s",
                   int(points.size()), int(orders.size()), outpath.c_str());
    }
    return true;
}

bool write_metric_rank_snapshot(const cGH* cctkGH, int snap, int rank, int nprocs, const std::vector<GridSpec>& grids) {
    DECLARE_CCTK_PARAMETERS;
    std::array<CCTK_INT, RAW_FIELDS> vi;
    if (!var_indices(RAW_VAR_NAMES, vi)) return false;
    std::string stem = metric_snapshot_stem(snap), suff = rank_suffix(rank, nprocs);
    std::ofstream bout((std::string(nearfield_output_dir)+"/"+stem+suff+".f32").c_str(), std::ios::binary);
    std::ofstream stats((std::string(nearfield_output_dir)+"/"+stem+suff+".stats.txt").c_str());
    if (!bout || !stats) return false;
    stats << std::setprecision(17) << "# rank nprocs grid_index family layer begin end count field finite_count nonfinite_count min max mean rms\n";
    std::vector<CCTK_REAL> xs, ys, zs;
    for (std::size_t gi=0; gi<grids.size(); ++gi) {
        const GridSpec& g = grids[gi]; long long b = chunk_begin(g.points, rank, nprocs), e = chunk_end(g.points, rank, nprocs), count = e-b;
        fill_grid_coords_chunk(g, b, count, xs, ys, zs);
        std::vector<std::vector<CCTK_REAL>> raw(RAW_FIELDS);
        if (!interpolate_fields(cctkGH, int(count), xs, ys, zs, RAW_FIELDS, vi.data(), raw)) return false;

        std::vector<std::vector<float>> field_buffers(METRIC_FIELDS, std::vector<float>(static_cast<std::size_t>(count)));
        std::array<FieldStats, METRIC_FIELDS> st;
        double vals[METRIC_FIELDS];
        for (long long q=0; q<count; ++q) {
            compute_metric_fields_at_point(raw, static_cast<std::size_t>(q), vals);
            for (int f=0; f<METRIC_FIELDS; ++f) {
                float vf = float(vals[f]);
                field_buffers[f][static_cast<std::size_t>(q)] = vf;
                st[f].add(double(vf));
            }
        }
        for (int f=0; f<METRIC_FIELDS; ++f) {
            bout.write(reinterpret_cast<const char*>(field_buffers[f].data()), std::streamsize(field_buffers[f].size()*sizeof(float)));
            if (!bout) return false;
            stats << rank << " " << nprocs << " " << gi << " " << g.family << " " << g.layer << " " << b << " " << e << " " << count << " " << METRIC_FIELD_NAMES[f] << " " << st[f].finite_count << " " << st[f].nonfinite_count << " " << st[f].min << " " << st[f].max << " " << st[f].mean() << " " << st[f].rms() << "\n";
        }
    }
    return true;
}

bool write_psi_sphere_rank_snapshot(const cGH* cctkGH, int snap, int rank, int nprocs, const std::vector<SphereSpec>& spheres) {
    DECLARE_CCTK_PARAMETERS;
    std::array<CCTK_INT, PSI_FIELDS> vi;
    if (!psi_var_indices(vi)) return false;
    std::string stem = sphere_snapshot_stem(snap), suff = rank_suffix(rank, nprocs);
    std::ofstream bout((std::string(nearfield_output_dir)+"/"+stem+suff+".f32").c_str(), std::ios::binary);
    std::ofstream stats((std::string(nearfield_output_dir)+"/"+stem+suff+".stats.txt").c_str());
    if (!bout || !stats) return false;
    stats << std::setprecision(17) << "# rank nprocs sphere_index family layer radius begin end count field finite_count nonfinite_count min max mean rms\n";
    std::vector<CCTK_REAL> xs, ys, zs;
    for (std::size_t si=0; si<spheres.size(); ++si) {
        const SphereSpec& s = spheres[si]; long long b = chunk_begin(s.points, rank, nprocs), e = chunk_end(s.points, rank, nprocs), count = e-b;
        fill_sphere_coords_chunk(s, b, count, xs, ys, zs);
        std::vector<std::vector<CCTK_REAL>> out(PSI_FIELDS);
        if (!interpolate_fields(cctkGH, int(count), xs, ys, zs, PSI_FIELDS, vi.data(), out)) return false;
        std::vector<float> fbuf(static_cast<std::size_t>(count));
        for (int f=0; f<PSI_FIELDS; ++f) {
            FieldStats st;
            for (long long q=0; q<count; ++q) { float vf = float(out[f][static_cast<std::size_t>(q)]); fbuf[static_cast<std::size_t>(q)] = vf; st.add(double(vf)); }
            bout.write(reinterpret_cast<const char*>(fbuf.data()), std::streamsize(fbuf.size()*sizeof(float)));
            if (!bout) return false;
            stats << rank << " " << nprocs << " " << si << " " << s.family << " " << s.layer << " " << s.radius << " " << b << " " << e << " " << count << " " << PSI_FIELD_NAMES[f] << " " << st.finite_count << " " << st.nonfinite_count << " " << st.min << " " << st.max << " " << st.mean() << " " << st.rms() << "\n";
        }
    }
    return true;
}

} // namespace

extern "C" void NearFieldExtract_Output(CCTK_ARGUMENTS) {
    DECLARE_CCTK_ARGUMENTS; DECLARE_CCTK_PARAMETERS;
    static bool init = false; static double next_time = 0.0; static int snap = 0;
    if (!init) { next_time = double(nearfield_grid_output_start_time); snap = 0; init = true; }
    if (!nearfield_enable || nearfield_out_every <= 0 || (cctk_iteration % nearfield_out_every) != 0) return;
    int rank = CCTK_MyProc(cctkGH), nprocs = CCTK_nProcs(cctkGH);

    auto get_centroid = [&](int si, bool require_valid, Vec3& out)->bool {
        if (si < 0) return false;
        if (require_valid && sf_valid[si] <= 0) return false;
        Vec3 c{double(sf_centroid_x[si]), double(sf_centroid_y[si]), double(sf_centroid_z[si])};
        if (!finite_vec(c)) return false;
        out = c;
        return true;
    };

    CenterInfo centers{Vec3{double(nearfield_bh1_center_x),double(nearfield_bh1_center_y),double(nearfield_bh1_center_z)}, Vec3{double(nearfield_bh2_center_x),double(nearfield_bh2_center_y),double(nearfield_bh2_center_z)}, "fallback", "fallback", "fallback", 0.0};
    Vec3 c1, c2; bool used_ah=false;
    if (nearfield_use_ah_centers) {
        bool a1=get_centroid(int(nearfield_ah_bh1_surface_index), true, c1), a2=get_centroid(int(nearfield_ah_bh2_surface_index), true, c2);
        if (a1 && a2 && norm(sub(c1,c2)) > 1e-8) { centers.bh1=c1; centers.bh2=c2; centers.bh1_source="AH_SphericalSurface"; centers.bh2_source="AH_SphericalSurface"; centers.center_mode="AH"; used_ah=true; }
    }
    if (!used_ah) {
        bool p1=get_centroid(int(nearfield_puncture_bh1_surface_index), false, c1), p2=get_centroid(int(nearfield_puncture_bh2_surface_index), false, c2);
        if (p1 && p2 && norm(sub(c1,c2)) > 1e-8) { centers.bh1=c1; centers.bh2=c2; centers.bh1_source="Puncture_SphericalSurface"; centers.bh2_source="Puncture_SphericalSurface"; centers.center_mode="puncture"; }
    }
    centers.separation = norm(sub(centers.bh1, centers.bh2));

    std::vector<GridSpec> grids = make_grids(centers);
    std::vector<SphereSpec> spheres = make_spheres();
    long long total_grid_points = 0; for (auto& g: grids) total_grid_points += g.points;
    long long total_sphere_points = 0; for (auto& s: spheres) total_sphere_points += s.points;

    if (rank == 0) {
        std::ofstream st((std::string(nearfield_output_dir)+"/"+std::string(nearfield_test_filename)).c_str(), std::ios::app);
        if (st) st << std::setprecision(17) << "iteration " << cctk_iteration << " time " << double(cctk_time) << " center_mode " << centers.center_mode << " separation " << centers.separation << " ngrids " << grids.size() << " nspheres " << spheres.size() << " total_grid_points " << total_grid_points << " total_sphere_points " << total_sphere_points << " metric_fields " << METRIC_FIELDS << " psi_sphere_fields " << PSI_FIELDS << " nprocs " << nprocs << " next_output_time " << next_time << "\n";
        if (nearfield_write_geometry) {
            std::ofstream geo((std::string(nearfield_output_dir)+"/"+std::string(nearfield_geometry_filename)).c_str(), std::ios::app);
            if (geo) {
                geo << std::setprecision(17) << "iteration " << cctk_iteration << " time " << double(cctk_time) << " center_mode " << centers.center_mode << " separation " << centers.separation << " ngrids " << grids.size() << " nspheres " << spheres.size() << " nprocs " << nprocs << "\n";
                for (auto& g: grids) { geo << "  grid family " << g.family << " layer " << g.layer << " center_x " << g.center.x << " center_y " << g.center.y << " center_z " << g.center.z << " half_width " << g.half_width << " requested_dx " << g.requested_dx << " nx " << g.nx << " ny " << g.ny << " nz " << g.nz << " actual_dx " << g.actual_dx << " points " << g.points << "\n"; if (nearfield_write_axis_samples_only) write_grid_corners(geo, g); }
                for (auto& s: spheres) { geo << "  sphere family " << s.family << " layer " << s.layer << " center_x " << s.center.x << " center_y " << s.center.y << " center_z " << s.center.z << " radius " << s.radius << " points " << s.points << "\n"; if (nearfield_write_axis_samples_only) write_sphere_axis_samples(geo, s); }
            }
        }
    }

    double t = double(cctk_time);
    if (t >= next_time && t <= double(nearfield_grid_output_stop_time)) {
        if (nearfield_verbose) CCTK_VInfo(CCTK_THORNSTRING, "NearFieldExtract v7 rank %d/%d writing snapshot %d at iteration %d time %.17g", rank, nprocs, snap, cctk_iteration, t);
        if (nearfield_debug_interp_enable) {
            bool ok = write_debug_interp_snapshot(cctkGH, cctk_iteration, t, rank, nprocs);
            if (!ok) CCTK_VWarn(1, __LINE__, __FILE__, CCTK_THORNSTRING, "NearFieldExtract v7 rank %d failed to write interpolation debug output", rank);
        }
        if (nearfield_write_grid_fields) {
            if (rank == 0) write_metric_global_meta(snap, cctk_iteration, t, nprocs, centers, grids);
            bool ok = write_metric_rank_snapshot(cctkGH, snap, rank, nprocs, grids);
            if (!ok) CCTK_VWarn(1, __LINE__, __FILE__, CCTK_THORNSTRING, "NearFieldExtract v7 rank %d failed to write metric snapshot", rank);
        }
        if (nearfield_write_psi_spheres) {
            if (rank == 0) write_sphere_global_meta(snap, cctk_iteration, t, nprocs, spheres);
            bool ok = write_psi_sphere_rank_snapshot(cctkGH, snap, rank, nprocs, spheres);
            if (!ok) CCTK_VWarn(1, __LINE__, __FILE__, CCTK_THORNSTRING, "NearFieldExtract v7 rank %d failed to write Psi sphere snapshot", rank);
        }
        ++snap;
        next_time += double(nearfield_grid_output_dt);
    }
    if (rank == 0 && nearfield_verbose) CCTK_VInfo(CCTK_THORNSTRING, "NearFieldExtract v7 status at iteration %d time %.17g; ngrids=%d grid_points=%lld nspheres=%d sphere_points=%lld nprocs=%d", cctk_iteration, double(cctk_time), int(grids.size()), total_grid_points, int(spheres.size()), total_sphere_points, nprocs);
}
