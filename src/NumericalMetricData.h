#pragma once

#include "Vec3.h"

#include <cstddef>
#include <string>
#include <vector>

struct NumericalMetricGridMeta {
    std::string family;
    int familyId = 0; // 1=BH1, 2=BH2, 0=other
    int layer = 0;
    Vec3 center;
    double halfWidth = 0.0;
    double requestedDx = 0.0;
    double actualDx = 0.0;
    int nx = 0;
    int ny = 0;
    int nz = 0;
    long long points = 0;

    // Rank chunks, indexed by MPI rank.
    std::vector<long long> begin;
    std::vector<long long> end;

    // Offsets used by the CUDA uploader.
    long long dataOffsetFloats = 0; // offset into grid-major, field-major snapshot data
    long long pointOffset = 0;      // offset into all-grid point-major arrays
};

struct NumericalMetricSnapshot {
    std::string stem;
    int snapshotIndex = -1;
    int iteration = -1;
    double time = 0.0;
    int nprocs = 0;
    int nfields = 0;
    long long totalGridPoints = 0;
    long long expectedGlobalF32Bytes = 0;
    std::vector<std::string> fieldOrder;
    std::vector<NumericalMetricGridMeta> grids;

    // Layout after loading:
    //   for grid in metadata order:
    //     for field in fieldOrder:
    //       all grid points, global linear order ix + nx*(iy + ny*iz)
    std::vector<float> data;
};

struct NumericalHorizonSnapshot {
    int snapshotIndex = -1;
    int metricIteration = -1;
    double metricTime = 0.0;

    bool ah1Valid = false;
    Vec3 ah1Center;
    double ah1RMin = 0.0;
    double ah1RMax = 0.0;
    double ah1RMean = 0.0;

    bool ah2Valid = false;
    Vec3 ah2Center;
    double ah2RMin = 0.0;
    double ah2RMax = 0.0;
    double ah2RMean = 0.0;

    bool commonValid = false;
};

struct NumericalMetricBundle {
    NumericalMetricSnapshot metric;
    NumericalHorizonSnapshot horizon;
    bool horizonValid = false;
};

NumericalMetricSnapshot loadNumericalMetricSnapshot(const std::string& snapshotStem);
NumericalHorizonSnapshot loadNumericalHorizonForSnapshot(const std::string& horizonPath,
                                                         int snapshotIndex);
NumericalMetricBundle loadNumericalMetricBundle(const std::string& snapshotStem,
                                                const std::string& horizonPath);
