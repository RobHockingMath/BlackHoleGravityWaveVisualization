#include "Params.h"

#include <cmath>
#include <iostream>

void printParams(const Params& p) {
    std::cerr << "Params:\n";
    std::cerr << "  image: " << p.width << " x " << p.height << " spp=" << p.samplesPerPixel << "\n";
    std::cerr << "  modeFile: " << p.modeFile << "\n";
    std::cerr << "  panorama: " << p.panoramaPath
              << " yaw=" << p.panoramaYawDegrees
              << " exposure=" << p.panoramaExposure << "\n";

    std::cerr << "  renderMode: " << p.renderMode << "\n";

    std::cerr << "  wave sphere: radius=" << p.waveVolumeRadius
              << " step=" << p.stepSize
              << " tmin=" << p.rayTMin
              << " tmax=";
    if (std::isinf(p.rayTMax)) std::cerr << "inf\n";
    else std::cerr << p.rayTMax << "\n";

    std::cerr << "  rInner=" << p.rInner
              << " innerWaveScaleRadius=" << p.innerWaveScaleRadius << "\n";

    std::cerr << "  retarded time: " << (p.useTortoiseRetardedTime ? "t-r*" : "t-r")
              << " tortoiseMass=" << p.tortoiseMass
              << " tortoiseRadiusFloor=" << p.tortoiseRadiusFloor
              << " tortoiseSafetyEps=" << p.tortoiseSafetyEps << "\n";

    std::cerr << "  extraction spheres:";
    if (p.use_extraction_sphere_100) std::cerr << " 100";
    if (p.use_extraction_sphere_115) std::cerr << " 115";
    if (p.use_extraction_sphere_136) std::cerr << " 136";
    if (p.use_extraction_sphere_167) std::cerr << " 167";
    if (p.use_extraction_sphere_214) std::cerr << " 214";
    if (p.use_extraction_sphere_300) std::cerr << " 300";
    if (p.use_extraction_sphere_500) std::cerr << " 500";
    std::cerr << "\n";

    std::cerr << "  volume: densityScale=" << p.densityScale
              << " ampCutoff=" << p.ampCutoff
              << " maxStepAlpha=" << p.maxStepAlpha << "\n";

    std::cerr << "  frame max normalization: " << (p.normalizeAmplitudePerFrame ? "on" : "off")
              << " sampleCount=" << p.frameAmplitudeSampleCount << "\n";

    std::cerr << "  axisMask=" << (p.axisMaskEnabled ? "on" : "off")
              << " inner=" << p.axisMaskInnerRadius
              << " outer=" << p.axisMaskOuterRadius << "\n";

    std::cerr << "  color: colorMode=" << p.colorMode
              << " saturation=" << p.colorSaturation
              << " waveBrightness=" << p.waveBrightness
              << " outputGamma=" << p.outputGamma << "\n";

    std::cerr << "  paraview peaks: n=" << p.paraviewPeaksNumPeaks
              << " firstPos=" << p.paraviewPeaksFirstPosition
              << " lastPos=" << p.paraviewPeaksLastPosition
              << " firstOpacity=" << p.paraviewPeaksFirstOpacity
              << " lastOpacity=" << p.paraviewPeaksLastOpacity
              << " sigma=" << p.paraviewPeaksSigma
              << " strength=" << p.paraviewPeaksStrength
              << " scalarOpacityUnitDistance=" << p.paraviewScalarOpacityUnitDistance << "\n";

    std::cerr << "  paraview scalar: radialMode=" << p.paraviewScalarRadialMode
              << " opacityRadialFalloff=" << (p.paraviewOpacityRadialFalloffEnabled ? "on" : "off")
              << " opacityReferenceRadius=" << p.paraviewOpacityReferenceRadius
              << " opacityFalloffPower=" << p.paraviewOpacityFalloffPower
              << " outerFadeWidth=" << p.paraviewOuterFadeWidth << "\n";

    std::cerr << "  paraview adaptive: maxStep=" << p.paraviewAdaptiveMaxStep
              << " safety=" << p.paraviewAdaptiveSafetyFactor
              << " fineBandSigmas=" << p.paraviewAdaptiveFineBandSigmas
              << " derivativeFloor=" << p.paraviewAdaptiveDerivativeFloor << "\n";

    std::cerr << "  gwpv peaks: n=" << p.gwpvPeaksNumPeaks
              << " firstPos=" << p.gwpvPeaksFirstPosition
              << " lastPos=" << p.gwpvPeaksLastPosition
              << " firstOpacity=" << p.gwpvPeaksFirstOpacity
              << " lastOpacity=" << p.gwpvPeaksLastOpacity
              << " strength=" << p.gwpvPeaksStrength
              << " scalarOpacityUnitDistance=" << p.gwpvScalarOpacityUnitDistance << "\n";

    std::cerr << "  gwpv opacity gain: " << (p.gwpvOpacityGainEnabled ? "on" : "off")
              << " csv=\"" << p.gwpvOpacityGainCsvPath << "\""
              << " multiplier=" << p.gwpvOpacityGainMultiplier << "\n";

    std::cerr << "  gwpv wavelength comp: " << (p.gwpvWavelengthCompEnabled ? "on" : "off")
              << " csv=\"" << p.gwpvWavelengthCsvPath << "\""
              << " csvBasePower=" << p.gwpvWavelengthCsvBasePower
              << " Rref=" << p.gwpvWavelengthReferenceRadius << "\n";

    std::cerr << "  gwpv wavelength shape: secondPlateau="
              << (p.gwpvWavelengthSecondPlateauEnabled ? "on" : "off")
              << " start=" << p.gwpvWavelengthSecondPlateauStartTime
              << " end=" << p.gwpvWavelengthSecondPlateauEndTime
              << " opacityPower=" << p.gwpvWavelengthOpacityPower
              << " opacityGainMin=" << p.gwpvWavelengthOpacityGainMin
              << " opacityGainMax=" << p.gwpvWavelengthOpacityGainMax
              << " colorPower=" << p.gwpvWavelengthColorPower
              << " colorGainMin=" << p.gwpvWavelengthColorGainMin
              << " colorGainMax=" << p.gwpvWavelengthColorGainMax
              << " peakWidth=" << (p.gwpvWavelengthPeakWidthEnabled ? "on" : "off")
              << " peakWidthPower=" << p.gwpvWavelengthPeakWidthPower
              << " peakWidthGainMin=" << p.gwpvWavelengthPeakWidthGainMin
              << " peakWidthGainMax=" << p.gwpvWavelengthPeakWidthGainMax << "\n";

    std::cerr << "  gwpv wavelength stepping: "
              << (p.gwpvWavelengthStepScalingEnabled ? "on" : "off")
              << " stepPower=" << p.gwpvWavelengthStepPower
              << " minStep=" << p.gwpvWavelengthMinStep
              << " maxSkipMultiplier=" << p.gwpvWavelengthMaxSkipMultiplier << "\n";

    std::cerr << "  gwpv modifiers: axisMask=" << (p.gwpvUseAxisMask ? "on" : "off")
              << " opacityRadialEnvelope=" << (p.gwpvUseOpacityRadialEnvelope ? "on" : "off") << "\n";

    std::cerr << "  gwpv adaptive: " << (p.gwpvAdaptiveEnabled ? "on" : "off")
              << " maxStep=" << p.paraviewAdaptiveMaxStep
              << " safety=" << p.paraviewAdaptiveSafetyFactor
              << " guardPeakDecays=" << p.gwpvAdaptiveGuardPeakDecays
              << " derivativeFloor=" << p.paraviewAdaptiveDerivativeFloor << "\n";


    std::cerr << "  metric lensing: " << (p.metricLensingEnabled ? "on" : "off")
              << " useRadiusScaledStrain=" << (p.metricUseRadiusScaledStrain ? "on" : "off")
              << " scale=" << p.metricPerturbationScale
              << " derivatives=" << (p.metricUseAnalyticDerivatives ? "analytic" : "finite-difference")
              << " derivativeEps=" << p.metricDerivativeEps << "\n";

    std::cerr << "  metric RK: h0=" << p.metricGeodesicInitialStep
              << " hMin=" << p.metricGeodesicMinStep
              << " hMaxInner=" << p.metricGeodesicMaxStepInner
              << " hMaxOuter=" << p.metricGeodesicMaxStepOuter
              << " hMaxRadii=(" << p.metricGeodesicMaxStepInnerRadius
              << "," << p.metricGeodesicMaxStepOuterRadius << ")"
              << " absTol=" << p.metricGeodesicAbsTol
              << " relTol=" << p.metricGeodesicRelTol
              << " colorStep=" << p.metricColorStep
              << " renormNull=" << (p.metricRenormalizeNullSpeed ? "on" : "off")
              << " maxAccepted=" << p.metricMaxAcceptedSteps << "\n";

    std::cerr << "  numerical metric: " << (p.numericalMetricEnabled ? "on" : "off")
              << " snapshot=\"" << p.numericalMetricSnapshotStem << "\""
              << " horizon=\"" << p.numericalMetricHorizonPath << "\""
              << " maxLayer=" << p.numericalMetricMaxLayer
              << " precomputedGamma=on"
              << " useDtFlag=" << (p.numericalMetricUseTimeDerivatives ? "on" : "off")
              << "(ignored)"
              << " horizonSafety=" << p.numericalMetricHorizonSafetyFactor << "\n";


    std::cerr << "  black holes: " << (p.blackHolesEnabled ? "on" : "off")
              << " traj=\"" << p.blackHoleTrajectoryCsvPath << "\""
              << " texture=\"" << p.blackHoleTexturePath << "\""
              << " masses=(" << p.blackHolePlusMass << "," << p.blackHoleMinusMass << ")"
              << " renderRadiusScale=" << p.blackHoleRenderRadiusScale
              << " captureRadiusScale=" << p.blackHoleCaptureRadiusScale << "\n";

    std::cerr << "  MP metric: " << (p.metricUseMajumdarPapapetrou ? "on" : "off")
              << " massScale=" << p.metricMPMassScale
              << " softening=" << p.metricMPSoftening << "\n";

    if (p.activeModes.empty()) {
        std::cerr << "  active harmonics: all\n";
    } else {
        std::cerr << "  active harmonics:";
        for (const auto& lm : p.activeModes) std::cerr << " (" << lm.l << "," << lm.m << ")";
        std::cerr << "\n";
    }
}
