// emotion_model.h - Generated automatically by emotion_training.py
// Header-only int8 emotion estimator. Matches the Python quantizer (zero-point 0 + ReLU).
#pragma once
#include <math.h>
#include <stdint.h>

#define EMOTION_NUM_FEATURES 18
#define EMOTION_NUM_CLASSES 4
#define EMOTION_WINDOW_EPOCHS 180
#define EMOTION_EPOCH_SEC 5

enum EmotionStatus : uint8_t {
    EMOTION_OK = 0,
    EMOTION_LOW_CONFIDENCE = 1,
    EMOTION_HIGH_MOTION = 2,
    EMOTION_NO_DATA = 3,
};

static const uint8_t EMOTION_NONE = 255;
static const char* const EMOTION_NAMES[EMOTION_NUM_CLASSES] = {"calm", "tense", "upbeat", "subdued"};

struct EmotionResult {
    uint8_t status;
    uint8_t emotion;
    uint8_t confidence_pct;
    uint8_t intensity_pct;
};

// ---------- Scaler ----------
static const float kEmotionMean[18] = {
    1.191586018e+00f, -2.869473994e-01f, 3.565438464e-02f, 3.961068392e+00f, 2.836739779e+00f, -3.239322901e-01f, 4.256844223e-01f, 3.891646147e+00f, 2.141262531e+00f, -2.226742059e-01f, 4.883952439e-02f, 5.183603764e+00f, 1.063718915e+00f, -6.664302200e-02f, 6.187747046e-02f, 5.163547516e+00f, 2.469219863e-01f, 6.931471825e-01f
};
static const float kEmotionStd[18] = {
    3.314231634e-01f, 1.630297422e+00f, 2.566689998e-02f, 4.079641402e-01f, 5.080509782e-01f, 2.145584106e+00f, 9.891676158e-02f, 3.329486251e-01f, 4.733018875e-01f, 1.733144999e+00f, 2.381089330e-02f, 1.200083256e+00f, 1.119166166e-01f, 9.653975964e-01f, 1.205722615e-02f, 1.186577797e+00f, 1.350605786e-01f, 1.000000000e+00f
};
static const double kEmotionInputScale = 0.03937007859349251;
static const double kEmotionLogitScale = 0.05308373518816129;
static const double kEmotionLevelScale = 0.008925645370183028;
static const double kEmotionTemperature = 1.0;
static const uint8_t kEmotionMinConfidence = 61;

// ---------- Quantized layers (weights, bias, mult, shift, zeros) ----------

static const int8_t kHeartWeights[24] = {
    -44, -87, 2, 32, -112, 0, -51, 14, 51, 127, 5, 16, -73, -14, 22, -54, 2, -95, 62, -58, 55, -49, 60, 51
};
static const int32_t kHeartBias[6] = {
    -880, 56, 1372, -385, -414, 152
};
static const int32_t kHeartMult = 1382634568;
static const uint8_t kHeartShift = 37;
static const int32_t kHeartInZero = 0;
static const int32_t kHeartOutZero = 0;
static const int32_t kHeartOutMin = 0;


static const int8_t kMotionWeights[24] = {
    36, 70, 2, 7, 33, 102, -79, -1, 20, 3, 53, -57, -75, 100, 68, 5, -77, -127, 55, 1, -124, 55, 48, 28
};
static const int32_t kMotionBias[6] = {
    -1692, 2232, -1667, 194, 128, 1043
};
static const int32_t kMotionMult = 1123685551;
static const uint8_t kMotionShift = 37;
static const int32_t kMotionInZero = 0;
static const int32_t kMotionOutZero = 0;
static const int32_t kMotionOutMin = 0;


static const int8_t kVoiceWeights[80] = {
    15, 39, 49, -15, 8, 35, -32, 2, 15, 49, 30, -8, 69, -3, 4, 53, 13, 4, 27, -42, 62, -1, -99, -5, -36, -42, 47, 0, 5, 37, -2, -29, -73, -51, -55, -28, -16, -3, 49, 32, 8, -68, -87, -2, -14, -82, 14, -7, -7, 20, -36, 116, 20, -20, 20, -24, -19, -4, -2, 7, 29, -81, 59, 0, 6, 33, -29, -13, -13, 1, -29, -27, -127, 24, 34, -14, 29, -27, 50, -25
};
static const int32_t kVoiceBias[8] = {
    790, 1750, 545, -37, -2976, 1, 2069, 303
};
static const int32_t kVoiceMult = 1461775416;
static const uint8_t kVoiceShift = 37;
static const int32_t kVoiceInZero = 0;
static const int32_t kVoiceOutZero = 0;
static const int32_t kVoiceOutMin = 0;


static const int8_t kDeep1Weights[320] = {
    27, -14, 12, -4, -6, -3, 27, 45, -7, 15, 44, -29, -79, -16, 19, -4, -27, -75, -29, -15, 9, 19, 24, -23, 23, -8, -7, -1, -24, 4, 13, -12, 17, -30, -18, 17, -74, 55, -74, 8, 3, -26, 73, 17, -5, -46, -10, -2, -23, 6, 15, -35, 6, 25, -92, -25, -23, 14, 1, -59, 43, 47, -28, 34, 1, 16, 16, 26, 10, 1, 34, -9, 28, 42, 16, 15, 52, -20, 15, 18, 25, -1, -24, 15, 24, 18, 32, -2, 13, -36, 68, -18, 2, 11, 2, -8, 71, 35, -21, -19, -19, 10, -31, 3, 9, -14, -6, 53, 14, 21, -11, -10, -1, 51, 21, -13, -39, -44, 27, 13, 47, 31, -58, 11, 31, 27, -41, -20, -16, -53, 54, -7, 5, 31, -31, -45, 1, -44, -50, -66, -38, -51, 5, -23, -28, -20, 36, 50, -13, 48, -2, 36, 54, 31, -26, -11, 4, -32, 12, -54, 0, -82, 60, -69, -41, -16, 9, 21, -30, 25, 5, -41, -3, -12, -65, -3, -6, -6, 8, -41, -51, -40, 41, 2, -27, -62, 9, 33, -14, 26, -7, -6, 2, 36, 38, -9, -51, -19, 51, 13, -24, 13, -15, -9, -11, 4, -26, -18, -9, 12, -12, 9, 21, -3, -24, -6, 4, -11, -27, 6, 21, -4, 57, 29, 13, -19, -9, -13, -24, 4, 33, -13, -10, -16, 27, 27, 61, 65, -46, 26, 22, 7, 5, -14, -15, -27, 18, 20, -21, -5, 21, 14, 23, 48, 21, -16, 19, -4, 8, 0, 36, 34, 1, 3, 12, 32, -20, 14, 7, -21, 23, 11, -33, 8, 68, 14, -30, -24, 34, 34, -62, -31, -7, -61, -8, -3, 51, 18, 4, 5, -3, 6, 64, 24, -25, -127, -23, 19, 13, -55, 3, -9, -4, -2, 16, -9, 20, 42, -21, 3, -4, 8, 15, 59, 23, 10, 20, -31, -20, -4
};
static const int32_t kDeep1Bias[16] = {
    2069, 850, 1025, -507, -2178, 1688, -1192, 1350, 818, 1449, -741, -922, 1404, 1078, 1307, -764
};
static const int32_t kDeep1Mult = 1292872846;
static const uint8_t kDeep1Shift = 37;
static const int32_t kDeep1InZero = 0;
static const int32_t kDeep1OutZero = 0;
static const int32_t kDeep1OutMin = 0;


static const int8_t kDeep2Weights[192] = {
    0, 17, 49, -13, -5, -7, -20, 44, 40, 31, -23, 21, 31, -44, 14, 5, 21, -127, 7, 30, 33, 20, 46, 21, -24, -20, 8, -46, 22, -6, 34, 26, 5, -7, 1, -32, -57, 35, -31, 21, 26, 41, 4, -15, 23, 9, 15, 10, -3, -10, 31, 7, 40, -15, 65, 1, 17, -9, -9, 33, 16, 4, 6, 8, 20, 10, 8, -27, -65, 21, -34, 8, -19, 12, -13, -20, -5, 25, -38, -22, 23, -22, -20, 2, 6, 13, 53, -18, -5, 15, -15, 9, 13, 31, -38, 0, -14, 8, -10, 3, 0, -16, -20, -7, 11, -5, -5, -15, -15, -3, -6, 8, 52, -5, 5, -10, -42, 19, -46, 44, 30, 34, 14, -27, -4, 7, 34, -2, 25, 91, -19, -11, -43, -1, -53, -16, -46, 34, 8, -18, 5, 32, -39, -21, 41, -74, -36, 43, -38, 29, -14, 15, -33, 15, -10, -12, 31, 37, -9, 14, 12, -20, -3, 11, -21, 2, -28, -18, 3, -19, -12, 8, -8, -20, 13, -25, 22, -60, -4, 16, 48, 9, 27, 2, -5, -14, 11, 16, 4, 23, 1, 6
};
static const int32_t kDeep2Bias[12] = {
    -105, -379, 1214, 885, 1481, 1110, -108, 1043, 844, 328, 81, 360
};
static const int32_t kDeep2Mult = 1942178406;
static const uint8_t kDeep2Shift = 38;
static const int32_t kDeep2InZero = 0;
static const int32_t kDeep2OutZero = 0;
static const int32_t kDeep2OutMin = 0;


static const int8_t kDeep3Weights[96] = {
    -12, -101, 44, 29, 72, 22, 22, 40, 94, -53, 12, -17, 9, -4, -21, -29, -24, 4, 38, -15, -33, -36, 16, 17, -34, 27, 41, -80, 49, -15, -16, 54, 61, 67, 31, -20, 80, -17, 67, -3, 0, -25, 6, 78, -46, -28, 21, -34, -19, 127, -5, 23, -23, 23, 25, 31, -103, 30, -37, 49, 31, 40, 18, -3, -39, -26, 24, 39, 57, -18, -25, 5, 53, -46, -48, 51, -43, -14, -23, -53, 19, -7, -11, 40, -95, 38, -5, 44, 18, 79, -25, -33, 69, 78, -33, 59
};
static const int32_t kDeep3Bias[8] = {
    490, -606, 140, 258, -257, -541, 260, 729
};
static const int32_t kDeep3Mult = 1481545280;
static const uint8_t kDeep3Shift = 38;
static const int32_t kDeep3InZero = 0;
static const int32_t kDeep3OutZero = 0;
static const int32_t kDeep3OutMin = 0;


static const int8_t kEmotionWeights[32] = {
    92, 8, 66, -100, -45, -80, -127, 23, 69, 33, -54, 70, -78, 9, -29, -53, -98, 40, 82, 13, 28, 32, -39, -98, -46, 33, -125, -32, 92, 8, 75, 107
};
static const int32_t kEmotionBias[4] = {
    867, 388, 134, -1282
};
static const int32_t kEmotionMult = 1988950370;
static const uint8_t kEmotionShift = 38;
static const int32_t kEmotionInZero = 0;
static const int32_t kEmotionOutZero = 0;
static const int32_t kEmotionOutMin = -127;


static const int8_t kIntensityWeights[8] = {
    94, -74, -49, 77, 122, -127, 100, 24
};
static const int32_t kIntensityBias[1] = {
    -798
};
static const int32_t kIntensityMult = 1974225851;
static const uint8_t kIntensityShift = 37;
static const int32_t kIntensityInZero = 0;
static const int32_t kIntensityOutZero = 0;
static const int32_t kIntensityOutMin = -127;


// ---------- Feature-extraction constants (same as Python) ----------
static const uint16_t kEmotionMinEpochs[4] = {90, 90, 6, 6};
static const double kEmotionMeanFloor[4] = {1.0, 10.0, 10.0, 1.0};
static const double kEmotionBaselinePrior[4] = {70.0, 30.0, 160.0, 30.0};
static const double kEmotionPeakSigma = 0.5;
static const double kEmotionActivityGateMg = 150.0;
static const double kEmotionBaselineAlphaMin = 0.01;

// ---------- Integer dense layer (matches Python Layer.__call__) ----------
static inline bool emotion_present(double x) { return x - x == 0.0; }

static inline void emotion_dense(const int8_t* input, uint8_t n_in, int32_t in_zero,
                                 const int8_t* weights, const int32_t* bias, int32_t mult,
                                 uint8_t shift, uint8_t n_out, int32_t out_zero, int32_t out_min,
                                 int8_t* output) {
    for (uint8_t k = 0; k < n_out; ++k) {
        int32_t sum = bias[k];
        for (uint8_t i = 0; i < n_in; ++i) {
            sum += (int32_t)(*weights++) * ((int32_t)input[i] - in_zero);
        }
        int64_t value = (int64_t)sum * mult;
        value = (value + ((int64_t)1 << (shift - 1))) >> shift;
        value += out_zero;
        // out_min implements ReLU when out_min == out_zero (usually 0)
        if (value < out_min) value = out_min;
        if (value > 127) value = 127;
        output[k] = (int8_t)value;
    }
}

static inline int8_t emotion_quantize(double value, uint8_t index) {
    if (!emotion_present(value)) return 0;
    const double compressed = copysign(log1p(fabs(value)), value);
    const double z = (compressed - kEmotionMean[index]) / kEmotionStd[index];
    const double q = floor(z / kEmotionInputScale + 0.5);
    if (q <= -127.0) return -127;
    if (q >= 127.0) return 127;
    return (int8_t)q;
}

static inline void emotion_forward(const int8_t input[EMOTION_NUM_FEATURES],
                                   int8_t logits[EMOTION_NUM_CLASSES], int8_t* intensity) {
    int8_t encoded[20];
    // Encoder branches (ReLU via out_min == 0)
    emotion_dense(input, 4, kHeartInZero, kHeartWeights, kHeartBias, kHeartMult,
                  kHeartShift, 6, kHeartOutZero, kHeartOutMin, encoded);
    emotion_dense(input + 4, 4, kMotionInZero, kMotionWeights, kMotionBias, kMotionMult,
                  kMotionShift, 6, kMotionOutZero, kMotionOutMin, encoded + 6);
    emotion_dense(input + 8, 10, kVoiceInZero, kVoiceWeights, kVoiceBias, kVoiceMult,
                  kVoiceShift, 8, kVoiceOutZero, kVoiceOutMin, encoded + 12);

    int8_t deep1[16];
    emotion_dense(encoded, 20, kDeep1InZero, kDeep1Weights, kDeep1Bias, kDeep1Mult,
                  kDeep1Shift, 16, kDeep1OutZero, kDeep1OutMin, deep1);

    int8_t deep2[12];
    emotion_dense(deep1, 16, kDeep2InZero, kDeep2Weights, kDeep2Bias, kDeep2Mult,
                  kDeep2Shift, 12, kDeep2OutZero, kDeep2OutMin, deep2);

    int8_t deep3[8];
    emotion_dense(deep2, 12, kDeep3InZero, kDeep3Weights, kDeep3Bias, kDeep3Mult,
                  kDeep3Shift, 8, kDeep3OutZero, kDeep3OutMin, deep3);

    // Heads (no ReLU, out_min = -127)
    emotion_dense(deep3, 8, kEmotionInZero, kEmotionWeights, kEmotionBias, kEmotionMult,
                  kEmotionShift, 4, kEmotionOutZero, kEmotionOutMin, logits);
    emotion_dense(deep3, 8, kIntensityInZero, kIntensityWeights, kIntensityBias, kIntensityMult,
                  kIntensityShift, 1, kIntensityOutZero, kIntensityOutMin, intensity);
}

static inline EmotionResult emotion_decide(const int8_t logits[EMOTION_NUM_CLASSES], int8_t level) {
    EmotionResult result;
    result.emotion = EMOTION_NONE;
    result.intensity_pct = 0;
    uint8_t top = 0;
    for (uint8_t j = 1; j < EMOTION_NUM_CLASSES; ++j) {
        if (logits[j] > logits[top]) top = j;
    }
    double total = 0.0;
    for (uint8_t j = 0; j < EMOTION_NUM_CLASSES; ++j) {
        const double gap = (double)((int16_t)logits[j] - (int16_t)logits[top]);
        total += exp(gap * kEmotionLogitScale / kEmotionTemperature);
    }
    result.confidence_pct = (uint8_t)floor(100.0 / total + 0.5);
    if (result.confidence_pct < kEmotionMinConfidence) {
        result.status = EMOTION_LOW_CONFIDENCE;
        return result;
    }
    double percent = floor((double)level * kEmotionLevelScale * 100.0 + 0.5);
    if (percent < 0.0) percent = 0.0;
    if (percent > 100.0) percent = 100.0;
    result.status = EMOTION_OK;
    result.emotion = top;
    result.intensity_pct = (uint8_t)percent;
    return result;
}

static inline EmotionResult predict_emotion(const double features[EMOTION_NUM_FEATURES]) {
    if (!emotion_present(features[0]) || !emotion_present(features[4])) {
        EmotionResult result;
        result.status = EMOTION_NO_DATA;
        result.emotion = EMOTION_NONE;
        result.confidence_pct = 0;
        result.intensity_pct = 0;
        return result;
    }
    int8_t input[EMOTION_NUM_FEATURES];
    for (uint8_t i = 0; i < EMOTION_NUM_FEATURES; ++i) {
        input[i] = emotion_quantize(features[i], i);
    }
    int8_t logits[EMOTION_NUM_CLASSES];
    int8_t level;
    emotion_forward(input, logits, &level);
    return emotion_decide(logits, level);
}

// ---------- Sliding 15-minute window (same logic as Python) ----------
class EmotionWindow {
public:
    EmotionWindow() : filled_(0), mic_epochs_(0) {
        for (uint8_t c = 0; c < 4; ++c) {
            baseline_[c] = kEmotionBaselinePrior[c];
            count_[c] = 0;
        }
        for (uint8_t i = 0; i < EMOTION_NUM_FEATURES; ++i) features_[i] = NAN;
    }
    bool push(float hr_bpm, float motion_mg, float pitch_hz, float energy_db, bool mic_on,
              EmotionResult& result) {
        buffer_[0][filled_] = clean(hr_bpm);
        buffer_[1][filled_] = clean(motion_mg);
        buffer_[2][filled_] = mic_on ? clean(pitch_hz) : NAN;
        buffer_[3][filled_] = mic_on ? clean(energy_db) : NAN;
        if (mic_on) ++mic_epochs_;
        if (++filled_ < EMOTION_WINDOW_EPOCHS) return false;
        finish(result);
        filled_ = 0;
        mic_epochs_ = 0;
        return true;
    }
    const double* features() const { return features_; }
private:
    static float clean(float x) { return emotion_present(x) ? x : NAN; }
    void summarize(uint8_t channel, double& mean, double& spread, uint16_t& count) const {
        double total = 0.0;
        count = 0;
        for (uint16_t i = 0; i < EMOTION_WINDOW_EPOCHS; ++i) {
            const double x = buffer_[channel][i];
            if (emotion_present(x)) {
                total += x;
                ++count;
            }
        }
        const uint16_t divisor = count > 0 ? count : 1;
        mean = total / divisor;
        double squares = 0.0;
        for (uint16_t i = 0; i < EMOTION_WINDOW_EPOCHS; ++i) {
            const double x = buffer_[channel][i];
            if (emotion_present(x)) squares += (x - mean) * (x - mean);
        }
        spread = sqrt(squares / divisor);
    }
    double peakInterval(uint8_t channel, double mean, double spread) const {
        const double threshold = mean + kEmotionPeakSigma * spread;
        const float* x = buffer_[channel];
        uint16_t first = 0, last = 0, peaks = 0;
        for (uint16_t i = 1; i + 1 < EMOTION_WINDOW_EPOCHS; ++i) {
            const double before = x[i - 1], here = x[i], after = x[i + 1];
            if (here > before && here >= after && here > threshold) {
                if (peaks == 0) first = i;
                last = i;
                ++peaks;
            }
        }
        if (peaks < 2) return (double)EMOTION_WINDOW_EPOCHS * EMOTION_EPOCH_SEC;
        return (double)(last - first) / (peaks - 1) * EMOTION_EPOCH_SEC;
    }
    void finish(EmotionResult& result) {
        double mean[4], spread[4];
        uint16_t count[4];
        bool usable[4];
        for (uint8_t c = 0; c < 4; ++c) {
            summarize(c, mean[c], spread[c], count[c]);
            usable[c] = count[c] >= kEmotionMinEpochs[c];
        }
        const bool voice_ok = usable[2] && usable[3];
        for (uint8_t i = 0; i < EMOTION_NUM_FEATURES; ++i) features_[i] = NAN;
        for (uint8_t c = 0; c < 4; ++c) {
            if (!usable[c] || (c >= 2 && !voice_ok)) continue;
            const double magnitude = fabs(mean[c]);
            const double divisor = magnitude > kEmotionMeanFloor[c] ? magnitude : kEmotionMeanFloor[c];
            features_[c * 4] = spread[c];
            features_[c * 4 + 1] = mean[c] - baseline_[c];
            features_[c * 4 + 2] = spread[c] / divisor;
            features_[c * 4 + 3] = peakInterval(c, mean[c], spread[c]);
        }
        if (voice_ok) {
            features_[16] = (double)count[2] / (mic_epochs_ > 0 ? mic_epochs_ : 1);
            features_[17] = 1.0;
        } else {
            features_[17] = 0.0;
        }
        if (!usable[0] || !usable[1]) {
            result.status = EMOTION_NO_DATA;
            result.emotion = EMOTION_NONE;
            result.confidence_pct = 0;
            result.intensity_pct = 0;
        } else if (mean[1] >= kEmotionActivityGateMg) {
            result.status = EMOTION_HIGH_MOTION;
            result.emotion = EMOTION_NONE;
            result.confidence_pct = 0;
            result.intensity_pct = 0;
        } else {
            result = predict_emotion(features_);
        }
        if (usable[1] && mean[1] < kEmotionActivityGateMg) {
            for (uint8_t c = 0; c < 4; ++c) {
                if (!usable[c]) continue;
                ++count_[c];
                double alpha = 1.0 / (count_[c] + 1.0);
                if (alpha < kEmotionBaselineAlphaMin) alpha = kEmotionBaselineAlphaMin;
                baseline_[c] += alpha * (mean[c] - baseline_[c]);
            }
        }
    }
    float buffer_[4][EMOTION_WINDOW_EPOCHS];
    uint16_t filled_;
    uint16_t mic_epochs_;
    double baseline_[4];
    uint32_t count_[4];
    double features_[EMOTION_NUM_FEATURES];
};