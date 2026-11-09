// emotion_model.h - Generated automatically by emotion_training.py
// 8 emotions + stability / intensity / burstiness / wavelength per 15-min window
#pragma once
#include <math.h>
#include <stdint.h>

#define EMOTION_NUM_FEATURES 18
#define EMOTION_NUM_CLASSES 8
#define EMOTION_WINDOW_EPOCHS 180
#define EMOTION_EPOCH_SEC 5

enum EmotionStatus : uint8_t {
    EMOTION_OK = 0,
    EMOTION_LOW_CONFIDENCE = 1,
    EMOTION_HIGH_MOTION = 2,
    EMOTION_NO_DATA = 3,
};

static const uint8_t EMOTION_NONE = 255;
static const char* const EMOTION_NAMES[EMOTION_NUM_CLASSES] = {
    "calm", "tense", "upbeat", "subdued", "happy", "sad", "angry", "tired"
};

// Final output for every 15-minute window
struct EmotionResult {
    uint8_t status;           // OK / LOW_CONFIDENCE / HIGH_MOTION / NO_DATA
    uint8_t emotion;          // 0..7 or 255
    uint8_t confidence_pct;   // 0..100
    float   stability;        // HR stability (std)
    float   intensity;        // HR intensity (mean - baseline)
    float   burstiness;       // HR burstiness (std / |mean|)
    float   wavelength;       // HR peak interval (seconds)
};

// ---------- Scaler ----------
static const float kEmotionMean[18] = {
    1.211127639e+00f, -2.210136652e-01f, 3.499699757e-02f, 3.963840246e+00f, 2.802989006e+00f, -3.463999331e-01f, 4.305724204e-01f, 3.899808168e+00f, 2.172212839e+00f, -1.777093560e-01f, 5.164336041e-02f, 5.296133518e+00f, 1.068146110e+00f, -1.783337444e-02f, 6.221369281e-02f, 5.296145916e+00f, 2.263343185e-01f, 6.931471825e-01f
};
static const float kEmotionStd[18] = {
    3.416048884e-01f, 1.738614082e+00f, 2.459247410e-02f, 4.077167213e-01f, 5.976523161e-01f, 2.272948027e+00f, 1.099099144e-01f, 3.710335195e-01f, 5.490010381e-01f, 2.144970655e+00f, 2.614963055e-02f, 1.229776144e+00f, 1.333916485e-01f, 1.297980785e+00f, 1.576537080e-02f, 1.227338910e+00f, 1.319180131e-01f, 1.000000000e+00f
};
static const double kEmotionInputScale = 0.03937007859349251;
static const double kEmotionLogitScale = 0.14616628331462228;
static const double kEmotionLevelScale = 0.00815317574448473;
static const double kEmotionTemperature = 1.05;
static const uint8_t kEmotionMinConfidence = 67;

static const int8_t kHeartWeights[24] = {
    -16, -78, 45, 14, -92, -1, 11, 23, 17, 127, 4, -16, -98, -21, 32, -41, -2, -84, 73, -57, 56, 59, 11, 42
};
static const int32_t kHeartBias[6] = {
    -756, 91, 1154, 619, -886, -1332
};
static const int32_t kHeartMult = 1406681125;
static const uint8_t kHeartShift = 37;
static const int32_t kHeartInZero = 0;
static const int32_t kHeartOutZero = 0;
static const int32_t kHeartOutMin = 0;

static const int8_t kMotionWeights[24] = {
    61, 99, -43, 17, 10, 86, -54, 13, 26, 21, 23, -75, -91, 108, 68, -33, -57, -127, 47, -6, -77, 110, 17, 18
};
static const int32_t kMotionBias[6] = {
    -2049, 2462, -2848, 109, -364, 500
};
static const int32_t kMotionMult = 1086316340;
static const uint8_t kMotionShift = 37;
static const int32_t kMotionInZero = 0;
static const int32_t kMotionOutZero = 0;
static const int32_t kMotionOutMin = 0;

static const int8_t kVoiceWeights[80] = {
    2, 14, 14, -18, 6, 109, -6, -25, 24, 16, 27, -22, 40, 5, -13, 60, 21, 20, 3, -35, 0, -27, 15, 1, -7, -60, 41, 6, -28, 25, -36, -44, -65, 8, -42, -73, -16, 19, -19, 22, -11, -62, -41, 0, -9, -66, 35, -8, -11, 13, -23, 39, 1, 24, 21, 51, -27, 22, -60, 5, -28, -53, 58, -12, 36, -12, -27, -1, 19, 7, -28, -36, -127, -10, 42, -14, 14, 10, 17, -31
};
static const int32_t kVoiceBias[8] = {
    121, 1228, -675, -6, -1879, -15, 703, 0
};
static const int32_t kVoiceMult = 1924454739;
static const uint8_t kVoiceShift = 37;
static const int32_t kVoiceInZero = 0;
static const int32_t kVoiceOutZero = 0;
static const int32_t kVoiceOutMin = 0;

static const int8_t kDeep1Weights[320] = {
    35, -5, 15, -15, -18, 5, 21, 5, -23, 13, 33, 5, -34, 13, 9, -11, 30, -20, -11, -54, 38, 11, -19, -1, 8, -18, -14, -11, -40, 27, 21, 9, 10, 18, 25, 5, 27, -78, -17, -14, 59, 15, 4, 20, 26, -46, 5, 14, 20, -2, 33, 5, -12, 32, -31, -127, -44, -20, -3, -66, 36, 23, -9, 13, 7, 25, 1, -6, 23, 3, 17, -12, 32, 16, -18, 6, 26, -37, -9, 2, 32, 32, -16, 40, 24, 8, 36, 28, 14, 5, 33, 6, -19, -2, -19, -2, 24, 1, -35, 21, -35, -10, 11, -16, -2, 16, -5, 18, -10, 41, -15, 5, -10, 43, -26, -49, -17, 29, 34, -24, -17, 22, -34, -2, 23, -8, -36, -9, -17, 0, 1, 21, 33, 33, -13, -45, -28, 10, -15, -93, -24, -32, 2, -24, -22, 1, 26, 35, -42, 34, -20, 24, -6, 21, -22, -16, -24, -28, 4, -67, 30, -16, 54, -48, -28, 15, 0, 1, -5, 21, 20, -7, -4, -37, 23, -10, -17, 5, -23, -8, -24, -27, 30, -4, -10, -26, 3, 14, -17, 16, 10, 3, 18, 30, -23, -19, -43, 34, 35, -21, -3, 25, -21, 13, 10, 12, -35, -30, -15, 10, 19, 11, 43, 1, -24, 0, 2, 40, -43, 22, 34, 7, 20, 23, 18, -3, 7, 7, 15, 2, 40, 7, -53, -8, -4, 21, 11, 29, -18, 27, -2, 17, -3, -2, 14, -26, 17, 3, -3, -24, 36, 5, -21, 24, -4, 1, 6, 5, 1, 5, -13, -1, 25, -26, 4, -6, -8, -11, -27, -46, -19, -8, -2, 6, 35, 24, 41, 1, 47, -10, -50, 10, 12, -16, -27, 8, 32, 25, -14, 6, -14, 7, 24, 9, -26, 7, -35, 32, 33, 5, -15, -4, 4, -8, 8, -1, 22, 23, -40, -1, -3, 19, -44, 23, 15, 10, -36, -40, 1, 2
};
static const int32_t kDeep1Bias[16] = {
    601, 95, 929, -899, -449, 783, -180, 532, 315, 778, -843, 59, 1002, 25, 1450, 68
};
static const int32_t kDeep1Mult = 1868545101;
static const uint8_t kDeep1Shift = 37;
static const int32_t kDeep1InZero = 0;
static const int32_t kDeep1OutZero = 0;
static const int32_t kDeep1OutMin = 0;

static const int8_t kDeep2Weights[192] = {
    -47, -25, 5, 15, 12, -36, -44, 20, -37, -49, -46, 40, 10, -32, -33, -26, -4, -16, 26, 44, 42, 22, 88, 31, -74, -11, 127, 10, 34, -53, 32, 38, 0, -27, 4, -29, -29, 49, 1, 29, 62, 49, -12, -18, 45, 60, 30, 31, 4, -5, 46, 48, 12, -28, 49, 0, -29, -23, 25, 49, 16, 3, 15, 9, 51, 29, 80, 10, -13, 24, 43, 17, -6, 16, -21, -16, 3, 70, -38, -10, 20, -15, 25, 6, -24, 6, 36, -2, -54, 19, -35, 17, 29, 49, -24, 4, -19, 7, -11, 1, -4, -17, -23, -6, 10, -6, -6, -20, -18, -6, -11, 9, 60, -33, 9, 0, -35, 35, 7, 68, 73, 41, 47, -53, 4, 12, 46, 34, 89, 90, 34, 41, 44, -41, 12, -23, -12, -7, 27, 13, -8, 72, -94, -1, 21, -103, 8, 1, -26, 35, 45, 37, -31, 33, 54, -21, 24, -3, -15, 9, 28, -24, 7, 8, 2, 21, -50, -40, 27, -16, 15, 25, -21, -40, 15, -69, 54, 14, 54, 44, 40, 3, 59, 25, -27, -13, 90, 36, -11, -22, -5, 18
};
static const int32_t kDeep2Bias[12] = {
    16, 182, 1348, 925, 909, 1244, -183, 653, -1446, 8, 305, -173
};
static const int32_t kDeep2Mult = 1661437005;
static const uint8_t kDeep2Shift = 38;
static const int32_t kDeep2InZero = 0;
static const int32_t kDeep2OutZero = 0;
static const int32_t kDeep2OutMin = 0;

static const int8_t kDeep3Weights[96] = {
    -21, -55, 37, 31, 39, 27, 14, 46, 13, 0, 16, 2, 7, -3, -17, -24, -19, 3, 30, -12, -27, -29, 13, 14, -95, 77, 18, -37, 52, 3, -16, 35, -77, 73, 25, 37, 123, -8, 48, 14, -3, 38, 5, 46, -30, 44, -53, 6, 19, 88, -24, 43, -1, 47, 27, -4, -77, 14, -29, 57, 6, 32, 28, -49, -55, -103, 19, 59, -29, -3, -79, 2, 26, 9, -65, 62, 77, 13, -23, -45, 127, -36, 1, 88, 4, 68, -16, 29, -51, -16, -23, -10, 9, 22, -15, 30
};
static const int32_t kDeep3Bias[8] = {
    764, -418, -1211, 245, 778, -731, -839, -125
};
static const int32_t kDeep3Mult = 1521708580;
static const uint8_t kDeep3Shift = 38;
static const int32_t kDeep3InZero = 0;
static const int32_t kDeep3OutZero = 0;
static const int32_t kDeep3OutMin = 0;

static const int8_t kEmotionWeights[64] = {
    11, 5, -77, -17, 60, -21, -55, 26, 48, 21, 0, 39, -127, -13, -127, -48, -7, 25, 51, 24, 0, -40, -53, -45, -16, 20, -46, -17, 27, 64, 24, 35, -15, -1, 48, -8, 19, -47, -64, 39, 25, -23, 1, -45, -1, -79, 25, -32, -32, 8, 29, 20, -52, 72, 1, -3, -22, 9, 3, -64, 7, 62, 37, 39
};
static const int32_t kEmotionBias[8] = {
    1867, 286, -692, -509, -263, 350, -613, -707
};
static const int32_t kEmotionMult = 1627672920;
static const uint8_t kEmotionShift = 38;
static const int32_t kEmotionInZero = 0;
static const int32_t kEmotionOutZero = 0;
static const int32_t kEmotionOutMin = -127;

static const int8_t kIntensityWeights[8] = {
    127, 42, 0, -37, 9, 15, 5, 69
};
static const int32_t kIntensityBias[1] = {
    -582
};
static const int32_t kIntensityMult = 1474831156;
static const uint8_t kIntensityShift = 36;
static const int32_t kIntensityInZero = 0;
static const int32_t kIntensityOutZero = 0;
static const int32_t kIntensityOutMin = -127;

// ---------- Feature-extraction constants ----------
static const uint16_t kEmotionMinEpochs[4] = {90, 90, 6, 6};
static const double kEmotionMeanFloor[4] = {1.0, 10.0, 10.0, 1.0};
static const double kEmotionBaselinePrior[4] = {70.0, 30.0, 160.0, 30.0};
static const double kEmotionPeakSigma = 0.5;
static const double kEmotionActivityGateMg = 150.0;
static const double kEmotionBaselineAlphaMin = 0.01;

// ---------- Integer dense layer ----------
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

    emotion_dense(deep3, 8, kEmotionInZero, kEmotionWeights, kEmotionBias, kEmotionMult,
                  kEmotionShift, 8, kEmotionOutZero, kEmotionOutMin, logits);
    emotion_dense(deep3, 8, kIntensityInZero, kIntensityWeights, kIntensityBias, kIntensityMult,
                  kIntensityShift, 1, kIntensityOutZero, kIntensityOutMin, intensity);
}

static inline EmotionResult emotion_decide(const int8_t logits[EMOTION_NUM_CLASSES],
                                           const double features[EMOTION_NUM_FEATURES]) {
    EmotionResult result;
    result.stability  = (float)features[0];
    result.intensity  = (float)features[1];
    result.burstiness = (float)features[2];
    result.wavelength = (float)features[3];

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
    result.emotion = top;
    result.status  = (result.confidence_pct < kEmotionMinConfidence)
                         ? EMOTION_LOW_CONFIDENCE
                         : EMOTION_OK;

    return result;
}
static inline EmotionResult predict_emotion(const double features[EMOTION_NUM_FEATURES]) {
    if (!emotion_present(features[0]) || !emotion_present(features[4])) {
        EmotionResult result;
        result.status = EMOTION_NO_DATA;
        result.emotion = EMOTION_NONE;
        result.confidence_pct = 0;
        result.stability = result.intensity = result.burstiness = result.wavelength = 0.0f;
        return result;
    }
    int8_t input[EMOTION_NUM_FEATURES];
    for (uint8_t i = 0; i < EMOTION_NUM_FEATURES; ++i) {
        input[i] = emotion_quantize(features[i], i);
    }
    int8_t logits[EMOTION_NUM_CLASSES];
    int8_t level;
    emotion_forward(input, logits, &level);
    return emotion_decide(logits, features);
}

// ---------- Sliding 15-minute window ----------
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
            if (emotion_present(x)) { total += x; ++count; }
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
            features_[c * 4 + 0] = spread[c];
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
            result.stability = result.intensity = result.burstiness = result.wavelength = 0.0f;
        } else if (mean[1] >= kEmotionActivityGateMg) {
            result.status = EMOTION_HIGH_MOTION;
            result.emotion = EMOTION_NONE;
            result.confidence_pct = 0;
            result.stability  = (float)features_[0];
            result.intensity  = (float)features_[1];
            result.burstiness = (float)features_[2];
            result.wavelength = (float)features_[3];
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
