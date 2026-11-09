#include <iostream>
#include <iomanip>
#include <cmath>
#include "emotion_model.h"

int main() {
    EmotionWindow window;
    EmotionResult result;

    std::cout << "Starting emotion model test (8 classes)...\n\n";

    // Simulating 15 minutes of data (relatively calm state)
    for (int i = 0; i < EMOTION_WINDOW_EPOCHS; ++i) {
        float hr     = 68.0f + 2.0f * std::sin(i * 0.05f);
        float motion = 25.0f + 8.0f * std::sin(i * 0.1f);
        float pitch  = 180.0f + 10.0f * std::sin(i * 0.08f);
        float energy = 32.0f + 3.0f * std::sin(i * 0.07f);
        bool  mic_on = true;

        bool ready = window.push(hr, motion, pitch, energy, mic_on, result);

        if (ready) {
            std::cout << "===== Result After 15 Minutes =====\n";
            std::cout << "Status      : ";
            switch (result.status) {
                case EMOTION_OK:             std::cout << "OK\n"; break;
                case EMOTION_LOW_CONFIDENCE: std::cout << "LOW_CONFIDENCE\n"; break;
                case EMOTION_HIGH_MOTION:    std::cout << "HIGH_MOTION\n"; break;
                case EMOTION_NO_DATA:        std::cout << "NO_DATA\n"; break;
                default:                     std::cout << "UNKNOWN\n"; break;
            }

            if (result.emotion != EMOTION_NONE)
                std::cout << "Emotion     : " << EMOTION_NAMES[result.emotion] << "\n";
            else
                std::cout << "Emotion     : NONE\n";

            std::cout << "Confidence  : " << (int)result.confidence_pct << " %\n";
            std::cout << std::fixed << std::setprecision(3);
            std::cout << "stability   : " << result.stability << "\n";
            std::cout << "intensity   : " << result.intensity << "\n";
            std::cout << "burstiness  : " << result.burstiness << "\n";
            std::cout << "wavelength  : " << result.wavelength << "\n";
            std::cout << "=================================\n\n";
        }
    }

    // Second test: Excessive motion
    std::cout << "Second test: High motion...\n";
    for (int i = 0; i < EMOTION_WINDOW_EPOCHS; ++i) {
        float hr     = 95.0f;
        float motion = 280.0f;   // Above 150
        float pitch  = 200.0f;
        float energy = 40.0f;
        bool  mic_on = true;

        bool ready = window.push(hr, motion, pitch, energy, mic_on, result);
        if (ready) {
            std::cout << "Status: ";
            if (result.status == EMOTION_HIGH_MOTION)
                std::cout << "HIGH_MOTION (Worked correctly)\n";
            else
                std::cout << (int)result.status << "\n";
        }
    }

    return 0;
}
