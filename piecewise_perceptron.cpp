#include "piecewise_perceptron.hpp"
#include <iostream>
#include <cmath>

PiecewiseLinearPredictor::PiecewiseLinearPredictor(uint64_t perceptron_count, uint8_t ghist_width, uint8_t hash_bits)
    : perceptron_count(perceptron_count), ghist_width(ghist_width), hash_mask((1 << hash_bits) - 1)
{
    theta = std::round(1.93 * ghist_width + 14);  // 学習制御用しきい値
}

void PiecewiseLinearPredictor::setup() {
    weights.resize(perceptron_count);
    for (auto& perceptron : weights) {
        perceptron.resize(ghist_width + 1); // +1 for bias
        for (auto& entry : perceptron) {
            entry.resize(hash_mask + 1, 0); // initialize hashed weights
        }
    }
    std::cout << "Initialized piecewise linear predictor with "
              << perceptron_count << " perceptrons, history width "
              << unsigned(ghist_width) << ", hash bits "
              << std::log2(hash_mask + 1) << std::endl;
}

void PiecewiseLinearPredictor::terminate() {}

uint64_t PiecewiseLinearPredictor::get_unique_inst_id(uint64_t seq_no, uint8_t piece) const {
    return (seq_no << 4) | (piece & 0xF);
}

int PiecewiseLinearPredictor::compute_dot_product(const std::vector<std::vector<int>>& perceptron_weights, uint64_t PC, uint64_t ghist) {
    int result = perceptron_weights[0][PC & hash_mask]; // bias uses hashed PC too

    for (int i = 0; i < ghist_width; ++i) {
        int bit = (ghist >> i) & 1;
        uint32_t index = ((PC >> i) ^ (ghist >> i)) & hash_mask;
        // uint32_t index = (PC ^ (ghist >> i)) & hash_mask;
        int input = bit ? +1 : -1;
        result += perceptron_weights[i + 1][index] * input;
    }
    return result;
}

bool PiecewiseLinearPredictor::predict(uint64_t seq_no, uint8_t piece, uint64_t PC) {
    uint64_t index = PC % perceptron_count;
    int dot = compute_dot_product(weights[index], PC, active_hist.ghist);
    active_hist.pred = dot >= 0;

    pred_time_histories[get_unique_inst_id(seq_no, piece)] = active_hist;
    return active_hist.pred;
}

bool PiecewiseLinearPredictor::predict_using_given_hist(uint64_t, uint8_t, uint64_t, const PiecewiseHist& hist, const bool) {
    return hist.pred;
}

bool PiecewiseLinearPredictor::predict_confidence(uint64_t seq_no, uint8_t piece, uint64_t PC) {
    uint64_t index = PC % perceptron_count;
    int dot = compute_dot_product(weights[index], PC, active_hist.ghist);
    return std::abs(dot) >= theta * 2;
}

void PiecewiseLinearPredictor::history_update(uint64_t, uint8_t, uint64_t, bool taken, uint64_t) {
    active_hist.ghist = ((active_hist.ghist << 1) | taken) & ((1ULL << ghist_width) - 1);
}

void PiecewiseLinearPredictor::update(uint64_t seq_no, uint8_t piece, uint64_t PC, bool actual_taken) {
    auto id = get_unique_inst_id(seq_no, piece);
    auto it = pred_time_histories.find(id);
    if (it != pred_time_histories.end()) {
        update(PC, actual_taken, it->second);
        pred_time_histories.erase(it);
    }
}

void PiecewiseLinearPredictor::update(uint64_t PC, bool actual_taken, const PiecewiseHist& hist) {
    uint64_t index = PC % perceptron_count;
    auto& perceptron_weights = weights[index];
    int t = actual_taken ? +1 : -1;

    int y = compute_dot_product(perceptron_weights, PC, hist.ghist);
    bool pred = y >= 0;

    if (pred != actual_taken || std::abs(y) <= theta) {
        // update bias
        perceptron_weights[0][PC & hash_mask] += t;

        // update weights for history bits
        for (int i = 0; i < ghist_width; ++i) {
            int bit = (hist.ghist >> i) & 1;
            int input = bit ? +1 : -1;
            uint32_t idx = ((PC >> i) ^ (hist.ghist >> i)) & hash_mask;
            // uint32_t idx = (PC ^ (hist.ghist >> i)) & hash_mask;
            perceptron_weights[i + 1][idx] += t * input;
        }
    }
}
