#include "perceptron.hpp"
#include <iostream>
#include <fstream>
#include <map>
PerceptronPredictor::PerceptronPredictor(uint64_t num_perceptrons, uint8_t ghist_width)
    : perceptron_count(num_perceptrons), ghist_width(ghist_width)
{
    theta = 1.93 * ghist_width + 14; // standard threshold
}

void PerceptronPredictor::setup() {
    weights.resize(perceptron_count, std::vector<int16_t>(ghist_width + 1, 0)); // +1 for bias
    std::cout << "Initialized perceptron predictor with " << perceptron_count
              << " perceptrons and history width " << unsigned(ghist_width) << std::endl;
}

void PerceptronPredictor::terminate() {
    std::cout << "Terminating perceptron predictor." << std::endl;
    // std::string filename = "perceptron_weights.csv";
    // std::ofstream file(filename);
    // for (size_t i = 0; i < takens.size(); ++i) {
    //     file << dots[i] << "," << (takens[i] ? 1 : -1) << std::endl;
    // }
    // file.close();

}


uint64_t PerceptronPredictor::get_unique_inst_id(uint64_t seq_no, uint8_t piece) const {
    assert(piece < 16);
    return (seq_no << 4) | (piece & 0xF);
}

int PerceptronPredictor::compute_dot_product(const std::vector<int16_t>& w, uint64_t ghist) {
    int result = w[0]; // bias
    for (int i = 0; i < ghist_width; ++i) {
        int bit = (ghist >> i) & 1;
        result += w[i + 1] * (bit ? 1 : -1);
    }
    return result;
}

bool PerceptronPredictor::predict(uint64_t seq_no, uint8_t piece, uint64_t PC) {
    uint64_t index = PC % perceptron_count;
    int dot = compute_dot_product(weights[index], active_hist.ghist);
    active_hist.pred = dot >= 0;
    // dots.push_back(dot);

    pred_time_histories.emplace(get_unique_inst_id(seq_no, piece), active_hist);
    return active_hist.pred;
}

bool PerceptronPredictor::predict_confidence(uint64_t seq_no, uint8_t piece, uint64_t PC) {
    uint64_t index = PC % perceptron_count;
    int dot = compute_dot_product(weights[index], active_hist.ghist);
    return abs(dot) >= theta * 2;
}

bool PerceptronPredictor::predict_using_given_hist(uint64_t, uint8_t, uint64_t, const PerceptronHist& hist_to_use, const bool) {
    return hist_to_use.pred;
}

void PerceptronPredictor::history_update(uint64_t, uint8_t, uint64_t, bool taken, uint64_t) {
    active_hist.ghist = (active_hist.ghist << 1 | taken) & ((1ULL << ghist_width) - 1);
    // takens.push_back(taken);
}

void PerceptronPredictor::update(uint64_t seq_no, uint8_t piece, uint64_t PC, bool actual_taken, std::map<int, int>& mp) {
    const auto id = get_unique_inst_id(seq_no, piece);
    auto it = pred_time_histories.find(id);
    if (it != pred_time_histories.end()) {
        update(PC, actual_taken, it->second, mp);
        pred_time_histories.erase(it);
    }
}

void PerceptronPredictor::update(uint64_t PC, bool actual_taken, const PerceptronHist& hist_to_use, std::map<int, int>& mp) {
    uint64_t index = PC % perceptron_count;
    auto& w = weights[index];
    int t = actual_taken ? 1 : -1;
    int y = compute_dot_product(w, hist_to_use.ghist);
    bool pred = y >= 0;
    if (pred != actual_taken)
        ++mp[PC];;
    if (pred != actual_taken || abs(y) <= theta) {
        w[0] += t; // bias
        for (int i = 0; i < ghist_width; ++i) {
            int bit = (hist_to_use.ghist >> i) & 1;
            w[i + 1] += (bit ? t : -t);
        }
    }
}
