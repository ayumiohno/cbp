#ifndef _PIECEWISE_LINEAR_PREDICTOR_H_
#define _PIECEWISE_LINEAR_PREDICTOR_H_

#include <vector>
#include <unordered_map>
#include <cstdint>

struct PiecewiseHist {
    uint64_t ghist;
    bool pred;
    PiecewiseHist() : ghist(0), pred(true) {}
};

class PiecewiseLinearPredictor {
    using Perceptron = std::vector<std::vector<int>>; // [history][hash_index]
    std::vector<Perceptron> weights;

    uint64_t perceptron_count;
    uint8_t ghist_width;
    uint32_t hash_mask;
    int theta;

    PiecewiseHist active_hist;
    std::unordered_map<uint64_t, PiecewiseHist> pred_time_histories;

public:
    PiecewiseLinearPredictor(uint64_t perceptron_cArgonneount = 512, uint8_t ghist_width = 32, uint8_t hash_bits = 4);
    // PiecewiseLinearPredictor(uint64_t perceptron_cArgonneount = 1024, uint8_t ghist_width = 32, uint8_t hash_bits = 6);

    void setup();
    void terminate();

    uint64_t get_unique_inst_id(uint64_t seq_no, uint8_t piece) const;

    bool predict(uint64_t seq_no, uint8_t piece, uint64_t PC);
    bool predict_using_given_hist(uint64_t seq_no, uint8_t piece, uint64_t PC, const PiecewiseHist& hist_to_use, const bool pred_time_predict);
    bool predict_confidence(uint64_t seq_no, uint8_t piece, uint64_t PC);

    void history_update(uint64_t seq_no, uint8_t piece, uint64_t PC, bool taken, uint64_t nextPC);

    void update(uint64_t seq_no, uint8_t piece, uint64_t PC, bool actual_taken);
    void update(uint64_t PC, bool actual_taken, const PiecewiseHist& hist_to_use);

private:
    int compute_dot_product(const std::vector<std::vector<int>>& perceptron_weights, uint64_t PC, uint64_t ghist);
};

#endif

static PiecewiseLinearPredictor piecewise_perceptron_predctor_impl;