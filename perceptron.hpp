#ifndef _PERCEPTRON_PREDICTOR_H_
#define _PERCEPTRON_PREDICTOR_H_

#include <vector>
#include <unordered_map>
#include <cassert>
#include <cstdint>

struct PerceptronHist {
    uint32_t ghist;
    bool pred;
    PerceptronHist() : ghist(0), pred(true) {}
};

class PerceptronPredictor {
    PerceptronHist active_hist;
    std::unordered_map<uint64_t, PerceptronHist> pred_time_histories;

    uint64_t perceptron_count;
    uint8_t ghist_width;
    int theta; // training threshold
    std::vector<std::vector<int16_t>> weights; // perceptron table

    // std::vector<int> dots;
    // std::vector<bool> takens;

public:
    PerceptronPredictor(uint64_t num_perceptrons = 1024, uint8_t ghist_width = 32);

    void setup();
    void terminate();

    uint64_t get_unique_inst_id(uint64_t seq_no, uint8_t piece) const;
    bool predict(uint64_t seq_no, uint8_t piece, uint64_t PC);
    bool predict_using_given_hist(uint64_t seq_no, uint8_t piece, uint64_t PC, const PerceptronHist& hist_to_use, const bool pred_time_predict);
    bool predict_confidence(uint64_t seq_no, uint8_t piece, uint64_t PC);

    void history_update(uint64_t seq_no, uint8_t piece, uint64_t PC, bool taken, uint64_t nextPC);
    void update(uint64_t seq_no, uint8_t piece, uint64_t PC, bool actual_taken);
    void update(uint64_t PC, bool actual_taken, const PerceptronHist& hist_to_use);

private:
    int compute_dot_product(const std::vector<int16_t>& w, uint64_t ghist);
};

#endif

static PerceptronPredictor perceptron_predctor_impl;