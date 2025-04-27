#include "hierarchical_perceptron.hpp"
#include <iostream>
#include <cmath>

HiearchicalPredictor::HiearchicalPredictor(uint64_t perceptron_count, uint8_t ghist_width)
    : perceptron_count(perceptron_count), ghist_width(ghist_width)
{
    theta = std::round(1.93 * ghist_width + 14);
    ghist_widths[0] = 4;
    ghist_widths[1] = 8;
    ghist_widths[2] = 16;
    ghist_widths[3] = 32;
}

void HiearchicalPredictor::setup()
{
    weights.resize(perceptron_count);
    for (auto &perceptron : weights)
    {
        perceptron.resize(ghist_width + 2); // +1 for bias and +1 for confidence
        for (auto &entry : perceptron)
        {
            entry.resize(4, 0); // initialize hashed weights
        }
    }
    std::cout << "Initialized piecewise linear predictor with "
              << perceptron_count << " perceptrons, history width "
              << unsigned(ghist_width) << ", hash bits " << std::endl;
}

void HiearchicalPredictor::terminate() {}

uint64_t HiearchicalPredictor::get_unique_inst_id(uint64_t seq_no, uint8_t piece) const
{
    return (seq_no << 4) | (piece & 0xF);
}

int HiearchicalPredictor::internel_compute_dot_product(const std::vector<std::vector<int8_t>> &perceptron_weights, uint64_t PC, uint64_t ghist, int index)
{
    int result = perceptron_weights[0][index];
    for (int i = 0; i < ghist_widths[index]; ++i)
    {
        uint64_t bit = (ghist >> i) & 1;
        int input = bit ? +1 : -1;
        result += perceptron_weights[i + 1][index] * input;
    }
    return result;
}

std::pair<int, int> HiearchicalPredictor::compute_dot_product(const std::vector<std::vector<int8_t>> &perceptron_weights, uint64_t PC, uint64_t ghist)
{
    std::array<int, 4> scores;
    for (int index = 0; index < 4; index++)
    {
        scores[index] = internel_compute_dot_product(perceptron_weights, PC, ghist, index);
    }
    // int max_result = scores[0];
    // int max_idx = 0;
    // for (int index = 1; index < 4; index++)
    // {
    //     if (abs(max_result * ((4 - max_idx))) <= abs(scores[index] * ((4 - index))))
    //     {
    //         max_result = scores[index];
    //         max_idx = index;
    //     }
    // }
    // std::cout << "index:" << max_idx << " score:" << scores[max_idx] << std::endl;
    int result = scores[0];
    int best_idx = 0;
    int max_conf = perceptron_weights[ghist_widths[0] + 1][0];
    for (int index = 0; index < 4; index++)
    {
        int my_conf = perceptron_weights[ghist_widths[index] + 1][index];
        if (my_conf >= max_conf)
        {
            max_conf = my_conf;
            best_idx = index;
            result = scores[index];
        }
    }
    // if (best_idx != 3)
    //     std::cout << "index:" << best_idx << " score:" << max_conf << std::endl;
    return {result, best_idx};
}

bool HiearchicalPredictor::predict(uint64_t seq_no, uint8_t piece, uint64_t PC)
{
    uint64_t index = (PC >> 2) % perceptron_count;
    auto [dot, max_idx] = compute_dot_product(weights[index], PC, active_hist.ghist);
    active_hist.pred = dot >= 0;

    pred_time_histories[get_unique_inst_id(seq_no, piece)] = active_hist;
    return active_hist.pred;
}

bool HiearchicalPredictor::predict_using_given_hist(uint64_t, uint8_t, uint64_t, const PiecewiseHist &hist, const bool)
{
    return hist.pred;
}

bool HiearchicalPredictor::predict_confidence(uint64_t seq_no, uint8_t piece, uint64_t PC)
{
    uint64_t index = (PC >> 2) % perceptron_count;
    auto [dot, max_idx] = compute_dot_product(weights[index], PC, active_hist.ghist);
    return std::abs(dot) >= theta * 2;
}

void HiearchicalPredictor::history_update(uint64_t, uint8_t, uint64_t, bool taken, uint64_t)
{
    active_hist.ghist = ((active_hist.ghist << 1) | taken) & ((1ULL << ghist_width) - 1);
}

void HiearchicalPredictor::update(uint64_t seq_no, uint8_t piece, uint64_t PC, bool actual_taken)
{
    auto id = get_unique_inst_id(seq_no, piece);
    auto it = pred_time_histories.find(id);
    if (it != pred_time_histories.end())
    {
        update(PC, actual_taken, it->second);
        pred_time_histories.erase(it);
    }
}

void HiearchicalPredictor::update(uint64_t PC, bool actual_taken, const PiecewiseHist &hist)
{
    uint64_t index = (PC >> 2) % perceptron_count;
    auto &perceptron_weights = weights[index];
    int t = actual_taken ? +1 : -1;

    for (int idx = 0; idx < 4; idx++)
    {
        auto y = internel_compute_dot_product(perceptron_weights, PC, hist.ghist, idx);
        bool pred = y >= 0;
        if (pred != actual_taken || std::abs(y) <= std::round(1.93 * ghist_widths[idx] + 14))
        {
            // update bias
            perceptron_weights[0][idx] += t;

            // update weights for history bits
            for (int i = 0; i < ghist_widths[idx]; ++i)
            {
                int bit = (hist.ghist >> i) & 1;
                int input = bit ? +1 : -1;
                if (perceptron_weights[i + 1][idx] >= 127 && t * input == 1)
                    continue;
                if (perceptron_weights[i + 1][idx] <= -127 && t * input == -1)
                    continue;
                perceptron_weights[i + 1][idx] += t * input;
            }
        }
        if (pred == actual_taken)
        {
            // std::cout << "hit" << perceptron_weights[ghist_widths[idx] + 1][idx] << std::endl;
            if (perceptron_weights[ghist_widths[idx] + 1][idx] >= 127)
                continue;
            perceptron_weights[ghist_widths[idx]+1][idx] += 1;
        } 
        else
        {
            // std::cout << "miss " << (int)perceptron_weights[ghist_widths[idx] + 1][idx] << std::endl;
            // if (perceptron_weights[ghist_widths[idx] + 1][idx] <= -127)
            //     continue;
            perceptron_weights[ghist_widths[idx]+1][idx] /= 4;
        }
    }
}
