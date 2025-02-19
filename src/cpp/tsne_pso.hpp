#ifndef TSNE_PSO_CPP_H
#define TSNE_PSO_CPP_H

#include <vector>
#include <memory>
#include <stdexcept>

extern "C" {
#include "tsne_pso.h"
}

namespace tsne_pso {

class TSNE_PSO {
public:
    TSNE_PSO(int n_components = 2,
             double perplexity = 30.0,
             int n_particles = 100,
             int max_iter = 1000,
             double learning_rate = 200.0,
             double early_exaggeration = 12.0,
             int random_seed = 42,
             int n_threads = -1,
             bool verbose = false)
    {
        config_ = tsne_pso_init_config();
        config_.n_components = n_components;
        config_.perplexity = perplexity;
        config_.n_particles = n_particles;
        config_.max_iter = max_iter;
        config_.learning_rate = learning_rate;
        config_.early_exaggeration = early_exaggeration;
        config_.random_seed = random_seed;
        config_.n_threads = n_threads;
        config_.verbose = verbose;
    }

    std::vector<std::vector<double>> fit_transform(const std::vector<std::vector<double>>& X) {
        if (X.empty()) {
            throw std::invalid_argument("Input data X is empty");
        }

        size_t n_samples = X.size();
        size_t n_features = X[0].size();

        // Convert input data to contiguous array
        std::vector<double> X_contiguous(n_samples * n_features);
        for (size_t i = 0; i < n_samples; ++i) {
            if (X[i].size() != n_features) {
                throw std::invalid_argument("Inconsistent number of features in input data");
            }
            std::copy(X[i].begin(), X[i].end(), X_contiguous.begin() + i * n_features);
        }

        // Run t-SNE
        tsne_pso_result* result = tsne_pso_fit(X_contiguous.data(), n_samples, n_features, &config_);
        if (!result) {
            throw std::runtime_error("t-SNE fitting failed");
        }

        // Convert result to vector of vectors
        std::vector<std::vector<double>> embedding(n_samples, std::vector<double>(config_.n_components));
        for (size_t i = 0; i < n_samples; ++i) {
            for (size_t j = 0; j < config_.n_components; ++j) {
                embedding[i][j] = result->embedding[i * config_.n_components + j];
            }
        }

        // Clean up
        tsne_pso_free_result(result);

        return embedding;
    }

    static std::string version() {
        return tsne_pso_version();
    }

private:
    tsne_pso_config config_;
};

} // namespace tsne_pso

#endif // TSNE_PSO_CPP_H