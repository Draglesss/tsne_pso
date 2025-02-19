#ifndef TSNE_PSO_H
#define TSNE_PSO_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/**
 * @brief Configuration parameters for TSNE-PSO algorithm
 */
typedef struct {
    int n_components;        /**< Number of dimensions in the embedded space */
    double perplexity;      /**< Perplexity parameter for t-SNE */
    int n_particles;        /**< Number of particles in PSO */
    int max_iter;           /**< Maximum number of iterations */
    double learning_rate;   /**< Learning rate for gradient updates */
    double early_exaggeration; /**< Early exaggeration factor */
    double min_gain;        /**< Minimum gain for adaptive learning */
    int random_seed;        /**< Random seed for reproducibility */
    int n_threads;          /**< Number of threads for parallel processing */
    double theta;           /**< Barnes-Hut approximation parameter */
    int verbose;            /**< Verbosity level */
} tsne_pso_config;

/**
 * @brief Result structure containing the embedded coordinates
 */
typedef struct {
    double* embedding;      /**< Pointer to the embedded coordinates */
    size_t n_samples;       /**< Number of samples */
    size_t n_components;    /**< Number of dimensions in the embedding */
    double kl_divergence;   /**< Final KL divergence */
    int n_iter;            /**< Number of iterations performed */
} tsne_pso_result;

/**
 * @brief Initialize default configuration parameters
 * @return tsne_pso_config with default values
 */
tsne_pso_config tsne_pso_init_config(void);

/**
 * @brief Run t-SNE with PSO optimization
 * @param X Input data matrix (n_samples × n_features)
 * @param n_samples Number of samples
 * @param n_features Number of features
 * @param config Configuration parameters
 * @return Result structure containing the embedding
 */
tsne_pso_result* tsne_pso_fit(const double* X, size_t n_samples, size_t n_features, 
                             const tsne_pso_config* config);

/**
 * @brief Free memory allocated for result structure
 * @param result Pointer to result structure
 */
void tsne_pso_free_result(tsne_pso_result* result);

/**
 * @brief Get version string
 * @return Version string
 */
const char* tsne_pso_version(void);

#ifdef __cplusplus
}
#endif

#endif /* TSNE_PSO_H */