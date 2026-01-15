#include "tsne_pso.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <float.h>
#include <time.h>
#include <stdint.h>

/*
 * TSNE-PSO core
 * ------------
 * This file implements a t-SNE-style objective (KL(P||Q)) and searches the
 * embedding using a particle swarm, nudged by gradient information.
 *
 * Notes:
 * - The public API is defined in `include/tsne_pso.h`.
 * - OpenMP is used for coarse-grained parallelism in the O(n^2) kernels.
 */

#define TSNE_PSO_VERSION "1.0.0"
#define EPS 1e-7

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---- RNG (thread-local, deterministic per `random_seed`) --------------------
//
// `rand()` is not thread-safe; we use a small, fast PRNG per OpenMP thread.
static inline uint64_t splitmix64_next(uint64_t* x) {
    uint64_t z = (*x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static inline uint64_t xorshift64star_next(uint64_t* x) {
    uint64_t z = *x;
    z ^= z >> 12;
    z ^= z << 25;
    z ^= z >> 27;
    *x = z;
    return z * 2685821657736338717ULL;
}

static inline double rng_uniform01(uint64_t* state) {
    // 53 bits to double in [0, 1)
    const uint64_t r = xorshift64star_next(state);
    return (r >> 11) * (1.0 / 9007199254740992.0);
}

static inline double rng_normal01(uint64_t* state) {
    // Box–Muller transform (avoid log(0)).
    double u1 = rng_uniform01(state);
    double u2 = rng_uniform01(state);
    if (u1 < 1e-12) u1 = 1e-12;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

// Default configuration for the public C API.
tsne_pso_config tsne_pso_init_config(void) {
    tsne_pso_config config;
    config.n_components = 2;
    config.perplexity = 30.0;
    config.n_particles = 100;
    config.max_iter = 1000;
    config.learning_rate = 200.0;
    config.early_exaggeration = 12.0;
    config.min_gain = 0.01;
    config.random_seed = 42;
    // Default to 1 thread for reproducibility; callers can opt into parallelism via `n_threads`.
    config.n_threads = 1;
    config.theta = 0.5;
    config.verbose = 0;
    return config;
}

// Internal state for PSO-based optimization of the t-SNE objective.
typedef struct {
    double* position;       // Current embedding candidate (flattened).
    double* velocity;       // PSO velocity term.
    double* best_position;  // Particle-best position so far.
    double best_fitness;    // Particle-best objective value (lower is better).
    uint64_t rng_state;     // Per-particle RNG state (deterministic across threads).
} Particle;

typedef struct {
    double* global_best;
    double global_best_fitness;
    size_t global_best_index;
    Particle* particles;
    size_t n_particles;
    size_t n_dimensions;
    double w;           // Inertia weight.
    double c1;          // Cognitive coefficient.
    double c2;          // Social coefficient.
    double v_max;       // Velocity clamp (stability guardrail).
} PSO_Swarm;

// (Random helpers replaced by thread-local RNG above.)

// Allocation that fails fast with a useful message.
static void* safe_malloc(size_t size) {
    void* ptr = malloc(size);
    if (!ptr) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

// Initialize a particle near the origin (small Gaussian noise).
static void init_particle(Particle* particle, size_t n_dimensions, uint64_t seed) {
    particle->position = (double*)safe_malloc(n_dimensions * sizeof(double));
    particle->velocity = (double*)safe_malloc(n_dimensions * sizeof(double));
    particle->best_position = (double*)safe_malloc(n_dimensions * sizeof(double));
    particle->rng_state = seed;
    
    for (size_t i = 0; i < n_dimensions; i++) {
        particle->position[i] = rng_normal01(&particle->rng_state) * 1e-4;
        particle->velocity[i] = 0.0;
        particle->best_position[i] = particle->position[i];
    }
    particle->best_fitness = INFINITY;
}

// Allocate and initialize swarm-level state.
static PSO_Swarm* init_swarm(size_t n_particles, size_t n_dimensions, uint64_t master_seed) {
    PSO_Swarm* swarm = (PSO_Swarm*)safe_malloc(sizeof(PSO_Swarm));
    swarm->n_particles = n_particles;
    swarm->n_dimensions = n_dimensions;
    swarm->global_best = (double*)safe_malloc(n_dimensions * sizeof(double));
    swarm->particles = (Particle*)safe_malloc(n_particles * sizeof(Particle));
    
    // PSO hyperparameters (classic defaults; tuned for stability over speed).
    swarm->w = 0.9;    // Inertia weight
    swarm->c1 = 2.0;   // Cognitive parameter
    swarm->c2 = 2.0;   // Social parameter
    swarm->v_max = 5.0; // Maximum velocity
    swarm->global_best_fitness = INFINITY;
    swarm->global_best_index = 0;
    
    for (size_t i = 0; i < n_particles; i++) {
        // Derive a stable per-particle seed from the master seed and particle index.
        uint64_t s = master_seed ^ (0x9e3779b97f4a7c15ULL * (uint64_t)(i + 1));
        s = splitmix64_next(&s);
        init_particle(&swarm->particles[i], n_dimensions, s);
    }
    
    return swarm;
}

// Full pairwise distance matrix (symmetric, diagonal unused).
static void compute_pairwise_distances(const double* X, size_t n_samples, size_t n_features,
                                     double* distances) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n_samples; i++) {
        for (size_t j = i + 1; j < n_samples; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < n_features; k++) {
                double diff = X[i * n_features + k] - X[j * n_features + k];
                sum += diff * diff;
            }
            // Store squared distance (this is what the Gaussian kernel expects).
            distances[i * n_samples + j] = sum;
            distances[j * n_samples + i] = sum;
        }
    }
}

// Compute joint probabilities P_ij by matching per-row entropy to `perplexity`,
// then symmetrizing. The `P` buffer is used as scratch during the row pass.
static void compute_joint_probabilities(const double* distances, size_t n_samples,
                                     double perplexity, double* P) {
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < n_samples; i++) {
        double beta = 1.0;
        double beta_min = -INFINITY;
        double beta_max = INFINITY;
        
        // Binary search for beta s.t. H(P_i) ~= log(perplexity).
        for (int iter = 0; iter < 50; iter++) {
            double sum_P = 0.0;
            double sum_dP = 0.0;
            double H = 0.0;
            
            // Unnormalized Gaussian row (diagonal skipped).
            for (size_t j = 0; j < n_samples; j++) {
                if (i != j) {
                    const double dist2 = distances[i * n_samples + j];
                    const double P_ij = exp(-beta * dist2);
                    P[i * n_samples + j] = P_ij;
                    sum_P += P_ij;
                    sum_dP += dist2 * P_ij;
                }
            }

            // If everything underflowed, fall back to a uniform row.
            if (sum_P <= 0.0 || !isfinite(sum_P)) {
                const double inv = 1.0 / (double)(n_samples - 1);
                for (size_t j = 0; j < n_samples; j++) {
                    if (i != j) {
                        P[i * n_samples + j] = inv;
                    }
                }
                H = log((double)(n_samples - 1) + EPS);
                break;
            }
            
            // Normalize + compute Shannon entropy.
            for (size_t j = 0; j < n_samples; j++) {
                if (i != j) {
                    P[i * n_samples + j] /= sum_P;
                    H -= P[i * n_samples + j] * log(P[i * n_samples + j] + EPS);
                }
            }
            
            // Adjust beta to move entropy toward target.
            double H_diff = H - log(perplexity);
            if (fabs(H_diff) < 1e-5) break;
            
            if (H_diff > 0) {
                beta_min = beta;
                beta = (beta_max == INFINITY) ? beta * 2 : (beta + beta_max) / 2;
            } else {
                beta_max = beta;
                beta = (beta_min == -INFINITY) ? beta / 2 : (beta + beta_min) / 2;
            }
        }
    }
    
    // Symmetrize: P_ij = (P_ij + P_ji) / (2n). (Normalization is implicit.)
    double sum_P = 0.0;
    for (size_t i = 0; i < n_samples; i++) {
        for (size_t j = 0; j < i; j++) {
            P[i * n_samples + j] = (P[i * n_samples + j] + P[j * n_samples + i]) / (2 * n_samples);
            P[j * n_samples + i] = P[i * n_samples + j];
            sum_P += 2 * P[i * n_samples + j];
        }
    }

    // Normalize to make sum_{i!=j} P_ij = 1 (standard t-SNE convention).
    if (sum_P > 0.0) {
        const double inv = 1.0 / sum_P;
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n_samples; i++) {
            for (size_t j = 0; j < n_samples; j++) {
                if (i != j) {
                    P[i * n_samples + j] *= inv;
                } else {
                    P[i * n_samples + j] = 0.0;
                }
            }
        }
    } else {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n_samples; i++) {
            for (size_t j = 0; j < n_samples; j++) {
                P[i * n_samples + j] = 0.0;
            }
        }
    }
}

// KL(P||Q) objective for a particle embedding Y.
// Computes KL without materializing Q: we accumulate sum_Q and the p-weighted log(q_unnorm).
static double compute_kl_loss(const double* P,
                              size_t n_samples, size_t n_components,
                              const double* Y,
                              double sum_p_log_p,
                              double sum_p_total) {
    // Deterministic accumulation: OpenMP reductions are order-dependent and can break reproducibility.
    const int max_threads = omp_get_max_threads();
    double* partial_sum_Q = (double*)safe_malloc((size_t)max_threads * sizeof(double));
    double* partial_sum_p_log_q = (double*)safe_malloc((size_t)max_threads * sizeof(double));
    for (int t = 0; t < max_threads; t++) {
        partial_sum_Q[t] = 0.0;
        partial_sum_p_log_q[t] = 0.0;
    }

    int used_threads = 1;
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        #pragma omp single
        { used_threads = omp_get_num_threads(); }

        double local_sum_Q = 0.0;
        double local_sum_p_log_q = 0.0;

        #pragma omp for schedule(static)
        for (size_t i = 0; i < n_samples; i++) {
            for (size_t j = i + 1; j < n_samples; j++) {
                double dist2 = 0.0;
                for (size_t d = 0; d < n_components; d++) {
                    const double diff = Y[i * n_components + d] - Y[j * n_components + d];
                    dist2 += diff * diff;
                }
                const double q_unnorm = 1.0 / (1.0 + dist2);
                local_sum_Q += 2.0 * q_unnorm;

                const double p = P[i * n_samples + j];
                if (p > 0.0) {
                    local_sum_p_log_q += 2.0 * p * log(q_unnorm + EPS);
                }
            }
        }

        partial_sum_Q[tid] = local_sum_Q;
        partial_sum_p_log_q[tid] = local_sum_p_log_q;
    }

    double sum_Q = 0.0;
    double sum_p_log_q = 0.0;
    for (int t = 0; t < used_threads; t++) {
        sum_Q += partial_sum_Q[t];
        sum_p_log_q += partial_sum_p_log_q[t];
    }
    free(partial_sum_Q);
    free(partial_sum_p_log_q);

    // KL = sum p log(p) - sum p log(q_unnorm) + log(sum_Q) * sum p
    return sum_p_log_p - sum_p_log_q + log(sum_Q + EPS) * sum_p_total;
}

static void compute_p_stats(const double* P, size_t n_samples, double* out_sum_p_log_p, double* out_sum_p_total) {
    const int max_threads = omp_get_max_threads();
    double* partial_sum_p_log_p = (double*)safe_malloc((size_t)max_threads * sizeof(double));
    double* partial_sum_p_total = (double*)safe_malloc((size_t)max_threads * sizeof(double));
    for (int t = 0; t < max_threads; t++) {
        partial_sum_p_log_p[t] = 0.0;
        partial_sum_p_total[t] = 0.0;
    }

    int used_threads = 1;
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        #pragma omp single
        { used_threads = omp_get_num_threads(); }

        double local_logp = 0.0;
        double local_sum = 0.0;

        #pragma omp for schedule(static)
        for (size_t i = 0; i < n_samples; i++) {
            for (size_t j = i + 1; j < n_samples; j++) {
                const double p = P[i * n_samples + j];
                if (p > 0.0) {
                    local_logp += 2.0 * p * log(p + EPS);
                    local_sum += 2.0 * p;
                }
            }
        }

        partial_sum_p_log_p[tid] = local_logp;
        partial_sum_p_total[tid] = local_sum;
    }

    double sum_p_log_p = 0.0;
    double sum_p_total = 0.0;
    for (int t = 0; t < used_threads; t++) {
        sum_p_log_p += partial_sum_p_log_p[t];
        sum_p_total += partial_sum_p_total[t];
    }
    free(partial_sum_p_log_p);
    free(partial_sum_p_total);

    *out_sum_p_log_p = sum_p_log_p;
    *out_sum_p_total = sum_p_total;
}

// PSO update: v <- w v + c1 r1 (B - Y) + c2 r2 (G - Y), then Y <- Y + v.
static void update_particle_pso(Particle* particle,
                                const double* global_best_snapshot,
                                size_t n_dimensions,
                                const PSO_Swarm* swarm,
                                double c1, double c2) {
    for (size_t i = 0; i < n_dimensions; i++) {
        const double r1 = rng_uniform01(&particle->rng_state);
        const double r2 = rng_uniform01(&particle->rng_state);
        const double cognitive = c1 * r1 * (particle->best_position[i] - particle->position[i]);
        const double social = c2 * r2 * (global_best_snapshot[i] - particle->position[i]);

        particle->velocity[i] = swarm->w * particle->velocity[i] + cognitive + social;

        // Clamp to keep exploration bounded.
        if (particle->velocity[i] > swarm->v_max)
            particle->velocity[i] = swarm->v_max;
        else if (particle->velocity[i] < -swarm->v_max)
            particle->velocity[i] = -swarm->v_max;

        particle->position[i] += particle->velocity[i];
    }
}

// Release all heap allocations owned by the swarm.
static void free_swarm(PSO_Swarm* swarm) {
    if (swarm) {
        free(swarm->global_best);
        for (size_t i = 0; i < swarm->n_particles; i++) {
            free(swarm->particles[i].position);
            free(swarm->particles[i].velocity);
            free(swarm->particles[i].best_position);
        }
        free(swarm->particles);
        free(swarm);
    }
}

tsne_pso_result* tsne_pso_fit(const double* X, size_t n_samples, size_t n_features,
                             const tsne_pso_config* config) {
    // Pipeline: compute P from X, then optimize embedding positions via PSO.
    uint64_t seed = (uint64_t)config->random_seed;
    uint64_t master_seed = splitmix64_next(&seed);
    
    omp_set_num_threads(config->n_threads);
    
    tsne_pso_result* result = (tsne_pso_result*)safe_malloc(sizeof(tsne_pso_result));
    result->n_samples = n_samples;
    result->n_components = config->n_components;
    result->embedding = (double*)safe_malloc(n_samples * config->n_components * sizeof(double));
    
    double* distances = (double*)safe_malloc(n_samples * n_samples * sizeof(double));
    compute_pairwise_distances(X, n_samples, n_features, distances);
    
    double* P = (double*)safe_malloc(n_samples * n_samples * sizeof(double));
    compute_joint_probabilities(distances, n_samples, config->perplexity, P);
    double sum_p_log_p = 0.0;
    double sum_p_total = 0.0;
    compute_p_stats(P, n_samples, &sum_p_log_p, &sum_p_total);
    
    // Early exaggeration (common t-SNE heuristic).
    for (size_t i = 0; i < n_samples * n_samples; i++) {
        P[i] *= config->early_exaggeration;
    }
    compute_p_stats(P, n_samples, &sum_p_log_p, &sum_p_total);
    
    PSO_Swarm* swarm = init_swarm(config->n_particles, n_samples * config->n_components, master_seed);
    
    double best_kl_divergence = INFINITY;
    int best_iter = 0;

    // Dynamic coefficients: start cognitive-heavy, then gradually shift weight to the global best.
    const double h = 50.0;
    const double f = (config->max_iter > 0) ? (1.0 / (double)config->max_iter) : 1.0;

    // Evaluate initial particles to seed (B_k, G) before any PSO updates.
    // Always initialize G to a valid position to avoid undefined behavior if losses become NaN/Inf.
    swarm->global_best_index = 0;
    memcpy(swarm->global_best, swarm->particles[0].position, swarm->n_dimensions * sizeof(double));
    swarm->global_best_fitness = compute_kl_loss(P, n_samples, config->n_components,
                                                 swarm->particles[0].position, sum_p_log_p, sum_p_total);
    if (!isfinite(swarm->global_best_fitness)) {
        swarm->global_best_fitness = INFINITY;
    }
    best_kl_divergence = swarm->global_best_fitness;
    best_iter = 0;
    #pragma omp parallel for schedule(static)
    for (size_t p = 0; p < swarm->n_particles; p++) {
        Particle* particle = &swarm->particles[p];
        const size_t n_dim = n_samples * config->n_components;
        const double loss0 = compute_kl_loss(P, n_samples, config->n_components,
                                             particle->position, sum_p_log_p, sum_p_total);

        #pragma omp critical
        {
            if (isfinite(loss0) && loss0 < particle->best_fitness) {
                particle->best_fitness = loss0;
                memcpy(particle->best_position, particle->position, n_dim * sizeof(double));
            }
            if (isfinite(loss0) &&
                (loss0 < swarm->global_best_fitness ||
                 (loss0 == swarm->global_best_fitness && p < swarm->global_best_index))) {
                swarm->global_best_fitness = loss0;
                swarm->global_best_index = p;
                memcpy(swarm->global_best, particle->position, n_dim * sizeof(double));
                best_kl_divergence = loss0;
                best_iter = 0;
            }
        }
    }
    
    for (int iter = 0; iter < config->max_iter; iter++) {
        // Drop exaggeration after a short burn-in.
        if (iter == 100) {
            for (size_t i = 0; i < n_samples * n_samples; i++) {
                P[i] /= config->early_exaggeration;
            }
            compute_p_stats(P, n_samples, &sum_p_log_p, &sum_p_total);
        }

        // Dynamic update of c1/c2
        // Choose schedule so c1 == c2 at t = f and c1 + c2 = h.
        const double t = (double)(iter + 1);
        const double c2 = h / (1.0 + f * t);
        const double c1 = h - c2;

        // Snapshot global best for this iteration to avoid mid-iteration drift across threads.
        const size_t n_dim = n_samples * config->n_components;
        double* global_best_snapshot = (double*)safe_malloc(n_dim * sizeof(double));
        memcpy(global_best_snapshot, swarm->global_best, n_dim * sizeof(double));

        #pragma omp parallel for schedule(static)
        for (size_t p = 0; p < swarm->n_particles; p++) {
                Particle* particle = &swarm->particles[p];

                // PSO position/velocity update (Eq. 5/6), then loss evaluation (Eq. 4).
                update_particle_pso(particle, global_best_snapshot, n_dim, swarm, c1, c2);
                const double loss = compute_kl_loss(P, n_samples, config->n_components,
                                                    particle->position, sum_p_log_p, sum_p_total);

                // Track particle-best and swarm-best (guarded; this section is small).
                #pragma omp critical
                {
                    if (isfinite(loss) && loss < particle->best_fitness) {
                        particle->best_fitness = loss;
                        memcpy(particle->best_position, particle->position, n_dim * sizeof(double));
                    }
                    if (isfinite(loss) &&
                        (loss < swarm->global_best_fitness ||
                         (loss == swarm->global_best_fitness && p < swarm->global_best_index))) {
                        swarm->global_best_fitness = loss;
                        swarm->global_best_index = p;
                        memcpy(swarm->global_best, particle->position, n_dim * sizeof(double));
                        if (loss < best_kl_divergence) {
                            best_kl_divergence = loss;
                            best_iter = iter;
                        }
                    }
                }
        }

        free(global_best_snapshot);
        
        if (config->verbose && (iter + 1) % 50 == 0) {
            printf("Iteration %d: KL divergence = %.6f\n",
                   iter + 1, swarm->global_best_fitness);
        }
    }
    
    memcpy(result->embedding, swarm->global_best,
           n_samples * config->n_components * sizeof(double));
    result->kl_divergence = best_kl_divergence;
    result->n_iter = best_iter + 1;
    
    free(distances);
    free(P);
    free_swarm(swarm);
    
    return result;
}

void tsne_pso_free_result(tsne_pso_result* result) {
    if (result) {
        free(result->embedding);
        free(result);
    }
}

const char* tsne_pso_version(void) {
    return TSNE_PSO_VERSION;
} 