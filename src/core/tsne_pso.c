#include "tsne_pso.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <float.h>
#include <time.h>

#define TSNE_PSO_VERSION "1.0.0"
#define EPS 1e-7
#define MACHINE_EPSILON DBL_EPSILON

// Initialize default configuration
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
    config.n_threads = omp_get_max_threads();
    config.theta = 0.5;
    config.verbose = 0;
    return config;
}

// Internal structures for PSO
typedef struct {
    double* position;
    double* velocity;
    double* best_position;
    double best_fitness;
    double* gradient;
    double* gains;
} Particle;

typedef struct {
    double* global_best;
    double global_best_fitness;
    Particle* particles;
    size_t n_particles;
    size_t n_dimensions;
    double w;           // Inertia weight
    double c1;          // Cognitive parameter
    double c2;          // Social parameter
    double v_max;       // Maximum velocity
} PSO_Swarm;

// Random number generation
static double rand_uniform(void) {
    return (double)rand() / RAND_MAX;
}

static double rand_normal(void) {
    double u1 = rand_uniform();
    double u2 = rand_uniform();
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

// Memory allocation with error checking
static void* safe_malloc(size_t size) {
    void* ptr = malloc(size);
    if (!ptr) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

// Initialize particle
static void init_particle(Particle* particle, size_t n_dimensions) {
    particle->position = (double*)safe_malloc(n_dimensions * sizeof(double));
    particle->velocity = (double*)safe_malloc(n_dimensions * sizeof(double));
    particle->best_position = (double*)safe_malloc(n_dimensions * sizeof(double));
    particle->gradient = (double*)safe_malloc(n_dimensions * sizeof(double));
    particle->gains = (double*)safe_malloc(n_dimensions * sizeof(double));
    
    // Initialize with random values from normal distribution
    for (size_t i = 0; i < n_dimensions; i++) {
        particle->position[i] = rand_normal() * 1e-4;
        particle->velocity[i] = 0.0;
        particle->best_position[i] = particle->position[i];
        particle->gains[i] = 1.0;
    }
    particle->best_fitness = INFINITY;
}

// Initialize PSO swarm
static PSO_Swarm* init_swarm(size_t n_particles, size_t n_dimensions) {
    PSO_Swarm* swarm = (PSO_Swarm*)safe_malloc(sizeof(PSO_Swarm));
    swarm->n_particles = n_particles;
    swarm->n_dimensions = n_dimensions;
    swarm->global_best = (double*)safe_malloc(n_dimensions * sizeof(double));
    swarm->particles = (Particle*)safe_malloc(n_particles * sizeof(Particle));
    
    // Set PSO parameters
    swarm->w = 0.9;    // Inertia weight
    swarm->c1 = 2.0;   // Cognitive parameter
    swarm->c2 = 2.0;   // Social parameter
    swarm->v_max = 5.0; // Maximum velocity
    swarm->global_best_fitness = INFINITY;
    
    // Initialize particles
    for (size_t i = 0; i < n_particles; i++) {
        init_particle(&swarm->particles[i], n_dimensions);
    }
    
    return swarm;
}

// Compute pairwise distances
static void compute_pairwise_distances(const double* X, size_t n_samples, size_t n_features,
                                     double* distances) {
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < n_samples; i++) {
        for (size_t j = i + 1; j < n_samples; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < n_features; k++) {
                double diff = X[i * n_features + k] - X[j * n_features + k];
                sum += diff * diff;
            }
            distances[i * n_samples + j] = sqrt(sum);
            distances[j * n_samples + i] = distances[i * n_samples + j];
        }
    }
}

// Compute joint probabilities (P_ij)
static void compute_joint_probabilities(const double* distances, size_t n_samples,
                                     double perplexity, double* P) {
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < n_samples; i++) {
        double beta = 1.0;
        double beta_min = -INFINITY;
        double beta_max = INFINITY;
        
        // Binary search for beta
        for (int iter = 0; iter < 50; iter++) {
            double sum_P = 0.0;
            double sum_dP = 0.0;
            double H = 0.0;
            
            // Compute Gaussian kernel row
            for (size_t j = 0; j < n_samples; j++) {
                if (i != j) {
                    double sqdist = distances[i * n_samples + j];
                    double P_ij = exp(-beta * sqdist);
                    P[i * n_samples + j] = P_ij;
                    sum_P += P_ij;
                    sum_dP += sqdist * P_ij;
                }
            }
            
            // Normalize row
            for (size_t j = 0; j < n_samples; j++) {
                if (i != j) {
                    P[i * n_samples + j] /= sum_P;
                    H -= P[i * n_samples + j] * log(P[i * n_samples + j] + EPS);
                }
            }
            
            // Update beta
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
    
    // Symmetrize P and normalize
    double sum_P = 0.0;
    for (size_t i = 0; i < n_samples; i++) {
        for (size_t j = 0; j < i; j++) {
            P[i * n_samples + j] = (P[i * n_samples + j] + P[j * n_samples + i]) / (2 * n_samples);
            P[j * n_samples + i] = P[i * n_samples + j];
            sum_P += 2 * P[i * n_samples + j];
        }
    }
}

// Compute Q distribution and gradients
static double compute_q_and_gradients(PSO_Swarm* swarm, const double* P,
                                    size_t n_samples, size_t n_components,
                                    double* Q, Particle* particle) {
    double sum_Q = 0.0;
    size_t n_elements = n_samples * n_samples;
    
    // Compute Q distribution
    #pragma omp parallel for reduction(+:sum_Q)
    for (size_t i = 0; i < n_samples; i++) {
        for (size_t j = i + 1; j < n_samples; j++) {
            double q_ij = 0.0;
            for (size_t d = 0; d < n_components; d++) {
                double diff = particle->position[i * n_components + d] -
                            particle->position[j * n_components + d];
                q_ij += diff * diff;
            }
            q_ij = 1.0 / (1.0 + q_ij);  // t-distribution
            Q[i * n_samples + j] = q_ij;
            Q[j * n_samples + i] = q_ij;
            sum_Q += 2 * q_ij;
        }
    }
    
    // Compute gradients
    double kl_divergence = 0.0;
    memset(particle->gradient, 0, n_samples * n_components * sizeof(double));
    
    #pragma omp parallel for reduction(+:kl_divergence)
    for (size_t i = 0; i < n_samples; i++) {
        for (size_t j = 0; j < n_samples; j++) {
            if (i != j) {
                double p_ij = P[i * n_samples + j];
                double q_ij = Q[i * n_samples + j] / sum_Q;
                
                // Accumulate KL divergence
                kl_divergence += p_ij * log((p_ij + EPS) / (q_ij + EPS));
                
                // Compute gradient
                double grad_mult = 4 * (p_ij - q_ij * sum_Q) * q_ij;
                for (size_t d = 0; d < n_components; d++) {
                    double diff = particle->position[i * n_components + d] -
                                particle->position[j * n_components + d];
                    particle->gradient[i * n_components + d] += grad_mult * diff;
                }
            }
        }
    }
    
    return kl_divergence;
}

// Update particle position and velocity
static void update_particle(Particle* particle, const double* global_best,
                          size_t n_dimensions, const PSO_Swarm* swarm,
                          double learning_rate, double min_gain) {
    for (size_t i = 0; i < n_dimensions; i++) {
        // Update gains
        if (particle->gradient[i] * particle->velocity[i] >= 0) {
            particle->gains[i] *= 0.95;
        } else {
            particle->gains[i] += 0.05;
        }
        if (particle->gains[i] < min_gain) particle->gains[i] = min_gain;
        
        // Update velocity using PSO equation with momentum
        double cognitive = swarm->c1 * rand_uniform() * 
                         (particle->best_position[i] - particle->position[i]);
        double social = swarm->c2 * rand_uniform() * 
                       (global_best[i] - particle->position[i]);
        
        particle->velocity[i] = swarm->w * particle->velocity[i] +
                              cognitive + social -
                              particle->gains[i] * particle->gradient[i] * learning_rate;
        
        // Apply velocity clamping
        if (particle->velocity[i] > swarm->v_max)
            particle->velocity[i] = swarm->v_max;
        else if (particle->velocity[i] < -swarm->v_max)
            particle->velocity[i] = -swarm->v_max;
        
        // Update position
        particle->position[i] += particle->velocity[i];
    }
}

// Free swarm memory
static void free_swarm(PSO_Swarm* swarm) {
    if (swarm) {
        free(swarm->global_best);
        for (size_t i = 0; i < swarm->n_particles; i++) {
            free(swarm->particles[i].position);
            free(swarm->particles[i].velocity);
            free(swarm->particles[i].best_position);
            free(swarm->particles[i].gradient);
            free(swarm->particles[i].gains);
        }
        free(swarm->particles);
        free(swarm);
    }
}

// Main t-SNE with PSO optimization
tsne_pso_result* tsne_pso_fit(const double* X, size_t n_samples, size_t n_features,
                             const tsne_pso_config* config) {
    // Set random seed
    srand(config->random_seed);
    
    // Set number of threads
    omp_set_num_threads(config->n_threads);
    
    // Allocate result structure
    tsne_pso_result* result = (tsne_pso_result*)safe_malloc(sizeof(tsne_pso_result));
    result->n_samples = n_samples;
    result->n_components = config->n_components;
    result->embedding = (double*)safe_malloc(n_samples * config->n_components * sizeof(double));
    
    // Initialize distances matrix
    double* distances = (double*)safe_malloc(n_samples * n_samples * sizeof(double));
    compute_pairwise_distances(X, n_samples, n_features, distances);
    
    // Compute joint probabilities
    double* P = (double*)safe_malloc(n_samples * n_samples * sizeof(double));
    compute_joint_probabilities(distances, n_samples, config->perplexity, P);
    
    // Early exaggeration
    for (size_t i = 0; i < n_samples * n_samples; i++) {
        P[i] *= config->early_exaggeration;
    }
    
    // Initialize PSO swarm
    PSO_Swarm* swarm = init_swarm(config->n_particles, n_samples * config->n_components);
    
    // Allocate memory for Q distribution
    double* Q = (double*)safe_malloc(n_samples * n_samples * sizeof(double));
    
    // Main optimization loop
    double best_kl_divergence = INFINITY;
    int best_iter = 0;
    
    for (int iter = 0; iter < config->max_iter; iter++) {
        // Remove early exaggeration at 100 iterations
        if (iter == 100) {
            for (size_t i = 0; i < n_samples * n_samples; i++) {
                P[i] /= config->early_exaggeration;
            }
        }
        
        // Update each particle
        #pragma omp parallel for schedule(dynamic)
        for (size_t p = 0; p < swarm->n_particles; p++) {
            Particle* particle = &swarm->particles[p];
            
            // Compute Q distribution and gradients
            double kl_divergence = compute_q_and_gradients(swarm, P, n_samples,
                                                         config->n_components, Q, particle);
            
            // Update particle's best position
            #pragma omp critical
            {
                if (kl_divergence < particle->best_fitness) {
                    particle->best_fitness = kl_divergence;
                    memcpy(particle->best_position, particle->position,
                           n_samples * config->n_components * sizeof(double));
                    
                    // Update global best
                    if (kl_divergence < swarm->global_best_fitness) {
                        swarm->global_best_fitness = kl_divergence;
                        memcpy(swarm->global_best, particle->position,
                               n_samples * config->n_components * sizeof(double));
                        
                        if (kl_divergence < best_kl_divergence) {
                            best_kl_divergence = kl_divergence;
                            best_iter = iter;
                        }
                    }
                }
            }
            
            // Update particle position and velocity
            update_particle(particle, swarm->global_best,
                          n_samples * config->n_components, swarm,
                          config->learning_rate, config->min_gain);
        }
        
        // Print progress if verbose
        if (config->verbose && (iter + 1) % 50 == 0) {
            printf("Iteration %d: KL divergence = %.6f\n",
                   iter + 1, swarm->global_best_fitness);
        }
    }
    
    // Copy best solution to result
    memcpy(result->embedding, swarm->global_best,
           n_samples * config->n_components * sizeof(double));
    result->kl_divergence = best_kl_divergence;
    result->n_iter = best_iter + 1;
    
    // Clean up
    free(distances);
    free(P);
    free(Q);
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