#include <gtest/gtest.h>
#include "tsne_pso.hpp"
#include <vector>
#include <random>
#include <chrono>

using namespace tsne_pso;

class TSNEPSOTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Generate random test data
        std::mt19937 gen(42);
        std::normal_distribution<double> dist(0.0, 1.0);
        
        n_samples = 100;
        n_features = 10;
        test_data.resize(n_samples, std::vector<double>(n_features));
        
        for (auto& sample : test_data) {
            for (auto& feature : sample) {
                feature = dist(gen);
            }
        }
    }
    
    std::vector<std::vector<double>> test_data;
    size_t n_samples;
    size_t n_features;
};

// Test basic fitting functionality
TEST_F(TSNEPSOTest, BasicFit) {
    TSNE_PSO tsne(2, 30.0, 50, 250, 200.0, 12.0, 42, 1, false);
    
    auto embedding = tsne.fit_transform(test_data);
    
    ASSERT_EQ(embedding.size(), n_samples);
    ASSERT_EQ(embedding[0].size(), 2);
    
    // Check that embedding is not all zeros
    bool all_zeros = true;
    for (const auto& sample : embedding) {
        for (const auto& coord : sample) {
            if (std::abs(coord) > 1e-10) {
                all_zeros = false;
                break;
            }
        }
        if (!all_zeros) break;
    }
    EXPECT_FALSE(all_zeros);
}

// Test that different random seeds produce different results
TEST_F(TSNEPSOTest, RandomSeedEffect) {
    TSNE_PSO tsne1(2, 30.0, 50, 100, 200.0, 12.0, 42);
    TSNE_PSO tsne2(2, 30.0, 50, 100, 200.0, 12.0, 43);
    
    auto embedding1 = tsne1.fit_transform(test_data);
    auto embedding2 = tsne2.fit_transform(test_data);
    
    // Check that results are different
    bool all_same = true;
    for (size_t i = 0; i < n_samples; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            if (std::abs(embedding1[i][j] - embedding2[i][j]) > 1e-10) {
                all_same = false;
                break;
            }
        }
        if (!all_same) break;
    }
    EXPECT_FALSE(all_same);
}

// Test that output dimensions are correct
TEST_F(TSNEPSOTest, OutputDimensions) {
    std::vector<int> test_dimensions = {2, 3, 5};
    
    for (int dim : test_dimensions) {
        TSNE_PSO tsne(dim);
        auto embedding = tsne.fit_transform(test_data);
        
        EXPECT_EQ(embedding.size(), n_samples);
        EXPECT_EQ(embedding[0].size(), dim);
    }
}

// Test behavior with different perplexity values
TEST_F(TSNEPSOTest, PerplexityEffect) {
    std::vector<double> perplexities = {5.0, 30.0, 50.0};
    std::vector<std::vector<std::vector<double>>> embeddings;
    
    for (double perp : perplexities) {
        TSNE_PSO tsne(2, perp);
        embeddings.push_back(tsne.fit_transform(test_data));
    }
    
    // Check that different perplexities give different results
    for (size_t i = 1; i < embeddings.size(); ++i) {
        bool all_same = true;
        for (size_t j = 0; j < n_samples; ++j) {
            for (size_t k = 0; k < 2; ++k) {
                if (std::abs(embeddings[i][j][k] - embeddings[i-1][j][k]) > 1e-10) {
                    all_same = false;
                    break;
                }
            }
            if (!all_same) break;
        }
        EXPECT_FALSE(all_same);
    }
}

// Test parallel execution
TEST_F(TSNEPSOTest, ParallelExecution) {
    std::vector<int> thread_counts = {1, 2, 4};
    std::vector<double> execution_times;
    
    for (int threads : thread_counts) {
        TSNE_PSO tsne(2, 30.0, 50, 100, 200.0, 12.0, 42, threads);
        
        auto start = std::chrono::high_resolution_clock::now();
        tsne.fit_transform(test_data);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        execution_times.push_back(duration);
    }
    
    // Check that more threads generally means faster execution
    if (execution_times.size() >= 2) {
        double avg_speedup = 0.0;
        for (size_t i = 1; i < execution_times.size(); ++i) {
            avg_speedup += execution_times[0] / execution_times[i];
        }
        avg_speedup /= (execution_times.size() - 1);
        
        // Expect some speedup with more threads
        EXPECT_GT(avg_speedup, 1.0);
    }
}

// Test version string
TEST_F(TSNEPSOTest, VersionString) {
    std::string version = TSNE_PSO::version();
    EXPECT_FALSE(version.empty());
} 