import numpy as np
import pytest
import time
from tsne_pso import TSNE_PSO

def test_basic_functionality():
    # Generate random data
    rng = np.random.RandomState(42)
    X = rng.randn(100, 10)
    
    # Initialize t-SNE
    tsne = TSNE_PSO(
        n_components=2,
        perplexity=30.0,
        n_particles=50,
        max_iter=250,
        random_state=42,
        verbose=False
    )
    
    # Fit and transform
    embedding = tsne.fit_transform(X)
    
    # Check output shape
    assert embedding.shape == (100, 2)
    
    # Check that output is not all zeros
    assert not np.allclose(embedding, 0)

def test_different_dimensions():
    rng = np.random.RandomState(42)
    X = rng.randn(50, 20)
    
    for n_components in [2, 3, 5]:
        tsne = TSNE_PSO(n_components=n_components)
        embedding = tsne.fit_transform(X)
        assert embedding.shape == (50, n_components)

def test_perplexity_effect():
    rng = np.random.RandomState(42)
    X = rng.randn(75, 15)
    
    embeddings = []
    perplexities = [5.0, 30.0, 50.0]
    
    for perp in perplexities:
        tsne = TSNE_PSO(perplexity=perp, random_state=42)
        embeddings.append(tsne.fit_transform(X))
    
    # Check that different perplexities give different results
    for i in range(1, len(embeddings)):
        assert not np.allclose(embeddings[i], embeddings[i-1])

def test_random_state():
    X = np.random.randn(50, 10)
    
    # Same random state should give same results
    tsne1 = TSNE_PSO(random_state=42)
    tsne2 = TSNE_PSO(random_state=42)
    
    embedding1 = tsne1.fit_transform(X)
    embedding2 = tsne2.fit_transform(X)
    
    assert np.allclose(embedding1, embedding2)
    
    # Different random states should give different results
    tsne3 = TSNE_PSO(random_state=43)
    embedding3 = tsne3.fit_transform(X)
    
    assert not np.allclose(embedding1, embedding3)

def test_input_validation():
    tsne = TSNE_PSO()
    
    # Test empty input
    with pytest.raises(ValueError):
        tsne.fit_transform(np.array([]))
    
    # Test inconsistent dimensions
    X = np.random.randn(10, 5)
    X = np.vstack([X, np.random.randn(1, 6)])  # Add row with different dimension
    with pytest.raises(ValueError):
        tsne.fit_transform(X)

def test_parallel_execution():
    X = np.random.randn(100, 20)
    
    # Test with different numbers of threads
    times = []
    for n_jobs in [1, 2, 4]:
        tsne = TSNE_PSO(n_jobs=n_jobs, max_iter=100)
        
        start_time = time.time()
        tsne.fit_transform(X)
        times.append(time.time() - start_time)
    
    # More threads should generally be faster
    if len(times) >= 2:
        avg_speedup = np.mean([times[0] / t for t in times[1:]])
        assert avg_speedup > 1.0

def test_version():
    tsne = TSNE_PSO()
    assert isinstance(tsne.version(), str)
    assert len(tsne.version()) > 0 