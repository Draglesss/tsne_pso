import pytest
import numpy as np

@pytest.fixture
def random_data():
    rng = np.random.RandomState(42)
    return rng.randn(100, 10)

@pytest.fixture
def tsne_instance():
    from tsne_pso import TSNE_PSO
    return TSNE_PSO(
        n_components=2,
        perplexity=30.0,
        n_particles=50,
        max_iter=250,
        random_state=42,
        verbose=False
    ) 