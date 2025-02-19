# cython: language_level=3
# distutils: language = c
# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp

import numpy as np
cimport numpy as np
np.import_array()
from libc.stdlib cimport malloc, free

# Import the C declarations
cdef extern from "tsne_pso.h":
    ctypedef struct tsne_pso_config:
        int n_components
        double perplexity
        int n_particles
        int max_iter
        double learning_rate
        double early_exaggeration
        double min_gain
        int random_seed
        int n_threads
        double theta
        int verbose

    ctypedef struct tsne_pso_result:
        double* embedding
        size_t n_samples
        size_t n_components
        double kl_divergence
        int n_iter

    tsne_pso_config tsne_pso_init_config()
    tsne_pso_result* tsne_pso_fit(const double* X, size_t n_samples, size_t n_features,
                                 const tsne_pso_config* config)
    void tsne_pso_free_result(tsne_pso_result* result)
    const char* tsne_pso_version()

cdef class TSNE_PSO:
    """
    t-SNE with Particle Swarm Optimization
    
    Parameters
    ----------
    n_components : int, default=2
        Dimension of the embedded space.
    perplexity : float, default=30.0
        The perplexity is related to the number of nearest neighbors.
    n_particles : int, default=100
        Number of particles in the PSO swarm.
    max_iter : int, default=1000
        Maximum number of iterations.
    learning_rate : float, default=200.0
        Learning rate for t-SNE.
    early_exaggeration : float, default=12.0
        Early exaggeration factor.
    random_state : int, default=None
        Random seed for reproducibility.
    n_jobs : int, default=None
        Number of parallel jobs. None means using all processors.
    verbose : bool, default=False
        Whether to print progress messages.
    """
    cdef tsne_pso_config _config

    def __init__(self, n_components=2, perplexity=30.0, n_particles=100,
                 max_iter=1000, learning_rate=200.0, early_exaggeration=12.0,
                 random_state=None, n_jobs=None, verbose=False):
        self._config = tsne_pso_init_config()
        self._config.n_components = n_components
        self._config.perplexity = perplexity
        self._config.n_particles = n_particles
        self._config.max_iter = max_iter
        self._config.learning_rate = learning_rate
        self._config.early_exaggeration = early_exaggeration
        self._config.random_seed = random_state if random_state is not None else 42
        self._config.n_threads = n_jobs if n_jobs is not None else self._config.n_threads
        self._config.verbose = 1 if verbose else 0

    def fit_transform(self, X):
        """
        Fit t-SNE to X and return the embedded coordinates.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        embedding : array-like of shape (n_samples, n_components)
            Embedded coordinates.
        """
        # Convert input to numpy array
        X = np.asarray(X, dtype=np.float64, order='C')
        
        # Input validation
        if X.size == 0:
            raise ValueError("Empty input array")
        
        if X.ndim != 2:
            raise ValueError("Expected 2D array, got %dD" % X.ndim)
        
        # Get dimensions
        cdef size_t n_samples = X.shape[0]
        cdef size_t n_features = X.shape[1]
        
        # Get pointer to data
        cdef double* X_ptr = <double*>np.PyArray_DATA(X)
        
        # Call C function
        cdef tsne_pso_result* result = tsne_pso_fit(X_ptr, n_samples, n_features, &self._config)
        if result == NULL:
            raise RuntimeError("t-SNE fitting failed")
        
        # Convert result to numpy array
        cdef np.ndarray[double, ndim=2] embedding = np.zeros((n_samples, self._config.n_components), dtype=np.float64)
        cdef double* result_ptr = result.embedding
        cdef size_t i, j
        
        for i in range(n_samples):
            for j in range(self._config.n_components):
                embedding[i, j] = result_ptr[i * self._config.n_components + j]
        
        # Clean up
        tsne_pso_free_result(result)
        
        return embedding

    @staticmethod
    def version():
        """Return the version string of the C library."""
        return tsne_pso_version().decode('utf-8') 