# TSNE-PSO: High-Performance t-SNE with Particle Swarm Optimization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![Build Status](https://github.com/Draglesss/tsne_pso/workflows/CI/badge.svg)](https://github.com/Draglesss/tsne_pso/actions)

A blazingly fast implementation of t-SNE (t-Distributed Stochastic Neighbor Embedding) using Particle Swarm Optimization (PSO) for dimensionality reduction, based on the research paper by Allaoui et al. This implementation combines the power of C/C++ with Python bindings for maximum performance and usability.

## 🚀 Features

- **High Performance**: Core implementation in C with SIMD optimizations
- **Parallel Processing**: Multi-threaded implementation using OpenMP
- **Python Integration**: Seamless Python bindings via Cython
- **Memory Efficient**: Optimized memory usage for large datasets
- **Enterprise Ready**: Comprehensive test suite and documentation
- **Easy to Use**: Scikit-learn compatible API

## 📋 Requirements

### System Requirements
- CMake >= 3.15
- C/C++ compiler with C11/C++17 support
- OpenMP-compatible compiler
- Python >= 3.8

### Python Dependencies
- NumPy >= 1.20.0
- SciPy >= 1.6.0
- Cython >= 0.29.0 (build only)
- pytest >= 6.0.0 (testing only)

## 🔧 Installation

### Quick Install (Recommended)

```bash
# Clone the repository
git clone https://github.com/Draglesss/tsne_pso.git
cd tsne_pso

# Run the automated setup script
./setup.sh
```

### Manual Installation

1. **Create and activate a virtual environment (recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or
   .\venv\Scripts\activate  # Windows
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Build and install the package**:
   ```bash
   mkdir build && cd build
   cmake ..
   make
   pip install -e ..
   ```

## 💻 Usage

```python
from tsne_pso import TSNE_PSO
import numpy as np

# Initialize the model
tsne = TSNE_PSO(
    n_components=2,
    perplexity=30.0,
    n_particles=100,
    max_iter=1000,
    random_state=42,
    verbose=True
)

# Generate sample data
X = np.random.randn(1000, 50)

# Fit and transform the data
X_embedded = tsne.fit_transform(X)
```

## 🔍 API Reference

### TSNE_PSO Class

```python
TSNE_PSO(
    n_components=2,
    perplexity=30.0,
    n_particles=100,
    max_iter=1000,
    learning_rate=200.0,
    early_exaggeration=12.0,
    random_state=None,
    n_jobs=None,
    verbose=False
)
```

#### Parameters:
- `n_components` (int): Dimension of the embedded space
- `perplexity` (float): The perplexity is related to the number of nearest neighbors
- `n_particles` (int): Number of particles in PSO optimization
- `max_iter` (int): Maximum number of iterations
- `learning_rate` (float): Learning rate for gradient updates
- `early_exaggeration` (float): Early exaggeration factor
- `random_state` (int): Random seed for reproducibility
- `n_jobs` (int): Number of parallel jobs
- `verbose` (bool): Whether to print progress messages

## 🧪 Testing

Run the test suite:

```bash
python -m pytest tsne_pso/tests/
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

This implementation is based on the research paper:

```bibtex
@article{allaoui2025tsne,
    title={t-SNE-PSO: Optimizing t-SNE using particle swarm optimization},
    author={Allaoui, Mebarka and Belhaouari, Samir Brahim and Hedjam, Rachid and Bouanane, Khadra and Kherfi, Mohammed Lamine},
    journal={Expert Systems with Applications},
    year={2025},
    doi={10.1016/j.eswa.2025.126398},
    publisher={Elsevier}
}
```

If you use this implementation in your research, please cite both the original paper and this implementation:

```bibtex
@software{tsne_pso2025,
    author = {Otmane Fatteh},
    title = {TSNE-PSO: High-Performance t-SNE with Particle Swarm Optimization},
    year = {2025},
    publisher = {GitHub},
    url = {https://github.com/Draglesss/tsne_pso}
}
```

## 🙏 Acknowledgments

- **Original Paper Authors**:
  - Mebarka Allaoui (Bishop's University, Canada)
  - Samir Brahim Belhaouari (Hamad Bin Khalifa University, Qatar)
  - Rachid Hedjam (Sultan Qaboos University)
  - Khadra Bouanane (Kasdi Merbah University, Algeria)
  - Mohammed Lamine Kherfi (Sultan Qaboos University & University of Quebec)