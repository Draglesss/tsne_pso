 #!/bin/bash

# Exit on error
set -e

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

echo "Setting up development environment for TSNE-PSO..."

# Check for Python
if ! command_exists python3; then
    echo "Python 3 not found. Please install Python 3 first."
    exit 1
fi

# Check for pip
if ! command_exists pip; then
    echo "pip not found. Please install pip first."
    exit 1
fi

# Check for CMake
if ! command_exists cmake; then
    echo "CMake not found. Please install CMake first."
    exit 1
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install numpy cython pytest matplotlib scikit-learn

# Create build directory
echo "Creating build directory..."
mkdir -p build
cd build

# Configure CMake
echo "Configuring CMake..."
cmake ..

# Build
echo "Building..."
cmake --build .

echo "Development environment setup complete!"
echo "You can now build the project by running:"
echo "  cd build"
echo "  make"
# Check Python version
check_python_version
print_status "Python version check passed"

# Create virtual environment if not in one
if [[ -z "${VIRTUAL_ENV}" ]]; then
    print_warning "No virtual environment detected"
    if command_exists python3 -m venv; then
        print_status "Creating virtual environment..."
        python3 -m venv venv
        source venv/bin/activate
    else
        print_warning "python3-venv not found, continuing with system Python"
    fi
fi

# Upgrade pip
print_status "Upgrading pip..."
python3 -m pip install --upgrade pip

# Install Python dependencies
print_status "Installing Python dependencies..."
python3 -m pip install numpy cython pytest matplotlib scikit-learn build wheel setuptools

# Clean previous build if exists
if [ -d "build" ]; then
    print_status "Cleaning previous build..."
    rm -rf build
fi

# Create build directory
print_status "Creating build directory..."
mkdir -p build
cd build

# Configure CMake
print_status "Configuring CMake..."
cmake -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_PYTHON_BINDINGS=ON \
      -DBUILD_TESTS=ON \
      -DENABLE_OPENMP=ON \
      ..

# Build
print_status "Building..."
cmake --build . -- -j$(nproc)

# Run tests if they were built
if [ -f "tests/tsne_pso_test" ]; then
    print_status "Running tests..."
    ctest --output-on-failure
fi

cd ..

print_status "Development environment setup complete!"
echo
echo "You can now:"
echo "1. Build the project:        cd build && make"
echo "2. Run tests:                cd build && ctest"
echo "3. Install the package:      pip install ."
echo
print_warning "Note: Make sure to activate the virtual environment (if created) with:"
echo "source venv/bin/activate"
    print_status "Using virtual environment: ${VIRTUAL_ENV}"
elif [[ -n "${CONDA_DEFAULT_ENV}" ]]; then
    print_status "Using conda environment: ${CONDA_DEFAULT_ENV}"
else
    print_warning "Note: Make sure to activate the virtual environment with:"
    echo "source venv/bin/activate"
fi