#!/bin/bash

# Set strict error handling
set -euo pipefail
IFS=$'\n\t'

# Define colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status messages
print_status() {
    echo -e "${GREEN}[✓] $1${NC}"
}

# Function to print warning messages
print_warning() {
    echo -e "${YELLOW}[!] $1${NC}"
}

# Function to print error messages
print_error() {
    echo -e "${RED}[✗] $1${NC}"
}

# Function to check if a command exists
check_command() {
    if ! command -v "$1" &> /dev/null; then
        print_error "$1 is required but not installed."
        exit 1
    fi
}

# Function to check Python version
check_python_version() {
    local required_version="3.8"
    local python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    
    if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
        print_error "Python version must be >= $required_version (found $python_version)"
        exit 1
    fi
}

# Print banner
echo "================================================"
echo "   TSNE-PSO Project Setup and Test Script"
echo "================================================"
echo

# Check required tools
print_status "Checking required tools..."
check_command python3
check_command pip
check_command cmake
check_command git

# Check Python version
check_python_version
print_status "Python version check passed"

# Create virtual environment if not in one
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    print_warning "No virtual environment detected"
    if command -v python3 -m venv &> /dev/null; then
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

# Install the package
print_status "Installing the package..."
cd ..
pip install -e .

# Run tests
print_status "Running tests..."
python -m pytest tsne_pso/tests/ -v

print_status "Setup completed successfully!"
echo
echo "You can now use TSNE-PSO in your Python environment."
echo
print_warning "Note: If you created a new virtual environment, remember to activate it with:"
echo "source venv/bin/activate" 