# Find NumPy
# This module defines:
# NUMPY_FOUND - system has numpy
# NUMPY_INCLUDE_DIRS - the numpy include directory
# NUMPY_VERSION - numpy version

if(NOT NUMPY_FOUND)
if(NOT PYTHON_EXECUTABLE)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
endif()

execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c
    "import numpy as np; print(np.__version__); print(np.get_include());"
    RESULT_VARIABLE _NUMPY_SEARCH_SUCCESS
    OUTPUT_VARIABLE _NUMPY_VALUES_OUTPUT
    ERROR_VARIABLE _NUMPY_ERROR_VALUE
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(_NUMPY_SEARCH_SUCCESS MATCHES 0)
    string(REGEX REPLACE ";" "\\\\;" _NUMPY_VALUES ${_NUMPY_VALUES_OUTPUT})
    string(REGEX REPLACE "\n" ";" _NUMPY_VALUES ${_NUMPY_VALUES})
    list(GET _NUMPY_VALUES 0 NUMPY_VERSION)
    list(GET _NUMPY_VALUES 1 NUMPY_INCLUDE_DIRS)

    string(REGEX MATCH "^[0-9]+\\.[0-9]+\\.[0-9]+" _VER_CHECK "${NUMPY_VERSION}")
    if("${_VER_CHECK}" STREQUAL "")
        # The output from Python was unexpected. Raise an error always
        # here, because we found NumPy, but it appears to be corrupted somehow.
        message(FATAL_ERROR "Requested version and include location from NumPy, got instead:\n${_NUMPY_VALUES_OUTPUT}\n")
        return()
    endif()

    set(NUMPY_FOUND TRUE)
    message(STATUS "NumPy ver. ${NUMPY_VERSION} found (include: ${NUMPY_INCLUDE_DIRS})")
else()
    message(STATUS "NumPy import failure:\n${_NUMPY_ERROR_VALUE}")
endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NumPy
REQUIRED_VARS NUMPY_INCLUDE_DIRS
VERSION_VAR NUMPY_VERSION
)