# Find Cython
find_program(CYTHON_EXECUTABLE NAMES cython cython.bat cython3)
if(NOT CYTHON_EXECUTABLE)
    message(FATAL_ERROR "Cython not found. Please install Cython first.")
endif()

# Generate C source from Cython
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/tsne_pso.c
    COMMAND ${CYTHON_EXECUTABLE} 
    ARGS --output-file ${CMAKE_CURRENT_BINARY_DIR}/tsne_pso.c
         ${CMAKE_CURRENT_SOURCE_DIR}/tsne_pso.pyx
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tsne_pso.pyx
    COMMENT "Cythonizing ${CMAKE_CURRENT_SOURCE_DIR}/tsne_pso.pyx"
)

# Create Python extension module
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

add_library(tsne_pso_python MODULE
    ${CMAKE_CURRENT_BINARY_DIR}/tsne_pso.c
)

target_link_libraries(tsne_pso_python PRIVATE
    tsne_pso_core
    Python3::Python
    Python3::NumPy
    OpenMP::OpenMP_CXX
)

target_include_directories(tsne_pso_python PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
)

set_target_properties(tsne_pso_python PROPERTIES
    PREFIX ""
    OUTPUT_NAME "tsne_pso"
    SUFFIX ".so"
)

# Installation
install(TARGETS tsne_pso_python
    LIBRARY DESTINATION ${Python3_SITEARCH}/tsne_pso
)