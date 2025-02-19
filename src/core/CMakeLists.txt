# Create the core library
add_library(tsne_pso_core SHARED
    tsne_pso.c
)

# Set include directories
target_include_directories(tsne_pso_core
    PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# Set compile options
if(MSVC)
    target_compile_options(tsne_pso_core PRIVATE /O2 /openmp)
else()
    target_compile_options(tsne_pso_core PRIVATE -O3 -march=native -fopenmp)
endif()

# Link against OpenMP
target_link_libraries(tsne_pso_core
    PUBLIC
    OpenMP::OpenMP_C
    m
)

# Install library
install(TARGETS tsne_pso_core
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

set_target_properties(tsne_pso_core PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)