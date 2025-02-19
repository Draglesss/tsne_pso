# Find GTest package
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

enable_testing()

add_executable(test_tsne_pso unit/test_tsne_pso.cpp)

target_link_libraries(test_tsne_pso PRIVATE
    tsne_pso_core
    GTest::GTest
    GTest::Main
    OpenMP::OpenMP_CXX
)

target_include_directories(test_tsne_pso PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/cpp
)

if(MSVC)
    target_compile_options(test_tsne_pso PRIVATE /O2 /openmp)
else()
    target_compile_options(test_tsne_pso PRIVATE -O3 -march=native -fopenmp)
endif()

add_test(NAME test_tsne_pso COMMAND test_tsne_pso) 