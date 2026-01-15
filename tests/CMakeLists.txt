# Prefer a system GTest, but fall back to FetchContent (no sudo required).
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    include(FetchContent)
    # Avoid overriding parent project compiler/linker settings.
    set(gtest_force_shared_crt OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

    # Stable, widely-packaged tag.
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    FetchContent_MakeAvailable(googletest)
endif()

add_executable(test_tsne_pso unit/test_tsne_pso.cpp)

target_link_libraries(test_tsne_pso PRIVATE
    tsne_pso_core
    GTest::gtest
    GTest::gtest_main
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