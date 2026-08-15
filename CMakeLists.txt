cmake_minimum_required(VERSION 3.20)
project(matmult)

set(CMAKE_CXX_STANDARD 26)

add_executable(matmult main.cpp)

# Use standard Clang dash flags (no MSVC slash flags)
target_compile_options(matmult PRIVATE -O3 -ffast-math -march=native -fopenmp)
target_link_options(matmult PRIVATE -O3 -ffast-math -march=native -fopenmp)

# Automatically copy LLVM's OpenMP runtime DLL to the output folder
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    get_filename_component(COMPILER_BIN_DIR ${CMAKE_CXX_COMPILER} DIRECTORY)
    find_file(LIBOMP_DLL
            NAMES libomp.dll
            PATHS "${COMPILER_BIN_DIR}/../lib" "${COMPILER_BIN_DIR}"
            NO_DEFAULT_PATH
    )
    if(LIBOMP_DLL)
        add_custom_command(TARGET matmult POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${LIBOMP_DLL}" $<TARGET_FILE_DIR:matmult>
        )
    endif()
endif()