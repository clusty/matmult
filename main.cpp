#include <chrono>
#include <iostream>
#include <memory>
#include <stdexcept> // Required for std::out_of_range and std::invalid_argument
#include <functional>
#include <algorithm>
#include <omp.h>

struct Matrix {
    std::unique_ptr<float[]> data;
    size_t rows;
    size_t cols;

    Matrix(size_t rows, size_t cols) : rows(rows), cols(cols), data(new float[rows * cols]) {}

    [[nodiscard]] float& at(size_t row, size_t col) {
        return data[row * cols + col];
    }

    [[nodiscard]] const float& at(size_t row, size_t col) const {
        return data[row * cols + col];
    }
};

void reset_matrix(Matrix& m) {
    for (size_t i = 0; i < m.rows * m.cols; ++i) {
        m.data[i] = 0;
    }
}

// Function to perform naive matrix multiplication
void naive_matrix_mult(const Matrix& a, const Matrix& b, Matrix& c) {
    // Validate matrix dimensions for multiplication
    if (a.cols != b.rows) {
        throw std::invalid_argument("Matrix dimensions mismatch: a.cols must equal b.rows");
    }
    if (c.rows != a.rows || c.cols != b.cols) {
        throw std::invalid_argument("Result matrix c has incorrect dimensions for multiplication");
    }

    for (size_t i = 0; i < a.rows; ++i) {
        for (size_t j = 0; j < b.cols; ++j) {
            c.at(i, j) = 0;
            for (size_t k = 0; k < a.cols; ++k) {
                c.at(i, j) += a.at(i, k) * b.at(k, j);
            }
        }
    }
}

// Function to perform stride-optimized matrix multiplication
void stride_optimized_matrix_mult(const Matrix& a, const Matrix& b, Matrix& c) {
    // Validate matrix dimensions for multiplication
    if (a.cols != b.rows) {
        throw std::invalid_argument("Matrix dimensions mismatch: a.cols must equal b.rows");
    }
    if (c.rows != a.rows || c.cols != b.cols) {
        throw std::invalid_argument("Result matrix c has incorrect dimensions for multiplication");
    }

    reset_matrix(c);
    for (size_t i = 0; i < a.rows; ++i) {
        for (size_t k = 0; k < a.cols; ++k) {
            for (size_t j = 0; j < b.cols; ++j) {
                c.at(i, j) += a.at(i, k) * b.at(k, j);
            }
        }
    }
}

void tiled_matrix_mult(const Matrix& a, const Matrix& b, Matrix& c) {
    const size_t TILE_SIZE = 80;

    // Macro-kernel: Traverse the tiles in i-k-j order
    for (size_t i = 0; i < a.rows; i += TILE_SIZE) {
        size_t i_max = std::min(i + TILE_SIZE, a.rows);

        for (size_t k = 0; k < a.cols; k += TILE_SIZE) {
            size_t k_max = std::min(k + TILE_SIZE, a.cols);

            for (size_t j = 0; j < b.cols; j += TILE_SIZE) {
                size_t j_max = std::min(j + TILE_SIZE, b.cols);

                // Micro-kernel: Process the tile in stride-optimized ii-kk-jj order
                for (size_t ii = i; ii < i_max; ++ii) {
                    for (size_t kk = k; kk < k_max; ++kk) {

                        // Hoist the constant to prevent memory fetches in the inner loop
                        int a_ik = a.at(ii, kk);

                        // Tell the compiler and OpenMP runtime to unroll the continuous row loop
                        #pragma omp unroll
                        for (size_t jj = j; jj < j_max; ++jj) {
                            c.at(ii, jj) += a_ik * b.at(kk, jj);
                        }
                    }
                }
            }
        }
    }
}

// Single-Threaded, Packed, Auto-Vectorized Matrix Multiplication
void single_threaded_packed_matrix_mult(const Matrix& a, const Matrix& b, Matrix& c) {
    if (a.cols != b.rows) {
        throw std::invalid_argument("Matrix dimensions mismatch");
    }

    // Cache-friendly block dimensions (optimized for L2/L3 cache tiers)
    const size_t TILE_M = 96;
    const size_t TILE_N = 96;
    const size_t TILE_K = 96;

    const size_t a_rows = a.rows;
    const size_t a_cols = a.cols;
    const size_t b_cols = b.cols;

    const float* a_ptr = a.data.get();
    const float* b_ptr = b.data.get();
    float* c_ptr = c.data.get();

    // Initialize entire result matrix to zero
    std::fill_n(c_ptr, a_rows * b_cols, 0.0f);

    // Pre-allocate temporary packing buffers once outside the loop to avoid reallocations
    std::vector<float> b_packed(TILE_K * TILE_N, 0.0f);
    std::vector<float> c_tile(TILE_M * TILE_N, 0.0f);

    // Single-threaded block loops
    for (size_t i = 0; i < a_rows; i += TILE_M) {
        size_t i_max = std::min(i + TILE_M, a_rows);
        size_t height_i = i_max - i;

        for (size_t j = 0; j < b_cols; j += TILE_N) {
            size_t j_max = std::min(j + TILE_N, b_cols);
            size_t width_j = j_max - j;

            // Clear local tile workspace
            std::fill(c_tile.begin(), c_tile.begin() + height_i * width_j, 0.0f);

            for (size_t k = 0; k < a_cols; k += TILE_K) {
                size_t k_max = std::min(k + TILE_K, a_cols);
                size_t width_k = k_max - k;

                // --- 1. PANEL PACKING B ---
                // Copy non-contiguous source rows into a 100% linear contiguous buffer
                for (size_t kk = 0; kk < width_k; ++kk) {
                    size_t global_kk = k + kk;
                    const float* b_src_row = b_ptr + global_kk * b_cols + j;
                    float* b_dst_row = b_packed.data() + kk * TILE_N;

                    for (size_t jj = 0; jj < width_j; ++jj) {
                        b_dst_row[jj] = b_src_row[jj];
                    }
                }

                // --- 2. MICRO-KERNEL COMPUTATION ---
                for (size_t ii = 0; ii < height_i; ++ii) {
                    size_t global_ii = i + ii;
                    const float* a_row = a_ptr + global_ii * a_cols;
                    float* c_row_tile = c_tile.data() + ii * TILE_N;

                    for (size_t kk = 0; kk < width_k; ++kk) {
                        float a_val = a_row[k + kk];
                        const float* b_packed_row = b_packed.data() + kk * TILE_N;

                        // Instruct compiler to auto-vectorize this contiguous loop (AVX-512)
                        #pragma omp simd
                        for (size_t jj = 0; jj < width_j; ++jj) {
                            c_row_tile[jj] += a_val * b_packed_row[jj];
                        }
                    }
                }
            }

            // --- 3. WRITE BACK TO GLOBAL C ---
            for (size_t ii = 0; ii < height_i; ++ii) {
                size_t global_ii = i + ii;
                float* c_global_row = c_ptr + global_ii * b_cols + j;
                const float* c_row_tile = c_tile.data() + ii * TILE_N;

                for (size_t jj = 0; jj < width_j; ++jj) {
                    c_global_row[jj] += c_row_tile[jj];
                }
            }
        }
    }
}

// Multi-Threaded, Packed, Auto-Vectorized Matrix Multiplication
void multithreaded_packed_matrix_mult(const Matrix& a, const Matrix& b, Matrix& c) {
    if (a.cols != b.rows) {
        throw std::invalid_argument("Matrix dimensions mismatch");
    }

    const size_t TILE_M = 96;
    const size_t TILE_N = 96;
    const size_t TILE_K = 96;

    const size_t a_rows = a.rows;
    const size_t a_cols = a.cols;
    const size_t b_cols = b.cols;

    const float* a_ptr = a.data.get();
    const float* b_ptr = b.data.get();
    float* c_ptr = c.data.get();

    // Parallelize outer block rows across available CPU cores safely
    #pragma omp parallel for schedule(guided)
    for (size_t i = 0; i < a_rows; i += TILE_M) {
        size_t i_max = std::min(i + TILE_M, a_rows);
        size_t height_i = i_max - i;

        // Thread-local aligned buffers (guarantees zero race conditions)
        alignas(64) std::vector<float> b_packed(TILE_K * TILE_N, 0.0f);
        alignas(64) std::vector<float> c_tile(TILE_M * TILE_N, 0.0f);

        for (size_t j = 0; j < b_cols; j += TILE_N) {
            size_t j_max = std::min(j + TILE_N, b_cols);
            size_t width_j = j_max - j;

            // Clear local tile workspace
            std::fill(c_tile.begin(), c_tile.begin() + height_i * width_j, 0.0f);

            for (size_t k = 0; k < a_cols; k += TILE_K) {
                size_t k_max = std::min(k + TILE_K, a_cols);
                size_t width_k = k_max - k;

                // --- 1. PANEL PACKING B ---
                for (size_t kk = 0; kk < width_k; ++kk) {
                    size_t global_kk = k + kk;
                    const float* b_src_row = b_ptr + global_kk * b_cols + j;
                    float* b_dst_row = b_packed.data() + kk * TILE_N;

                    for (size_t jj = 0; jj < width_j; ++jj) {
                        b_dst_row[jj] = b_src_row[jj];
                    }
                }

                // --- 2. MICRO-KERNEL COMPUTATION ---
                for (size_t ii = 0; ii < height_i; ++ii) {
                    size_t global_ii = i + ii;
                    const float* a_row = a_ptr + global_ii * a_cols;
                    float* c_row_tile = c_tile.data() + ii * TILE_N;

                    for (size_t kk = 0; kk < width_k; ++kk) {
                        float a_val = a_row[k + kk];
                        const float* b_packed_row = b_packed.data() + kk * TILE_N;

                        #pragma omp simd
                        for (size_t jj = 0; jj < width_j; ++jj) {
                            c_row_tile[jj] += a_val * b_packed_row[jj];
                        }
                    }
                }
            }

            // --- 3. WRITE BACK TO GLOBAL C ---
            for (size_t ii = 0; ii < height_i; ++ii) {
                size_t global_ii = i + ii;
                float* c_global_row = c_ptr + global_ii * b_cols + j;
                const float* c_row_tile = c_tile.data() + ii * TILE_N;

                for (size_t jj = 0; jj < width_j; ++jj) {
                    c_global_row[jj] = c_row_tile[jj];
                }
            }
        }
    }
}

void benchmark(const std::string& name,
               const std::function<void(const Matrix&, const Matrix&, Matrix&)>& func,
               const Matrix& a, const Matrix& b, Matrix& c) {
    reset_matrix(c);
    auto start = std::chrono::high_resolution_clock::now();
    func(a, b, c);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << name << " completed in " << duration.count() << " seconds." << std::endl;
    std::cout << "Result c[0][0]: " << c.at(0, 0) << std::endl;
}

int main() {

    #pragma omp parallel
    {
    #pragma omp single
        std::cout << "Active OpenMP threads: " << omp_get_num_threads() << std::endl;
    }
    // Matrix dimensions
    const size_t N = 4000;

    // Initialize matrices a and b
    Matrix a(N, N);
    Matrix b(N, N);
    Matrix c(N, N);

    // Initialize matrices a and b with some values
    // Initialize matrices a and b with bounded values to prevent overflow
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            a.at(i, j) = static_cast<int>(i + j) % 100;
            b.at(i, j) = static_cast<int>(i - j) % 100;
        }
    }
    //benchmark("Naive matrix multiplication", naive_matrix_mult, a, b, c);
    //benchmark("Stride-optimized matrix multiplication", stride_optimized_matrix_mult, a, b, c);
    //benchmark("Tiled matrix multiplication", tiled_matrix_mult, a, b, c);
    benchmark("Tiled matrix multiplication copy", single_threaded_packed_matrix_mult, a, b, c);
    //benchmark("Tiled matrix multiplication copy multi-threaded", multithreaded_packed_matrix_mult, a, b, c);

    return 0;
}
