// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_CUDA4DNN_CSL_CUBLAS_HPP
#define OPENCV_DNN_CUDA4DNN_CSL_CUBLAS_HPP

#include <opencv2/dnn/csl/cublas.hpp>

#include "pointer.hpp"

#include <cstddef>
#include <type_traits>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl { namespace cublas {

    /** @brief GEMM for row-major matrices
     *
     * \f$ C = \alpha AB + \beta C \f$
     *
     * @tparam          T           matrix element type (must be `float` or `double`)
     *
     * @param           handle      valid cuBLAS Handle
     * @param           transa      use transposed matrix of A for computation
     * @param           transb      use transposed matrix of B for computation
     * @param           rows_c      number of rows in C
     * @param           cols_c      number of columns in C
     * @param           common_dim  common dimension of A and B (columns of A or rows of B)
     * @param           alpha       scale factor for AB
     * @param[in]       A           pointer to row-major matrix A in device memory
     * @param           lda         leading dimension of matrix A
     * @param[in]       B           pointer to row-major matrix B in device memory
     * @param           ldb         leading dimension of matrix B
     * @param           beta        scale factor for C
     * @param[in,out]   C           pointer to row-major matrix C in device memory
     * @param           ldc         leading dimension of matrix C
     *
     * Exception Guarantee: Basic
     */
    template <class T>
    typename std::enable_if<std::is_same<T, float>::value || std::is_same<T, double>::value, void>
    ::type gemm(Handle handle,
        bool transa, bool transb,
        std::size_t rows_c, std::size_t cols_c, std::size_t common_dim,
        T alpha, const DevicePtr<const T> A, std::size_t lda,
        const DevicePtr<const T> B, std::size_t ldb,
        T beta, const DevicePtr<T> C, std::size_t ldc);

    /** @brief strided batched GEMM for row-major matrices
     *
     * \f$ C = \alpha AB + \beta C \f$
     *
     * @tparam          T           matrix element type (must be `float` or `double`)
     *
     * @param           handle      valid cuBLAS Handle
     * @param           transa      use transposed matrix of A for computation
     * @param           transb      use transposed matrix of B for computation
     * @param           rows_c      number of rows in C
     * @param           cols_c      number of columns in C
     * @param           common_dim  common dimension of A and B (columns of A or rows of B)
     * @param           alpha       scale factor for AB
     * @param[in]       A           pointer to row-major matrix A in device memory
     * @param           lda         leading dimension of matrix A
     * @param           strideA     stride for adjacent batch A matrixes (in bytes)
     * @param[in]       B           pointer to row-major matrix B in device memory
     * @param           ldb         leading dimension of matrix B
     * @param           strideB     stride for adjacent batch B matrixes (in bytes)
     * @param           beta        scale factor for C
     * @param[in,out]   C           pointer to row-major matrix C in device memory
     * @param           ldc         leading dimension of matrix C
     * @param           strideC     stride for adajcent batch result matrixes (in bytes)
     *
     * Exception Guarantee: Basic
     */
    template <class T>
    typename std::enable_if<std::is_same<T, float>::value || std::is_same<T, double>::value, void>
    ::type gemmStridedBatched(Handle handle,
        bool transa, bool transb,
        std::size_t rows_c, std::size_t cols_c, std::size_t common_dim,
        T alpha, const DevicePtr<const T> A, std::size_t lda, std::size_t strideA,
        const DevicePtr<const T> B, std::size_t ldb, std::size_t strideB,
        T beta, const DevicePtr<T> C, std::size_t ldc, std::size_t strideC,
        std::size_t batchCount);

    /** @brief GEAM for column-major matrices
     *
     * \f$ C = \alpha A + \beta B \f$
     *
     * @tparam      T       matrix element type (must be `float` or `double`)
     *
     * @param       handle  valid cuBLAS Handle
     * @param       transa  use transposed matrix of A for computation
     * @param       transb  use transposed matrix of B for computation
     * @param       rows    number of rows in A/B/C
     * @param       cols    number of columns in A/B/C
     * @param       alpha   scale factor for A
     * @param[in]   A       pointer to column-major matrix A in device memory
     * @param       lda     leading dimension of matrix A
     * @param       beta    scale factor for C
     * @param[in]   B       pointer to column-major matrix B in device memory
     * @param       ldb     leading dimension of matrix B
     * @param[out]  C       pointer to column-major matrix C in device memory
     * @param       ldc     leading dimension of matrix C
     *
     * Exception Guarantee: Basic
     */
    template <class T>
    typename std::enable_if<std::is_same<T, float>::value || std::is_same<T, double>::value, void>
    ::type geam(Handle handle,
        bool transa, bool transb,
        std::size_t rows, std::size_t cols,
        T alpha, const DevicePtr<const T> A, std::size_t lda,
        T beta, const DevicePtr<const T> B, std::size_t ldb,
        const DevicePtr<T> C, std::size_t ldc);

}}}}} /* cv::dnn::cuda4dnn::csl::cublas */

#endif /* OPENCV_DNN_CUDA4DNN_CSL_CUBLAS_HPP */
