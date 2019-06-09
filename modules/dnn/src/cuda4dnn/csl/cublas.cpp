// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "cublas.hpp"
#include "stream.hpp"
#include "pointer.hpp"

#include <cublas_v2.h>

#include <cstddef>
#include <memory>

#define CUDA4DNN_CHECK_CUBLAS(call) \
    ::cv::dnn::cuda4dnn::csl::cublas::check((call), CV_Func, __FILE__, __LINE__)

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl { namespace cublas {

    static void check(cublasStatus_t status, const char* func, const char* file, int line) {
        auto cublasGetErrorString = [](cublasStatus_t err) {
            switch (err) {
            case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
            case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
            case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
            case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
            case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
            case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
            case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
            case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
            case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
            case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
            }
            return "UNKNOWN_CUBLAS_ERROR";
        };

        if (status != CUBLAS_STATUS_SUCCESS)
            throw cuBLASException(Error::GpuApiCallError, cublasGetErrorString(status), func, file, line);
    }

    /** noncopyable cuBLAS smart handle
     *
     * UniqueHandle is a smart non-sharable wrapper for cuBLAS handle which ensures that the handle
     * is destroyed after use. The handle can be associated with a CUDA stream by specifying the
     * stream during construction. By default, the handle is associated with the default stream.
     */
    class Handle::UniqueHandle {
    public:
        UniqueHandle() { CUDA4DNN_CHECK_CUBLAS(cublasCreate(&handle)); }
        UniqueHandle(UniqueHandle&) = delete;
        UniqueHandle(UniqueHandle&& other) noexcept
            : stream(std::move(other.stream)), handle{ other.handle } {
            other.handle = nullptr;
        }

        UniqueHandle(Stream strm) : stream(std::move(strm)) {
            CUDA4DNN_CHECK_CUBLAS(cublasCreate(&handle));
            try {
                CUDA4DNN_CHECK_CUBLAS(cublasSetStream(handle, StreamAccessor::get(stream)));
            } catch (...) {
                /* cublasDestroy won't throw if a valid handle is passed */
                CUDA4DNN_CHECK_CUBLAS(cublasDestroy(handle));
                throw;
            }
        }

        ~UniqueHandle() noexcept {
            if (handle != nullptr) {
                /* cublasDestroy won't throw if a valid handle is passed */
                CUDA4DNN_CHECK_CUBLAS(cublasDestroy(handle));
            }
        }

        UniqueHandle& operator=(const UniqueHandle&) = delete;
        UniqueHandle& operator=(UniqueHandle&& other) noexcept {
            stream = std::move(other.stream);
            handle = other.handle;
            other.handle = nullptr;
            return *this;
        }

        //!< returns the raw cuBLAS handle
        cublasHandle_t get() const noexcept { return handle; }

    private:
        Stream stream;
        cublasHandle_t handle;
    };

    /* used to access the raw cuBLAS handle held by Handle */
    class HandleAccessor {
    public:
        static cublasHandle_t get(Handle& handle) {
            CV_Assert(handle);
            return handle.handle->get();
        }
    };

    Handle::Handle() : handle(std::make_shared<Handle::UniqueHandle>()) { }
    Handle::Handle(Stream strm) : handle(std::make_shared<Handle::UniqueHandle>(std::move(strm))) { }
    Handle::operator bool() const noexcept { return static_cast<bool>(handle); }

    template <>
    void gemm<float>(Handle handle,
        bool transa, bool transb,
        std::size_t rows_c, std::size_t cols_c, std::size_t common_dim,
        float alpha, const DevicePtr<const float> A, std::size_t lda,
        const DevicePtr<const float> B, std::size_t ldb,
        float beta, const DevicePtr<float> C, std::size_t ldc)
    {
        CV_Assert(handle);
        CV_Assert(transa == false && transb == false); /* other options unsupported yet */

        auto opa = transa ? CUBLAS_OP_T : CUBLAS_OP_N,
            opb = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
        int irows_c = static_cast<int>(rows_c),
            icols_c = static_cast<int>(cols_c),
            icommon_dim = static_cast<int>(common_dim),
            ilda = static_cast<int>(lda),
            ildb = static_cast<int>(ldb),
            ildc = static_cast<int>(ldc);

        if (!transa && !transb) {
            CUDA4DNN_CHECK_CUBLAS(
                cublasSgemm(
                    HandleAccessor::get(handle),
                    opb, opa,
                    irows_c, icols_c, icommon_dim,
                    &alpha, B.get(), ildb,
                    A.get(), ilda,
                    &beta, C.get(), ildc
                )
            );
        }
    }

    template <>
    void gemm<double>(Handle handle,
        bool transa, bool transb,
        std::size_t rows_c, std::size_t cols_c, std::size_t common_dim,
        double alpha, const DevicePtr<const double> A, std::size_t lda,
        const DevicePtr<const double> B, std::size_t ldb,
        double beta, const DevicePtr<double> C, std::size_t ldc)
    {
        CV_Assert(handle);
        CV_Assert(transa == false && transb == false); /* other options unsupported */

        auto opa = transa ? CUBLAS_OP_T : CUBLAS_OP_N,
            opb = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
        int irows_c = static_cast<int>(rows_c),
            icols_c = static_cast<int>(cols_c),
            icommon_dim = static_cast<int>(common_dim),
            ilda = static_cast<int>(lda),
            ildb = static_cast<int>(ldb),
            ildc = static_cast<int>(ldc);

        if (!transa && !transb) {
            CUDA4DNN_CHECK_CUBLAS(
                cublasDgemm(
                    HandleAccessor::get(handle),
                    opb, opa,
                    irows_c, icols_c, icommon_dim,
                    &alpha, B.get(), ildb,
                    A.get(), ilda,
                    &beta, C.get(), ildc
                )
            );
        }
    }

    template <>
    void geam<float>(Handle handle,
        bool transa, bool transb,
        std::size_t rows, std::size_t cols,
        float alpha, const DevicePtr<const float> A, std::size_t lda,
        float beta, const DevicePtr<const float> B, std::size_t ldb,
        const DevicePtr<float> C, std::size_t ldc)
    {
        CV_Assert(handle);

        auto opa = transa ? CUBLAS_OP_T : CUBLAS_OP_N,
             opb = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
        int irows = static_cast<int>(rows),
            icols = static_cast<int>(cols),
            ilda = static_cast<int>(lda),
            ildb = static_cast<int>(ldb),
            ildc = static_cast<int>(ldc);

        CUDA4DNN_CHECK_CUBLAS(
            cublasSgeam(
                HandleAccessor::get(handle),
                opa, opb,
                irows, icols,
                &alpha, A.get(), ilda,
                &beta, B.get(), ildb,
                C.get(), ildc
            )
        );
    }

    template <>
    void geam<double>(Handle handle,
        bool transa, bool transb,
        std::size_t rows, std::size_t cols,
        double alpha, const DevicePtr<const double> A, std::size_t lda,
        double beta, const DevicePtr<const double> B, std::size_t ldb,
        const DevicePtr<double> C, std::size_t ldc)
    {
        CV_Assert(handle);

        auto opa = transa ? CUBLAS_OP_T : CUBLAS_OP_N,
             opb = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
        int irows = static_cast<int>(rows),
            icols = static_cast<int>(cols),
            ilda = static_cast<int>(lda),
            ildb = static_cast<int>(ldb),
            ildc = static_cast<int>(ldc);

        CUDA4DNN_CHECK_CUBLAS(
            cublasDgeam(
                HandleAccessor::get(handle),
                opa, opb,
                irows, icols,
                &alpha, A.get(), ilda,
                &beta, B.get(), ildb,
                C.get(), ildc
            )
        );
    }

}}}}} /* namespace cv::dnn::cuda4dnn::csl::cublas */
