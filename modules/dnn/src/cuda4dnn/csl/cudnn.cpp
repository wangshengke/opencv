// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "cudnn.hpp"
#include "stream.hpp"

#include <cudnn.h>
#include <memory>

#define CUDA4DNN_CHECK_CUDNN(call) \
    ::cv::dnn::cuda4dnn::csl::cudnn::check((call), CV_Func, __FILE__, __LINE__)

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl { namespace cudnn {

    static void check(cudnnStatus_t status, const char* func, const char* file, int line) {
        if (status != CUDNN_STATUS_SUCCESS)
            throw cuDNNException(Error::GpuApiCallError, cudnnGetErrorString(status), func, file, line);
    }

    /** @brief noncopyable cuDNN smart handle
     *
     * UniqueHandle is a smart non-sharable wrapper for cuDNN handle which ensures that the handle
     * is destroyed after use. The handle can be associated with a CUDA stream by specifying the
     * stream during construction. By default, the handle is associated with the default stream.
     */
    class Handle::UniqueHandle {
    public:
        UniqueHandle() { CUDA4DNN_CHECK_CUDNN(cudnnCreate(&handle)); }
        UniqueHandle(UniqueHandle&) = delete;
        UniqueHandle(UniqueHandle&& other) noexcept
            : stream(std::move(other.stream)), handle{ other.handle } {
            other.handle = nullptr;
        }

        UniqueHandle(Stream strm) : stream(std::move(strm)) {
            CUDA4DNN_CHECK_CUDNN(cudnnCreate(&handle));
            try {
                CUDA4DNN_CHECK_CUDNN(cudnnSetStream(handle, StreamAccessor::get(stream)));
            } catch (...) {
                /* cudnnDestroy won't throw if a valid handle is passed */
                CUDA4DNN_CHECK_CUDNN(cudnnDestroy(handle));
                throw;
            }
        }

        ~UniqueHandle() noexcept {
            if (handle != nullptr) {
                /* cudnnDestroy won't throw if a valid handle is passed */
                CUDA4DNN_CHECK_CUDNN(cudnnDestroy(handle));
            }
        }

        UniqueHandle& operator=(const UniqueHandle&) = delete;
        UniqueHandle& operator=(UniqueHandle&& other) noexcept {
            stream = std::move(other.stream);
            handle = other.handle;
            other.handle = nullptr;
            return *this;
        }

        //!< returns the raw cuDNN handle
        cudnnHandle_t get() const noexcept { return handle; }

    private:
        Stream stream;
        cudnnHandle_t handle;
    };

    /** used to access the raw cuDNN handle held by Handle */
    class HandleAccessor {
    public:
        static cudnnHandle_t get(Handle& handle) {
            CV_Assert(handle);
            return handle.handle->get();
        }
    };

    Handle::Handle() : handle(std::make_shared<Handle::UniqueHandle>()) { }
    Handle::Handle(Stream strm) : handle(std::make_shared<Handle::UniqueHandle>(std::move(strm))) { }
    Handle::operator bool() const noexcept { return static_cast<bool>(handle); }

}}}}} /* namespace cv::dnn::cuda4dnn::csl::cudnn */
