// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <opencv2/dnn/csl/stream.hpp>

#include "error.hpp"
#include "stream.hpp"

#include <memory>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl {

    /** @brief noncopyable smart CUDA stream
     *
     * UniqueStream is a smart non-sharable wrapper for CUDA stream handle which ensures that
     * the handle is destroyed after use. Unless explicitly specified by a constructor argument,
     * the stream object represents the default stream.
     */
    class Stream::UniqueStream {
    public:
        UniqueStream() noexcept : stream{ 0 } { }
        UniqueStream(UniqueStream&) = delete;
        UniqueStream(UniqueStream&& other) noexcept {
            stream = other.stream;
            other.stream = 0;
        }

        UniqueStream(bool create) : stream{ 0 } {
            if (create) {
                CUDA4DNN_CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            }
        }

        ~UniqueStream() {
            if (stream != nullptr)
                CUDA4DNN_CHECK_CUDA(cudaStreamDestroy(stream));
        }

        UniqueStream& operator=(const UniqueStream&) = delete;
        UniqueStream& operator=(UniqueStream&& other) noexcept {
            stream = other.stream;
            other.stream = 0;
            return *this;
        }

        //!< returns the raw CUDA stream handle
        cudaStream_t get() const noexcept { return stream; }

        void synchronize() const { CUDA4DNN_CHECK_CUDA(cudaStreamSynchronize(stream)); }
        bool busy() const {
            auto status = cudaStreamQuery(stream);
            if (status == cudaErrorNotReady)
                return true;
            CUDA4DNN_CHECK_CUDA(status);
            return false;
        }

    private:
        cudaStream_t stream;
    };

    Stream::Stream() : stream(std::make_shared<Stream::UniqueStream>()) { }
    Stream::Stream(bool create) : stream(std::make_shared<Stream::UniqueStream>(create)) { }
    void Stream::synchronize() const { stream->synchronize(); }
    bool Stream::busy() const { return stream->busy(); }
    Stream::operator bool() const noexcept { return static_cast<bool>(stream); }

    cudaStream_t StreamAccessor::get(const Stream& stream) {
        CV_Assert(stream);
        return stream.stream->get();
    }

}}}} /* namespace cv::dnn::cuda4dnn::csl */
