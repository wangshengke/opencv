// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <cuda_runtime.h>

#include "../cuda4dnn/csl/math.hpp"
#include "../cuda4dnn/csl/kernel_utils.hpp"

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl  { namespace kernels {

    namespace detail {
        template <class T> __device__ T abs(T val);
        template <> __device__ float abs(float val) { return fabsf(val); }
        template <> __device__ double abs(double val) { return fabs(val); }

        template <class T> __device__ T exp(T val);
        template <> __device__ float exp(float val) { return expf(val); }
        template <> __device__ double exp(double val) { return exp(val); }

        template <class T> __device__ T max(T x, T y);
        template <> __device__ float max(float x, float y) { return fmaxf(x, y); }
        template <> __device__ double max(double x, double y) { return fmax(x, y); }

        template <class T> __device__ T min(T x, T y);
        template <> __device__ float min(float x, float y) { return fminf(x, y); }
        template <> __device__ double min(double x, double y) { return fmin(x, y); }

        template <class T> __device__ T log(T val);
        template <> __device__ float log(float val) { return logf(val); }
        template <> __device__ double log(double val) { return log(val); }

        template <class T> __device__ T tanh(T val);
        template <> __device__ float tanh(float val) { return tanhf(val); }
        template <> __device__ double tanh(double val) { return tanh(val); }

        template <class T> __device__ T pow(T val, T exp);
        template <> __device__ float pow(float val, float exp) { return powf(val, exp); }
        template <> __device__ double pow(double val, double exp) { return pow(val, exp); }

        template <class T>
        __device__ T sigmoid(T val) { return T(1) / (1 + exp(-val)); }
    }

    namespace raw {
        template <class T>
        __global__ void abs(view<T> src, span<T> dest) {
            assert(src.size() >= dest.size());
            for (auto i : grid_stride_range(dest.size())) {
                using detail::abs;
                dest[i] = abs(src[i]);
            }
        }

        template <class T>
        __global__ void tanh(view<T> src, span<T> dest) {
            assert(src.size() >= dest.size());
            for (auto i : grid_stride_range(dest.size())) {
                using detail::tanh;
                dest[i] = tanh(src[i]);
            }
        }

        template <class T>
        __global__ void sigmoid(view<T> src, span<T> dest) {
            assert(src.size() >= dest.size());
            for (auto i : grid_stride_range(dest.size())) {
                using detail::sigmoid;
                dest[i] = sigmoid(src[i]);
            }
        }

        template <class T>
        __global__ void bnll(view<T> src, span<T> dest) {
            assert(src.size() >= dest.size());
            for (auto i : grid_stride_range(dest.size())) {
                using detail::log;
                using detail::exp;
                dest[i] = log(1 + exp(-src[i]));
            }
        }

        template <class T>
        __global__ void elu(view<T> src, span<T> dest) {
            assert(src.size() >= dest.size());
            for (auto i : grid_stride_range(dest.size())) {
                using detail::exp;
                dest[i] = src[i] >= 0 ? src[i] : (exp(src[i]) - 1);
            }
        }

        template <class T>
        __global__ void relu(view<T> src, span<T> dest, T slope) {
            assert(src.size() >= dest.size());
            for (auto i : grid_stride_range(dest.size())) {
                dest[i] = src[i] >= 0.0 ? src[i] : slope * src[i];
            }
        }

        template <class T>
        __global__ void clipped_relu(view<T> src, span<T> dest, T ceiling, T floor) {
            assert(src.size() >= dest.size());
            assert(floor <= ceiling);
            for (auto i : grid_stride_range(dest.size())) {
                using detail::max;
                using detail::min;
                dest[i] = min(max(src[i], floor), ceiling);
            }
        }

        template <class T>
        __global__ void power(view<T> src, span<T> dest, T exp, T scale, T shift) {
            assert(src.size() >= dest.size());
            for (auto i : grid_stride_range(dest.size())) {
                using detail::pow;
                dest[i] = pow(shift + scale * src[i], exp);
            }
        }
    }

    template <class T>
    void abs(const Stream& stream, span<T> dest, view<T> src) {
        CV_Assert(src.size() >= dest.size());

        auto policy = make_optimal_policy(raw::abs<T>, 0, stream);
        launch_kernel(raw::abs<T>, policy, src, dest);
    }

    template void abs<float>(const Stream& stream, span<float> dest, view<float> src);
    template void abs<double>(const Stream& stream, span<double> dest, view<double> src);

    template <class T>
    void tanh(const Stream& stream, span<T> dest, view<T> src) {
        CV_Assert(src.size() >= dest.size());

        auto policy = make_optimal_policy(raw::tanh<T>, 0, stream);
        launch_kernel(raw::tanh<T>, policy, src, dest);
    }

    template void tanh<float>(const Stream& stream, span<float> dest, view<float> src);
    template void tanh<double>(const Stream& stream, span<double> dest, view<double> src);

    template <class T>
    void sigmoid(const Stream& stream, span<T> dest, view<T> src) {
        CV_Assert(src.size() >= dest.size());

        auto policy = make_optimal_policy(raw::sigmoid<T>, 0, stream);
        launch_kernel(raw::sigmoid<T>, policy, src, dest);
    }

    template void sigmoid<float>(const Stream& stream, span<float> dest, view<float> src);
    template void sigmoid<double>(const Stream& stream, span<double> dest, view<double> src);

    template <class T>
    void bnll(const Stream& stream, span<T> dest, view<T> src) {
        CV_Assert(src.size() >= dest.size());

        auto policy = make_optimal_policy(raw::bnll<T>, 0, stream);
        launch_kernel(raw::bnll<T>, policy, src, dest);
    }

    template void bnll<float>(const Stream& stream, span<float> dest, view<float> src);
    template void bnll<double>(const Stream& stream, span<double> dest, view<double> src);

    template <class T>
    void elu(const Stream& stream, span<T> dest, view<T> src) {
        CV_Assert(src.size() >= dest.size());

        auto policy = make_optimal_policy(raw::elu<T>, 0, stream);
        launch_kernel(raw::elu<T>, policy, src, dest);
    }

    template void elu<float>(const Stream& stream, span<float> dest, view<float> src);
    template void elu<double>(const Stream& stream, span<double> dest, view<double> src);

    template <class T>
    void relu(const Stream& stream, span<T> dest, view<T> src, T slope) {
        CV_Assert(src.size() >= dest.size());

        auto policy = make_optimal_policy(raw::relu<T>, 0, stream);
        launch_kernel(raw::relu<T>, policy, src, dest, slope);
    }

    template void relu<float>(const Stream& stream, span<float> dest, view<float> src, float slope);
    template void relu<double>(const Stream& stream, span<double> dest, view<double> src, double slope);

    template <class T>
    void clipped_relu(const Stream& stream, span<T> dest, view<T> src, T floor, T ceiling) {
        CV_Assert(src.size() >= dest.size());
        CV_Assert(floor <= ceiling);

        auto policy = make_optimal_policy(raw::clipped_relu<T>, 0, stream);
        launch_kernel(raw::clipped_relu<T>, policy, src, dest, floor, ceiling);
    }

    template void clipped_relu<float>(const Stream& stream, span<float> dest, view<float> src, float floor, float ceiling);
    template void clipped_relu<double>(const Stream& stream, span<double> dest, view<double> src, double floor, double ceiling);

    template <class T>
    void power(const Stream& stream, span<T> dest, view<T> src, T exp, T scale, T shift) {
        CV_Assert(src.size() >= dest.size());

        auto policy = make_optimal_policy(raw::power<T>, 0, stream);
        launch_kernel(raw::power<T>, policy, src, dest, exp, scale, shift);
    }

    template void power<float>(const Stream& stream, span<float> dest, view<float> src, float exp, float scale, float shift);
    template void power<double>(const Stream& stream, span<double> dest, view<double> src, double exp, double scale, double shift);

}}}}} /*  cv::dnn::cuda4dnn::csl::kernels */
