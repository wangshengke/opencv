// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "math.hpp"
#include "grid_stride_range.hpp"
#include "execution.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/span.hpp"

#include <opencv2/core.hpp>

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    namespace raw {
        template <class T>
        __global__ void eltwise_max_2(span<T> output, view<T> x, view<T> y) {
            for (auto i : grid_stride_range(output.size())) {
                using device::max;
                output[i] = max(x[i], y[i]);
            }
        }

        template <class T>
        __global__ void eltwise_sum_2(span<T> output, view<T> x, view<T> y) {
            for (auto i : grid_stride_range(output.size()))
                output[i] = x[i] + y[i];
        }

        template <class T>
        __global__ void eltwise_sum_coeff_2(span<T> output, T coeff_x, view<T> x, T coeff_y, view<T> y) {
            for (auto i : grid_stride_range(output.size()))
                output[i] = coeff_x * x[i] + coeff_y * y[i];
        }

        template <class T>
        __global__ void eltwise_prod_2(span<T> output, view<T> x, view<T> y) {
            for (auto i : grid_stride_range(output.size()))
                output[i] = x[i] * y[i];
        }
    }

    template <class T>
    void eltwise_max_2(const Stream& stream, span<T> output, view<T> x, view<T> y) {
        CV_Assert(x.size() == y.size());
        CV_Assert(x.size() == output.size());

        auto kernel = raw::eltwise_max_2<T>;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, output, x, y);
    }

    template void eltwise_max_2(const Stream& stream, span<__half> output, view<__half> x, view<__half> y);
    template void eltwise_max_2(const Stream& stream, span<float> output, view<float> x, view<float> y);

    template <class T>
    void eltwise_sum_2(const Stream& stream, span<T> output, view<T> x, view<T> y) {
        CV_Assert(x.size() == y.size());
        CV_Assert(x.size() == output.size());

        auto kernel = raw::eltwise_sum_2<T>;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, output, x, y);
    }

    template void eltwise_sum_2(const Stream& stream, span<__half> output, view<__half> x, view<__half> y);
    template void eltwise_sum_2(const Stream& stream, span<float> output, view<float> x, view<float> y);

    template <class T>
    void eltwise_sum_coeff_2(const Stream& stream, span<T> output, T coeff_x, view<T> x, T coeff_y, view<T> y) {
        CV_Assert(x.size() == y.size());
        CV_Assert(x.size() == output.size());

        if (static_cast<float>(coeff_x) == 1.0f && static_cast<float>(coeff_y) == 1.0f) {
            eltwise_sum_2(stream, output, x, y);
            return;
        }

        auto kernel = raw::eltwise_sum_coeff_2<T>;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, output, coeff_x, x, coeff_y, y);
    }

    template void eltwise_sum_coeff_2(const Stream&, span<__half>, __half, view<__half>, __half, view<__half>);
    template void eltwise_sum_coeff_2(const Stream&, span<float>, float, view<float>, float, view<float>);

    template <class T>
    void eltwise_prod_2(const Stream& stream, span<T> output, view<T> x, view<T> y) {
        CV_Assert(x.size() == y.size());
        CV_Assert(x.size() == output.size());

        auto kernel = raw::eltwise_prod_2<T>;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, output, x, y);
    }

    template void eltwise_prod_2(const Stream& stream, span<__half> output, view<__half> x, view<__half> y);
    template void eltwise_prod_2(const Stream& stream, span<float> output, view<float> x, view<float> y);

}}}} /* cv::dnn::cuda4dnn::kernels */
