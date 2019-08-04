// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "math.hpp"
#include "types.hpp"
#include "vector_traits.hpp"
#include "grid_stride_range.hpp"
#include "execution.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/span.hpp"

#include "../cuda4dnn/kernels/scale_shift.hpp"

#include <opencv2/core.hpp>

#include <cstddef>

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn  { namespace kernels {

    namespace raw {
        template <class T>
        __global__ void abs(span<T> output, view<T> input) {
            for (auto i : grid_stride_range(output.size())) {
                using device::abs;
                output[i] = abs(input[i]);
            }
        }

        template <class T>
        __global__ void tanh(span<T> output, view<T> input) {
            for (auto i : grid_stride_range(output.size())) {
                using device::tanh;
                output[i] = tanh(input[i]);
            }
        }

        template <class T>
        __global__ void sigmoid(span<T> output, view<T> input) {
            for (auto i : grid_stride_range(output.size())) {
                using device::sigmoid;
                output[i] = sigmoid(input[i]);
            }
        }

        template <class T>
        __global__ void bnll(span<T> output, view<T> input) {
            for (auto i : grid_stride_range(output.size())) {
                using device::log1pexp;
                output[i] = input[i] > T(0) ? input[i] + log1pexp(-input[i]) : log1pexp(input[i]);
            }
        }

        template <class T>
        __global__ void elu(span<T> output, view<T> input) {
            for (auto i : grid_stride_range(output.size())) {
                using device::exp;
                output[i] = input[i] >= T(0) ? input[i] : expm1(input[i]);
            }
        }

        template <class T, std::size_t N>
        __global__ void relu_vec(span<T> output, view<T> input, T slope) {
            using vector_type = get_vector_type_t<T, N>;

            auto output_vPtr = vector_type::get_pointer(output.data());
            auto input_vPtr = vector_type::get_pointer(input.data());

            for (auto i : grid_stride_range(output.size() / vector_type::size())) {
                vector_type vec;
                v_load(vec, input_vPtr[i]);
                for(int j = 0; j < vector_type::size(); j++)
                    vec.data[j] = vec.data[j] >= T(0) ? vec.data[j] : slope * vec.data[j];
                v_store(output_vPtr[i], vec);
            }
        }

        template <class T, std::size_t N>
        __global__ void clipped_relu_vec(span<T> output, view<T> input, T floor, T ceiling) {
            using vector_type = get_vector_type_t<T, N>;

            auto output_vPtr = vector_type::get_pointer(output.data());
            auto input_vPtr = vector_type::get_pointer(input.data());

            for (auto i : grid_stride_range(output.size() / vector_type::size())) {
                using device::clamp;

                vector_type vec;
                v_load(vec, input_vPtr[i]);
                for (int j = 0; j < vector_type::size(); j++)
                    vec.data[j] = clamp(vec.data[j], floor, ceiling);
                v_store(output_vPtr[i], vec);
            }
        }

        template <class T, std::size_t N>
        __global__ void axiswise_relu_vec(span<T> output, view<T> input, size_type inner_size, view<T> slope) {
            using vector_type = get_vector_type_t<T, N>;

            auto output_vPtr = vector_type::get_pointer(output.data());
            auto input_vPtr = vector_type::get_pointer(input.data());

            inner_size /= vector_type::size();
            for (auto i : grid_stride_range(output.size() / vector_type::size())) {
                const index_type c = (i / inner_size) % static_cast<size_type>(slope.size());

                vector_type vec;
                v_load(vec, input_vPtr[i]);
                for (int j = 0; j < vector_type::size(); j++)
                    vec.data[j] = vec.data[j] > T(0) ? vec.data[j] : vec.data[j] * slope[c];
                v_store(output_vPtr[i], vec);
            }
        }

        template <class T, std::size_t N>
        __global__ void power_vec(span<T> output, view<T> input, T exp, T scale, T shift) {
            using vector_type = get_vector_type_t<T, N>;

            auto output_vPtr = vector_type::get_pointer(output.data());
            auto input_vPtr = vector_type::get_pointer(input.data());

            for (auto i : grid_stride_range(output.size() / vector_type::size())) {
                using device::pow;

                vector_type vec;
                v_load(vec, input_vPtr[i]);
                for (int j = 0; j < vector_type::size(); j++)
                    vec.data[j] = pow(shift + scale * vec.data[j], exp);
                v_store(output_vPtr[i], vec);
            }
        }
    }

    template <class T>
    void abs(const Stream& stream, span<T> output, view<T> input) {
        CV_Assert(input.size() == output.size());

        auto kernel = raw::abs<T>;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, output, input);
    }

    template void abs<__half>(const Stream& stream, span<__half> output, view<__half> input);
    template void abs<float>(const Stream& stream, span<float> output, view<float> input);
    template void abs<double>(const Stream& stream, span<double> output, view<double> input);

    template <class T>
    void tanh(const Stream& stream, span<T> output, view<T> input) {
        CV_Assert(input.size() == output.size());

        auto kernel = raw::tanh<T>;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, output, input);
    }

    template void tanh<__half>(const Stream&, span<__half>, view<__half>);
    template void tanh<float>(const Stream&, span<float>, view<float>);
    template void tanh<double>(const Stream&, span<double>, view<double>);

    template <class T>
    void sigmoid(const Stream& stream, span<T> output, view<T> input) {
        CV_Assert(input.size() == output.size());

        auto kernel = raw::sigmoid<T>;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, output, input);
    }

    template void sigmoid<__half>(const Stream&, span<__half>, view<__half>);
    template void sigmoid<float>(const Stream&, span<float>, view<float>);
    template void sigmoid<double>(const Stream&, span<double>, view<double>);

    template <class T>
    void bnll(const Stream& stream, span<T> output, view<T> input) {
        CV_Assert(input.size() == output.size());

        auto kernel = raw::bnll<T>;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, output, input);
    }

    template void bnll<__half>(const Stream&, span<__half>, view<__half>);
    template void bnll<float>(const Stream&, span<float>, view<float>);
    template void bnll<double>(const Stream&, span<double>, view<double>);

    template <class T>
    void elu(const Stream& stream, span<T> output, view<T> input) {
        CV_Assert(input.size() == output.size());

        auto kernel = raw::elu<T>;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, output, input);
    }

    template void elu<__half>(const Stream&, span<__half>, view<__half>);
    template void elu<float>(const Stream&, span<float>, view<float>);
    template void elu<double>(const Stream&, span<double>, view<double>);

    template <class T, std::size_t N>
    void launch_vectorized_relu(const Stream& stream, span<T> output, view<T> input, T slope) {
        CV_Assert(is_fully_aligned<T>(output, N));
        CV_Assert(is_fully_aligned<T>(input, N));

        auto kernel = raw::relu_vec<T, N>;
        auto policy = make_policy(kernel, output.size() / N, 0, stream);
        launch_kernel(kernel, policy, output, input, slope);
    }

    template <class T>
    void relu(const Stream& stream, span<T> output, view<T> input, T slope) {
        CV_Assert(input.size() == output.size());

        if(is_fully_aligned<T>(output, 4) && is_fully_aligned<T>(input, 4)) {
            launch_vectorized_relu<T, 4>(stream, output, input, slope);
        } else if (is_fully_aligned<T>(output, 2) && is_fully_aligned<T>(input, 2)) {
            launch_vectorized_relu<T, 2>(stream, output, input, slope);
        } else {
            launch_vectorized_relu<T, 1>(stream, output, input, slope);
        }
    }

    template void relu<__half>(const Stream&, span<__half>, view<__half>, __half);
    template void relu<float>(const Stream&, span<float>, view<float>, float);
    template void relu<double>(const Stream&, span<double>, view<double>, double);

    template <class T, std::size_t N>
    void launch_vectorized_clipped_relu(const Stream& stream, span<T> output, view<T> input, T floor, T ceiling) {
        CV_Assert(is_fully_aligned<T>(output, N));
        CV_Assert(is_fully_aligned<T>(input, N));

        auto kernel = raw::clipped_relu_vec<T, N>;
        auto policy = make_policy(kernel, output.size() / N, 0, stream);
        launch_kernel(kernel, policy, output, input, floor, ceiling);
    }

    template <class T>
    void clipped_relu(const Stream& stream, span<T> output, view<T> input, T floor, T ceiling) {
        CV_Assert(input.size() == output.size());
        CV_Assert(static_cast<double>(floor) <= static_cast<double>(ceiling));

        if(is_fully_aligned<T>(output, 4) && is_fully_aligned<T>(input, 4)) {
            launch_vectorized_clipped_relu<T, 4>(stream, output, input, floor, ceiling);
        } else if (is_fully_aligned<T>(output, 2) && is_fully_aligned<T>(input, 2)) {
            launch_vectorized_clipped_relu<T, 2>(stream, output, input, floor, ceiling);
        } else {
            launch_vectorized_clipped_relu<T, 1>(stream, output, input, floor, ceiling);
        }
    }

    template void clipped_relu<__half>(const Stream&, span<__half>, view<__half>, __half, __half);
    template void clipped_relu<float>(const Stream&, span<float>, view<float>, float, float);
    template void clipped_relu<double>(const Stream&, span<double>, view<double>, double, double);

    template <class T, std::size_t N>
    void launch_vectorized_axiswise_relu(const Stream& stream, span<T> output, view<T> input, std::size_t inner_size, view<T> slope) {
        CV_Assert(is_fully_aligned<T>(output, N));
        CV_Assert(is_fully_aligned<T>(input, N));
        CV_Assert(inner_size % N == 0);

        auto kernel = raw::axiswise_relu_vec<T, N>;
        auto policy = make_policy(kernel, output.size() / N, 0, stream);
        launch_kernel(kernel, policy, output, input, inner_size, slope);
    }

    template <class T>
    void axiswise_relu(const Stream& stream, span<T> output, view<T> input, std::size_t inner_size, view<T> slope) {
        CV_Assert(input.size() == output.size());
        CV_Assert(slope.size() == input.size() / inner_size);

        if (is_fully_aligned<T>(output, 4) && is_fully_aligned<T>(input, 4) && inner_size % 4 == 0) {
            launch_vectorized_axiswise_relu<T, 4>(stream, output, input, inner_size, slope);
        } else if (is_fully_aligned<T>(output, 2) && is_fully_aligned<T>(input, 2) && inner_size % 2 == 0) {
            launch_vectorized_axiswise_relu<T, 2>(stream, output, input, inner_size, slope);
        } else {
            launch_vectorized_axiswise_relu<T, 1>(stream, output, input, inner_size, slope);
        }
    }

    template void axiswise_relu<__half>(const Stream&, span<__half>, view<__half>, std::size_t, view<__half>);
    template void axiswise_relu<float>(const Stream&, span<float>, view<float>, std::size_t, view<float>);
    template void axiswise_relu<double>(const Stream&, span<double>, view<double>, std::size_t, view<double>);

    template <class T, std::size_t N>
    void launch_vectorized_power(const Stream& stream, span<T> output, view<T> input, T exp, T scale, T shift) {
        CV_Assert(is_fully_aligned<T>(output, N));
        CV_Assert(is_fully_aligned<T>(input, N));

        auto kernel = raw::power_vec<T, N>;
        auto policy = make_policy(kernel, output.size() / N, 0, stream);
        launch_kernel(kernel, policy, output, input, exp, scale, shift);
    }

    template <class T>
    void power(const Stream& stream, span<T> output, view<T> input, T exp, T scale, T shift) {
        CV_Assert(input.size() == output.size());

        if (static_cast<float>(exp) == 1.0f) {
            scale1_with_bias1(stream, output, input, scale, shift);
            return;
        }

        if (is_fully_aligned<T>(output, 4) && is_fully_aligned<T>(input, 4) && output.size()) {
            launch_vectorized_power<T, 4>(stream, output, input, exp, scale, shift);
        } else if (is_fully_aligned<T>(output, 2) && is_fully_aligned<T>(input, 2) && output.size()) {
            launch_vectorized_power<T, 2>(stream, output, input, exp, scale, shift);
        } else {
            launch_vectorized_power<T, 1>(stream, output, input, exp, scale, shift);
        }
    }

    template void power<__half>(const Stream&, span<__half>, view<__half>, __half, __half, __half);
    template void power<float>(const Stream&, span<float>, view<float>, float, float, float);
    template void power<double>(const Stream&, span<double>, view<double>, double, double, double);

}}}} /* cv::dnn::cuda4dnn::kernels */
