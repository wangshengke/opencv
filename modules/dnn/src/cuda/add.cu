// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../cuda4dnn/csl/kernels.hpp"

#include "../cuda4dnn/csl/tensor.hpp"
#include "../cuda4dnn/csl/kernel_utils.hpp"

#include <cuda_runtime.h>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl  { namespace kernels {

    namespace raw {
        template <class T>
        __global__ void add_with_expansion(
            std::size_t n,
            DevicePtr<T> output,
            DevicePtr<const T> larger_x, std::size_t size_x,
            DevicePtr<const T> smaller_y, std::size_t size_y)
        {
            for (auto i : grid_stride_range(n)) {
                const auto idx = (i / size_x) % size_y;
                output[i] = larger_x[i] + smaller_y[idx];
            }
        }
    }

    template <class T>
    void add_with_expansion(
        const Stream& stream,
        TensorSpan<T> output,
        TensorView<T> x, std::size_t inner_size,
        TensorView<T> y)
    {
        auto policy = make_policy(raw::add_with_expansion<T>, 0, stream);
        launch_kernel(raw::add_with_expansion<T>, policy,
            output.size(),
            output.get(),
            x.get(), inner_size,
            y.get(), y.size());
    }

    template void add_with_expansion(
        const Stream& stream,
        TensorSpan<float> output,
        TensorView<float> x, std::size_t inner_size,
        TensorView<float> y);

    template void add_with_expansion<double>(
        const Stream& stream,
        TensorSpan<double> output,
        TensorView<double> x, std::size_t inner_size,
        TensorView<double> y);

}}}}} /* cv::dnn::cuda4dnn::csl::kernels */
