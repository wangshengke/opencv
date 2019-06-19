// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_CUDA4DNN_KERNELS_HPP
#define OPENCV_DNN_CUDA4DNN_KERNELS_HPP

#include "cuda4dnn/csl/stream.hpp"
#include "cuda4dnn/csl/tensor.hpp"

#include <cstddef>
#include <type_traits>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl { namespace kernels {

    template <class T>
    void concat(
        const Stream& stream,
        TensorSpan<T> output, TensorView<T> input,
        std::size_t concat_size, std::size_t input_concat_axis_size,
        std::size_t output_concat_axis_size, std::size_t output_offset_concat_axis);

}}}}} /* namespace cv::dnn::cuda4dnn::csl::kernels */

#endif /* OPENCV_DNN_CUDA4DNN_KERNELS_HPP */
