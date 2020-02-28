// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_KERNELS_DETECTION_OUTPUT_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_KERNELS_DETECTION_OUTPUT_HPP

#include "../csl/stream.hpp"
#include "../csl/span.hpp"
#include "../csl/tensor.hpp"

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    template <class T>
    void decode_bboxes(const csl::Stream& stream, csl::Span<T> output, csl::View<T> locations, csl::View<T> priors,
        std::size_t num_loc_classes, bool share_location, std::size_t background_label_id,
        bool transpose_location, bool variance_encoded_in_target,
        bool corner_true_or_center_false, bool normalized_bbox,
        bool clip_box, float clip_width, float clip_height);

    template <class T>
    void findTopK(const csl::Stream& stream, csl::TensorSpan<int> indices, csl::TensorSpan<int> count, csl::TensorView<T> scores, std::size_t top_k, std::size_t background_label_id, float threshold);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_KERNELS_DETECTION_OUTPUT_HPP */
