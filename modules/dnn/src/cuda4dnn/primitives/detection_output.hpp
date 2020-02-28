// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_DETECTION_OUTPUT_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_DETECTION_OUTPUT_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"
#include "../csl/tensor.hpp"

#include "../kernels/permute.hpp"
#include "../kernels/detection_output.hpp"

#include <cstddef>
#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    struct DetectionOutputConfiguration {
        std::size_t batch_size;

        enum class CodeType {
            CORNER,
            CENTER_SIZE
        };
        CodeType code_type;

        bool share_location;
        std::size_t num_priors;
        std::size_t num_classes;
        std::size_t background_label_id;

        bool transpose_location;
        bool variance_encoded_in_target;
        bool normalized_bbox;

        bool clip_box;

        std::size_t classwise_topK;
        float confidence_threshold;
    };

    template <class T>
    class DetectionOutputOp final : public CUDABackendNode {
    public:
        using wrapper_type = GetCUDABackendWrapperType<T>;

        DetectionOutputOp(csl::Stream stream_, const DetectionOutputConfiguration& config)
            : stream(std::move(stream_))
        {
            corner_true_or_center_false = (config.code_type == DetectionOutputConfiguration::CodeType::CORNER);

            share_location = config.share_location;
            num_priors = config.num_priors;
            num_classes = config.num_classes;
            background_label_id = config.background_label_id;

            transpose_location = config.transpose_location;
            variance_encoded_in_target = config.variance_encoded_in_target;
            normalized_bbox = config.normalized_bbox;

            clip_box = config.clip_box;

            classwise_topK = config.classwise_topK;
            if (classwise_topK == -1)
                classwise_topK = num_priors;
            confidence_threshold = config.confidence_threshold;

            auto num_loc_classes = (share_location ? 1 : num_classes);

            csl::WorkspaceBuilder builder;
            builder.require<T>(config.batch_size * num_priors * num_loc_classes * 4); /* decoded boxes */
            builder.require<T>(config.batch_size * num_priors * num_classes); /* transposed scores */
            builder.require<int>(config.batch_size * num_classes * classwise_topK); /* indices */
            builder.require<int>(config.batch_size * num_classes); /* classwise topK count */
            scratch_mem_in_bytes = builder.required_workspace_size();
        }

        void forward(
            const std::vector<cv::Ptr<BackendWrapper>>& inputs,
            const std::vector<cv::Ptr<BackendWrapper>>& outputs,
            csl::Workspace& workspace) override
        {
            CV_Assert((inputs.size() == 3 || inputs.size() == 4) && outputs.size() == 1);

            auto locations_wrapper = inputs[0].dynamicCast<wrapper_type>();
            auto locations = locations_wrapper->getView();

            auto scores_wrapper = inputs[1].dynamicCast<wrapper_type>();
            auto scores = scores_wrapper->getView();
            scores.unsqueeze();
            scores.reshape(-1, num_priors, num_classes);

            auto priors_wrapper = inputs[2].dynamicCast<wrapper_type>();
            auto priors = priors_wrapper->getView();

            auto output_wrapper = outputs[0].dynamicCast<wrapper_type>();
            auto output = output_wrapper->getSpan();

            auto batch_size = locations.get_axis_size(0);
            auto num_loc_classes = (share_location ? 1 : num_classes);
            locations.unsqueeze();
            locations.unsqueeze();
            locations.reshape(-1, num_priors, num_loc_classes, 4);

            csl::TensorSpan<T> decoded_boxes;
            {
                auto shape = std::vector<std::size_t>{batch_size, num_priors, num_loc_classes, 4};
                csl::WorkspaceAllocator allocator(workspace);
                decoded_boxes = allocator.get_tensor_span<T>(std::begin(shape), std::end(shape));
                CV_Assert(is_shape_same(decoded_boxes, locations));
            }

            float clip_width = 0.0, clip_height = 0.0;
            if (clip_box)
            {
                if (normalized_bbox)
                {
                    clip_width = clip_height = 1.0f;
                }
                else
                {
                    auto image_wrapper = inputs[3].dynamicCast<wrapper_type>();
                    auto image_shape = image_wrapper->getShape();

                    CV_Assert(image_shape.size() == 4);
                    clip_width = image_shape[3] - 1;
                    clip_height = image_shape[2] - 1;
                }
            }

            kernels::decode_bboxes<T>(stream, decoded_boxes, locations, priors,
                num_loc_classes, share_location, background_label_id,
                transpose_location, variance_encoded_in_target,
                corner_true_or_center_false, normalized_bbox,
                clip_box, clip_width, clip_height);

            csl::TensorSpan<T> scores_permuted;
            {
                auto shape = std::vector<std::size_t>{batch_size, num_classes, num_priors};
                csl::WorkspaceAllocator allocator(workspace);
                scores_permuted = allocator.get_tensor_span<T>(std::begin(shape), std::end(shape));
            }

            kernels::permute<T>(stream, scores_permuted, scores, {0, 2, 1});

            csl::TensorSpan<int> indices;
            {
                auto shape = std::vector<std::size_t>{decoded_boxes.get_axis_size(0), num_classes, classwise_topK};
                csl::WorkspaceAllocator allocator(workspace);
                indices = allocator.get_tensor_span<int>(std::begin(shape), std::end(shape));
            }

            csl::TensorSpan<int> count;
            {
                auto shape = std::vector<std::size_t>{batch_size, num_classes};
                csl::WorkspaceAllocator allocator(workspace);
                count = allocator.get_tensor_span<int>(std::begin(shape), std::end(shape));
            }

            kernels::findTopK<T>(stream, indices, count, scores_permuted, classwise_topK, background_label_id, confidence_threshold);
        }

        std::size_t get_workspace_memory_in_bytes() const noexcept override { return scratch_mem_in_bytes; }

    private:
        csl::Stream stream;
        std::size_t scratch_mem_in_bytes;

        bool share_location;
        std::size_t num_priors;
        std::size_t num_classes;
        std::size_t background_label_id;

        bool transpose_location;
        bool variance_encoded_in_target;
        bool corner_true_or_center_false;
        bool normalized_bbox;
        bool clip_box;

        std::size_t classwise_topK;
        float confidence_threshold;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_DETECTION_OUTPUT_HPP */
