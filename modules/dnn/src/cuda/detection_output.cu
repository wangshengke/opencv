// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "math.hpp"
#include "grid_stride_range.hpp"
#include "execution.hpp"
#include "vector_traits.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/span.hpp"
#include "../cuda4dnn/csl/tensor.hpp"

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    namespace raw {
        template <class T, bool VARIANCE_ENCODED_IN_TARGET, bool CORNER_TRUE_CENTER_FALSE, bool CLIP_BBOX>
        __global__ void decode_bbox(
            Span<T> decoded_bboxes, View<T> locations, View<T> priors,
            bool share_location, bool transpose_location, bool normalized_bbox,
            size_type num_loc_classes, index_type background_label_id,
            float clip_width, float clip_height
        )
        {
            struct BoundingBox {
                float xmin, ymin, xmax, ymax;
            };

            // decoded_bboxes: [batch_size, num_priors, num_loc_classes, 4]
            // locations: [batch_size, num_priors, num_loc_classes, 4]

            // priors: [1, 2, num_priors, 4]
            size_type num_priors = priors.size() / 8; /* 4 bbox values + 4 variance values per prior */

            using vector_type = get_vector_type_t<T, 4>;
            auto locations_vPtr = vector_type::get_pointer(locations.data());
            auto priors_vPtr = vector_type::get_pointer(priors.data());
            auto decoded_bboxes_vPtr = vector_type::get_pointer(decoded_bboxes.data());

            const auto boxes_per_batch = num_priors * num_loc_classes;
            for (auto idx : grid_stride_range(decoded_bboxes.size() / 4))
            {
                auto p = (idx % boxes_per_batch) / num_loc_classes;
                auto c = idx % num_loc_classes;

                if (!share_location && c == background_label_id)
                    continue;

                BoundingBox bbox;
                {
                    vector_type location;
                    v_load(location, locations_vPtr[idx]);

                    if (transpose_location)
                    {
                        bbox.ymin = location.data[0];
                        bbox.xmin = location.data[1];
                        bbox.ymax = location.data[2];
                        bbox.xmax = location.data[3];
                    }
                    else
                    {
                        bbox.xmin = location.data[0];
                        bbox.ymin = location.data[1];
                        bbox.xmax = location.data[2];
                        bbox.ymax = location.data[3];
                    }
                }

                if (!VARIANCE_ENCODED_IN_TARGET)
                {
                    vector_type prior_variance;
                    v_load(prior_variance, priors_vPtr[num_priors + p]);

                    bbox.xmin *= static_cast<float>(prior_variance.data[0]);
                    bbox.ymin *= static_cast<float>(prior_variance.data[1]);
                    bbox.xmax *= static_cast<float>(prior_variance.data[2]);
                    bbox.ymax *= static_cast<float>(prior_variance.data[3]);
                }

                BoundingBox prior;
                {
                    vector_type prior_box;
                    v_load(prior_box, priors_vPtr[p]);

                    prior.xmin = prior_box.data[0];
                    prior.ymin = prior_box.data[1];
                    prior.xmax = prior_box.data[2];
                    prior.ymax = prior_box.data[3];
                }

                BoundingBox decoded_bbox;
                if (CORNER_TRUE_CENTER_FALSE)
                {
                    decoded_bbox.xmin = prior.xmin + bbox.xmin;
                    decoded_bbox.ymin = prior.ymin + bbox.ymin;
                    decoded_bbox.xmax = prior.xmax + bbox.xmax;
                    decoded_bbox.ymax = prior.ymax + bbox.ymax;
                }
                else
                {
                    auto prior_width = prior.xmax - prior.xmin;
                    auto prior_height = prior.ymax - prior.ymin;
                    if (!normalized_bbox)
                    {
                        prior_width += 1;
                        prior_height += 1;
                    }

                    auto prior_center_x = prior.xmin + prior_width * 0.5f;
                    auto prior_center_y = prior.ymin + prior_height * 0.5f;

                    auto decode_bbox_center_x = bbox.xmin * prior_width + prior_center_x;
                    auto decode_bbox_center_y = bbox.ymin * prior_height + prior_center_y;

                    using device::exp;
                    float decode_bbox_width = exp(bbox.xmax) * prior_width;
                    float decode_bbox_height = exp(bbox.ymax) * prior_height;

                    decoded_bbox.xmin = decode_bbox_center_x - decode_bbox_width * 0.5f;
                    decoded_bbox.ymin = decode_bbox_center_y - decode_bbox_height * 0.5f;
                    decoded_bbox.xmax = decode_bbox_center_x + decode_bbox_width * 0.5f;
                    decoded_bbox.ymax = decode_bbox_center_y + decode_bbox_height * 0.5f;
                }

                vector_type decoded_bbox_vec;
                if (CLIP_BBOX)
                {
                    decoded_bbox_vec.data[0] = clamp(decoded_bbox.xmin, 0.0f, clip_width);
                    decoded_bbox_vec.data[1] = clamp(decoded_bbox.ymin, 0.0f, clip_height);
                    decoded_bbox_vec.data[2] = clamp(decoded_bbox.xmax, 0.0f, clip_width);
                    decoded_bbox_vec.data[3] = clamp(decoded_bbox.ymax, 0.0f, clip_height);
                }
                else
                {
                    decoded_bbox_vec.data[0] = decoded_bbox.xmin;
                    decoded_bbox_vec.data[1] = decoded_bbox.ymin;
                    decoded_bbox_vec.data[2] = decoded_bbox.xmax;
                    decoded_bbox_vec.data[3] = decoded_bbox.ymax;
                }

                v_store(decoded_bboxes_vPtr[idx], decoded_bbox_vec);
            }
        }

        template <class T, std::size_t MANTISSA_SBITS>
        __global__ void findTopK(Span<int> indices, Span<int> count, View<T> scores, T threshold, size_type top_k, size_type num_classes, size_type class_offset, size_type num_priors, index_type background_label_id)
        {
            // indices: [batch_size, num_classes, topK]
            // count: [batch_size, num_classes]
            // scores: [batch_size, num_priors, num_classes]

            /* We need to sort boxes based on their confidence scores. The confidence scores fall in
             * the range [0.0, 1.0]. We break the range into bins and perform count sort. This is an
             * approximate algorithm but it almost always selects the boxes which would have been
             * selected by an exact sorting algorithm.
             *
             * Each block handles a particular class of a particular batch item. Since we do not require
             * many bins to accurately count sort the confidence scores, we can easily keep bin count array
             * in shared memory.
             */
            auto b = (blockIdx.x + class_offset) / num_classes;
            auto c = (blockIdx.x + class_offset) % num_classes;
            if (c == background_label_id)
                return;

            /* The floating point numbers are represented in the following format:
             *      [sign][exponent][mantissa]
             * fp32    1      8         23
             * fp16    1      5         10
             *
             * Since the confidence scores are always positive and at most one:
             * - sign is always 0
             * - exponent is always 0 (except for 1.0)
             *
             * Hence, we are only interested in the mantissa part (and exponent for 1.0).
             *
             * The most significant bits of the mantissa represent larger fractions, just like
             * the most significant bits of a binary integer represent higher powers of two.
             * For the purpose of comparing two floating point values, we can interpret the
             * mantissa as an integer.
             *
             * We use a power of two number of bins (say 2^n). This allows us to index the `bins`
             * array using the `n` most significant bits of the mantissa.
             *
             * Note that smaller scores will have a smaller index, i.e. the `bins` are ordered in
             * ascending order.
             */
            constexpr int BINS = 1 << MANTISSA_SBITS;

            __shared__ unsigned int bins[BINS];
            for (int i = threadIdx.x; i < BINS; i += blockDim.x)
                bins[i] = 0;

            __syncthreads();

            for (int i = threadIdx.x; i < num_priors; i += blockDim.x)
            {
                auto confidence = scores[b * (num_classes * num_priors) + num_classes * c + i];
                if (confidence > threshold)
                {
                    using device::extract_mantissa_bits;
                    int mantissa_index = extract_mantissa_bits<T>(confidence, MANTISSA_SBITS);
                    if (confidence == static_cast<T>(1.0))
                        mantissa_index = BINS - 1; /* exponent is one; put it in the last bin */

                    atomicAdd(&bins[mantissa_index], 1);
                }
            }

            __syncthreads();

            /* We have the counts of confidence scores in the bins. Now need to store the indices
             * of the `top_k` confidence values in the `indices` array.
             *
             * We use a little trick to parallelize the process of filling up the `indices` array.
             * We want every thread in the block to participate in the process. To do so, we compute
             * the reverse prefix sum of the bins array (the reason will be explained later).
             */
            if (threadIdx.x == 0)
            {
                for (int i = BINS - 1; i > 0; i--)
                    bins[i - 1] += bins[i];
            }

            if (threadIdx.x == 0)
                count[b * num_classes + c] = 0;

            __syncthreads();

            for (int i = threadIdx.x; i < num_priors; i += blockDim.x)
            {
                auto confidence = scores[b * (num_classes * num_priors) + num_classes * c + i];
                if (confidence > threshold)
                {
                    using device::extract_mantissa_bits;
                    int mantissa_index = extract_mantissa_bits<T>(confidence, MANTISSA_SBITS);
                    if (confidence == static_cast<T>(1.0))
                        mantissa_index = BINS - 1;

                    /* This bounding box is eligible to be selected unless it does not fall in
                     * the `top_k`. If it did, we would have to compute the location where it needs
                     * to be stored.
                     *
                     * Suppose we had just 4 bins and say the following were the counts:
                     * BIN0 2
                     * BIN1 1
                     * BIN2 3
                     * BIN3 2
                     *
                     * We will try our best to store in a nearly sorted order in the `indices` array.
                     * This requires that the boxes corresponding to the BIN3 must be stored first followed
                     * by BIN2, BIN1 and then BIN0.
                     *
                     * By computing the prefix sums, we can obtain the starting index for the bounding boxes
                     * corresponding to that bin.
                     */
                    index_type idx = bins[mantissa_index];
                    if (idx < top_k)
                    {
                        indices[b * (top_k * num_classes) + c * top_k + idx] = i;

                        /* We don't want a box to overwrite another box in its bin, so we increment the prefix sum
                         * corresponding to that bin by one. This ensures that the next box in the bin will be stored
                         * in the next empty slot.
                         */
                        atomicAdd(&bins[mantissa_index], 1);
                        atomicAdd(&count[b * num_classes + c], 1);
                    }
                }
            }

            __syncthreads();
        }
    }

    template <class T, bool NORMALIZED_BBOX>
    __global__ void blockwise_class_nms(Span<int> indices, View<int> count, View<T> decoded_bboxes, size_type num_classes, size_type class_offset, size_type top_k, index_type background_label_id, float nms_threshold)
    {
        // indices: [batch_size, num_classes, topK]
        // count: [batch_size, num_classes]

        auto b = (blockIdx.x + class_offset) / num_classes;
        auto c = (blockIdx.x + class_offset) % num_classes;
        if (c == background_label_id)
            return;

        using vector_type = get_vector_type_t<T, 4>;
        auto decoded_bboxes_vPtr = vector_type::get_pointer(decoded_bboxes.data());

        if (threadIdx.x == 0)
        {
            struct BoundingBox {
                float xmin, ymin, xmax, ymax;
            };

            const auto boxes = count[b * num_classes + c];
            for (int i = 0; i < boxes; i++)
            {
                auto idxA = indices[b * num_classes * top_k + c * num_classes + i];
                if (idxA == -1)
                    continue;

                vector_type boxA;
                v_load(boxA, decoded_bboxes_vPtr[idxA]);

                BoundingBox bbox1;
                bbox1.xmin = boxA.data[0];
                bbox1.ymin = boxA.data[1];
                bbox1.xmax = boxA.data[2];
                bbox1.ymax = boxA.data[3];

                for (int j = i; j < boxes; j++)
                {
                    auto idxB = indices[b * num_classes * top_k + c * num_classes + j];
                    if (idxB == -1)
                        continue;

                    vector_type boxB;
                    v_load(boxB, decoded_bboxes_vPtr[idxB]);

                    BoundingBox bbox2;
                    bbox2.xmin = boxB.data[0];
                    bbox2.ymin = boxB.data[1];
                    bbox2.xmax = boxB.data[2];
                    bbox2.ymax = boxB.data[3];

                    using device::min;
                    using device::max;

                    BoundingBox intersect_bbox;
                    intersect_bbox.xmin = std::max(bbox1.xmin, bbox2.xmin);
                    intersect_bbox.ymin = std::max(bbox1.ymin, bbox2.ymin);
                    intersect_bbox.xmax = std::min(bbox1.xmax, bbox2.xmax);
                    intersect_bbox.ymax = std::min(bbox1.ymax, bbox2.ymax);

                    float intersect_size = 0.0;
                    if (!(intersect_bbox.xmax < intersect_bbox.xmin || intersect_bbox.ymax < intersect_bbox.ymin))
                    {
                        float width = intersect_bbox.xmax - intersect_bbox.xmin;
                        float height = intersect_bbox.ymax - intersect_bbox.ymin;
                        if (NORMALIZED_BBOX)
                        {
                            intersect_size = width * height;
                        }
                        else
                        {
                            intersect_size = (width + 1) * (height + 1);
                        }
                    }

                    float overlap = 0;
                    if (intersect_size > 0)
                    {
                        float bbox1_size = 0, bbox2_size = 0;
                        if (!(bbox1.xmax < bbox1.xmin || bbox1.ymax < bbox1.ymin))
                        {
                            float width = bbox1.xmax - bbox1.xmin;
                            float height = bbox1.ymax - bbox1.ymin;
                            if (NORMALIZED_BBOX)
                            {
                                bbox1_size = width * height;
                            }
                            else
                            {
                                bbox1_size = (width + 1) * (height + 1);
                            }
                        }

                        if (!(bbox2.xmax < bbox2.xmin || bbox2.ymax < bbox2.ymin))
                        {
                            float width = bbox2.xmax - bbox2.xmin;
                            float height = bbox2.ymax - bbox2.ymin;
                            if (NORMALIZED_BBOX)
                            {
                                bbox2_size = width * height;
                            }
                            else
                            {
                                bbox2_size = (width + 1) * (height + 1);
                            }
                        }

                        overlap = intersect_size / (bbox1_size + bbox2_size - intersect_size);
                    }

                    if (overlap > nms_threshold)
                        indices[b * num_classes * top_k + c * num_classes + j] = -1;
                }
            }
        }
    }

    template <class T, bool VARIANCE_ENCODED_IN_TARGET, bool CORNER_TRUE_CENTER_FALSE, bool CLIP_BBOX>
    void launch_decode_boxes_kernel(const Stream& stream, Span<T> decoded_bboxes, View<T> locations, View<T> priors,
        bool share_location, bool transpose_location, bool normalized_bbox,
        size_type num_loc_classes, index_type background_label_id,
        float clip_width, float clip_height)
    {
        auto kernel = raw::decode_bbox<T, VARIANCE_ENCODED_IN_TARGET, CORNER_TRUE_CENTER_FALSE, CLIP_BBOX>;
        auto policy = make_policy(kernel, decoded_bboxes.size() / 4, 0, stream);
        policy.grid = {1, 1, 1};
        policy.block = { 1, 1, 1};
        launch_kernel(kernel, policy, decoded_bboxes, locations, priors, share_location, transpose_location, normalized_bbox, num_loc_classes, background_label_id, clip_width, clip_height);
    }

    template <class T, std::size_t current, class ...Args> static
    typename std::enable_if<current == 0, void>
    ::type dispatch_decode_bboxes(int selector, Args&& ...args) {
        if(selector == 0)
            launch_decode_boxes_kernel<T, 0, 0, 0>(std::forward<Args>(args)...);
    }

    template <class T, std::size_t current, class ...Args> static
    typename std::enable_if<current != 0, void>
    ::type dispatch_decode_bboxes(int selector, Args&& ...args) {
        if(selector == current)
            launch_decode_boxes_kernel<T, current & 4, current & 2, current & 1>(std::forward<Args>(args)...);
        else
            dispatch_decode_bboxes<T, current - 1, Args...>(selector, std::forward<Args>(args)...);
    }

    template <class T>
    void decode_bboxes(const Stream& stream, Span<T> output, View<T> locations, View<T> priors,
        std::size_t num_loc_classes,
        bool share_location, std::size_t background_label_id,
        bool transpose_location, bool variance_encoded_in_target,
        bool corner_true_or_center_false, bool normalized_bbox,
        bool clip_box, float clip_width, float clip_height)
    {
        unsigned int config = (variance_encoded_in_target << 2 | corner_true_or_center_false << 1 | clip_box);
        dispatch_decode_bboxes<T, 7>(config, stream, output, locations, priors, share_location, transpose_location, normalized_bbox, num_loc_classes, background_label_id, clip_width, clip_height);
    }

    template void decode_bboxes(const Stream&, Span<__half>, View<__half>, View<__half>, std::size_t, bool, std::size_t, bool, bool, bool, bool, bool, float, float);
    template void decode_bboxes(const Stream&, Span<float>, View<float>, View<float>, std::size_t, bool, std::size_t, bool, bool, bool, bool, bool, float, float);

    template <class T>
    void findTopK(const Stream& stream, TensorSpan<int> indices, TensorSpan<int> count, TensorView<T> scores, std::size_t top_k, std::size_t background_label_id, float threshold)
    {
        auto batch_size = scores.get_axis_size(0);
        auto num_classes = scores.get_axis_size(1);
        auto num_priors = scores.get_axis_size(2);

        auto num_blocks = num_classes;
        auto num_threads = std::min<std::size_t>(1024, num_priors);
        auto class_offset = 0;
        index_type optimized_bg_label_id = background_label_id;

        /* we can perform optimize cases where:
         * - `background_label_id` is the first class
         * - `background_label_id` is the last class
         */
        if (background_label_id == num_classes - 1)
        {
            /* skip the last class */
            num_blocks -= 1;
            optimized_bg_label_id = -1;
        }

        if (background_label_id == 0)
        {
            /* skip the first class and inform the kernel that the real classes start from one */
            class_offset = 1;
            num_blocks -= 1;
            optimized_bg_label_id = -1;
        }

        dim3 grid_size(num_blocks);
        dim3 block_size(num_threads);
        auto policy = execution_policy(grid_size, block_size, stream);

        auto kernel = raw::findTopK<T, 7>;
        launch_kernel(kernel, policy, indices, count, scores, threshold, top_k, num_classes, class_offset, num_priors, optimized_bg_label_id);
    }

    template void findTopK(const Stream&, TensorSpan<int>, TensorSpan<int>, TensorView<__half>, std::size_t, std::size_t, float);
    template void findTopK(const Stream&, TensorSpan<int>, TensorSpan<int>, TensorView<float>, std::size_t, std::size_t, float);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
