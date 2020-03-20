// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "math.hpp"
#include "stride_range.hpp"
#include "execution.hpp"
#include "vector_traits.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/span.hpp"
#include "../cuda4dnn/csl/tensor.hpp"

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    namespace raw {
        struct BoundingBox {
            float xmin, ymin, xmax, ymax;
        };

        template <class T, bool VARIANCE_ENCODED_IN_TARGET, bool CORNER_TRUE_CENTER_FALSE, bool CLIP_BBOX>
        __global__ void decode_bbox(Span<T> decoded_bboxes, View<T> locations, View<T> priors,
            bool share_location, bool transpose_location, bool normalized_bbox,
            size_type num_loc_classes, index_type background_label_id,
            float clip_width, float clip_height)
        {
            // decoded_bboxes: [batch_size, num_priors, num_loc_classes, 4]
            // locations: [batch_size, num_priors, num_loc_classes, 4]
            // priors: [1, 2, num_priors, 4]
            const size_type num_priors = priors.size() / 8; /* 4 bbox values + 4 variance values per prior */

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

        template <class T, std::size_t BINS>
        __global__ void findTopK(Span<int> indices, Span<int> count, View<T> scores, float threshold, size_type top_k, size_type num_classes, size_type class_offset, size_type num_priors, index_type background_label_id)
        {
            // indices: [batch_size, num_classes, topK]
            // count: [batch_size, num_classes]
            // scores: [batch_size, num_classes, num_priors]

            /* We need to sort boxes based on their confidence scores. The confidence scores fall in
             * the range [0.0, 1.0]. We break the range into bins and perform count sort. This is an
             * approximate algorithm but it almost always selects the boxes which would have been
             * selected by an exact sorting algorithm.
             *
             * Each block handles a particular class of a particular batch item.
             *
             * If the background class is the first or the last class, we can optimize and avoid wasting
             * a block. To know the exact details of `class_offset`, look at the kernel launch function.
             */
            const auto b = (blockIdx.x + class_offset) / num_classes;
            const auto c = (blockIdx.x + class_offset) % num_classes;
            if (c == background_label_id)
                return;

            /* We do not require a large number of bins to find the top K confidence scores. We will use
             * a reasonable number of bins which will fit in the shared memory.
             *
             * Note that smaller scores will have a smaller index, i.e. the `bins` are ordered in
             * ascending order.
             */

            __shared__ unsigned int bins[BINS];
            for (auto i : block_stride_range(BINS))
                bins[i] = 0;

            __syncthreads();

            for (auto i : block_stride_range(num_priors))
            {
                const float confidence = scores[b * (num_classes * num_priors) + c * num_priors + i];
                if (confidence > threshold)
                {
                    auto conf_scaled = (confidence - threshold)/(1 - threshold);

                    using device::clamp;
                    int bin_index = conf_scaled * BINS;
                    bin_index = clamp<int>(bin_index, 0, BINS - 1);

                    atomicAdd(&bins[bin_index], 1);
                }
            }

            __syncthreads();

            /* We have the counts of confidence scores in the bins. Our ultimate goal is to store the indices
             * of the `top_k` confidence values in the `indices` array.
             *
             * We use a little trick to parallelize the process of filling up the `indices` array.
             * We want every thread in the block to participate in the process. To do so, we shift the array
             * one place to the left and then compute the suffix sum of the bins array. The reason will be
             * explained later.
             */
            if (threadIdx.x < warpSize)
            {
                for (int i = threadIdx.x; i < BINS - 1; i += warpSize)
                {
                    auto temp = bins[i + 1];
                    __syncwarp();
                    bins[i] = temp;
                    // bins[i] won't be read again => no need for __syncwarp here
                }

                if (threadIdx.x == 0)
                    bins[BINS - 1] = 0;

                /* We can compute suffix sum of an array in groups of N numbers.
                 * Let N be 4 for this example.
                 *
                 * 1) Last 4 numbers
                 *                      1   2   3   4   |   5   6   7   8   |   9   10  11  12
                 * local suffix sum:                                            42  33  23  12
                 *
                 * 2) Middle 4 numbers
                 *                      1   2   3   4   |   5   6   7   8   |   9   10  11  12
                 * local suffix sum:                    |   26  21  15  8   |
                 *
                 * We add `42` (first element in the previous local group) to each element to get:
                 *
                 *                      1   2   3   4   |   5   6   7   8   |   9   10  11  12
                 *                                      |   68  63  57  50  |   42  33  23  12
                 * 3) First 4 numbers
                 *
                 *                      1   2   3   4   |   5   6   7   8   |   9   10  11  12
                 * local suffix sum:    10  9   7   4   |
                 *
                 * We add `68` (first element in the previous local group) to each element to get:
                 *
                 *                      1   2   3   4   |   5   6   7   8   |   9   10  11  12
                 * local suffix sum:    78  77  75  72  |   68  63  57  50  |   42  33  23  12
                 *
                 * What we are left with now is the suffix sum of the entire array.
                 *
                 * We use the aforementioned logic but work in groups of `warpSize`.
                 */
                constexpr int group_size = 32; /* must be equal to warpSize */
                assert(group_size == warpSize);

                static_assert(BINS % group_size == 0, "number of bins must be a multiple of warpSize");

                const auto inverse_lane_id = group_size - threadIdx.x - 1;
                unsigned int previous_group_first_element = 0;

                for (int warp_id = BINS / group_size - 1; warp_id >= 0; warp_id--)
                {
                    auto idx = warp_id * group_size + threadIdx.x;
                    auto value = bins[idx];

                    for (int i = 1; i < group_size; i *= 2)
                    {
                        int n = __shfl_down_sync(0xFFFFFFFF, value, i);
                        if (inverse_lane_id >= i)
                            value += n;
                    }

                    value += previous_group_first_element;
                    bins[idx] = value;

                    previous_group_first_element = __shfl_sync(0xFFFFFFFF, value, 0);
                }
            }

            if (threadIdx.x == 0)
                count[b * num_classes + c] = 0;

            __syncthreads();

            for (auto i : block_stride_range(num_priors))
            {
                const float confidence = scores[b * (num_classes * num_priors) + c * num_priors + i];
                if (confidence > threshold)
                {
                    auto conf_scaled = (confidence - threshold)/(1 - threshold);

                    int bin_index = conf_scaled * BINS;
                    bin_index = clamp<int>(bin_index, 0, BINS - 1);

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
                     * We first shift values in the `bins` array to the left by one. This gives us:
                     * BIN0 1
                     * BIN1 3
                     * BIN2 2
                     * BIN3 0
                     *
                     * We now compute the suffix sum of the array. This gives us:
                     * BIN0 6
                     * BIN1 5
                     * BIN2 2
                     * BIN3 0
                     *
                     * The bins now give us the location in the `indices` array from which the indices of the
                     * scores corresponding to that bin would be stored. We atomically increment the bin count
                     * everytime we store an index corresponding to that bin. Therefore, the value in the bins
                     * now give the index in the `indices` array where the next index corresponding to a score
                     * in that bin must be put.
                     */

                    const index_type idx = atomicAdd(&bins[bin_index], 1);
                    if (idx < top_k)
                    {
                        indices[b * num_classes * top_k + c * top_k + idx] = i;
                        atomicAdd(&count[b * num_classes + c], 1);
                    }
                }
            }
        }

        template <bool NORMALIZED_BBOX>
        __device__ float compute_size(BoundingBox bbox)
        {
            if (bbox.xmax < bbox.xmin || bbox.ymax < bbox.ymin)
                return 0.0;

            float width = bbox.xmax - bbox.xmin;
            float height = bbox.ymax - bbox.ymin;
            if (NORMALIZED_BBOX)
                return width * height;
            else
                return (width + 1) * (height + 1);
        }

        template <class T, bool NORMALIZED_BBOX>
        __global__ void blockwise_class_nms(Span<int> indices, Span<int> count, View<T> decoded_bboxes, bool share_location, size_type num_priors, size_type num_classes, size_type class_offset, size_type classwise_topK, index_type background_label_id, float nms_threshold)
        {
            // indices: [batch_size, num_classes, classwise_topK]
            // count: [batch_size, num_classes]
            // decoded_bboxes: [batch_size, num_priors, num_loc_classes, 4]

            auto b = (blockIdx.x + class_offset) / num_classes;
            auto c = (blockIdx.x + class_offset) % num_classes;
            if (c == background_label_id)
                return;

            using vector_type = get_vector_type_t<T, 4>;
            auto decoded_bboxes_vPtr = vector_type::get_pointer(decoded_bboxes.data());

            const auto boxes = count[b * num_classes + c];
            for (int i = 0; i < boxes; i++)
            {
                auto prior_id = indices[b * num_classes * classwise_topK + c * classwise_topK + i];
                if (prior_id != -1)
                {
                    const index_type idxA = share_location ?
                        b * num_priors + prior_id :
                        b * num_priors * num_classes + prior_id * num_classes + c;

                    vector_type boxA;
                    v_load(boxA, decoded_bboxes_vPtr[idxA]);

                    BoundingBox bbox1;
                    bbox1.xmin = boxA.data[0];
                    bbox1.ymin = boxA.data[1];
                    bbox1.xmax = boxA.data[2];
                    bbox1.ymax = boxA.data[3];

                    for (auto j : block_stride_range(i + 1, boxes))
                    {
                        prior_id = indices[b * num_classes * classwise_topK + c * classwise_topK + j];
                        if (prior_id == -1)
                            continue;

                        const index_type idxB = share_location ?
                            b * num_priors + prior_id :
                            b * num_priors * num_classes + prior_id * num_classes + c;

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
                        intersect_bbox.xmin = max(bbox1.xmin, bbox2.xmin);
                        intersect_bbox.ymin = max(bbox1.ymin, bbox2.ymin);
                        intersect_bbox.xmax = min(bbox1.xmax, bbox2.xmax);
                        intersect_bbox.ymax = min(bbox1.ymax, bbox2.ymax);

                        float intersect_size = compute_size<NORMALIZED_BBOX>(intersect_bbox), overlap = 0.0;
                        if (intersect_size > 0)
                        {
                            float bbox1_size = compute_size<NORMALIZED_BBOX>(bbox1);
                            float bbox2_size = compute_size<NORMALIZED_BBOX>(bbox2);
                            overlap = intersect_size / (bbox1_size + bbox2_size - intersect_size);
                        }

                        if (overlap > nms_threshold)
                            indices[b * num_classes * classwise_topK + c * classwise_topK + j] = -1;
                    }
                }

                __syncthreads();
            }

            if (threadIdx.x == 0)
                count[b * num_classes + c] = 0;

            __syncthreads();

            for (auto i : block_stride_range(boxes))
            {
                auto prior_id = indices[b * num_classes * classwise_topK + c * classwise_topK + i];
                if(prior_id != -1)
                {
                    const index_type idx = atomicAdd(&count[b * num_classes + c], 1);
                    indices[b * num_classes * classwise_topK + c * classwise_topK + idx] = prior_id;
                }
            }
        }

        template <class T, std::size_t BINS>
        __global__ void nms_collect(
            Span<int> kept_indices, Span<int> kept_count, View<int> indices, View<int> count, View<T> scores,
            size_type num_classes, size_type num_priors, size_type classwise_topK, size_type keepTopK, index_type background_label_id)
        {
            // kept_indices: [batch_size, keepTopK]
            // kept_count: [batch_size]

            // indices: [batch_size, num_classes, topK]
            // count: [batch_size, num_classes]
            // scores: [batch_size, num_classes, num_priors]

            const auto b = blockIdx.x;

            __shared__ unsigned int bins[BINS];
            for (auto i : block_stride_range(BINS))
                bins[i] = 0;

            __syncthreads();

            for (int c = 0; c < num_classes; c++)
            {
                if (c == background_label_id)
                    continue;

                auto boxes = count[b * num_classes + c];
                for (auto i : block_stride_range(boxes))
                {
                    auto prior_id = indices[b * num_classes * classwise_topK + c * classwise_topK + i];
                    const float confidence = scores[b * (num_classes * num_priors) + c * num_priors + prior_id];

                    using device::clamp;
                    int bin_index = confidence * BINS;
                    bin_index = clamp<int>(bin_index, 0, BINS - 1);

                    atomicAdd(&bins[bin_index], 1);
                }
            }

            __syncthreads();

            if (threadIdx.x < warpSize)
            {
                for (int i = threadIdx.x; i < BINS - 1; i += warpSize)
                {
                    auto temp = bins[i + 1];
                    __syncwarp();
                    bins[i] = temp;
                    // bins[i] won't be read again => no need for __syncwarp here
                }

                if (threadIdx.x == 0)
                    bins[BINS - 1] = 0;

                constexpr int group_size = 32; /* must be equal to warpSize */
                assert(group_size == warpSize);

                static_assert(BINS % group_size == 0, "number of bins must be a multiple of warpSize");

                const auto inverse_lane_id = group_size - threadIdx.x - 1;
                unsigned int previous_group_first_element = 0;

                for (int warp_id = BINS / group_size - 1; warp_id >= 0; warp_id--)
                {
                    auto idx = warp_id * group_size + threadIdx.x;
                    auto value = bins[idx];

                    for (int i = 1; i < group_size; i *= 2)
                    {
                        int n = __shfl_down_sync(0xFFFFFFFF, value, i);
                        if (inverse_lane_id >= i)
                            value += n;
                    }

                    value += previous_group_first_element;
                    bins[idx] = value;

                    previous_group_first_element = __shfl_sync(0xFFFFFFFF, value, 0);
                }
            }

            if (threadIdx.x == 0)
                kept_count[b] = 0;

            __syncthreads();

            for (int c = 0; c < num_classes; c++)
            {
                if (c == background_label_id)
                    continue;

                auto boxes = count[b * num_classes + c];
                for (auto i : block_stride_range(boxes))
                {
                    auto prior_id = indices[b * num_classes * classwise_topK + c * classwise_topK + i];
                    const float confidence = scores[b * (num_classes * num_priors) + c * num_priors + prior_id];

                    int bin_index = confidence * BINS;
                    bin_index = clamp<int>(bin_index, 0, BINS - 1);

                    const index_type idx = atomicAdd(&bins[bin_index], 1);
                    if (idx < keepTopK)
                    {
                        kept_indices[b * keepTopK + idx] = c * num_priors + prior_id;
                        atomicAdd(&kept_count[b], 1);
                    }
                }
            }
        }

        template <class T>
        __global__ void consolidate_detections(Span<T> output,
            View<int> kept_indices, View<int> kept_count, View<T> decoded_bboxes, View<T> scores, bool share_location,
            size_type batch_size, size_type num_classes, size_type num_priors, size_type keepTopK, DevicePtr<int> num_detections)
            {
                using vector_type = get_vector_type_t<T, 4>;
                auto decoded_bboxes_vPtr = vector_type::get_pointer(decoded_bboxes.data());

                // output: [1, 1, batch_size * keepTopK, 7]
                // kept_indices: [batch_size, keepTopK]
                // kept_count: [batch_size]
                // decoded_bboxes: [batch_size, num_priors, num_loc_classes, 4]
                // scores: [batch_size, num_classes, num_priors]

                for (int b = 0; b < batch_size; b++)
                {
                    for (auto i : grid_stride_range(kept_count[b]))
                    {
                        auto score_id = kept_indices[b * keepTopK + i];
                        auto c = score_id / num_priors;
                        auto prior_id = score_id % num_priors;

                        const auto confidence = scores[b * num_classes * num_priors + score_id];

                        index_type bbox_id;
                        if (share_location)
                        {
                            // decoded_bboxes: [batch_size, num_priors, 1, 4]
                            bbox_id = b * num_priors + prior_id;
                        }
                        else
                        {
                            // decoded_bboxes: [batch_size, num_priors, num_classes, 4]
                            bbox_id = b * num_priors * num_classes + prior_id * num_classes + c;
                        }

                        vector_type bbox;
                        v_load(bbox, decoded_bboxes_vPtr[bbox_id]);

                        auto output_id = atomicAdd(num_detections.get(), 1);
                        output[output_id * 7 + 0] = b;
                        output[output_id * 7 + 1] = c;
                        output[output_id * 7 + 2] = confidence;
                        output[output_id * 7 + 3] = bbox.data[0];
                        output[output_id * 7 + 4] = bbox.data[1];
                        output[output_id * 7 + 5] = bbox.data[2];
                        output[output_id * 7 + 6] = bbox.data[3];
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
        // indices: [batch_size, num_classes, topK]
        // count: [batch_size, num_classes]
        // scores: [batch_size, num_classes, num_priors]

        auto batch_size = scores.get_axis_size(0);
        auto num_classes = scores.get_axis_size(1);
        auto num_priors = scores.get_axis_size(2);

        auto num_blocks = batch_size * num_classes;
        auto num_threads = std::max<std::size_t>(std::min<std::size_t>(1024, num_priors), 32);
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

        auto kernel = raw::findTopK<T, 512>;
        launch_kernel(kernel, policy, indices, count, scores, threshold, top_k, num_classes, class_offset, num_priors, optimized_bg_label_id);
    }

    template void findTopK(const Stream&, TensorSpan<int>, TensorSpan<int>, TensorView<__half>, std::size_t, std::size_t, float);
    template void findTopK(const Stream&, TensorSpan<int>, TensorSpan<int>, TensorView<float>, std::size_t, std::size_t, float);

    template <class T>
    void blockwise_class_nms(const Stream& stream, TensorSpan<int> indices, TensorSpan<int> count, TensorView<T> decoded_bboxes,
        bool share_location, bool normalized_bbox, std::size_t background_label_id, float nms_threshold)
    {
        // indices: [batch_size, num_classes, topK]
        // count: [batch_size, num_classes]
        // decoded_bboxes: [batch_size, num_priors, num_pred_classes, 4]

        auto batch_size = indices.get_axis_size(0);
        auto num_classes = indices.get_axis_size(1);
        auto top_k = indices.get_axis_size(2);
        auto num_priors = decoded_bboxes.get_axis_size(1);

        CV_Assert(count.get_axis_size(0) == batch_size);
        CV_Assert(count.get_axis_size(1) == num_classes);
        CV_Assert(decoded_bboxes.get_axis_size(0) == batch_size);

        auto num_blocks = batch_size * num_classes;
        auto num_threads = std::max<std::size_t>(std::min<std::size_t>(1024, top_k), 32);

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

        if (normalized_bbox)
        {
            auto kernel = raw::blockwise_class_nms<T, true>;
            launch_kernel(kernel, policy, indices, count, decoded_bboxes, share_location, num_priors, num_classes, class_offset, top_k, optimized_bg_label_id, nms_threshold);
        }
        else
        {
            auto kernel = raw::blockwise_class_nms<T, false>;
            launch_kernel(kernel, policy, indices, count, decoded_bboxes, share_location, num_priors, num_classes, class_offset, top_k, optimized_bg_label_id, nms_threshold);
        }
    }

    template void blockwise_class_nms(const Stream&, TensorSpan<int>, TensorSpan<int>, TensorView<__half>, bool, bool, std::size_t, float);
    template void blockwise_class_nms(const Stream&, TensorSpan<int>, TensorSpan<int>, TensorView<float>, bool, bool, std::size_t, float);

    template <class T>
    void nms_collect(const Stream& stream, TensorSpan<int> kept_indices, TensorSpan<int> kept_count,
        TensorView<int> indices, TensorView<int> count, TensorView<T> scores, std::size_t background_label_id)
    {
        // kept_indices: [batch_size, keepTopK]
        // kept_count: [batch_size]

        // indices: [batch_size, num_classes, classwise_topK]
        // count: [batch_size, num_classes]
        // scores: [batch_size, num_classes, num_priors]

        auto batch_size = kept_indices.get_axis_size(0);
        CV_Assert(kept_count.get_axis_size(0) == batch_size);
        CV_Assert(indices.get_axis_size(0) == batch_size);
        CV_Assert(count.get_axis_size(0) == batch_size);
        CV_Assert(scores.get_axis_size(0) == batch_size);

        auto keepTopK = kept_indices.get_axis_size(1);

        auto num_classes = indices.get_axis_size(1);
        CV_Assert(count.get_axis_size(1) == num_classes);
        CV_Assert(scores.get_axis_size(1) == num_classes);

        auto classwise_topK = indices.get_axis_size(2);
        auto num_priors = scores.get_axis_size(2);

        auto num_blocks = batch_size;
        auto num_threads = 1024; // TODO

        dim3 grid_size(num_blocks);
        dim3 block_size(num_threads);
        auto policy = execution_policy(grid_size, block_size, stream);

        auto kernel = raw::nms_collect<T, 512>;
        launch_kernel(kernel, policy, kept_indices, kept_count, indices, count, scores, num_classes, num_priors, classwise_topK, keepTopK, background_label_id);
    }

    template void nms_collect(const Stream&, TensorSpan<int>, TensorSpan<int>, TensorView<int>, TensorView<int>, TensorView<__half>, std::size_t);
    template void nms_collect(const Stream&, TensorSpan<int>, TensorSpan<int>, TensorView<int>, TensorView<int>, TensorView<float>, std::size_t);

    template <class T>
    void consolidate_detections(const Stream& stream, TensorSpan<T> output,
        TensorView<int> kept_indices, TensorView<int> kept_count,
        TensorView<T> decoded_bboxes, TensorView<T> scores, bool share_location, DevicePtr<int> num_detections)
    {
        // output: [1, 1, batch_size * keepTopK, 7]
        // kept_indices: [batch_size, keepTopK]
        // kept_count: [batch_size]
        // decoded_bboxes: [batch_size, num_priors, num_loc_classes, 4]
        // scores: [batch_size, num_classes, num_priors]

        auto batch_size = kept_indices.get_axis_size(0);
        auto keepTopK = kept_indices.get_axis_size(1);

        auto num_classes = scores.get_axis_size(1);
        auto num_priors = scores.get_axis_size(2);

        auto kernel = raw::consolidate_detections<T>;
        auto policy = make_policy(kernel, keepTopK, 0, stream);
        launch_kernel(kernel, policy, output, kept_indices, kept_count, decoded_bboxes, scores, share_location, batch_size, num_classes, num_priors, keepTopK, num_detections);
    }

    template void consolidate_detections(const Stream&, TensorSpan<__half>, TensorView<int>, TensorView<int>, TensorView<__half>, TensorView<__half>, bool, DevicePtr<int>);
    template void consolidate_detections(const Stream&, TensorSpan<float>, TensorView<int>, TensorView<int>, TensorView<float>, TensorView<float>, bool, DevicePtr<int>);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
