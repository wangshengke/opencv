// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_CUDA4DNN_PRIMITIVES_MAX_UNPOOLING_HPP
#define OPENCV_DNN_CUDA4DNN_PRIMITIVES_MAX_UNPOOLING_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"
#include "../csl/kernels.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <vector>
#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    struct MaxPoolingConfiguration {
        /* the size of the following vectors must be equal to the pooling order */
        std::vector<std::size_t> window_size;
        std::vector<std::size_t> strides;

        enum class padding_mode {
            manual, /* uses explicit padding values provided in `pads_begin` and `pads_end` */
            valid, /* no padding is added */
            same /* TensorFlow logic is used for same padding */
        };

        padding_mode padMode;

        /* explicit paddings are used if and only if padMode is set to manual */
        std::vector<std::size_t> pads_begin;

        /* full shape inclusive of channel and batch axis */
        std::vector<std::size_t> input_shape;
    };

    template <class T>
    class MaxPoolingOp final : public CUDABackendNode {
    public:
        using wrapper_type = GetCUDABackendWrapperType<T>;

        MaxPoolingOp(csl::Stream stream_, const MaxPoolingConfiguration& config)
            : stream(std::move(stream_))
        {
            window_size = config.window_size;

            const auto pooling_order = window_size.size();
            CV_Assert(pooling_order >= 1);

            strides = config.strides;
            CV_Assert(pooling_order == strides.size());

            if (pooling_order != 2 && pooling_order != 3)
                CV_Error(Error::StsNotImplemented, "Only 2D/3D max-pooling are supported.");

            padding_left.resize(pooling_order);
            if (config.padMode == MaxPoolingConfiguration::padding_mode::manual)
            {
                const auto& pads_begin = config.pads_begin;
                CV_Assert(pooling_order == pads_begin.size());

                padding_left.assign(std::begin(pads_begin), std::end(pads_begin));
            }
            else if (config.padMode == MaxPoolingConfiguration::padding_mode::valid)
            {
                /* nothing to do as the paddings are already preset to zero */
            }
            else if (config.padMode == MaxPoolingConfiguration::padding_mode::same)
            {
                /* TensorFlow Logic:
                 * total_padding[i] = (o[i] - 1) * s[i] + effective_k[i] - i[i]
                 *
                 * if total padding is odd, the extra is added towards the end
                 */
                const auto& input_shape = config.input_shape;
                CV_Assert(input_shape.size() == pooling_order + 2);

                for (int i = 0; i < pooling_order; i++)
                {
                    const auto output_dim = (input_shape[i + 2] - 1 + strides[i]) / strides[i];
                    const auto required_total_padding =
                        std::max<std::int64_t>(0, (output_dim - 1) * strides[i] + window_size[i] - input_shape[i + 2]);

                    padding_left[i] = required_total_padding / 2;
                }
            }
        }

        void forward(
            std::vector<cv::Ptr<BackendWrapper>>& inputs,
            std::vector<cv::Ptr<BackendWrapper>>& outputs,
            csl::Workspace& workspace) override
        {
            CV_Assert(inputs.size() == 1 && outputs.size() == 2);

            auto input_wrapper = inputs[0].dynamicCast<wrapper_type>();
            auto input_data = input_wrapper->getView();

            auto output_wrapper = outputs[0].dynamicCast<wrapper_type>();
            auto output_data = output_wrapper->getSpan();

            auto indices_wrapper = outputs[1].dynamicCast<wrapper_type>();
            auto output_indices = indices_wrapper->getSpan();

            csl::kernels::max_pooling_with_indices<T>(
                stream, output_data, output_indices, input_data, window_size, strides, padding_left
            );
        }

    private:
        csl::Stream stream;

        std::vector<std::size_t> window_size, strides, padding_left;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_CUDA4DNN_PRIMITIVES_MAX_UNPOOLING_HPP */
