// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_CUDA4DNN_CSL_TENSOR_HPP
#define OPENCV_DNN_CUDA4DNN_CSL_TENSOR_HPP

#include "nvcc_defs.hpp"
#include "memory.hpp"
#include "cublas.hpp"
#include "cudnn.hpp"

#include <opencv2/core.hpp>

#include <array>
#include <functional>
#include <algorithm>
#include <numeric>
#include <iterator>

#ifndef CSL_DEFAULT_TENSOR_RANK
    #define CSL_DEFAULT_TENSOR_RANK 4
#endif

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl {

    template <class T, std::size_t rank>
    class TensorSpan;

    template <class T, std::size_t rank>
    class TensorView;

    /** @brief multi-dimensional contiguous GPU tensor
     *
     * \tparam  T       type of data stored by the tensor
     * \tparam  rank    rank of the tensor
     */
    template <class T, std::size_t rank_ = CSL_DEFAULT_TENSOR_RANK>
    class Tensor {
        static_assert(std::is_standard_layout<T>::value, "T must staisfy StandardLayoutType");

    public:
        using value_type    = typename ManagedPtr<T>::element_type;
        using pointer       = typename ManagedPtr<value_type>::pointer;
        using const_pointer = typename ManagedPtr<value_type>::const_pointer;
        using size_type     = std::size_t;

        static constexpr auto rank = rank_;

        Tensor() noexcept { std::fill(std::begin(sizes), std::end(sizes), 0); }
        Tensor(const Tensor&) = delete;
        Tensor(Tensor&& other) noexcept {
            data = std::move(other.data);
            sizes = other.sizes;
            std::fill(std::begin(other.sizes), std::end(other.sizes), 0);
        }

        template <class ...Sizes>
        Tensor(Sizes... sizes) { resize(sizes...); }

        Tensor& operator=(const Tensor&) = delete;
        Tensor& operator=(Tensor&& other) noexcept {
            data = std::move(other.data);
            sizes = other.sizes;
            std::fill(std::begin(other.sizes), std::end(other.sizes), 0);
            return *this;
        }

        /** returns the total number of elements in the tensor */
        size_type size() const noexcept {
            return std::accumulate(std::begin(sizes), std::end(sizes), 1, std::multiplies<size_type>());
        }

        /** @brief returns the length of the axis
         *
         * Every axis is assigned a zero-based index which can be used to select an axis.
         * Negative index can be used to select an axis from the end.
         *
         * Examples:
         * > -1 represents the last axis
         * > 0 represents the first axis
         * > 1 represents the second axis
         *
         * Pre-conditions:
         * - the axis must be in the range [-rank, rank)
         */
        size_type get_axis_size(int axis) const noexcept {
            axis = axis < 0 ? rank + axis : axis;
            CV_Assert(axis >= 0 && axis < rank);
            return sizes[axis];
        }

        /** returns a device pointer to mutable device memory */
        pointer get() noexcept { return data.get(); }

        /** returns a device pointer to immutable device memory */
        const_pointer get() const noexcept { return data.get(); }

        /** @brief resizes the tensor
         *
         * Pre-conditions:
         * - [start, end) represents a range containing length of the axes in order
         * - number of axis sizes provided must be less than or equal to the tensor rank
         * - the sizes must be positive integers
         *
         * The length of unspecified axes will be assumed to be one.
         *
         * Exception Guarantee: Strong
         */
        template <class ForwardItr>
        typename std::enable_if<!std::is_integral<ForwardItr>::value, void> // TODO is_iterator
        ::type resize(ForwardItr start, ForwardItr end) {
            CV_Assert(start != end);
            CV_Assert(std::distance(start, end) <= rank);

            using ItrValueType = typename std::iterator_traits<ForwardItr>::value_type;
            auto total = std::accumulate(start, end, 1, std::multiplies<ItrValueType>());
            data.reset(total);

            /* length of the unspecified axes are assumed to be one */
            auto fill_sizes = rank - std::distance(start, end);
            std::fill_n(std::begin(sizes), fill_sizes, 1);
            std::copy(start, end, std::begin(sizes) + fill_sizes);
        }

        template <class ...Sizes>
        void resize(Sizes... new_sizes_) {
            static_assert(sizeof...(Sizes) <= rank, "number of axes exceeds the tensor rank");
            std::array<size_type, sizeof...(Sizes)> new_sizes = { static_cast<size_type>(new_sizes_)... };
            resize(std::begin(new_sizes), std::end(new_sizes));
        }

        /** @brief reshapes the tensor
         *
         * Length deduction:
         * The length of at most one axis can be deduced using the total size constraint. The axis can
         * be marked for deduction by specifying the size as -1.
         *
         * The axes for which no size was provided (excluding -1) will be assumed to be one.
         *
         * Pre-conditions:
         * - [start, end) represents a range containing length of the axes in order
         * - the number of axis lengths provided must be less than or equal to the tensor rank
         * - at most one axis length is allowed for length deduction
         * - the lengths provided must ensure that the total number of elements remains unchnged
         *
         * Post-conditions:
         * - if an axis was left unspecified, the value will be updated to have the correct size
         *
         * Exception Guarantee: Strong
         */
        template <class ForwardItr>
        typename std::enable_if<!std::is_integral<ForwardItr>::value, void> // TODO is_iterator
            ::type reshape(ForwardItr start, ForwardItr end) {
            CV_Assert(start != end);
            CV_Assert(std::distance(start, end) <= rank);

            using ItrValueType = typename std::iterator_traits<ForwardItr>::value_type;

            /* the user may leave at most one axis size for deduction by specifying -1 */
            auto sizes_to_deduce = std::count(start, end, -1);
            if (sizes_to_deduce > 1) { CV_Error(Error::StsBadArg, "only one axis size can be deduced"); }

            /* sizes must be positive numbers with the exception of -1 */
            auto invalid_sizes = std::count_if(start, end, [](ItrValueType x) {
                return !(x > 0 || x == -1);
                });
            if (invalid_sizes) { CV_Error(Error::StsBadArg, "invalid axis size"); }

            /* compute the total number of elements in the new tensor */
            size_type unknown_size = 0;
            auto total = std::accumulate(start, end, 1, std::multiplies<int>());
            if (total < 0) {
                /* there is an unknown size */
                if (std::abs(total) <= size()) {
                    unknown_size = size() / std::abs(total);
                    total = size();
                }
                /* Edge case: if `total` is already more than size(), skip the deduction as it's impossible
                ** Since `total` is negative, the size check which follows will fail and throw an error
                */
            }

            /* the number of elements before and after reshape must be exactly same */
            if (total != size()) {
                CV_Error(Error::StsBadArg, "new axes do not preserve the tensor element count");
            }

            /* we assume the size of the unspecified axes to be one */
            auto fill_sizes = rank - std::distance(start, end);
            std::fill_n(std::begin(sizes), fill_sizes, 1);
            std::copy(start, end, std::begin(sizes) + fill_sizes);

            /* replace the unknown axis with the correct value */
            std::replace(std::begin(sizes), std::end(sizes), size_type(-1), unknown_size);
        }

        template <class ...Sizes>
        void reshape(Sizes... new_sizes_) {
            static_assert(sizeof...(Sizes) <= rank, "number of axes exceeds the tensor rank");
            std::array<std::int64_t, sizeof...(Sizes)> new_sizes = { static_cast<std::int64_t>(new_sizes_)... };
            reshape(std::begin(new_sizes), std::end(new_sizes));
        }

        operator TensorSpan<T, rank_>() noexcept; /* defined later */
        operator TensorView<T, rank_>() noexcept; /* defined later */

        friend void swap(Tensor& lhs, Tensor& rhs) noexcept {
            using std::swap;
            swap(lhs.data, rhs.data);
            swap(lhs.sizes, rhs.sizes);
        }

    private:
        std::array<size_type, rank> sizes;
        ManagedPtr<value_type> data;
    };

    /** @brief spans a tensor
     *
     * \tparam  T       type of data stored by the tensor
     * \tparam  rank    rank of the tensor
     *
     * A span is valid if and only if the following hold true:
     * - the span is initialized
     * - parent tensor is still alive
     * - parent tensor holds a valid memory block
     * - parent tensor hasn't performed any resizing operation since the span was created
     */
    template <class T, std::size_t rank_ = CSL_DEFAULT_TENSOR_RANK>
    class TensorSpan {
    public:
        using tensor_type   = Tensor<T, rank_>;
        using value_type    = typename tensor_type::value_type;
        using pointer       = typename tensor_type::pointer;
        using const_pointer = typename tensor_type::const_pointer;
        using size_type     = typename tensor_type::size_type;

        static constexpr auto rank = rank_;

        TensorSpan() noexcept : ptr{ nullptr } { std::fill_n(sizes, rank, 0); }
        TensorSpan(const TensorSpan&) noexcept = default;
        TensorSpan(tensor_type& parent) noexcept : ptr{ parent.get() } {
            for (std::size_t i = 0; i < rank; i++)
                sizes[i] = parent.get_axis_size(i);
        }

        /* returns the total number of elements in the span */
        CUDA4DNN_HOST_DEVICE size_type size() const noexcept {
            return std::accumulate(std::begin(sizes), std::end(sizes), 1, std::multiplies<size_type>());
        }

        /** @brief returns the length of the axis
         *
         * Negative axis numbers can be used to select axis from the lower order.
         * Examples:
         * > -1 represents the last axis
         * > 0 represents the first axis
         * > 1 represents the second axis
         *
         * Pre-conditions:
         * - the axis must be in the range [-rank, rank)
         */
        CUDA4DNN_HOST_DEVICE size_type get_axis_size(int axis) const noexcept {
            axis = axis < 0 ? rank + axis : axis;
            CV_Assert(axis >= 0 && axis < rank);
            return sizes[axis];
        }

        /** returns a device pointer to mutable device memory */
        CUDA4DNN_HOST_DEVICE pointer get() noexcept { return ptr; }

        /** returns a device pointer to immutable device memory */
        CUDA4DNN_HOST_DEVICE const_pointer get() const noexcept { return ptr; }

        /** @brief reshapes the span
         *
         * Length deduction:
         * The length of at most one axis can be deduced using the total size constraint. The axis can
         * be marked for deduction by specifying the size as -1.
         *
         * The axes for which no size was provided (excluding -1) will be assumed to be one.
         *
         * Pre-conditions:
         * - [start, end) represents a range containing length of the axes in order
         * - the number of axis lengths provided must be less than or equal to the tensor rank
         * - at most one axis length is allowed for length deduction
         * - the lengths provided must ensure that the total number of elements remains unchnged
         *
         * Exception Guarantee: Strong
         */
        template <class ForwardItr> CUDA4DNN_HOST
        typename std::enable_if<!std::is_integral<ForwardItr>::value, void> // TODO is_iterator
        ::type reshape(ForwardItr start, ForwardItr end) {
            CV_Assert(start != end);
            CV_Assert(std::distance(start, end) <= rank);

            using ItrValueType = typename std::iterator_traits<ForwardItr>::value_type;

            /* the user may leave at most one axis size for deduction by specifying -1 */
            auto sizes_to_deduce = std::count(start, end, -1);
            if (sizes_to_deduce > 1) { CV_Error(Error::StsBadArg, "only one axis size can be deduced"); }

            /* sizes must be positive numbers with the exception of -1 */
            auto invalid_sizes = std::count_if(start, end, [](ItrValueType x) {
                return !(x > 0 || x == -1);
            });
            if (invalid_sizes) { CV_Error(Error::StsBadArg, "invalid axis size"); }

            /* compute the total number of elements in the new tensor */
            size_type unknown_size = 0;
            auto total = std::accumulate(start, end, 1, std::multiplies<int>());
            if (total < 0) {
                /* there is an unknown size */
                if (std::abs(total) <= size()) {
                    unknown_size = size() / std::abs(total);
                    total = size();
                }
                /* Edge case: if `total` is already more than size(), skip the deduction as it's impossible
                ** Since `total` is negative, the size check which follows will fail and throw an error
                */
            }

            /* the number of elements before and after reshape must be exactly same */
            if (total != size()) {
               CV_Error(Error::StsBadArg, "new axes do not preserve the tensor element count");
            }

            /* we assume the size of the unspecified axes to be one */
            auto fill_sizes = rank - std::distance(start, end);
            std::fill_n(std::begin(sizes), fill_sizes, 1);
            std::copy(start, end, std::begin(sizes) + fill_sizes);

            /* replace the unknown axis with the correct value */
            std::replace(std::begin(sizes), std::end(sizes), size_type(-1), unknown_size);
        }

        template <class ...Sizes>
        CUDA4DNN_HOST void reshape(Sizes... new_sizes_) {
            static_assert(sizeof...(Sizes) <= rank, "number of axes exceeds the tensor rank");
            std::array<std::int64_t, sizeof...(Sizes)> new_sizes = { static_cast<std::int64_t>(new_sizes_)... };
            reshape(std::begin(new_sizes), std::end(new_sizes));
        }

        /** @brief obtains a subspan of the span
         *
         * The axes for which no size was provided will be assumed to be one.
         *
         * Pre-conditions:
         * - the `offset` must be less than the size of the span
         * - [start, end) represents a range containing length of the subspan axes in order
         * - the number of axis lengths provided must be less than or equal to the tensor rank
         * - the lengths provided must ensure that the number of elements does not exceed (old size - offset)
         *
         * Exception Guarantee: Strong
         */
        template <class ForwardItr> CUDA4DNN_HOST
        typename std::enable_if<!std::is_integral<ForwardItr>::value, TensorSpan> // TODO is_iterator
        ::type subspan(size_type offset, ForwardItr start, ForwardItr end) {
            CV_Assert(start != end);
            CV_Assert(std::distance(start, end) <= rank);

            auto cur_size = size();
            CV_Assert(offset < cur_size);

            using ItrValueType = typename std::iterator_traits<ForwardItr>::value_type;

            /* sizes must be positive numbers */
            auto invalid_sizes = std::count_if(start, end, [](ItrValueType x) {
                return !(x > 0);
            });
            if (invalid_sizes) { CV_Error(Error::StsBadArg, "invalid axis size"); }

            /* the number of elements must be equal to the new size */
            auto max_size = (cur_size - offset);
            auto total = std::accumulate(start, end, 1, std::multiplies<ItrValueType>());
            if (total > max_size) {
                CV_Error(Error::StsBadArg, "axes lengths lead to OOB accesses");
            }

            TensorSpan temp;

            /* we assume the size of the unspecified axes to be one */
            auto fill_sizes = rank - std::distance(start, end);
            std::fill_n(std::begin(temp.sizes), fill_sizes, 1);
            std::copy(start, end, std::begin(temp.sizes) + fill_sizes);

            temp.ptr = ptr + offset;
            return temp;
        }

        template <class ...Sizes>
        CUDA4DNN_HOST TensorSpan subspan(size_type offset, Sizes... new_sizes_) {
            static_assert(sizeof...(Sizes) <= rank, "number of axes exceeds the tensor rank");
            std::array<std::int64_t, sizeof...(Sizes)> new_sizes = { static_cast<std::int64_t>(new_sizes_)... };
            return subspan(offset, std::begin(new_sizes), std::end(new_sizes));
        }

        operator TensorView<T, rank_>() noexcept; /* defined later */

        friend void swap(TensorSpan& lhs, TensorSpan& rhs) noexcept {
            using std::swap;
            swap(lhs.ptr, rhs.ptr);
            swap(lhs.sizes, rhs.sizes);
        }

    private:
        size_type sizes[rank];
        pointer ptr;
    };

    template <class T, std::size_t rank_>
    Tensor<T, rank_>::operator TensorSpan<T, rank_>() noexcept {
        return TensorSpan<T, rank_>(*this);
    }

    /** @brief view of a tensor
     *
     * \tparam  T       type of data stored by the tensor
     * \tparam  rank    rank of the tensor
     *
     * A view is valid if and only if the following hold true:
     * - the view is initialized
     * - parent tensor is still alive
     * - parent tensor holds a valid memory block
     * - parent tensor hasn't performed any resizing operation since the view was created
     */
    template <class T, std::size_t rank_ = CSL_DEFAULT_TENSOR_RANK>
    class TensorView {
    public:
        using tensor_type = Tensor<T, rank_>;
        using value_type = typename tensor_type::value_type;
        using pointer = typename tensor_type::pointer;
        using const_pointer = typename tensor_type::const_pointer;
        using size_type = typename tensor_type::size_type;

        static constexpr auto rank = rank_;

        TensorView() noexcept : ptr{ nullptr } { std::fill_n(sizes, rank, 0); }
        TensorView(const TensorView&) noexcept = default;
        TensorView(const TensorSpan<T, rank_>& other) noexcept : ptr{ other.get() } {
            for (int i = 0; i < rank; i++)
                sizes[i] = other.get_axis_size(i);
        }
        TensorView(const tensor_type& parent) noexcept : ptr{ parent.get() } {
            for (std::size_t i = 0; i < rank; i++)
                sizes[i] = parent.get_axis_size(i);
        }

        TensorView& operator=(const TensorView&) = default;
        TensorView& operator=(const TensorSpan<T, rank_>& other) noexcept {
            TensorView tmp(other);
            swap(*this, tmp);
            return *this;
        }

        /* returns the total number of elements in the view */
        CUDA4DNN_HOST_DEVICE size_type size() const noexcept {
            return std::accumulate(std::begin(sizes), std::end(sizes), 1, std::multiplies<size_type>());
        }

        /** @brief returns the length of the axis
         *
         * Negative axis numbers can be used to select axis from the lower order.
         * Examples:
         * > -1 represents the last axis
         * > 0 represents the first axis
         * > 1 represents the second axis
         *
         * Pre-conditions:
         * - the axis must be in the range [-rank, rank)
         */
        CUDA4DNN_HOST_DEVICE size_type get_axis_size(int axis) const noexcept {
            axis = axis < 0 ? rank + axis : axis;
            CV_Assert(axis >= 0 && axis < rank);
            return sizes[axis];
        }

        /** returns a device pointer to immutable device memory */
        CUDA4DNN_HOST_DEVICE const_pointer get() const noexcept { return ptr; }

        /** @brief reshapes the view
         *
         * Length deduction:
         * The length of at most one axis can be deduced using the total size constraint. The axis can
         * be marked for deduction by specifying the size as -1.
         *
         * The axes for which no size was provided (excluding -1) will be assumed to be one.
         *
         * Pre-conditions:
         * - [start, end) represents a range containing length of the axes in order
         * - the number of axis lengths provided must be less than or equal to the tensor rank
         * - at most one axis length is allowed for length deduction
         * - the lengths provided must ensure that the total number of elements remains unchnged
         *
         * Exception Guarantee: Strong
         */
        template <class ForwardItr> CUDA4DNN_HOST
        typename std::enable_if<!std::is_integral<ForwardItr>::value, void>
        ::type reshape(ForwardItr start, ForwardItr end) {
            CV_Assert(start != end);
            CV_Assert(std::distance(start, end) <= rank);

            using ItrValueType = typename std::iterator_traits<ForwardItr>::value_type;

            /* the user may leave at most one axis size for deduction by specifying -1 */
            auto sizes_to_deduce = std::count(start, end, -1);
            if (sizes_to_deduce > 1) { CV_Error(Error::StsBadArg, "only one axis size can be deduced"); }

            /* sizes must be positive numbers with the exception of -1 */
            auto invalid_sizes = std::count_if(start, end, [](ItrValueType x) {
                return !(x > 0 || x == -1);
            });
            if (invalid_sizes) { CV_Error(Error::StsBadArg, "invalid axis size"); }

            /* compute the total number of elements in the new tensor */
            size_type unknown_size = 0;
            auto total = std::accumulate(start, end, 1, std::multiplies<int>());
            if (total < 0) {
                /* there is an unknown size */
                if (std::abs(total) <= size()) {
                    unknown_size = size() / std::abs(total);
                    total = size();
                }
                /* Edge case: if `total` is already more than size(), skip the deduction as it's impossible
                ** Since `total` is negative, the size check which follows will fail and throw an error
                */
            }

            /* the number of elements before and after reshape must be exactly same */
            if (total != size()) {
                // CV_Error(Error::StsBadArg, "new axes do not preserve the tensor element count");
            }

            /* we assume the size of the unspecified axes to be one */
            auto fill_sizes = rank - std::distance(start, end);
            std::fill_n(std::begin(sizes), fill_sizes, 1);
            std::copy(start, end, std::begin(sizes) + fill_sizes);

            /* replace the unknown axis with the correct value */
            std::replace(std::begin(sizes), std::end(sizes), size_type(-1), unknown_size);
        }

        template <class ...Sizes>
        CUDA4DNN_HOST void reshape(Sizes... new_sizes_) {
            static_assert(sizeof...(Sizes) <= rank, "number of axes exceeds the tensor rank");
            std::array<std::int64_t, sizeof...(Sizes)> new_sizes = { static_cast<std::int64_t>(new_sizes_)... };
            reshape(std::begin(new_sizes), std::end(new_sizes));
        }

        /** @brief obtains a subview of the view
         *
         * The axes for which no size was provided will be assumed to be one.
         *
         * Pre-conditions:
         * - the `offset` must be less than the size of the view
         * - [start, end) represents a range containing length of the subview axes in order
         * - the number of axis lengths provided must be less than or equal to the tensor rank
         * - the lengths provided must ensure that the number of elements does not exceed (old size - offset)
         *
         * Exception Guarantee: Strong
         */
        template <class ForwardItr> CUDA4DNN_HOST
        typename std::enable_if<!std::is_integral<ForwardItr>::value, TensorView> // TODO is_iterator
        ::type subview(size_type offset, ForwardItr start, ForwardItr end) {
            CV_Assert(start != end);
            CV_Assert(std::distance(start, end) <= rank);

            auto cur_size = size();
            CV_Assert(offset < cur_size);

            using ItrValueType = typename std::iterator_traits<ForwardItr>::value_type;

            /* sizes must be positive numbers */
            auto invalid_sizes = std::count_if(start, end, [](ItrValueType x) {
                return !(x > 0);
            });
            if (invalid_sizes) { CV_Error(Error::StsBadArg, "invalid axis size"); }

            /* the number of elements must be equal to the new size */
            auto max_size = (cur_size - offset);
            auto total = std::accumulate(start, end, 1, std::multiplies<ItrValueType>());
            if (total > max_size) {
                CV_Error(Error::StsBadArg, "axes lengths lead to OOB accesses");
            }

            TensorView temp;

            /* we assume the size of the unspecified axes to be one */
            auto fill_sizes = rank - std::distance(start, end);
            std::fill_n(std::begin(temp.sizes), fill_sizes, 1);
            std::copy(start, end, std::begin(temp.sizes) + fill_sizes);

            temp.ptr = ptr + offset;
            return temp;
        }

        template <class ...Sizes>
        CUDA4DNN_HOST TensorView subview(size_type offset, Sizes... new_sizes_) {
            static_assert(sizeof...(Sizes) <= rank, "number of axes exceeds the tensor rank");
            std::array<std::int64_t, sizeof...(Sizes)> new_sizes = { static_cast<std::int64_t>(new_sizes_)... };
            return subview(offset, std::begin(new_sizes), std::end(new_sizes));
        }

        friend void swap(TensorView& lhs, TensorView& rhs) noexcept {
            using std::swap;
            swap(lhs.ptr, rhs.ptr);
            swap(lhs.sizes, rhs.sizes);
        }

    private:
        size_type sizes[rank];
        const_pointer ptr;
    };

    template <class T, std::size_t rank_>
    Tensor<T, rank_>::operator TensorView<T, rank_>() noexcept {
        return TensorView<T, rank_>(*this);
    }

    template <class T, std::size_t rank_>
    TensorSpan<T, rank_>::operator TensorView<T, rank_>() noexcept {
        return TensorView<T, rank_>(*this);
    }

    namespace tensor_utils {
        /** returns true if the two Tensor/TensorSpan/TensorView objects have the same shape */
        template <class TensorType1, class TensorType2> inline
            bool is_same_shape(const TensorType1& x, const TensorType2& y) noexcept {
            constexpr auto rank1 = TensorType1::rank;
            constexpr auto rank2 = TensorType2::rank;

            if (rank1 != rank2)
                return false;

            for (int i = 0; i < rank1; i++)
                if (x.get_axis_size(i) != y.get_axis_size(i))
                    return false;
            return true;
        }

        /** returns the rank to which the given tensor can be squeezed to */
        template <class TensorType> inline
        std::size_t get_effective_rank(const TensorType& x) noexcept {
            constexpr auto rank = TensorType::rank;
            std::size_t effective_rank = rank;
            for (int i = 0; i < rank; i++, effective_rank--)
                if (x.get_axis_size(i) != 1)
                    break;
            return effective_rank;
        }
    }

    namespace tensor_ops {

        /** @brief performs matrix-multplication
         *
         * Pre-conditions:
         * - \p A and \p B must meet the mathematical requirements for matrix multiplication
         * - \p result must be large enough to hold the result
         *
         * Exception Gaurantee: Basic
         */
        template <class T> inline
        void multiply(const cublas::Handle& handle, TensorSpan<T> result, TensorView<T> A, TensorView<T> B) {
            /* matrix operations can be performed only on rank two tensors */
            using tensor_utils::get_effective_rank;
            CV_Assert(get_effective_rank(A) == 2 &&
                      get_effective_rank(B) == 2 &&
                      get_effective_rank(result) == 2);

            /* check dimension requirements for matrix multiplication */
            CV_Assert(A.get_axis_size(-2) == result.get_axis_size(-2));
            CV_Assert(A.get_axis_size(-1) == B.get_axis_size(-2));
            CV_Assert(B.get_axis_size(-1) == result.get_axis_size(-1));

            const auto dest_nr = result.get_axis_size(-2);
            const auto dest_nc = result.get_axis_size(-1);
            const auto A_nc = A.get_axis_size(-1);
            const auto B_nr = B.get_axis_size(-2);
            const auto B_nc = B.get_axis_size(-1);

            cublas::gemm<T>(handle,
                false, false,
                dest_nc, dest_nr, B_nr,
                1.0, A.get(), A_nc,
                B.get(), B_nc,
                0.0, result.get(), dest_nc);
        }

        /** @brief performs matrix-addition
         *
         * Pre-conditions:
         * - \p A and \p B must meet the mathematical requirements for matrix addition
         * - \p result must be large enough to hold the result
         *
         * Exception Gaurantee: Basic
         */
        template <class T> inline
        void add(const cublas::Handle& handle, TensorSpan<T> result, TensorView<T> A, TensorView<T> B) {
            /* matrix operations can be performed only on rank two tensors */
            using tensor_utils::get_effective_rank;
            CV_Assert(get_effective_rank(A) == 2 &&
                      get_effective_rank(B) == 2 &&
                      get_effective_rank(result) == 2);

            /* check dimension requirements for matrix addition */
            CV_Assert(tensor_utils::is_same_shape(A, B));
            CV_Assert(tensor_utils::is_same_shape(A, result));

            const auto dest_nr = result.get_axis_size(-2);
            const auto dest_nc = result.get_axis_size(-1);
            const auto A_nr = A.get_axis_size(-2);
            const auto B_nr = B.get_axis_size(-2);

            cublas::geam<T>(handle,
                false, false,
                dest_nr, dest_nc,
                1.0, A.get(), A_nr,
                1.0, B.get(), B_nr,
                result.get(), dest_nr);
        }

        template <class T> inline
        void softmax(const cudnn::Handle& handle, TensorSpan<T> output, TensorView<T> input, bool log) {
            CV_Assert(input.rank == 4 && output.rank == 4);
            CV_Assert(tensor_utils::is_same_shape(output, input));

            using cudnn::TensorDescriptor;
            auto input_desc = TensorDescriptor<T>(
                input.get_axis_size(0),
                input.get_axis_size(1),
                input.get_axis_size(2),
                input.get_axis_size(3)
            );

            auto output_desc = TensorDescriptor<T>(
                output.get_axis_size(0),
                output.get_axis_size(1),
                output.get_axis_size(2),
                output.get_axis_size(3)
            );

            cudnn::softmax(handle, output_desc, output.get(), input_desc, input.get(), log);
        }
    }

}}}} /* namespace cv::dnn::cuda4dnn::csl */

#endif /* OPENCV_DNN_CUDA4DNN_CSL_TENSOR_HPP*/
