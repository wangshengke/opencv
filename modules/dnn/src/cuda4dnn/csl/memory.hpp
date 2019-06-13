// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_CUDA4DNN_CSL_MEMORY_HPP
#define OPENCV_DNN_CUDA4DNN_CSL_MEMORY_HPP

#include "error.hpp"
#include "pointer.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>
#include <type_traits>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl {

    /* @brief smart device pointer with allocation/deallocation methods
     *
     * ManagedPtr is a smart shared device pointer which also handles memory allocation.
     *
     * The cv qualifications of T are ignored. It does not make sense to allocate immutable
     * or volatile device memory.
     */
    template <class T>
    class ManagedPtr {
        static_assert(std::is_standard_layout<T>::value, "T must satisfy StandardLayoutType");

    public:
        using element_type = typename std::remove_cv<T>::type;

        using pointer = DevicePtr<element_type>;
        using const_pointer = DevicePtr<typename std::add_const<element_type>::type>;

        using size_type = std::size_t;

        ManagedPtr() noexcept : wrapped{ nullptr }, n{ 0 } { }
        ManagedPtr(const ManagedPtr&) noexcept = default;
        ManagedPtr(ManagedPtr&& other) noexcept
            : wrapped{ std::move(other.wrapped) }, n{ other.n }
        {
            other.reset();
        }

        /** allocates device memory for \p count number of elements
         *
         * Exception Guarantee: Strong
         */
        ManagedPtr(std::size_t count) {
            if (count <= 0) {
                CV_Error(Error::StsBadArg, "number of elements is zero or negative");
            }

            void* temp = nullptr;
            CUDA4DNN_CHECK_CUDA(cudaMalloc(&temp, count * sizeof(element_type)));

            auto ptr = typename pointer::pointer(static_cast<element_type*>(temp));
            wrapped.reset(ptr, [](element_type* ptr) {
                if (ptr != nullptr) {
                    /* contract violation for std::shared_ptr if cudaFree throws */
                    CUDA4DNN_CHECK_CUDA(cudaFree(ptr));
                }
            });

            n = count;
        }

        ManagedPtr& operator=(ManagedPtr&& other) noexcept {
            wrapped = std::move(other.wrapped);
            n = other.n;
            other.reset();
            return *this;
        }

        size_type size() const noexcept { return n; }

        void reset() noexcept { wrapped.reset(); n = 0; }

        /**
         * deallocates any previously allocated memory and allocates device memory
         * for \p count number of elements
         *
         * @note might optimize to avoid unnecessary reallocation
         *
         * Exception Guarantee: Strong
         */
        void reset(std::size_t count) {
            /* avoid unncessary reallocations */
            if (count <= n && (count > n / 2 || n * sizeof(T) < 256))
                return;

            ManagedPtr tmp(count);
            swap(tmp, *this);
        }

        pointer get() const noexcept { return pointer(wrapped.get()); }

        explicit operator bool() const noexcept { return wrapped; }

        friend bool operator==(const ManagedPtr& lhs, const ManagedPtr& rhs) noexcept { return lhs.wrapped == rhs.wrapped; }
        friend bool operator!=(const ManagedPtr& lhs, const ManagedPtr& rhs) noexcept { return lhs.wrapped != rhs.wrapped; }

        friend void swap(ManagedPtr& lhs, ManagedPtr& rhs) noexcept {
            using std::swap;
            swap(lhs.wrapped, rhs.wrapped);
            swap(lhs.n, rhs.n);
        }

    private:
        std::shared_ptr<element_type> wrapped;
        std::size_t n;
    };

    /** copies entire memory block pointed by \p src to \p dest
     *
     * \param[in]   src     device pointer
     * \param[out]  dest    host pointer
     *
     * Pre-conditions:
     * - memory pointed by \p dest must be large enough to hold the entire block of memory held by \p src
     *
     * Exception Guarantee: Basic
     */
    template <class T>
    void memcpy(T *dest, const ManagedPtr<T>& src) {
        memcpy<T>(dest, src.get(), src.size());
    }

    /** copies data from memory pointed by \p src to fully fill \p dest
     *
     * \param[in]   src     host pointer
     * \param[out]  dest    device pointer
     *
     * Pre-conditions:
     * - memory pointed by \p src must be at least as big as the memory block held by \p dest
     *
     * Exception Guarantee: Basic
     */
    template <class T>
    void memcpy(const ManagedPtr<T>& dest, const T* src) {
        memcpy<T>(dest.get(), src, dest.size());
    }

    /** copies data from memory pointed by \p src to \p dest
     *
     * if the two \p src and \p  dest have different sizes, the number of elements copied is
     * equal to the size of the smaller memory
     *
     * \param[in]   src     device pointer
     * \param[out]  dest    device pointer
     *
     * Exception Guarantee: Basic
     */
    template <class T>
    void memcpy(const ManagedPtr<T>& dest, const ManagedPtr<T>& src) {
        memcpy<T>(dest.get(), src.get(), std::min(dest.size(), src.size()));
    }

    /** sets device memory block to a specific 8-bit value
     *
     * \param[in]   src     device pointer
     * \param[out]  ch      8-bit value to fill the device memory with
     *
     * Exception Guarantee: Basic
     */
    template <class T>
    void memset(const ManagedPtr<T>& dest, std::int8_t ch) {
        memset<T>(dest.get(), ch, dest.size());
    }

    /** copies entire memory block pointed by \p src to \p dest asynchronously
     *
     * \param[in]   src     device pointer
     * \param[out]  dest    host pointer
     * \param       stream  CUDA stream that has to be used for the memory transfer
     *
     * Pre-conditions:
     * - memory pointed by \p dest must be large enough to hold the entire block of memory held by \p src
     * - \p dest points to page-locked memory
     *
     * Exception Guarantee: Basic
     */
    template <class T>
    void memcpy(T *dest, const ManagedPtr<T>& src, const Stream& stream) {
        CV_Assert(stream);
        memcpy<T>(dest, src.get(), src.size(), stream);
    }

    /** copies data from memory pointed by \p src to \p dest asynchronously
     *
     * \param[in]   src     host pointer
     * \param[out]  dest    device pointer
     * \param       stream  CUDA stream that has to be used for the memory transfer
     *
     * Pre-conditions:
     * - memory pointed by \p dest must be large enough to hold the entire block of memory held by \p src
     * - \p src points to page-locked memory
     *
     * Exception Guarantee: Basic
     */
    template <class T>
    void memcpy(const ManagedPtr<T>& dest, const T* src, const Stream& stream) {
        CV_Assert(stream);
        memcpy<T>(dest.get(), src, dest.size(), stream);
    }

    /** copies data from memory pointed by \p src to \p dest asynchronously
     *
     * \param[in]   src     device pointer
     * \param[out]  dest    device pointer
     * \param       stream  CUDA stream that has to be used for the memory transfer
     *
     * if the two \p src and \p  dest have different sizes, the number of elements copied is
     * equal to the size of the smaller memory
     *
     * Exception Guarantee: Basic
     */
    template <class T>
    void memcpy(ManagedPtr<T>& dest, const ManagedPtr<T>& src, const Stream& stream) {
        CV_Assert(stream);
        memcpy<T>(dest.get(), src.get(), std::min(dest.size(), src.size()), stream);
    }

    /** sets device memory block to a specific 8-bit value asynchronously
     *
     * \param[in]   src     device pointer
     * \param[out]  ch      8-bit value to fill the device memory with
     * \param       stream  CUDA stream that has to be used for the memory operation
     *
     * Exception Guarantee: Basic
     */
    template <class T>
    void memset(const ManagedPtr<T>& dest, int ch, const Stream& stream) {
        CV_Assert(stream);
        memset<T>(dest.get(), ch, dest.size(), stream);
    }

    /** @brief registers host memory as page-locked and unregisters on destruction
     *
     * Pre-conditons:
     * - host memory should be unregistered
     */
    class MemoryLockGuard {
    public:
        MemoryLockGuard() noexcept : ptr { nullptr } { }
        MemoryLockGuard(const MemoryLockGuard&) = delete;
        MemoryLockGuard(MemoryLockGuard&& other) noexcept : ptr{ other.ptr } {
            other.ptr = nullptr;
        }
        MemoryLockGuard(void* ptr_, std::size_t size_in_bytes) : ptr{ ptr_ } {
            CUDA4DNN_CHECK_CUDA(cudaHostRegister(ptr_, size_in_bytes, cudaHostRegisterPortable));
        }

        MemoryLockGuard& operator=(MemoryLockGuard&& other) noexcept {
            ptr = other.ptr;
            other.ptr = nullptr;
            return *this;
        }
        MemoryLockGuard& operator=(const MemoryLockGuard&) = delete;

        ~MemoryLockGuard() {
            if(ptr != nullptr)
                CUDA4DNN_CHECK_CUDA(cudaHostUnregister(ptr));
        }

    private:
        void *ptr;
    };

}}}} /* namespace cv::dnn::cuda4dnn::csl */

#endif /* OPENCV_DNN_CUDA4DNN_CSL_MEMORY_HPP */
