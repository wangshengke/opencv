// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_CSL_GRAPH_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_CSL_GRAPH_HPP

#include "error.hpp"

#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>

#include <cuda_runtime.h>

#include <memory>
#include <sstream>
#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl {

    class BaseUniqueGraphNode;

    class UniqueGraph {
    public:
        UniqueGraph() noexcept : graph{ 0 } { }
        UniqueGraph(UniqueGraph&) = delete;
        UniqueGraph(UniqueGraph&& other) noexcept {
            graph = other.graph;
            other.graph = 0;
        }

        UniqueGraph(bool create) : graph{ 0 } {
            if (create) {
                CUDA4DNN_CHECK_CUDA(cudaGraphCreate(&graph, 0));
            }
        }

        ~UniqueGraph() {
            try {
                nodes.clear(); /* destroy nodes first */
                if (graph != 0)
                    CUDA4DNN_CHECK_CUDA(cudaGraphDestroy(graph));
            } catch (const CUDAException& ex) {
                std::ostringstream os;
                os << "Asynchronous exception caught during CUDA graph destruction.\n";
                os << ex.what();
                os << "Exception will be ignored.\n";
                CV_LOG_WARNING(0, os.str().c_str());
            }
        }

        UniqueGraph& operator=(const UniqueGraph&) = delete;
        UniqueGraph& operator=(UniqueGraph&& other) noexcept {
            graph = other.graph;
            other.graph = 0;
            return *this;
        }

        cudaGraph_t get() const noexcept { return graph; }

        void add_node(std::shared_ptr<BaseUniqueGraphNode> node) { nodes.emplace_back(std::move(node)); }

    private:
        friend class Graph;

        cudaGraph_t graph;
        std::vector <std::shared_ptr<BaseUniqueGraphNode>> nodes;
    };

    class Graph {
    public:
        Graph() : graph(std::make_shared<UniqueGraph>()) { }
        Graph(const Graph&) = default;
        Graph(Graph&&) = default;

        Graph(bool create) : graph(std::make_shared<UniqueGraph>(create)) { }

        Graph& operator=(const Graph&) = default;
        Graph& operator=(Graph&&) = default;

        cudaGraph_t get() const noexcept {
            CV_Assert(graph);
            return graph->get();
        }

        template <class Func>
        void capture(csl::Stream& stream, Func&& f) {
            graph = std::make_shared<UniqueGraph>();
            cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeThreadLocal);
            f();
            cudaStreamEndCapture(stream.get(), &graph->graph);

            // fix exception safety
        }

        void add_node(std::shared_ptr<BaseUniqueGraphNode> node) { graph->add_node(std::move(node)); }

    private:
        std::shared_ptr<UniqueGraph> graph;
    };

    class BaseUniqueGraphNode {
    public:
        BaseUniqueGraphNode() noexcept : node{ 0 } { }
        BaseUniqueGraphNode(cudaGraphNode_t node_) : node{node_} { }
        BaseUniqueGraphNode(BaseUniqueGraphNode&) = delete;
        BaseUniqueGraphNode(BaseUniqueGraphNode&& other) noexcept {
            node = other.node;
            other.node = 0;
        }

        virtual ~BaseUniqueGraphNode() {
            try {
                if (node != 0) {
                    CUDA4DNN_CHECK_CUDA(cudaGraphDestroyNode(node));
                }
            } catch (const CUDAException& ex) {
                std::ostringstream os;
                os << "Asynchronous exception caught during CUDA graph node destruction.\n";
                os << ex.what();
                os << "Exception will be ignored.\n";
                CV_LOG_WARNING(0, os.str().c_str());
            }
        }

        BaseUniqueGraphNode& operator=(const BaseUniqueGraphNode&) = delete;
        BaseUniqueGraphNode& operator=(BaseUniqueGraphNode&& other) noexcept {
            node = other.node;
            other.node = 0;
            return *this;
        }

        virtual cudaGraphNode_t get() const noexcept {
            CV_Assert(node);
            return node;
        }

    protected:
        cudaGraphNode_t node;
    };

    class BaseGraphNode {
    public:
        BaseGraphNode() = default;
        BaseGraphNode(BaseGraphNode&) = default;
        BaseGraphNode(BaseGraphNode&& other) noexcept : node(std::move(other.node)) { }

        template <class APIFunc, class ...Args>
        BaseGraphNode(Graph graph, APIFunc func, Args&& ...args)
            : owning_graph(std::move(graph))
        {
            cudaGraphNode_t pGraphNode;
            CUDA4DNN_CHECK_CUDA(func(&pGraphNode, owning_graph.get(), std::forward<Args>(args)...));
            try {
                node = std::make_shared<BaseUniqueGraphNode>(pGraphNode);
                owning_graph.add_node(node);
            } catch(...) {
                CUDA4DNN_CHECK_CUDA(cudaGraphDestroyNode(pGraphNode));
            }
        }

        virtual ~BaseGraphNode() { }

        BaseGraphNode& operator=(const BaseGraphNode&) = default;
        BaseGraphNode& operator=(BaseGraphNode&&) = default;

        template <class ParentNode>
        void add_dependency(ParentNode parent)
        {
            auto from = parent->get();
            auto to = node->get();
            CUDA4DNN_CHECK_CUDA(cudaGraphAddDependencies(owning_graph.get(), from, to, 1));
        }

        virtual cudaGraphNode_t get() const noexcept {
            CV_Assert(node);
            return node->get();
        }

    protected:
        std::shared_ptr<BaseUniqueGraphNode> node;
        Graph owning_graph;
    };

    class GraphEmptyNode : public BaseGraphNode {
    public:
        GraphEmptyNode() = default;
        GraphEmptyNode(GraphEmptyNode&) = default;
        GraphEmptyNode(GraphEmptyNode&& other) = default;
        GraphEmptyNode(Graph graph) : BaseGraphNode(std::move(graph), cudaGraphAddEmptyNode, nullptr, 0) { }

        GraphEmptyNode& operator=(const GraphEmptyNode&) = default;
        GraphEmptyNode& operator=(GraphEmptyNode&&) = default;
    };

    class GraphChildGraphNode : public BaseGraphNode {
    public:
        GraphChildGraphNode() = default;
        GraphChildGraphNode(GraphChildGraphNode&) = default;
        GraphChildGraphNode(GraphChildGraphNode&& other) = default;
        GraphChildGraphNode(Graph graph, Graph child) : BaseGraphNode(std::move(graph), cudaGraphAddChildGraphNode, nullptr, 0, child.get()) { }

        GraphChildGraphNode& operator=(const GraphChildGraphNode&) = default;
        GraphChildGraphNode& operator=(GraphChildGraphNode&&) = default;
    };

    class GraphHostNode : public BaseGraphNode {
    public:
        GraphHostNode() = default;
        GraphHostNode(GraphHostNode&) = default;
        GraphHostNode(GraphHostNode&& other) = default;
        GraphHostNode(Graph graph) : BaseGraphNode(std::move(graph), cudaGraphAddHostNode, nullptr, 0, nullptr) { }

        void set_params(void *args)
        {
            params.userData = args;
            CUDA4DNN_CHECK_CUDA(cudaGraphHostNodeSetParams(BaseGraphNode::get(), &params));
        }

        GraphHostNode& operator=(const GraphHostNode&) = default;
        GraphHostNode& operator=(GraphHostNode&&) = default;

    private:
        cudaHostNodeParams params;
    };

    class GraphKernelNode : public BaseGraphNode {
    public:
        GraphKernelNode() = default;
        GraphKernelNode(GraphKernelNode&) = default;
        GraphKernelNode(GraphKernelNode&& other) = default;
        GraphKernelNode(Graph graph) : BaseGraphNode(std::move(graph), cudaGraphAddKernelNode, nullptr, 0, nullptr) { }

        void set_params(dim3 blockDim, dim3 gridDim, void *kernel, void **kernelParamsArray)
        {
           params.blockDim = blockDim;
           params.gridDim = gridDim;
           params.func = kernel;
           params.kernelParams = kernelParamsArray;
           params.extra = NULL;
           CUDA4DNN_CHECK_CUDA(cudaGraphKernelNodeSetParams(BaseGraphNode::get(), &params));
        }

        GraphKernelNode& operator=(const GraphKernelNode&) = default;
        GraphKernelNode& operator=(GraphKernelNode&&) = default;

    private:
        cudaKernelNodeParams params;
    };

    class GraphMemcpyNode : public BaseGraphNode {
    public:
        GraphMemcpyNode() = default;
        GraphMemcpyNode(GraphMemcpyNode&) = default;
        GraphMemcpyNode(GraphMemcpyNode&& other) = default;
        GraphMemcpyNode(Graph graph) : BaseGraphNode(std::move(graph), cudaGraphAddMemcpyNode, nullptr, 0, nullptr) { }

        template <class T>
        void set_params(const T* srcPtr, T* dstPtr, std::size_t numel)
        {
           params = {0};
           params.srcArray = NULL;
           params.srcPos = make_cudaPos(0, 0, 0);
           params.srcPtr = make_cudaPitchedPtr(srcPtr, sizeof(T) * numel, numel, 1);
           params.dstArray = NULL;
           params.dstPos = make_cudaPos(0, 0, 0);
           params.dstPtr = make_cudaPitchedPtr(dstPtr, sizeof(T) * numel, numel, 1);
           params.extent = make_cudaExtent(sizeof(T) * numel, 1, 1);
           params.kind = cudaMemcpyDefault;
        }

        GraphMemcpyNode& operator=(const GraphMemcpyNode&) = default;
        GraphMemcpyNode& operator=(GraphMemcpyNode&&) = default;

    private:
        cudaMemcpy3DParms params;
    };

}}}} /* namespace cv::dnn::cuda4dnn::csl */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_CSL_GRAPH_HPP */
