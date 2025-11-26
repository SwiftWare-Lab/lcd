#pragma once
#include <iostream>
#include <vector>
#include <functional>
#include <map>
#include <string>
#include <set>
#include <queue>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>

// Robust Error Checking Macro
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " (" << err << ") at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
}

struct DagNode {
    int id;
    std::string name;
    std::function<void(cudaStream_t)> kernelOp;
    std::vector<int> parents;
    std::vector<int> children;
};

class CudaDagBuilder {
private:
    std::map<int, DagNode> nodes;
    int nextId = 0;

    cudaStream_t captureStream = nullptr; // Dedicated stream for capture control
    std::vector<cudaStream_t> streamPool; // Worker streams
    std::map<int, cudaEvent_t> nodeEvents;
    
    cudaEvent_t captureStartEvent = nullptr; // To anchor workers to capture stream

    // Viz Data
    std::map<int, int> lastScheduledStreamIdx;

public:
    CudaDagBuilder() { resizeResources(32); } 

    ~CudaDagBuilder() {
        if (captureStream) cudaStreamDestroy(captureStream);
        if (captureStartEvent) cudaEventDestroy(captureStartEvent);
        for (auto s : streamPool) cudaStreamDestroy(s);
        for (auto& pair : nodeEvents) cudaEventDestroy(pair.second);
    }

    void resizeResources(int count) {
        // Cleanup old
        if (captureStream) cudaStreamDestroy(captureStream);
        if (captureStartEvent) cudaEventDestroy(captureStartEvent);
        for (auto s : streamPool) cudaStreamDestroy(s);
        streamPool.clear();

        // 1. Create Dedicated Capture Controller
        CHECK_CUDA(cudaStreamCreateWithFlags(&captureStream, cudaStreamNonBlocking));
        CHECK_CUDA(cudaEventCreateWithFlags(&captureStartEvent, cudaEventDisableTiming));

        // 2. Create Worker Streams
        for(int i=0; i<count; i++) {
            cudaStream_t s; 
            CHECK_CUDA(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
            streamPool.push_back(s);
        }
    }

    int addNode(std::string name, std::function<void(cudaStream_t)> op, const std::vector<int>& dependencies) {
        int id = nextId++;
        DagNode node;
        node.id = id;
        node.name = name;
        node.kernelOp = op;
        node.parents = dependencies;
        nodes[id] = node;

        for (int pid : dependencies) {
            if (nodes.find(pid) != nodes.end()) {
                nodes[pid].children.push_back(id);
            }
        }
        return id;
    }

    void addDependencies(int nodeId, std::vector<int>& dependencies){
        if (nodes.find(nodeId) != nodes.end()){
            auto node = nodes[nodeId];
            node.parents = dependencies;
            for (auto pid: dependencies){
                if (nodes.find(pid) != nodes.end()){
                    nodes[pid].children.push_back(nodeId);
                }
            }
        }
    }

    cudaGraph_t captureGraph() {
        if (streamPool.empty()) resizeResources(16);

        std::cout << "[DEBUG] Starting capture setup..." << std::endl;
        CHECK_CUDA(cudaDeviceSynchronize());

        // 1. Event Refresh
        for (auto& pair : nodeEvents) cudaEventDestroy(pair.second);
        nodeEvents.clear();

        for (auto const& pair : nodes) {
            cudaEvent_t e;
            CHECK_CUDA(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
            nodeEvents[pair.first] = e;
        }

        // 2. Topological Sort
        std::vector<int> sortedOrder;
        std::map<int, int> inDegree;
        std::queue<int> zeroDegree;

        for (auto const& pair : nodes) {
            inDegree[pair.first] = pair.second.parents.size();
            if (pair.second.parents.empty()) zeroDegree.push(pair.first);
        }

        while (!zeroDegree.empty()) {
            int u = zeroDegree.front(); zeroDegree.pop();
            sortedOrder.push_back(u);
            for (int v : nodes[u].children) {
                if (--inDegree[v] == 0) zeroDegree.push(v);
            }
        }
        
        if (sortedOrder.size() != nodes.size()) {
            std::cerr << "Error: Cycle detected." << std::endl;
            exit(1);
        }

        // 3. Begin Capture using Anchor Pattern
        lastScheduledStreamIdx.clear();
        std::map<int, cudaStream_t> nodeToStream; 
        
        // Track the "Head" node of each stream to prevent accidental serialization of siblings
        std::vector<int> streamHeads(streamPool.size(), -1);
        int streamPoolIdx = 0; 

        // A. Start Capture on DEDICATED stream
        std::cout << "[DEBUG] Begin Capture on Dedicated Stream: " << captureStream << std::endl;
        CHECK_CUDA(cudaStreamBeginCapture(captureStream, cudaStreamCaptureModeGlobal));

        // B. Fork: Record event on CaptureStream and make ALL workers wait.
        CHECK_CUDA(cudaEventRecord(captureStartEvent, captureStream));
        for (auto s : streamPool) {
            CHECK_CUDA(cudaStreamWaitEvent(s, captureStartEvent, 0));
        }

        // C. Process Nodes
        for (int nodeId : sortedOrder) {
            DagNode& node = nodes[nodeId];
            std::cout << "[DEBUG] Processing Node " << nodeId << " ('" << node.name << "')" << std::endl;
            
            int myStreamIdx = -1;
            bool inherited = false;

            // 1. Try to inherit from a parent (Extend the stream)
            for (int parentId : node.parents) {
                if (lastScheduledStreamIdx.count(parentId)) {
                    int parentStreamIdx = lastScheduledStreamIdx[parentId];
                    
                    // CRITICAL CHECK: Only inherit if this stream is still pointing to MY parent.
                    // If streamHeads[idx] != parentId, it means a sibling already stole this stream!
                    if (streamHeads[parentStreamIdx] == parentId) {
                        myStreamIdx = parentStreamIdx;
                        inherited = true;
                        break; 
                    }
                }
            }

            // 2. If inheritance failed (or no parents), pick next available stream
            if (!inherited) {
                myStreamIdx = streamPoolIdx;
                streamPoolIdx = (streamPoolIdx + 1) % streamPool.size();
            }

            cudaStream_t myStream = streamPool[myStreamIdx];
            lastScheduledStreamIdx[nodeId] = myStreamIdx;
            nodeToStream[nodeId] = myStream;
            
            // Update the Head: This stream is now owned by ME.
            streamHeads[myStreamIdx] = nodeId; 

            // Sync with Parents
            for (int parentId : node.parents) {
                cudaStream_t parentStream = nodeToStream[parentId];
                if (parentStream != myStream) {
                    CHECK_CUDA(cudaStreamWaitEvent(myStream, nodeEvents[parentId], 0));
                }
            }

            // Execute
            std::cout << "  -> Launching Kernel Op on Stream " << myStreamIdx << "..." << std::endl;
            cudaGetLastError(); 
            node.kernelOp(myStream);
            
            cudaError_t kErr = cudaGetLastError();
            if (kErr != cudaSuccess) std::cerr << "!! Error launching node: " << cudaGetErrorString(kErr) << std::endl;

            // Record Completion
            CHECK_CUDA(cudaEventRecord(nodeEvents[nodeId], myStream));
        }

        // D. Join: Find Leaf Nodes and make CaptureStream wait on them.
        std::cout << "[DEBUG] Joining branches..." << std::endl;
        bool hasLeaf = false;
        for (auto const& pair : nodes) {
            if (pair.second.children.empty()) {
                std::cout << "  -> Joining Leaf Node " << pair.first << std::endl;
                CHECK_CUDA(cudaStreamWaitEvent(captureStream, nodeEvents[pair.first], 0));
                hasLeaf = true;
            }
        }
        
        if (!hasLeaf && !nodes.empty()) {}

        std::cout << "[DEBUG] Ending Capture..." << std::endl;
        cudaGraph_t graph;
        CHECK_CUDA(cudaStreamEndCapture(captureStream, &graph));
        
        std::cout << "[DEBUG] Capture Successful." << std::endl;
        return graph;
    }

    std::string exportDot() {
        if (lastScheduledStreamIdx.empty()) return "digraph G { label=\"Run capture first!\"; }";

        std::stringstream ss;
        ss << "digraph CudaGraph {\n";
        ss << "  compound=true;\n";
        ss << "  rankdir=LR;\n";
        ss << "  node [shape=box style=filled fillcolor=lightgrey fontname=\"Helvetica\"];\n";

        std::map<int, std::vector<int>> streamToNodes;
        for (auto const& pair : lastScheduledStreamIdx) {
            streamToNodes[pair.second].push_back(pair.first);
        }

        for (auto const& pair : streamToNodes) {
            int streamIdx = pair.first;
            const auto& nodeIds = pair.second;
            ss << "  subgraph cluster_stream" << streamIdx << " {\n";
            ss << "    label = \"CUDA Stream " << streamIdx << "\";\n";
            ss << "    style=filled; color=lightblue;\n";
            for (int nid : nodeIds) {
                ss << "    node_" << nid << " [label=\"" << nodes[nid].name << "\"];\n";
            }
            ss << "  }\n";
        }

        for (auto const& pair : nodes) {
            int id = pair.first;
            const DagNode& node = pair.second;
            if (lastScheduledStreamIdx.find(id) == lastScheduledStreamIdx.end()) continue;
            int sourceStream = lastScheduledStreamIdx[id];

            for (int childId : node.children) {
                if (lastScheduledStreamIdx.find(childId) == lastScheduledStreamIdx.end()) continue;
                int targetStream = lastScheduledStreamIdx[childId];
                
                ss << "  node_" << id << " -> node_" << childId;
                if (sourceStream == targetStream) {
                    ss << " [color=black];\n";
                } else {
                    ss << " [color=red style=dashed label=\"WaitEvent\" fontcolor=red fontsize=10];\n";
                }
            }
        }
        ss << "}\n";
        return ss.str();
    }
};
