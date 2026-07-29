/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/jit/Node.hpp>

#include <af/dim4.hpp>

#include <algorithm>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace arrayfire {
namespace cuda {
namespace jit {

const std::string &getKernelPreamble();

// Collects the backend-neutral JIT node machinery needed by a CUDA kernel
// which consumes an expression instead of materializing it. Consumer kernels
// deliberately start with linear trees; the normal JIT evaluator remains the
// fallback for broadcasting, shifted, gapped, and moddims expressions.
class Consumer {
    common::Node_ptr root_;
    common::Node_map_t node_map_;
    std::vector<common::Node *> nodes_;
    std::vector<common::Node_ids> ids_;
    int output_id_{-1};
    bool linear_{false};

   public:
    Consumer(common::Node_ptr root, const af::dim4 &dims)
        : root_(std::move(root)) {
        if (!root_) { return; }

        output_id_ = root_->getNodesMap(node_map_, nodes_, ids_);
        linear_    = std::all_of(nodes_.begin(), nodes_.end(),
                                 [&dims](const common::Node *node) {
                                  return node->getOp() != af_moddims_t &&
                                         node->isLinear(dims.get());
                              });
    }

    Consumer(const Consumer &)            = delete;
    Consumer &operator=(const Consumer &) = delete;
    Consumer(Consumer &&)                 = default;
    Consumer &operator=(Consumer &&)      = default;

    bool isLinear() const { return linear_; }
    int outputId() const { return output_id_; }
    const common::Node &outputNode() const { return *nodes_[output_id_]; }

    std::string kernelName(const std::string &suffix) const {
        std::vector<common::Node *> outputs{root_.get()};
        std::vector<int> output_ids{output_id_};
        return common::getFuncName(outputs, output_ids, nodes_, ids_, true,
                                   false, false, false, false) +
               suffix;
    }

    std::string parameterDeclarations() const {
        std::stringstream stream;
        for (size_t i = 0; i < nodes_.size(); ++i) {
            nodes_[i]->genParams(stream, ids_[i].id, true);
        }
        return stream.str();
    }

    std::string offsetsAndOperations() const {
        std::stringstream stream;
        for (size_t i = 0; i < nodes_.size(); ++i) {
            nodes_[i]->genOffsets(stream, ids_[i].id, true);
        }
        for (size_t i = 0; i < nodes_.size(); ++i) {
            nodes_[i]->genFuncs(stream, ids_[i]);
        }
        return stream.str();
    }

    size_t parameterBytes() const {
        size_t bytes = 0;
        for (const common::Node *node : nodes_) {
            node->setArgs(
                0, true, [&bytes](int, const void *, size_t size, bool) {
                    // CUDA kernel arguments are naturally aligned in the
                    // parameter buffer. Pointer-size alignment is
                    // conservative for the pointer, float, and double
                    // arguments accepted by the first consumer.
                    constexpr size_t alignment = sizeof(void *);
                    bytes = (bytes + alignment - 1) & ~(alignment - 1);
                    bytes += size;
                });
        }
        constexpr size_t alignment = sizeof(void *);
        return (bytes + alignment - 1) & ~(alignment - 1);
    }

    void appendArguments(std::vector<void *> &arguments) const {
        for (const common::Node *node : nodes_) {
            node->setArgs(0, true,
                          [&arguments](int, const void *ptr, size_t, bool) {
                              arguments.push_back(const_cast<void *>(ptr));
                          });
        }
    }
};

}  // namespace jit
}  // namespace cuda
}  // namespace arrayfire
