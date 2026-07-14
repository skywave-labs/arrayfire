/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Param.hpp>
#include <common/jit/ModdimNode.hpp>
#include <common/jit/Node.hpp>
#include <common/jit/NodeIterator.hpp>
#include <jit/BufferNode.hpp>
#include <jit/Node.hpp>
#include <jit/UnaryNode.hpp>
#include <parallel.hpp>
#include <platform.hpp>
#include <vector>

namespace arrayfire {
namespace cpu {
namespace kernel {

/// Clones node_index_map and update the child pointers
std::vector<std::shared_ptr<common::Node>> cloneNodes(
    const std::vector<common::Node *> &node_index_map,
    const std::vector<common::Node_ids> &ids) {
    using arrayfire::common::Node;
    // find all moddims in the tree
    std::vector<std::shared_ptr<Node>> node_clones;
    node_clones.reserve(node_index_map.size());
    transform(begin(node_index_map), end(node_index_map),
              back_inserter(node_clones), [](Node *n) { return n->clone(); });

    for (common::Node_ids id : ids) {
        auto &children = node_clones[id.id]->m_children;
        for (int i = 0; i < Node::kMaxChildren && children[i] != nullptr; i++) {
            children[i] = node_clones[id.child_ids[i]];
        }
    }
    return node_clones;
}

/// Sets the shape of the buffer node_index_map under the moddims node to the
/// new shape
void propagateModdimsShape(
    std::vector<std::shared_ptr<common::Node>> &node_clones) {
    using arrayfire::common::NodeIterator;
    for (auto &node : node_clones) {
        if (node->getOp() == af_moddims_t) {
            common::ModdimNode *mn =
                static_cast<common::ModdimNode *>(node.get());

            NodeIterator<> it(node.get());
            while (it != NodeIterator<>()) {
                it = std::find_if(it, NodeIterator<>(), common::isBuffer);
                if (it == NodeIterator<>()) { break; }

                it->setShape(mn->m_new_shape);

                ++it;
            }
        }
    }
}

/// Removes node_index_map whos operation matchs a unary operation \p op.
void removeNodeOfOperation(
    std::vector<std::shared_ptr<common::Node>> &node_index_map, af_op_t op) {
    using arrayfire::common::Node;

    for (size_t nid = 0; nid < node_index_map.size(); nid++) {
        auto &node = node_index_map[nid];

        for (int i = 0;
             i < Node::kMaxChildren && node->m_children[i] != nullptr; i++) {
            if (node->m_children[i]->getOp() == op) {
                // replace moddims
                auto moddim_node    = node->m_children[i];
                node->m_children[i] = moddim_node->m_children[0];
            }
        }
    }

    node_index_map.erase(remove_if(begin(node_index_map), end(node_index_map),
                                   [op](std::shared_ptr<Node> &node) {
                                       return node->getOp() == op;
                                   }),
                         end(node_index_map));
}

/// Returns the cloned output_nodes located in the node_clones array
///
/// This function returns the new cloned version of the output_nodes_ from
/// the node_clones array. If the output node is a moddim node, then it will
/// set the output node to be its first non-moddim node child
template<typename T>
std::vector<TNode<T> *> getClonedOutputNodes(
    const common::Node_map_t &node_index_map,
    const std::vector<std::shared_ptr<common::Node>> &node_clones,
    const std::vector<common::Node_ptr> &output_nodes_) {
    std::vector<TNode<T> *> cloned_output_nodes;
    cloned_output_nodes.reserve(output_nodes_.size());
    for (auto &n : output_nodes_) {
        TNode<T> *ptr;
        if (n->getOp() == af_moddims_t) {
            // if the output node is a moddims node, then set the output node
            // to be the child of the moddims node. This is necessary because
            // we remove the moddim node_index_map from the tree later
            int child_index = node_index_map.at(n->m_children[0].get());
            ptr = static_cast<TNode<T> *>(node_clones[child_index].get());
            while (ptr->getOp() == af_moddims_t) {
                ptr = static_cast<TNode<T> *>(ptr->m_children[0].get());
            }
        } else {
            int node_index = node_index_map.at(n.get());
            ptr = static_cast<TNode<T> *>(node_clones[node_index].get());
        }
        cloned_output_nodes.push_back(ptr);
    }
    return cloned_output_nodes;
}

template<typename T>
void evalMultiple(std::vector<Param<T>> arrays,
                  std::vector<common::Node_ptr> output_nodes_) {
    using arrayfire::common::ModdimNode;
    using arrayfire::common::Node;
    using arrayfire::common::Node_map_t;
    using arrayfire::common::NodeIterator;

    af::dim4 odims = arrays[0].dims();
    af::dim4 ostrs = arrays[0].strides();

    Node_map_t node_index_map;
    std::vector<T *> ptrs;
    std::vector<common::Node *> full_nodes;
    std::vector<common::Node_ids> ids;

    int narrays = static_cast<int>(arrays.size());
    ptrs.reserve(narrays);
    for (int i = 0; i < narrays; i++) {
        ptrs.push_back(arrays[i].get());
        output_nodes_[i]->getNodesMap(node_index_map, full_nodes, ids);
    }
    auto node_clones = cloneNodes(full_nodes, ids);

    std::vector<TNode<T> *> cloned_output_nodes =
        getClonedOutputNodes<T>(node_index_map, node_clones, output_nodes_);
    propagateModdimsShape(node_clones);
    removeNodeOfOperation(node_clones, af_moddims_t);

    bool is_linear = true;
    for (auto &node : node_clones) { is_linear &= node->isLinear(odims.get()); }

    const size_t num_nodes        = node_clones.size();
    const size_t num_output_nodes = cloned_output_nodes.size();
    const size_t dim0_chunks =
        (static_cast<size_t>(odims[0]) + jit::VECTOR_LENGTH - 1) /
        jit::VECTOR_LENGTH;
    const size_t total_chunks =
        is_linear
            ? (static_cast<size_t>(odims.elements()) + jit::VECTOR_LENGTH - 1) /
                  jit::VECTOR_LENGTH
            : dim0_chunks * static_cast<size_t>(odims[1]) *
                  static_cast<size_t>(odims[2]) * static_cast<size_t>(odims[3]);

    auto eval_chunk_range = [&](auto &nodes, auto &output_nodes,
                                size_t chunk_begin, size_t chunk_end) {
        if (is_linear) {
            const dim_t num = odims.elements();
            for (size_t chunk = chunk_begin; chunk < chunk_end; ++chunk) {
                const dim_t id = chunk * jit::VECTOR_LENGTH;
                const int lim  = static_cast<int>(
                    std::min<dim_t>(jit::VECTOR_LENGTH, num - id));
                for (size_t n = 0; n < num_nodes; ++n) {
                    nodes[n]->calc(static_cast<int>(id), lim);
                }
                for (size_t n = 0; n < num_output_nodes; ++n) {
                    std::copy(output_nodes[n]->m_val.begin(),
                              output_nodes[n]->m_val.begin() + lim,
                              ptrs[n] + id);
                }
            }
            return;
        }

        const int dim0 = static_cast<int>(odims[0]);
        for (size_t chunk = chunk_begin; chunk < chunk_end; ++chunk) {
            size_t row = chunk / dim0_chunks;
            const int x =
                static_cast<int>(chunk % dim0_chunks) * jit::VECTOR_LENGTH;
            const int y = static_cast<int>(row % odims[1]);
            row /= odims[1];
            const int z    = static_cast<int>(row % odims[2]);
            const int w    = static_cast<int>(row / odims[2]);
            const int lim  = std::min(jit::VECTOR_LENGTH, dim0 - x);
            const dim_t id = x + y * ostrs[1] + z * ostrs[2] + w * ostrs[3];

            for (size_t n = 0; n < num_nodes; ++n) {
                nodes[n]->calc(x, y, z, w, lim);
            }
            for (size_t n = 0; n < num_output_nodes; ++n) {
                std::copy(output_nodes[n]->m_val.begin(),
                          output_nodes[n]->m_val.begin() + lim, ptrs[n] + id);
            }
        }
    };

    // Each node evaluation processes VECTOR_LENGTH elements. Use the JIT tree
    // size when selecting a grain so complex expressions parallelize sooner,
    // while small expressions avoid scheduling and graph-cloning overhead.
    constexpr size_t min_node_chunks_per_task = 1024;
    const size_t chunks_per_task              = std::max<size_t>(
        1, min_node_chunks_per_task / std::max<size_t>(1, num_nodes));
    const size_t useful_tasks =
        (total_chunks + chunks_per_task - 1) / chunks_per_task;
    constexpr size_t min_parallel_elements = 1 << 16;
    const size_t task_count =
        odims.elements() < static_cast<dim_t>(min_parallel_elements)
            ? 1
            : std::max<size_t>(
                  1, std::min(getParallelThreadCount(), useful_tasks));

    parallelFor(task_count, [&](size_t task) {
        const size_t chunks_per_task = total_chunks / task_count;
        const size_t remainder       = total_chunks % task_count;
        const size_t chunk_begin =
            task * chunks_per_task + std::min(task, remainder);
        const size_t chunk_end =
            chunk_begin + chunks_per_task + (task < remainder ? 1 : 0);
        if (task == 0) {
            eval_chunk_range(node_clones, cloned_output_nodes, chunk_begin,
                             chunk_end);
            return;
        }

        auto task_nodes = cloneNodes(full_nodes, ids);
        auto task_output_nodes =
            getClonedOutputNodes<T>(node_index_map, task_nodes, output_nodes_);
        propagateModdimsShape(task_nodes);
        removeNodeOfOperation(task_nodes, af_moddims_t);
        eval_chunk_range(task_nodes, task_output_nodes, chunk_begin, chunk_end);
    });
}

}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
