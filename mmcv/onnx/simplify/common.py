import copy
import warnings

import onnx


def add_suffix2name(ori_model, suffix='__', verify=False):
    """Simplily add a suffix to the name of node, which has a numeric name."""
    # check if has special op, which has subgraph.
    special_ops = ('If', 'Loop')
    for node in ori_model.graph.node:
        if node.op_type in special_ops:
            warnings.warn(f'This model has special op: {node.op_type}.')
            return ori_model

    model = copy.deepcopy(ori_model)

    def need_update(name):
        return name.isnumeric()

    def update_name(nodes):
        for node in nodes:
            if need_update(node.name):
                node.name += suffix

    update_name(model.graph.initializer)
    update_name(model.graph.input)
    update_name(model.graph.output)

    for i, node in enumerate(ori_model.graph.node):
        # process input of node
        for j, name in enumerate(node.input):
            if need_update(name):
                model.graph.node[i].input[j] = name + suffix

        # process output of node
        for j, name in enumerate(node.output):
            if need_update(name):
                model.graph.node[i].output[j] = name + suffix
    if verify:
        onnx.checker.check_model(model)

    return model
