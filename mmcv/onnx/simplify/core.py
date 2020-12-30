# This file is modified from https://github.com/daquexian/onnx-simplifier
import copy
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Union

import numpy as np  # type: ignore
import onnx  # type: ignore
import onnx.helper  # type: ignore
import onnx.numpy_helper
import onnx.shape_inference  # type: ignore
import onnxoptimizer  # type: ignore
import onnxruntime as rt  # type: ignore

from .common import add_suffix2name

TensorShape = List[int]
TensorShapes = Dict[Optional[str], TensorShape]


def add_features_to_output(m: onnx.ModelProto,
                           nodes: List[onnx.NodeProto]) -> None:
    """Add features to output in pb, so that ONNX Runtime will output them.

    Args:
        m (onnx.ModelProto): Input ONNX model.
        nodes (List[onnx.NodeProto]): List of ONNX nodes, whose outputs
        will be added into the graph output.
    """
    for node in nodes:
        for output in node.output:
            m.graph.output.extend([onnx.ValueInfoProto(name=output)])


def get_shape_from_value_info_proto(v: onnx.ValueInfoProto) -> List[int]:
    return [dim.dim_value for dim in v.type.tensor_type.shape.dim]


def get_value_info_all(m: onnx.ModelProto,
                       name: str) -> Optional[onnx.ValueInfoProto]:
    for v in m.graph.value_info:
        if v.name == name:
            return v

    for v in m.graph.input:
        if v.name == name:
            return v

    for v in m.graph.output:
        if v.name == name:
            return v

    return None


def get_shape(m: onnx.ModelProto, name: str) -> TensorShape:
    """Get shape info of a node in a model.

    Args:
        m (onnx.ModelProto): Input model.
        name (str): Name of a node.

    Returns:
        TensorShape: Shape of a node.

    Note:
        This method relies on onnx shape inference, which is not reliable.
        So only use it on input or output tensors
    """
    v = get_value_info_all(m, name)
    if v is not None:
        return get_shape_from_value_info_proto(v)
    raise RuntimeError('Cannot get shape of "{}"'.format(name))


def get_elem_type(m: onnx.ModelProto, name: str) -> Optional[int]:
    v = get_value_info_all(m, name)
    if v is not None:
        return v.type.tensor_type.elem_type
    return None


def get_np_type_from_elem_type(elem_type: int) -> int:
    """Map element type from ONNX to dtype of numpy.

    Args:
        elem_type (int): Element type index in ONNX.

    Returns:
        int: Data type in numpy.
    """
    # from https://github.com/onnx/onnx/blob/e5e9a539f550f07ec156812484e8d4f33fb91f88/onnx/onnx.proto#L461 # noqa: E501
    sizes = (None, np.float32, np.uint8, np.int8, np.uint16, np.int16,
             np.int32, np.int64, str, np.bool, np.float16, np.double,
             np.uint32, np.uint64, np.complex64, np.complex128, np.float16)
    assert len(sizes) == 17
    size = sizes[elem_type]
    assert size is not None
    return size


def get_input_names(model: onnx.ModelProto) -> List[str]:
    """Get input names of a model.

    Args:
        model (onnx.ModelProto): Input ONNX model.

    Returns:
        List[str]: List of input names.
    """
    input_names = list(
        set([ipt.name for ipt in model.graph.input]) -
        set([x.name for x in model.graph.initializer]))
    return input_names


def add_initializers_into_inputs(model: onnx.ModelProto) -> onnx.ModelProto:
    """add initializers into inputs of a model.

    Args:
        model (onnx.ModelProto): Input ONNX model.

    Returns:
        onnx.ModelProto: Updated ONNX model.
    """
    for x in model.graph.initializer:
        input_names = [x.name for x in model.graph.input]
        if x.name not in input_names:
            shape = onnx.TensorShapeProto()
            for dim in x.dims:
                shape.dim.extend(
                    [onnx.TensorShapeProto.Dimension(dim_value=dim)])
            model.graph.input.extend([
                onnx.ValueInfoProto(
                    name=x.name,
                    type=onnx.TypeProto(
                        tensor_type=onnx.TypeProto.Tensor(
                            elem_type=x.data_type, shape=shape)))
            ])
    return model


def generate_rand_input(
        model: onnx.ModelProto,
        input_shapes: Optional[TensorShapes] = None) -> Dict[str, np.ndarray]:
    """Generate random input for a model.

    Args:
        model (onnx.ModelProto): Input ONNX model.
        input_shapes (TensorShapes, optional): Input shapes of the model.

    Returns:
        Dict[str, np.ndarray]: Generated inputs of `np.ndarray`.
    """
    if input_shapes is None:
        input_shapes = {}
    input_names = get_input_names(model)
    full_input_shapes = {ipt: get_shape(model, ipt) for ipt in input_names}
    assert None not in input_shapes
    full_input_shapes.update(input_shapes)  # type: ignore
    for key in full_input_shapes:
        if np.prod(full_input_shapes[key]) <= 0:
            raise RuntimeError(f'The shape of input "{key}" has dynamic size, \
                please determine the input size manually.')

    inputs = {
        ipt: np.array(
            np.random.rand(*full_input_shapes[ipt]),
            dtype=get_np_type_from_elem_type(get_elem_type(model, ipt)))
        for ipt in input_names
    }
    return inputs


def get_constant_nodes(m: onnx.ModelProto) -> List[onnx.NodeProto]:
    """Collect constant nodes from a model.

    Args:
        m (onnx.ModelProto): Input ONNX model.

    Returns:
        List[onnx.NodeProto]: List of constant nodes.
    """

    const_nodes = []
    const_tensors = [x.name for x in m.graph.initializer]
    const_tensors.extend([
        node.output[0] for node in m.graph.node if node.op_type == 'Constant'
    ])
    # The output shape of some node types is determined by the input value
    # we consider the output of this node doesn't have constant shape,
    # so we do not simplify a such node even if the node is Shape op
    dynamic_tensors = []

    def is_dynamic(node):
        if node.op_type in ['NonMaxSuppression', 'NonZero', 'Unique'
                            ] and node.input[0] not in const_tensors:
            return True
        if node.op_type in [
                'Reshape', 'Expand', 'Upsample', 'ConstantOfShape'
        ] and len(node.input) > 1 and node.input[1] not in const_tensors:
            return True
        if node.op_type in ['Resize'] and (
            (len(node.input) > 2 and node.input[2] not in const_tensors) or
            (len(node.input) > 3
             and node.input[3] not in const_tensors)):  # noqa: E129
            return True
        return False

    for node in m.graph.node:
        if any(x in dynamic_tensors for x in node.input):
            dynamic_tensors.extend(node.output)
        elif node.op_type == 'Shape':
            const_nodes.append(node)
            const_tensors.extend(node.output)
        elif is_dynamic(node):
            dynamic_tensors.extend(node.output)
        elif all([x in const_tensors for x in node.input]):
            const_nodes.append(node)
            const_tensors.extend(node.output)
    return copy.deepcopy(const_nodes)


def forward(
        model: onnx.ModelProto,
        inputs: Dict[str, np.ndarray] = None,
        input_shapes: Optional[TensorShapes] = None) -> Dict[str, np.ndarray]:
    """Run forward on a model.

    Args:
        model (onnx.ModelProto): Input ONNX model.
        inputs (Dict[str, np.ndarray], optional): Inputs of the model.
        input_shapes (TensorShapes, optional): Input shapes of the model.

    Returns:
        Dict[str, np.ndarray]: Outputs of the model.
    """
    if input_shapes is None:
        input_shapes = {}
    sess_options = rt.SessionOptions()
    # load custom lib for onnxruntime in mmcv
    ort_custom_op_path = ''
    try:
        from mmcv.ops import get_onnxruntime_op_path
        ort_custom_op_path = get_onnxruntime_op_path()
    except ImportError:
        pass
    if os.path.exists(ort_custom_op_path):
        sess_options.register_custom_ops_library(ort_custom_op_path)
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel(0)
    sess_options.log_severity_level = 3
    sess = rt.InferenceSession(
        model.SerializeToString(),
        sess_options=sess_options,
        providers=['CPUExecutionProvider'])
    if inputs is None:
        inputs = generate_rand_input(model, input_shapes=input_shapes)
    outputs = [x.name for x in sess.get_outputs()]
    run_options = rt.RunOptions()
    run_options.log_severity_level = 3
    res = OrderedDict(
        zip(outputs, sess.run(outputs, inputs, run_options=run_options)))
    return res


def forward_for_node_outputs(
        model: onnx.ModelProto,
        nodes: List[onnx.NodeProto],
        input_shapes: Optional[TensorShapes] = None,
        inputs: Optional[Dict[str,
                              np.ndarray]] = None) -> Dict[str, np.ndarray]:
    if input_shapes is None:
        input_shapes = {}
    model = copy.deepcopy(model)
    add_features_to_output(model, nodes)
    res = forward(model, inputs=inputs, input_shapes=input_shapes)
    return res


def insert_elem(repeated_container, index: int, element):
    repeated_container.extend([repeated_container[-1]])
    for i in reversed(range(index + 1, len(repeated_container) - 1)):
        repeated_container[i].CopyFrom(repeated_container[i - 1])
    repeated_container[index].CopyFrom(element)


def eliminate_const_nodes(model: onnx.ModelProto,
                          const_nodes: List[onnx.NodeProto],
                          res: Dict[str, np.ndarray]) -> onnx.ModelProto:
    """Eliminate redundant constant nodes from model.

    Args:
        model (onnx.ModelProto): The original ONNX model.
        const_nodes (List[onnx.NodeProto]):
            Constant nodes detected by `get_constant_nodes`.
        res (Dict[str, np.ndarray]): Outputs of the model.

    Returns:
        onnx.ModelProto: The simplified onnx model.
    """

    for i, node in enumerate(model.graph.node):
        if node in const_nodes:
            for output in node.output:
                new_node = copy.deepcopy(node)
                new_node.name = 'node_' + output
                new_node.op_type = 'Constant'
                new_attr = onnx.helper.make_attribute(
                    'value',
                    onnx.numpy_helper.from_array(res[output], name=output))
                del new_node.input[:]
                del new_node.attribute[:]
                del new_node.output[:]
                new_node.output.extend([output])
                new_node.attribute.extend([new_attr])
                insert_elem(model.graph.node, i + 1, new_node)
            del model.graph.node[i]

    return model


def optimize(model: onnx.ModelProto, skip_fuse_bn: bool,
             skipped_optimizers: Optional[Sequence[str]]) -> onnx.ModelProto:
    """Perform optimization on an ONNX model. Before simplifying, use this
    method to generate value_info. After simplifying, use this method to fold
    constants generated in previous step into initializer, and eliminate unused
    constants.

    Args:
        model (onnx.ModelProto): The input ONNX model.
        skip_fuse_bn (bool): Whether to skip fuse bn.
        skipped_optimizers (Sequence[str]): List of optimizers to be skipped.

    Returns:
        onnx.ModelProto: The optimized model.
    """
    # Due to a onnx bug, https://github.com/onnx/onnx/issues/2417,
    # we need to add missing initializers into inputs
    onnx.checker.check_model(model)
    input_num = len(model.graph.input)
    model = add_initializers_into_inputs(model)
    onnx.helper.strip_doc_string(model)
    onnx.checker.check_model(model)
    optimizers_list = [
        'eliminate_deadend', 'eliminate_nop_dropout', 'eliminate_nop_cast',
        'eliminate_nop_monotone_argmax', 'eliminate_nop_pad',
        'extract_constant_to_initializer', 'eliminate_unused_initializer',
        'eliminate_nop_transpose', 'eliminate_identity',
        'fuse_add_bias_into_conv', 'fuse_consecutive_concats',
        'fuse_consecutive_log_softmax', 'fuse_consecutive_reduce_unsqueeze',
        'fuse_consecutive_squeezes', 'fuse_consecutive_transposes',
        'fuse_matmul_add_bias_into_gemm', 'fuse_pad_into_conv',
        'fuse_transpose_into_gemm'
    ]
    if not skip_fuse_bn:
        optimizers_list.append('fuse_bn_into_conv')
    if skipped_optimizers is not None:
        for opt in skipped_optimizers:
            try:
                optimizers_list.remove(opt)
            except ValueError:
                pass

    model = onnxoptimizer.optimize(model, optimizers_list, fixed_point=True)
    if model.ir_version > 3:
        del model.graph.input[input_num:]
    onnx.checker.check_model(model)
    return model


def check(model_opt: onnx.ModelProto,
          model_ori: onnx.ModelProto,
          n_times: int = 5,
          input_shapes: Optional[TensorShapes] = None,
          inputs: Optional[List[Dict[str, np.ndarray]]] = None) -> bool:
    """Check model before and after simplify.

    Args:
        model_opt (onnx.ModelProto): Optimized model.
        model_ori (onnx.ModelProto): Original model.
        n_times (int, optional): Number of times to compare models.
        input_shapes (TensorShapes, optional): Input shapes of the model.
        inputs (List[Dict[str, np.ndarray]], optional): Inputs of the model.

    Returns:
        bool: `True` means the outputs of two models have neglectable
            numeric difference.
    """

    if input_shapes is None:
        input_shapes = {}
    onnx.checker.check_model(model_opt)
    if inputs is not None:
        n_times = min(n_times, len(inputs))
    for i in range(n_times):
        print(f'Checking {i}/{n_times}...')
        if inputs is None:
            model_input = generate_rand_input(
                model_opt, input_shapes=input_shapes)
        else:
            model_input = inputs[i]
        res_opt = forward(model_opt, inputs=model_input)
        res_ori = forward(model_ori, inputs=model_input)

        for name in res_opt.keys():
            if not np.allclose(
                    res_opt[name], res_ori[name], rtol=1e-4, atol=1e-5):
                print(
                    'Tensor {} changes after simplifying. The max diff is {}.'.
                    format(name,
                           np.max(np.abs(res_opt[name] - res_ori[name]))))
                print('Note that the checking is not always correct.')
                print('After simplifying:')
                print(res_opt[name])
                print('Before simplifying:')
                print(res_ori[name])
                print('----------------')
                return False
    return True


def clean_constant_nodes(const_nodes: List[onnx.NodeProto],
                         res: Dict[str, np.ndarray]):
    """Clean constant nodes.

    Args:
        const_nodes (List[onnx.NodeProto]): List of constant nodes.
        res (Dict[str, np.ndarray]): The forward result of model.

    Returns:
        List[onnx.NodeProto]:  The constant nodes which have an output in res.

    Notes:
        It seems not needed since commit 6f2a72, but maybe it still prevents
        some unknown bug.
    """

    return [node for node in const_nodes if node.output[0] in res]


def check_and_update_input_shapes(model: onnx.ModelProto,
                                  input_shapes: TensorShapes) -> TensorShapes:
    input_names = get_input_names(model)
    if None in input_shapes:
        if len(input_names) == 1:
            input_shapes[input_names[0]] = input_shapes[None]
            del input_shapes[None]
        else:
            raise RuntimeError('The model has more than 1 inputs!')
    for x in input_shapes:
        if x not in input_names:
            raise RuntimeError(f'The model doesn\'t have input named "{x}"')
    return input_shapes


def simplify(model: Union[str, onnx.ModelProto],
             inputs: Sequence[Dict[str, np.ndarray]] = None,
             output_file: str = None,
             perform_optimization: bool = True,
             skip_fuse_bn: bool = False,
             skip_shape_inference: bool = True,
             input_shapes: Dict[str, Sequence[int]] = None,
             skipped_optimizers: Sequence[str] = None) -> onnx.ModelProto:
    """Simplify and optimize an onnx model.

    For models from detection and segmentation, it is strongly suggested to
    input multiple input images for verification.

    Arguments:
        model (str or onnx.ModelProto): path of model or loaded model object.
        inputs (optional, Sequence[Dict[str, np.ndarray]]): inputs of model.
        output_file (optional, str): output file to save simplified model.
        perform_optimization (optional, bool): whether to perform optimization.
        skip_fuse_bn (optional, bool): whether to skip fusing bn layer.
        skip_shape_inference (optional, bool): whether to skip shape inference.
        input_shapes (optional, Dict[str, Sequence[int]]):
            the shapes of model inputs.
        skipped_optimizers (optional, Sequence[str]):
            the names of optimizer to be skipped.

    Returns:
        onnx.ModelProto: simplified and optimized onnx model.

    Example:
        >>> import onnx
        >>> import numpy as np
        >>>
        >>> from mmcv.onnx import simplify
        >>>
        >>> dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        >>> input = {'input':dummy_input}
        >>> input_file = 'sample.onnx'
        >>> output_file = 'slim.onnx'
        >>> model = simplify(input_file, [input], output_file)
    """
    if input_shapes is None:
        input_shapes = {}
    if isinstance(model, str):
        model = onnx.load(model)
    # rename op with numeric name for issue
    # https://github.com/onnx/onnx/issues/2613
    model = add_suffix2name(model)
    onnx.checker.check_model(model)
    model_ori = copy.deepcopy(model)
    numel_node_ori = len(model_ori.graph.node)
    if not skip_shape_inference:
        model = onnx.shape_inference.infer_shapes(model)

    input_shapes = check_and_update_input_shapes(model, input_shapes)

    if perform_optimization:
        model = optimize(model, skip_fuse_bn, skipped_optimizers)

    const_nodes = get_constant_nodes(model)
    feed_inputs = None if inputs is None else inputs[0]
    res = forward_for_node_outputs(
        model, const_nodes, input_shapes=input_shapes, inputs=feed_inputs)
    const_nodes = clean_constant_nodes(const_nodes, res)
    model = eliminate_const_nodes(model, const_nodes, res)
    onnx.checker.check_model(model)

    if perform_optimization:
        model = optimize(model, skip_fuse_bn, skipped_optimizers)

    check_ok = check(
        model_ori, model, input_shapes=input_shapes, inputs=inputs)

    assert check_ok, 'Check failed for the simplified model!'
    numel_node_slim = len(model.graph.node)
    print(f'Number of nodes: {numel_node_ori} -> {numel_node_slim}')

    if output_file is not None:
        save_dir, _ = os.path.split(output_file)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        onnx.save(model, output_file)
    return model
