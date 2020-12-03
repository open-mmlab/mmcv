import ctypes
import glob
import os

import tensorrt as trt


def get_tensorrt_op_path():
    wildcard = os.path.join(
        os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
        '_ext_trt.*.so')

    paths = glob.glob(wildcard)
    if len(paths) > 0:
        return paths[0]
    else:
        return None


plugin_is_loaded = False


def is_tensorrt_plugin_loaded():
    global plugin_is_loaded
    return plugin_is_loaded


def load_tensorrt_plugin():
    global plugin_is_loaded
    if not plugin_is_loaded:
        ctypes.CDLL(get_tensorrt_op_path())
        plugin_is_loaded = True


def onnx2trt(onnx_model,
             opt_shape_dict,
             log_level=trt.Logger.ERROR,
             fp16_mode=False,
             max_workspace_size=0,
             device_id=0):
    # import plugin
    load_tensorrt_plugin()

    device = torch.device('cuda:{}'.format(device_id))
    # create builder and network
    logger = trt.Logger(log_level)
    builder = trt.Builder(logger)
    EXPLICIT_BATCH = 1 << (int)(
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)

    # parse onnx
    parser = trt.OnnxParser(network, logger)

    if isinstance(onnx_model, str):
        assert parser.parse_from_file(onnx_model), 'parse onnx failed.'
    else:
        assert parser.parse(
            onnx_model.SerializeToString()), 'parse onnx failed.'

    # config builder
    builder.max_workspace_size = max_workspace_size

    config = builder.create_builder_config()
    config.max_workspace_size = max_workspace_size
    profile = builder.create_optimization_profile()

    for input_name, param in opt_shape_dict.items():
        min_shape = tuple(param[0][:])
        opt_shape = tuple(param[1][:])
        max_shape = tuple(param[2][:])
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    if fp16_mode:
        builder.fp16_mode = fp16_mode
        config.set_flag(trt.BuilderFlag.FP16)

    # create engine
    with torch.cuda.device(device):
        engine = builder.build_engine(network, config)

    return engine


def save_trt_engine(engine, path):
    with open(path, mode='wb') as f:
        f.write(bytearray(engine.serialize()))


def load_trt_engine(path):
    with trt.Logger() as logger, trt.Runtime(logger) as runtime:
        with open(path, mode='rb') as f:
            engine_bytes = f.read()
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        return engine


try:
    import torch

    def torch_dtype_from_trt(dtype):
        if dtype == trt.bool:
            return torch.bool
        elif dtype == trt.int8:
            return torch.int8
        elif dtype == trt.int32:
            return torch.int32
        elif dtype == trt.float16:
            return torch.float16
        elif dtype == trt.float32:
            return torch.float32
        else:
            raise TypeError('%s is not supported by torch' % dtype)

    def torch_device_from_trt(device):
        if device == trt.TensorLocation.DEVICE:
            return torch.device('cuda')
        elif device == trt.TensorLocation.HOST:
            return torch.device('cpu')
        else:
            return TypeError('%s is not supported by torch' % device)

    class TRTWraper(torch.nn.Module):
        def __init__(self, engine=None, input_names=None, output_names=None):
            super(TRTWraper, self).__init__()
            load_tensorrt_plugin()
            self._register_state_dict_hook(TRTWraper._on_state_dict)
            self.engine = engine
            if isinstance(self.engine, str):
                self.engine = load_trt_engine(engine)
            if self.engine is not None:
                self.context = self.engine.create_execution_context()

            self.input_names = input_names
            self.output_names = output_names

        def _on_state_dict(self, state_dict, prefix, local_metadata):
            state_dict[prefix + 'engine'] = bytearray(self.engine.serialize())
            state_dict[prefix + 'input_names'] = self.input_names
            state_dict[prefix + 'output_names'] = self.output_names

        def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                                  strict, missing_keys, unexpected_keys,
                                  error_msgs):
            engine_bytes = state_dict[prefix + 'engine']

            with trt.Logger() as logger, trt.Runtime(logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(engine_bytes)
                self.context = self.engine.create_execution_context()

            self.input_names = state_dict[prefix + 'input_names']
            self.output_names = state_dict[prefix + 'output_names']

        def forward(self, inputs):
            bindings = [None
                        ] * (len(self.input_names) + len(self.output_names))

            for input_name, input_tensor in inputs.items():
                idx = self.engine.get_binding_index(input_name)

                self.context.set_binding_shape(idx, tuple(input_tensor.shape))
                bindings[idx] = input_tensor.contiguous().data_ptr()

            # create output tensors
            # outputs = [None] * len(self.output_names)
            outputs = {}
            for i, output_name in enumerate(self.output_names):
                idx = self.engine.get_binding_index(output_name)
                dtype = torch_dtype_from_trt(
                    self.engine.get_binding_dtype(idx))
                shape = tuple(self.context.get_binding_shape(idx))

                device = torch_device_from_trt(self.engine.get_location(idx))
                output = torch.empty(size=shape, dtype=dtype, device=device)
                outputs[output_name] = output
                bindings[idx] = output.data_ptr()

            self.context.execute_async_v2(
                bindings,
                torch.cuda.current_stream().cuda_stream)

            return outputs

except ImportError:
    pass
