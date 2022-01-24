import torch
import numpy as np
import inspect
import copy
from typing import Any, Callable, Dict, Iterator, Optional, Union
from poptorch import PoplarExecutor, poptorch_core, __version__, identity_loss
from poptorch._args_parser import ArgsParser
from mmcv.parallel.data_container import DataContainer

class DictArgsParser(ArgsParser):
    def __init__(self, inputs):
        # Combine args and kwargs:
        self._has_variadic_arguments = True
        self._varnames = list(inputs.keys())
        self._defaults = [inspect.Parameter.empty for _ in self._varnames]
        self._warned_not_contiguous_input = False


# def get_train_step_wrapper(user_model,not_traced_inputs,traced_input_keys):
#     def train_step_wrapper(inputs_tuple):
#         # convert tuple back to kwargs
#         kwargs = {_key:_val for _key,_val in zip(traced_input_keys,inputs_tuple)}
#         kwargs = {**kwargs, **not_traced_inputs} # add back all inputs that will not be traced
#         optimizer = kwargs.pop('optimizer')
#         data = kwargs
#         outputs = user_model.train_step(data,optimizer)
#         return outputs
#     return train_step_wrapper

class TreeManager:
    def __init__(self, logger=None):
        self.fixed_data_types = (int, str, float, np.ndarray, type(None))
        self.logger = logger
        self.logger_on = False if logger is None else True 
        self.keys_of_changed_vals = []
        self.data_not_in_dict_changed = False

    def set_tree(self, _tree):
        # _tree: A composite data type containing dictionaries, lists, tensors and basic python data types
        if hasattr(self,'_tree'):
            self.update(_tree)
        else:
            self._tree = _tree
    
    def get_tree(self,):
        return self._tree

    def update(self, treeA, treeB='TreeManagerNone', strict=True, key=None):
        treeB = self._tree if treeB == 'TreeManagerNone' else treeB
        # Update with a tree with the same structure but different values(tensors and basic python data types)
        if isinstance(treeA, (tuple,list)):
            for idx in range(len(treeA)):
                assert isinstance(treeA[idx],type(treeB[idx])), 'data structure changed'
                if isinstance(treeA[idx],torch.Tensor):
                    treeB[idx] = treeA[idx]
                else:
                    self.update(treeA[idx], treeB[idx], strict)
        elif isinstance(treeA, dict):
            for k,v in treeA.items():
                assert isinstance(treeA[k],type(treeB[k])), 'data structure changed'
                if isinstance(v,torch.Tensor):
                    treeB[k] = treeA[k]
                else:
                    self.update(treeA[k], treeB[k], strict, key)
        elif isinstance(treeA, self.fixed_data_types):
            if strict:
                assert treeA==treeB, 'all data except torch.Tensor should be same!!!'
            elif self.logger_on:
                if key is None and not self.data_not_in_dict_changed:
                    self.logger.info('find a non-torch.Tensor data changed')
                    self.data_not_in_dict_changed = True
                elif key not in self.keys_of_changed_vals:
                    self.logger.warning('find a non-torch.Tensor data changed, and the key is {}'.format(str(key)))
                    self.keys_of_changed_vals.append(key)
                else:
                    pass
            else:
                raise RuntimeError("find a non-torch.Tensor data changed, and no logger record this problem")
        elif isinstance(treeA, DataContainer):
            assert isinstance(treeB, DataContainer)
            self.update(treeA.data, treeB.data, False)
        else:
            raise NotImplementedError('not supported datatype:{}'.format(str(treeA)))
    
    def get_tensors(self,):
        # get a list of tensor from self._tree
        tensors = []
        self._get_tensors(self._tree, tensors)
        return tensors

    def _get_tensors(self, _tree, tensors):
        if isinstance(_tree, (tuple,list)):
            for idx in range(len(_tree)):
                if isinstance(_tree[idx],torch.Tensor):
                    tensors.append(_tree[idx])
                else:
                    self._get_tensors(_tree[idx],tensors)
        elif isinstance(_tree, dict):
            for k,v in _tree.items():
                if isinstance(v,torch.Tensor):
                    tensors.append(_tree[k])
                else:
                    self._get_tensors(_tree[k],tensors)
        elif isinstance(_tree, self.fixed_data_types):
            pass
        elif isinstance(_tree, DataContainer):
            self._get_tensors(_tree.data,tensors)
        else:
            raise NotImplementedError('not supported datatype:{}'.format(str(_tree)))
    
    def set_tensors(self, tensors):
        self._set_tensors(self._tree, tensors)
        return self._tree
    
    def _set_tensors(self, _tree, tensors):
        if isinstance(_tree, (tuple,list)):
            for idx in range(len(_tree)):
                if isinstance(_tree[idx],torch.Tensor):
                    _tree[idx] = tensors.pop(0)
                else:
                    self._set_tensors(_tree[idx],tensors)
        elif isinstance(_tree, dict):
            for k,v in _tree.items():
                if isinstance(v,torch.Tensor):
                    _tree[k] = tensors.pop(0)
                else:
                    self._set_tensors(_tree[k],tensors)
        elif isinstance(_tree, self.fixed_data_types):
            pass
        elif isinstance(_tree, DataContainer):
            self._set_tensors(_tree.data,tensors)
        else:
            raise NotImplementedError('not supported datatype:{}'.format(str(_tree)))   


def wrap_model(model, inputs_tree_manager, outputs_tree_manager):
    class Net(torch.nn.Module):
        def __init__(self, model, inputs_tree_manager, outputs_tree_manager):
            super().__init__()
            self.model = model
            self.inputs_tree_manager = inputs_tree_manager
            self.outputs_tree_manager = outputs_tree_manager

        def forward(self, inputs_tuple):
            # convert tuple back to kwargs
            self.inputs_tree_manager.set_tensors(list(inputs_tuple))
            kwargs = {**(self.inputs_tree_manager.get_tree())}
            optimizer = kwargs.pop('optimizer')
            data = kwargs
            outputs = self.model.train_step(data,optimizer)

            # tell poptorch which loss will be used finally
            identity_loss(outputs['loss'],reduction='none')
            # record all the places of return tensors in the converting stage
            # while in the real run stage, all the tensor are changed inplace
            # that means the output can be obtained directly outside this functioon
            self.outputs_tree_manager.set_tree(outputs)
            plain_outputs = self.outputs_tree_manager.get_tensors()
            return plain_outputs

    return Net(model, inputs_tree_manager, outputs_tree_manager)


class PoplarExecutorForMMCV(PoplarExecutor):
    def __init__(self, model, logger=None, *args, **kwargs):
        self.inputs_tree_manager = TreeManager(logger=logger) # wrapped model only accept and output tuple, so TreeManager will convert dictionary to tuple and convert them back
        self.outputs_tree_manager = TreeManager(logger=logger)
        model = wrap_model(model, self.inputs_tree_manager, self.outputs_tree_manager) # make torch.jit.trace convert self._model
        super().__init__(model, *args, **kwargs)
        self._args_parser = None # overwrite self._args_parser in train_step or val_step

    def train_step(self, data, optimizer=None, **kwargs):
        assert len(kwargs) == 0 # TODO, support later if necessary

        data['optimizer'] = None # TODO we will ignore optimizer for it will not be used in model, support later if necessary

        # get tensors out of data and put them in a tuple
        self.inputs_tree_manager.set_tree(data)
        inputs_tuple = tuple(self.inputs_tree_manager.get_tensors())

        # parser args for first iter
        self._args_parser = DictArgsParser({'args':inputs_tuple}) if self._args_parser is None else self._args_parser

        # run or convert model, the plain_outputs will be used in converting stage
        plain_outputs = self(inputs_tuple)

        # put list of tensors back to the output dict according to the same order
        self.outputs_tree_manager.set_tensors(plain_outputs)
        # get the real output dictionary from self.outputs_tree_manager
        output_dic = self.outputs_tree_manager.get_tree()

        # outputs contained loss, log_vars, num_samples, only loss(torch.tensor) has been updated
        # remove all unchanged vars, left torch.tensor
        neat_output_dic = {'loss':output_dic['loss']}

        # re-parse outputs, get back log_vars and num_samples
        loss, log_vars = self.model._parse_losses(neat_output_dic)
        final_output_dic = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

        return final_output_dic


def trainingModel(model: Union['torch.nn.Module', 'poptorch.PoplarExecutor'],
                  options: Optional['poptorch.Options'] = None,
                  optimizer: Optional['torch.optim.Optimizer'] = None,
                  logger = None
                  ) -> 'poptorch.PoplarExecutor':
    """ Create a PopTorch training model, from a PyTorch model, to run on IPU
    hardware in training mode.

    .. note:: PopTorch makes a shallow copy of the model. Changes to the
        parameters in the returned training model affect the original model
        and vice versa. However, primitive variable types are not synced: for
        example calling ``model.train()`` on the original model, which
        changes the ``training`` bool of the model instance, will not alter the
        model returned by this function. You may need to call ``model.train()``
        on your model before you call this function for correct behaviour.

    :param model: The PyTorch model to wrap.
    :param options: The IPU specific options
    :param optimizer: The optimizers to apply during \
        training.

        Supported PyTorch optimizers: ``optim.SGD``, ``optim.Adam``, \
             ``optim.AdamW``, ``optim.RMSprop``.

        Supported PopTorch optimizers: :py:class:`poptorch.optim.SGD`, \
            :py:class:`poptorch.optim.Adam`, \
            :py:class:`poptorch.optim.AdamW`, \
            :py:class:`poptorch.optim.RMSprop`. \
            :py:class:`poptorch.optim.LAMB`.

    :returns: The :py:class:`poptorch.PoplarExecutor` wrapper to use in place
        of ``model``.
    """
    if isinstance(model, PoplarExecutor):
        model = model._user_model  # pylint: disable=protected-access

    # Create a copy of the original model in case it needs to be wrapped
    maybe_wrapped_model = copy.copy(model)

    return PoplarExecutorForMMCV(model=maybe_wrapped_model,
                          logger=logger,
                          options=options,
                          training=True,
                          optimizer=optimizer,
                          user_model=model,
                          poptorch_version=__version__,)