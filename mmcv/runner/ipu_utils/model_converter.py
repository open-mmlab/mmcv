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
            if isinstance(_tree, torch.Tensor):
                assert type(self._tree) == torch.Tensor
                self._tree = _tree
            else:
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
        if type(self._tree) == torch.Tensor:
            tensors = [self._tree]
        else:
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
        if type(self._tree) == torch.Tensor:
            assert len(tensors) == 1
            assert type(tensors[0]) == torch.Tensor
            self._tree = tensors[0]
        else:
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


class WrappedNet(torch.nn.Module):
    def __init__(self, model, inputs_tree_manager, outputs_tree_manager):
        super().__init__()
        self.model = model
        self.inputs_tree_manager = inputs_tree_manager
        self.outputs_tree_manager = outputs_tree_manager
        self.training = model.training

    def forward(self, inputs_tuple):
        # convert tuple back to kwargs
        self.inputs_tree_manager.set_tensors(list(inputs_tuple))
        kwargs = {**(self.inputs_tree_manager.get_tree())}
        if self.training:
            outputs = self.forward_train(kwargs)
            # tell poptorch which loss will be used finally
            identity_loss(outputs['loss'],reduction='none')
        else:
            outputs = self.forward_eval(kwargs)
            
        if isinstance(outputs, torch.Tensor):
            # currently not support single tensor output, need to wrap it with a dictionary, use a key word to identify this situation
            outputs = {'output of WrappedNet: single tensor': outputs}

        # record all the places of return tensors in the converting stage
        # while in the real run stage, all the tensor are changed inplace
        # that means the output can be obtained directly outside this functioon
        self.outputs_tree_manager.set_tree(outputs)
        plain_outputs = self.outputs_tree_manager.get_tensors()
        return plain_outputs

    def forward_train(self, kwargs):
        optimizer = kwargs.pop('optimizer')
        data = kwargs
        outputs = self.model.train_step(data,optimizer)
        return outputs
    
    def forward_eval(self, kwargs):
        img = kwargs.pop('img')
        img_metas = kwargs.pop('img_metas')
        return_loss = kwargs.pop('return_loss')
        assert not return_loss
        # TODO Temporarily hard-code to close post_process, otherwise, in the third trace, that is, _check_trace, post_process cannot detect the tracing state of jit, resulting in the automatic conversion of output tensor to numpy array, resulting in _check_trace failure
        outputs = self.model(img, img_metas=img_metas, return_loss=return_loss, post_process=False)
        return outputs


class PoplarExecutorForMMCV(PoplarExecutor):
    def __init__(self, model, logger=None, training=True, *args, **kwargs):
        # self.model == self._user_model: input pytorch model
        # self._model: wrapped model which is used to compile and update weights
        # these two models use same weights
        self.inputs_tree_manager = TreeManager(logger=logger) # wrapped model only accept and output tuple, so TreeManager will convert dictionary to tuple and convert them back
        self.outputs_tree_manager = TreeManager(logger=logger)
        self.logger = logger
        model = WrappedNet(model, self.inputs_tree_manager, self.outputs_tree_manager) # make torch.jit.trace convert self._model
        super().__init__(model, training=training, *args, **kwargs)
        self._args_parser = None # overwrite self._args_parser in train_step or val_step
        if training:
            assert self.training
        else:
            assert not self.training

    @property
    def training(self,):
        # If trying to get the attribute training of self, since the class has no training attribute, it will automatically look for the training attribute of self.model. However, the real attribute we want to check is self._training, self.model.training  and self._training are often inconsistent. It is not clear whether it is a Poptorch bug or a special design, temporarily use this function to fix the problem
        return self._training # comes from self.model._training

    def run_model(self, data_dict):
        # this function used to parse input_dict and convert to output_dict

        # get tensors out of data and put them in a tuple
        self.inputs_tree_manager.set_tree(data_dict)
        inputs_tuple = tuple(self.inputs_tree_manager.get_tensors())

        # parser args for first iter
        self._args_parser = DictArgsParser({'args':inputs_tuple}) if self._args_parser is None else self._args_parser

        # run or convert model, the plain_outputs will be used in converting stage
        plain_outputs = self(inputs_tuple)

        # put list of tensors back to the output dict according to the same order
        self.outputs_tree_manager.set_tensors(plain_outputs)
        # get the real output dictionary from self.outputs_tree_manager
        output_dic = self.outputs_tree_manager.get_tree()

        if 'output of WrappedNet: single tensor' in output_dic:
            assert len(output_dic) == 1
            assert type(output_dic['output of WrappedNet: single tensor']) == torch.Tensor
            output_dic = output_dic['output of WrappedNet: single tensor']

        return output_dic

    def train_step(self, data, optimizer=None, **kwargs):
        # arguments from mmcls/models/classifiers/base.py:BaseClassifier.train_step
        assert self.training
        assert len(kwargs) == 0 # TODO, support later if necessary

        data['optimizer'] = None # TODO we will ignore optimizer for it will not be used in model, support later if necessary

        output_dic = self.run_model(data)

        # outputs contained loss, log_vars, num_samples, only loss(torch.tensor) has been updated
        # remove all unchanged vars, left torch.tensor
        neat_output_dic = {'loss':output_dic['loss']}

        # re-parse outputs, get back log_vars and num_samples
        loss, log_vars = self.model._parse_losses(neat_output_dic)
        final_output_dic = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

        return final_output_dic
    
    def tmp_ussage_for_eval_call(self, img, img_metas, return_loss=True, **kwargs):
        # arguments from mmdet/models/detectors/base.py:BaseDetector.forward
        # tmp usssage for eval mode
        assert not self.training
        assert len(kwargs) == 0 # TODO, support later if necessary
        assert not return_loss
        data = {'img':img,'img_metas':img_metas,'return_loss':return_loss}

        output_dic = self.run_model(data)

        return output_dic

    def detachFromDevice(self,):
        if self.isCompiled() and self._is_attached:
            super().detachFromDevice()
    
    def attachToDevice(self,):
        if self.isCompiled() and not self._is_attached:
            super().attachToDevice()

    # def copyWeightsToDevice(self,):
    #     if self.isCompiled():
    #         super().copyWeightsToDevice()
    #     else:
    #         if self.logger is None:
    #             raise('model not complied')
    #         else:
    #             self.logger.warning('model not complied， so the weights are not copied to the IPU')


class TrainEvalModel:
    def __init__(self, model, options, optimizer, logger=None):
        self._train_executor = trainingModel(copy.copy(model), options=options, optimizer=optimizer, logger=logger)
        self._eval_executor = inferenceModel(copy.copy(model), options=options, logger=logger)
        self.training = True

    @property
    def executor(self,):
        if self.training:
            return self._train_executor
        else:
            return self._eval_executor

    def train(self, mode: bool = True):
        r"""Sets the module in training mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        if mode == self.training:
            return self
        else:
            self.copyWeightsToHost() # copy weights from IPU to cpu before off-load current session
            self.detachFromDevice() # detach the current session before change the mode, if is training mode and weights are updated, poptorch will copy weights from IPU to host

            self.training = mode # session will changed with mode changing
            self.model.train(mode)

            self.attachToDevice() # after changing mode, attach the current new session, and this function will copy weights of model to device
            # self.copyWeightsToDevice() # new session is loaded, then copy weights from cpu back to IPU(two modes correspond to two IPU sessions with two copy of weights on IPU)
            return self

    def eval(self):
        r"""Sets the module in evaluation mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        This is equivalent with :meth:`self.train(False) <torch.nn.Module.train>`.

        See :ref:`locally-disable-grad-doc` for a comparison between
        `.eval()` and several similar mechanisms that may be confused with it.

        Returns:
            Module: self
        """
        return self.train(False)

    # TODO Unified training and eval interface, merge train_step(train) and __call__(eval) together
    def train_step(self, data, optimizer=None, **kwargs):
        assert self.training, "not supported train_step on eval mode"
        return self._train_executor.train_step(data, optimizer, **kwargs)

    # TODO Unified training and eval interface, merge train_step(train) and __call__(eval) together
    def __call__(self, *args, **kwargs):
        if self.training:
            raise NotImplementedError('currently the training call is implemented on function train_step')
        else:
            # self._args_parser = DictArgsParser({'args':inputs_tuple}) if self._args_parser is None else self._args_parser
            return self._eval_executor.tmp_ussage_for_eval_call(*args, **kwargs)
    
    def __getattr__(self, attr):
        return getattr(self.executor, attr)


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


def inferenceModel(model: Union['torch.nn.Module', 'poptorch.PoplarExecutor'],
                   options: Optional['poptorch.Options'] = None,
                   logger = None
                   ) -> 'poptorch.PoplarExecutor':
    """Create a PopTorch inference model, from a PyTorch model, to run on IPU
    hardware in inference mode.

    .. note:: PopTorch makes a shallow copy of the model. Changes to the
        parameters in the returned inference model affect the original model
        and vice versa. However, primitive variable types are not synced: for
        example calling ``model.eval()`` on the original model will not alter
        the model returned by this function. You may need to call
        ``model.eval()`` on your model before you call this function for correct
        behaviour.

    :param model: The PyTorch model to wrap.
    :param options: The IPU specific options
    :returns: The :py:class:`poptorch.PoplarExecutor` wrapper to use in place
        of ``model``.
    """

    if isinstance(model, PoplarExecutor):
        model = model._user_model  # pylint: disable=protected-access

    return PoplarExecutorForMMCV(model=copy.copy(model),
                          logger=logger,
                          options=options,
                          training=False,
                          poptorch_version=__version__)