# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import poptorch
import numpy as np
import inspect
import copy
import warnings
from typing import Optional, Union
from collections import OrderedDict
from poptorch import PoplarExecutor, __version__, identity_loss
from poptorch._args_parser import ArgsParser
from mmcv.parallel.data_container import DataContainer
from ..fp16_utils import auto_fp16


class DictArgsParser(ArgsParser):
    def __init__(self, inputs):
        # Combine args and kwargs:
        self._has_variadic_arguments = True
        self._varnames = list(inputs.keys())
        self._defaults = [inspect.Parameter.empty for _ in self._varnames]
        self._warned_not_contiguous_input = False


class ComplexDataManager:
    """A class used to record data structure of input of model.

    At present, the input data structure accepted by IPU is limited,
    when the input data structure of mmcv varies.
    Here, an intermediate class is needed to convert and record
    the data structure.

    Args:
        logger (warnings.warn): logger used to print warning
    """
    def __init__(self, logger=None):
        self.fixed_data_types = (int, str, float, np.ndarray, type(None))
        self.warning = warnings.warn if logger is None else logger.warning
        self.keys_of_changed_vals = []
        self.non_dict_element_changed = False
        self.quick_mode = False
        self._tree = None

    def quick(self,):
        self.quick_mode = True

    def compare_fixed_type(self, a, b):
        if isinstance(a, np.ndarray):
            return np.all(a == b)
        else:
            return a == b

    def set_tree(self, _tree):
        # _tree: A composite data type containing dictionaries, lists,
        # tensors and basic python data types
        if self._tree is not None:
            if isinstance(_tree, torch.Tensor):
                assert type(self._tree) == torch.Tensor, \
                    'original complex data is not torch.tensor'
                self._tree = _tree
            else:
                self.update(_tree)
        else:
            self._tree = _tree

    def get_tree(self,):
        return self._tree

    def update(
            self,
            treeA,
            treeB='ComplexDataManagerNone',
            strict=True,
            key=None,
            address='data'
            ):
        treeB = self._tree if treeB == 'ComplexDataManagerNone' else treeB
        # Update with a tree with the same structure
        # but different values(tensors and basic python data types)
        if isinstance(treeA, (tuple, list)):
            for idx in range(len(treeA)):
                new_address = ''
                if not self.quick_mode:
                    new_address = address+f'[{str(idx)}]'
                    assert isinstance(treeA[idx], type(treeB[idx])),\
                        f'data structure changed: {new_address}'
                if isinstance(treeA[idx], torch.Tensor):
                    treeB[idx] = treeA[idx]
                else:
                    self.update(treeA[idx], treeB[idx],
                                strict, address=new_address)
        elif isinstance(treeA, dict):
            for k, v in treeA.items():
                new_address = ''
                if not self.quick_mode:
                    new_address = address + f'[{str(k)}]'
                    assert isinstance(treeA[k], type(treeB[k])),\
                        f'data structure changed: {new_address}'
                if isinstance(v, torch.Tensor):
                    treeB[k] = treeA[k]
                else:
                    self.update(treeA[k], treeB[k], strict,
                                key, address=new_address)
        elif isinstance(treeA, self.fixed_data_types):
            if not self.quick_mode:
                is_equal = self.compare_fixed_type(treeA, treeB)
                if strict:
                    assert is_equal, 'all data except torch.Tensor '\
                        'should be same, but data({}) is changed'.\
                        format(address)
                else:
                    self.warning(f'find a non-torch.Tensor data({type(treeA)}) \
                        changed, and the address is {address}')
        elif isinstance(treeA, DataContainer):
            if not self.quick_mode:
                assert isinstance(treeB, DataContainer)
                new_address = address + '.data'
                self.update(treeA.data, treeB.data, False, address=new_address)
        else:
            raise NotImplementedError(
                f'not supported datatype:{str(treeA)}, address is {address}')

    def get_tensors(self, target_tree=None):
        # get a list of tensor from self._tree
        target_tree = self._tree if target_tree is None else target_tree
        tensors = []
        if type(target_tree) == torch.Tensor:
            tensors = [target_tree]
        else:
            self._get_tensors(target_tree, tensors)
        return tensors

    def _get_tensors(self, _tree, tensors):
        if isinstance(_tree, (tuple, list)):
            for idx in range(len(_tree)):
                if isinstance(_tree[idx], torch.Tensor):
                    tensors.append(_tree[idx])
                else:
                    self._get_tensors(_tree[idx], tensors)
        elif isinstance(_tree, dict):
            for k, v in _tree.items():
                if isinstance(v, torch.Tensor):
                    tensors.append(_tree[k])
                else:
                    self._get_tensors(_tree[k], tensors)
        elif isinstance(_tree, self.fixed_data_types):
            pass
        elif isinstance(_tree, DataContainer):
            self._get_tensors(_tree.data, tensors)
        else:
            raise NotImplementedError(
                f'not supported datatype:{str(_tree)}')

    def set_tensors(self, tensors):
        if type(self._tree) == torch.Tensor:
            assert len(tensors) == 1
            assert type(tensors[0]) == torch.Tensor
            self._tree = tensors[0]
        else:
            self._set_tensors(self._tree, tensors)
        return self._tree

    def _set_tensors(self, _tree, tensors):
        if isinstance(_tree, (tuple, list)):
            for idx in range(len(_tree)):
                if isinstance(_tree[idx], torch.Tensor):
                    _tree[idx] = tensors.pop(0)
                else:
                    self._set_tensors(_tree[idx], tensors)
        elif isinstance(_tree, dict):
            for k, v in _tree.items():
                if isinstance(v, torch.Tensor):
                    _tree[k] = tensors.pop(0)
                else:
                    self._set_tensors(_tree[k], tensors)
        elif isinstance(_tree, self.fixed_data_types):
            pass
        elif isinstance(_tree, DataContainer):
            self._set_tensors(_tree.data, tensors)
        else:
            raise NotImplementedError(
                f'not supported datatype:{str(_tree)}')

    def clean_tensors(self,):
        self._clean_tensors(self._tree)

    def _clean_tensors(self, _tree):
        if isinstance(_tree, (tuple, list)):
            for idx in range(len(_tree)):
                if isinstance(_tree[idx], torch.Tensor):
                    _tree[idx] = None
                else:
                    self._clean_tensors(_tree[idx])
        elif isinstance(_tree, dict):
            for k, v in _tree.items():
                if isinstance(v, torch.Tensor):
                    _tree[k] = None
                else:
                    self._clean_tensors(_tree[k])
        elif isinstance(_tree, self.fixed_data_types):
            pass
        elif isinstance(_tree, DataContainer):
            self._clean_tensors(_tree.data)
        else:
            raise NotImplementedError(
                f'not supported datatype:{str(_tree)}')


class WrappedNet(nn.Module):
    def __init__(
            self,
            model,
            inputs_tree_manager,
            outputs_tree_manager,
            modules_to_record=[],
            hooked_features={}
            ):
        super().__init__()
        self.model = model
        self.inputs_tree_manager = inputs_tree_manager
        self.outputs_tree_manager = outputs_tree_manager
        self.training = model.training
        # Register a hook function to capture the intermediate features
        # generated by the network to align the outputs between ipu and cpu
        self.hooked_features = hooked_features
        for idx, (_name, _module) in enumerate(model.named_modules()):
            if _name in modules_to_record or idx in modules_to_record:
                _features_hook = self.get_input_output_hook(
                    _name, idx, self.hooked_features)
                _module.register_forward_hook(hook=_features_hook)

    def get_input_output_hook(self, name, idx, save_dic):
        def input_output_hook(module, fea_in, fea_out):
            if isinstance(fea_in, tuple):
                fea_in = list(fea_in)
            if isinstance(fea_out, tuple):
                fea_out = list(fea_out)
            save_dic[name] = {'fea_in': fea_in, 'fea_out': fea_out, 'idx': idx}
            return None
        return input_output_hook

    def forward(self, inputs_tuple):
        # convert tuple back to kwargs
        self.inputs_tree_manager.set_tensors(list(inputs_tuple))
        kwargs = {**(self.inputs_tree_manager.get_tree())}
        if self.training:
            outputs = self.forward_train(kwargs)
            # tell poptorch which loss will be used finally
            identity_loss(outputs['loss'], reduction='none')
        else:
            outputs = self.forward_eval(kwargs)

        if isinstance(outputs, torch.Tensor):
            # currently not support single tensor output,
            # need to wrap it with a dictionary,
            # use a keyword to identify this case
            outputs = {'output of WrappedNet: single tensor': outputs}

        # if there are some features need to be record, add extra outputs
        for _name in self.hooked_features:
            outputs[_name] = self.hooked_features[_name]

        # record all the places of return tensors in the converting stage
        # while in the real run stage, all the tensor are changed in-place
        # that means the output can be obtained directly outside this function
        self.outputs_tree_manager.set_tree(outputs)
        plain_outputs = self.outputs_tree_manager.get_tensors()
        return plain_outputs

    def forward_train(self, kwargs):
        optimizer = kwargs.pop('optimizer')
        data = kwargs
        outputs = self.train_step(data, optimizer)
        return outputs

    def train_step(self, data, optimizer=None, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating are also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict, optional): The
                optimizer of runner is passed to ``train_step()``. This
                argument is unused and reserved.

        Returns:
            dict: Dict of outputs. The following fields are contained.
                - loss (torch.Tensor): A tensor for back propagation, which \
                    can be a weighted sum of multiple losses.
                - log_vars (dict): Dict contains all the variables to be sent \
                    to the logger.
                - num_samples (int): Indicates the batch size (when the model \
                    is DDP, it means the batch size on each GPU), which is \
                    used for averaging the logs.
        """
        losses = self.model(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

        return outputs

    def _parse_losses(self, losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            elif isinstance(loss_value, dict):
                for name, value in loss_value.items():
                    log_vars[name] = value
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)
        log_vars['loss'] = loss

        return loss, log_vars

    def forward_eval(self, kwargs):
        img = kwargs.pop('img')
        img_metas = kwargs.pop('img_metas', None)
        return_loss = kwargs.pop('return_loss')
        assert not return_loss
        # TODO Temporarily hard-code to close post_process,
        # otherwise, in the third trace(_check_trace),
        # post_process will convert output tensor to numpy array automatically,
        # resulting in _check_trace failure
        outputs = self.model(
            img, img_metas=img_metas,
            return_loss=return_loss, post_process=False)
        return outputs


class PoplarExecutorForMMCV(PoplarExecutor):
    def __init__(
            self,
            model,
            logger=None,
            training=True,
            modules_to_record=[],
            *args,
            **kwargs
            ):
        # self.model == self._user_model: input pytorch model
        # self._model: wrapped model which is used to compile
        # and update weights, these two models use same weights
        # wrapped model only accept and output tuple, so ComplexDataManager
        # will convert dictionary to tuple and convert them back
        self.inputs_tree_manager = ComplexDataManager(logger=logger)
        self.outputs_tree_manager = ComplexDataManager(logger=logger)
        self.logger = logger
        self.hooked_features = {}
        self.hooked_features_ipu = {}
        self.compare_with_cpu = True if len(modules_to_record) > 0 else False
        # move model.fp16_enabled to self.fp16_enabled,
        # modify the position where the input is automatically casted to half
        if getattr(model, 'fp16_enabled', False):
            model.fp16_enabled = False
            self.fp16_enabled = True
        # make torch.jit.trace convert self._model
        model = WrappedNet(model, self.inputs_tree_manager,
                           self.outputs_tree_manager,
                           modules_to_record=modules_to_record,
                           hooked_features=self.hooked_features)
        super().__init__(model, training=training, *args, **kwargs)
        # overwrite self._args_parser in train_step or val_step
        self._args_parser = None
        if training:
            assert self.training
        else:
            assert not self.training

    @property
    def training(self,):
        # If trying to get the attribute(training) of self,
        # since the class has no training attribute,
        # it will automatically look for the training attribute of self.model.
        # However, the real attribute we want to check is self._training,
        # self.model.training  and self._training are often inconsistent.
        # It is not clear whether it is a Poptorch bug or a special design,
        # temporarily use this function to fix the problem
        return self._training  # comes from self.model._training

    @auto_fp16(supported_types=[PoplarExecutor])
    def run_model(self, data_dict):
        # this function used to parse input_dict
        # and convert to output_dict
        if self.isCompiled():
            self.inputs_tree_manager.set_tree(data_dict)
            inputs_tuple = tuple(self.inputs_tree_manager.get_tensors())
        else:
            # get tensors out of data and put them in a tuple
            self.inputs_tree_manager.set_tree(data_dict)
            inputs_tuple = tuple(self.inputs_tree_manager.get_tensors())
            # turn logger in tree manager off after compilation
            self.inputs_tree_manager.quick()
            self.outputs_tree_manager.quick()

        # parser args for first iter
        self._args_parser = DictArgsParser(
            {'args': inputs_tuple}) if self._args_parser is None\
            else self._args_parser

        # run or convert model
        # the plain_outputs will be used in converting stage
        plain_outputs = self(inputs_tuple)

        self.inputs_tree_manager.clean_tensors()

        # put list of tensors back to the output dict
        # according to the same order
        self.outputs_tree_manager.set_tensors(plain_outputs)
        # get the real output dictionary from self.outputs_tree_manager
        output_dic = self.outputs_tree_manager.get_tree()

        # split output_dic into hooked_features_ipu
        # and output of the torch model
        mmcv_model_output = {}
        for _name in output_dic:
            if _name in self.hooked_features:
                self.hooked_features_ipu[_name] = output_dic[_name]
            else:
                mmcv_model_output[_name] = output_dic[_name]

        if 'output of WrappedNet: single tensor' in output_dic:
            assert len(mmcv_model_output) == 1
            assert type(
                mmcv_model_output['output of WrappedNet: single tensor'])\
                == torch.Tensor
            mmcv_model_output = \
                mmcv_model_output['output of WrappedNet: single tensor']

        return mmcv_model_output

    def train_step(self, data, optimizer=None, **kwargs):
        # arguments from mmcls/models/classifiers/base.py:
        # BaseClassifier.train_step
        assert self.training
        assert len(kwargs) == 0  # TODO, support later if necessary

        # TODO support datacontainer as input
        # currently, auto_fp16 and ComplexDataManager take too much time on
        # traversing datacontainer
        data['img_metas'] = None
        num_samples = len(data['img'].data)

        # TODO we will ignore optimizer for it will not be used in model,
        # support later if necessary
        data['optimizer'] = None
        output_dic = self.run_model(data)

        # outputs contained loss, log_vars, num_samples,
        # only loss(torch.tensor) has been updated
        # remove all unchanged vars, left torch.tensor
        neat_output_dic = {'loss': output_dic['loss']}

        # re-parse outputs, get back log_vars and num_samples
        loss, log_vars = self.model._parse_losses(neat_output_dic)
        final_output_dic = dict(
            loss=loss, log_vars=log_vars, num_samples=num_samples)
        return final_output_dic

    def eval_call(self, img, img_metas=None, return_loss=True, **kwargs):
        # arguments from mmdet/models/detectors/base.py:BaseDetector.forward
        # tmp usssage for eval mode
        assert not self.training
        assert len(kwargs) == 0  # TODO, support later if necessary
        assert not return_loss
        data = {'img': img, 'img_metas': img_metas, 'return_loss': return_loss}

        output_dic = self.run_model(data)

        return output_dic

    def detachFromDevice(self,):
        if self.isCompiled() and self._is_attached:
            super().detachFromDevice()

    def attachToDevice(self,):
        if self.isCompiled() and not self._is_attached:
            super().attachToDevice()


def compare_feat(featA, featB, rtol=1e-3, atol=1e-5):
    try:
        np.testing.assert_allclose(featA, featB, rtol=rtol, atol=atol)
    except AssertionError as e:
        print(e)


class TrainEvalModel:
    def __init__(
            self,
            train_model,
            eval_model,
            options,
            optimizer,
            modules_to_record=[],
            logger=None):
        if train_model is None:
            self._train_executor = None
            self.training = False
        else:
            self._train_executor = trainingModel(
                train_model, options=options['training'], optimizer=optimizer,
                logger=logger, modules_to_record=modules_to_record)
            self.training = True
        self._eval_executor = inferenceModel(
            eval_model, options=options['inference'], logger=logger)

    @property
    def executor(self,):
        if self.training:
            return self._train_executor
        else:
            return self._eval_executor

    def train(self, mode: bool = True):
        r"""Sets the module in training mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in
        training/evaluation mode, if they are affected,
        e.g. :class:`Dropout`, :class:`BatchNorm`, etc.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        if not isinstance(mode, bool):
            raise ValueError('training mode is expected to be boolean')
        if self._train_executor is None and mode is True:
            raise RuntimeError(
                'The train_executor is not initialized.'
                'If you want to initialize train_executor,'
                'you need to input optimizer when converting pytorch model')

        if mode == self.training:
            self.model.train(mode)
            return self
        else:
            if self.isCompiled():
                # copy weights from IPU to cpu before off-load current session
                self.copyWeightsToHost()
                # detach the current session before change the mode,
                # if is training mode and weights are updated,
                # poptorch will copy weights from IPU to host
                self.detachFromDevice()

            self.training = mode  # session will changed with mode changing
            self.model.train(mode)

            # after changing mode, attach the current new session,
            # and this function will copy weights of model to device
            self.attachToDevice()
            return self

    def eval(self):
        r"""Sets the module in evaluation mode.

        This has any effect only on certain modules.
        See documentations of particular modules
        for details of their behaviors in training/evaluation mode,
        if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`, etc.

        This is equivalent with :meth:`self.train(False)
        <nn.Module.train>`.

        See :ref:`locally-disable-grad-doc` for a comparison between
        `.eval()` and several similar mechanisms that may be confused with it.

        Returns:
            Module: self
        """
        return self.train(False)

    def compare_data_between_ipu_and_cpu(
            self,
            hooked_features_cpu,
            hooked_features_ipu
            ):
        for _key, _val in hooked_features_cpu.items():
            fea_in_cpu_list = [_val['fea_in']] if isinstance(
                _val['fea_in'], torch.Tensor) else _val['fea_in']
            fea_in_ipu_list = [hooked_features_ipu[_key]['fea_in']] \
                if isinstance(_val['fea_in'], torch.Tensor) \
                else hooked_features_ipu[_key]['fea_in']

            fea_out_cpu_list = [_val['fea_out']] if isinstance(
                _val['fea_out'], torch.Tensor) else _val['fea_out']
            fea_out_ipu_list = [hooked_features_ipu[_key]['fea_out']] \
                if isinstance(_val['fea_out'], torch.Tensor) \
                else hooked_features_ipu[_key]['fea_out']

            print(_key)
            for idx, (featA, featB) in \
                    enumerate(zip(fea_in_cpu_list, fea_in_ipu_list)):
                print('fea_in, tensor ', idx)
                compare_feat(featA.detach().numpy(), featB.detach().numpy())
            for idx, (featA, featB) in \
                    enumerate(zip(fea_out_cpu_list, fea_out_ipu_list)):
                print('fea_out, tensor', idx)
                compare_feat(featA.detach().numpy(), featB.detach().numpy())

    # TODO Unified training and eval interface,
    # merge train_step(train) and __call__(eval) together
    def train_step(self, data, optimizer=None, **kwargs):
        assert self.training, 'not supported train_step on eval mode'
        hooked_features_cpu = {}
        if self._train_executor.isCompiled() \
                and self._train_executor.compare_with_cpu:
            self.copyWeightsToHost()
            self._train_executor.model.train_step(data, optimizer, **kwargs)
            hooked_features_cpu = {**(self._train_executor.hooked_features)}
        result = self._train_executor.train_step(data, optimizer, **kwargs)
        if self._train_executor.isCompiled() \
            and self._train_executor.compare_with_cpu \
                and len(hooked_features_cpu) > 0:
            self.compare_data_between_ipu_and_cpu(
                hooked_features_cpu, self._train_executor.hooked_features_ipu)
        return result

    # TODO Unified training and eval interface,
    # merge train_step(train) and __call__(eval) together
    def __call__(self, *args, **kwargs):
        if self.training:
            raise NotImplementedError(
                'use train_step rather than __call__')
        else:
            return self._eval_executor.eval_call(*args, **kwargs)

    def __getattr__(self, attr):
        return getattr(self.executor, attr)


def trainingModel(model: Union['nn.Module', 'poptorch.PoplarExecutor'],
                  options: Optional['poptorch.Options'] = None,
                  optimizer: Optional['torch.optim.Optimizer'] = None,
                  logger=None,
                  modules_to_record=[]
                  ) -> 'poptorch.PoplarExecutor':
    """Create a PopTorch training model from a PyTorch model, to run on IPU
    hardware in training mode.

    Note:
        PopTorch makes a shallow copy of the model. Changes to the
        parameters in the returned training model affect the original model
        and vice versa. However, primitive variable types are not synced: for
        example calling ``model.train()`` on the original model, which
        changes the ``training`` bool of the model instance, will not alter the
        model returned by this function. You may need to call ``model.train()``
        on your model before you call this function for correct behavior.

    :param model: The PyTorch model to wrap.
    :param options: The IPU specific options
    :param optimizer: The optimizers to apply during \
        training.

        - Supported PyTorch optimizers: ``optim.SGD``, ``optim.Adam``,
          ``optim.AdamW`` and ``optim.RMSprop``.

        - Supported PopTorch optimizers: ``poptorch.optim.SGD``,
           ``poptorch.optim.Adam``, ``poptorch.optim.AdamW``,
           ``poptorch.optim.RMSprop`` and ``poptorch.optim.LAMB``.

    returns:
        The :class:`poptorch.PoplarExecutor` wrapper to use in place
        of ``model``.
    """
    # Create a copy of the original model in case it needs to be wrapped
    maybe_wrapped_model = copy.copy(model)

    return PoplarExecutorForMMCV(model=maybe_wrapped_model,
                                 logger=logger,
                                 options=options,
                                 training=True,
                                 optimizer=optimizer,
                                 user_model=model,
                                 modules_to_record=modules_to_record,
                                 poptorch_version=__version__,)


def inferenceModel(model: Union['nn.Module', 'poptorch.PoplarExecutor'],
                   options: Optional['poptorch.Options'] = None,
                   logger=None
                   ) -> 'poptorch.PoplarExecutor':
    """Create a PopTorch inference model from a PyTorch model, to run on IPU
    hardware in inference mode.

    .. note:: PopTorch makes a shallow copy of the model. Changes to the
        parameters in the returned inference model affect the original model
        and vice versa. However, primitive variable types are not synced: for
        example calling ``model.eval()`` on the original model will not alter
        the model returned by this function. You may need to call
        ``model.eval()`` on your model before you call this function for
        correct behavior.

    :param model: The PyTorch model to wrap.
    :param options: The IPU specific options
    :returns: The :py:class:`poptorch.PoplarExecutor` wrapper to use in place
        of ``model``.
    """

    return PoplarExecutorForMMCV(model=copy.copy(model),
                                 logger=logger,
                                 options=options,
                                 training=False,
                                 poptorch_version=__version__)
