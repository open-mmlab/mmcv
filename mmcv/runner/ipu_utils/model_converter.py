import warnings
import re
import torch
import inspect
import copy
from typing import Any, Callable, Dict, Iterator, Optional, Union
from poptorch import PoplarExecutor, poptorch_core, __version__, identity_loss
from poptorch._args_parser import ArgsParser

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

class PlaceHolder:
    def __init__(self,data=None):
        self.data = data

def retrive_vals_from_tree(_tree,placeholders,vals):
    if isinstance(_tree, (tuple,list)):
        for idx in range(len(_tree)):
            if isinstance(_tree[idx], (tuple,list,dict)):
                retrive_vals_from_tree(_tree[idx],placeholders,vals)
            else:
                p = PlaceHolder(_tree[idx])
                if isinstance(_tree[idx],torch.Tensor):
                    vals.append(_tree[idx])
                    placeholders.append(p)
                _tree[idx] = p
    elif isinstance(_tree, dict):
        for k,v in _tree.items():
            if isinstance(v, (tuple,list,dict)):
                retrive_vals_from_tree(v,placeholders,vals)
            else:
                p = PlaceHolder(v)
                if isinstance(v,torch.Tensor):
                    vals.append(v)
                    placeholders.append(p)
                _tree[k] = p
    else:
        raise NotImplementedError        


def get_train_step_wrapper(user_model, not_traced_inputs, traced_input_keys, output_tree_and_placeholders):
    class Net(torch.nn.Module):
        def __init__(self, user_model,not_traced_inputs,traced_input_keys,output_tree_and_placeholders):
            super().__init__()
            self.model = user_model
            self.not_traced_inputs = not_traced_inputs
            self.traced_input_keys = traced_input_keys
            self.output_tree_and_placeholders = output_tree_and_placeholders # (output tree, placeholders contained in the output tree)

        def forward(self, inputs_tuple):
        # convert tuple back to kwargs
            kwargs = {_key:_val for _key,_val in zip(self.traced_input_keys,inputs_tuple)}
            kwargs = {**kwargs, **(self.not_traced_inputs)} # add back all inputs that will not be traced
            optimizer = kwargs.pop('optimizer')
            data = kwargs
            outputs = self.model.train_step(data,optimizer)
            identity_loss(outputs['loss'],reduction='none')
            plain_outputs = []
            place_holders = []
            retrive_vals_from_tree(outputs,place_holders,plain_outputs)
            self.output_tree_and_placeholders[0] = outputs
            self.output_tree_and_placeholders[1] = place_holders
            return plain_outputs

    return Net(user_model,not_traced_inputs,traced_input_keys, output_tree_and_placeholders)


class PoplarExecutorForMMCV(PoplarExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._self_specified_func = None # function will be converted by torch.jit.trace, default is self._model
        self._args_parser = None # overwrite self._args_parser in train_step or val_step
        self.not_traced_inputs = {} # all inputs that cannot be traced will be put here
        self.traced_input_keys = [] # flatten dictionary input to tuple, record all keys
        self.output_tree_and_placeholders = [None, None] # output dictionary, placeholders in the nodes of output dictionary

    def train_step(self, data, optimizer=None, **kwargs):
        assert len(kwargs) == 0
        self._self_specified_func = get_train_step_wrapper(self._user_model, self.not_traced_inputs, self.traced_input_keys, self.output_tree_and_placeholders) # make torch.jit.trace convert this function
        data['optimizer'] = optimizer
        # remove inputs that cannot be traced
        for key_cannot_be_traced in ['img_metas','optimizer']:
            self.not_traced_inputs[key_cannot_be_traced] = data.pop(key_cannot_be_traced)
        inputs_tuple = []
        for _key in data:
            self.traced_input_keys.append(_key)
            inputs_tuple.append(data[_key])
        self._args_parser = DictArgsParser({'args':inputs_tuple}) if self._args_parser is None else self._args_parser
        plain_outputs = self(inputs_tuple)
        output_dic, place_holders = self.output_tree_and_placeholders
        for output,place_holder in zip(plain_outputs,place_holders):
            place_holder.data = output
        return output_dic


    def _trace_with_warning_filter(self, in_tensors_trace_view_tuple):
        # Conditionally suppress the following jit warnings when the model
        # contains any non-deterministic nodes (e.g. dropout)
        rng_warnings = [
            "Trace had nondeterministic nodes",
            "the traced function does not match the corresponding output"
        ]

        def filterWarnings(warning):
            return not any([m in str(warning.message) for m in rng_warnings])

        warns = []
        with warnings.catch_warnings(record=True) as caught:
            try:
                traced = torch.jit.trace(self._model if self._self_specified_func is None else self._self_specified_func,
                                        in_tensors_trace_view_tuple)
            except RuntimeError as e:
                pattern = r'Type \'Tuple(\[.*\])\' cannot be traced'
                match = re.match(pattern, str(e))
                if match:
                    types = match.group(1)
                    raise TypeError(
                        "All forward function arguments used to compile and "
                        "run the model must be Tensors or (possibly nested) "
                        f"Lists and Tuples of Tensors (Got types: {types})."
                    ).with_traceback(e.__traceback__)
                raise e

            # pylint: disable=protected-access
            if poptorch_core.isGraphNondeterministic(traced._c):
                warns = list(filter(filterWarnings, caught))

        # Reissue remaining warnings
        for w in warns:
            warnings.warn_explicit(message=w.message,
                                category=w.category,
                                filename=w.filename,
                                lineno=w.lineno)

        return traced

def trainingModel(model: Union['torch.nn.Module', 'poptorch.PoplarExecutor'],
                  options: Optional['poptorch.Options'] = None,
                  optimizer: Optional['torch.optim.Optimizer'] = None
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
                          options=options,
                          training=True,
                          optimizer=optimizer,
                          user_model=model,
                          poptorch_version=__version__)