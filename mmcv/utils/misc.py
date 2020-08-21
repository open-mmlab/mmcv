# Copyright (c) Open-MMLab. All rights reserved.
import functools
import itertools
import subprocess
import warnings
from collections import abc
from importlib import import_module
from inspect import getfullargspec


def is_str(x):
    """Whether the input is an string instance.

    Note: This method is deprecated since python 2 is no longer supported.
    """
    return isinstance(x, str)


def iter_cast(inputs, dst_type, return_type=None):
    """Cast elements of an iterable object into some type.

    Args:
        inputs (Iterable): The input object.
        dst_type (type): Destination type.
        return_type (type, optional): If specified, the output object will be
            converted to this type, otherwise an iterator.

    Returns:
        iterator or specified type: The converted object.
    """
    if not isinstance(inputs, abc.Iterable):
        raise TypeError('inputs must be an iterable object')
    if not isinstance(dst_type, type):
        raise TypeError('"dst_type" must be a valid type')

    out_iterable = map(dst_type, inputs)

    if return_type is None:
        return out_iterable
    else:
        return return_type(out_iterable)


def list_cast(inputs, dst_type):
    """Cast elements of an iterable object into a list of some type.

    A partial method of :func:`iter_cast`.
    """
    return iter_cast(inputs, dst_type, return_type=list)


def tuple_cast(inputs, dst_type):
    """Cast elements of an iterable object into a tuple of some type.

    A partial method of :func:`iter_cast`.
    """
    return iter_cast(inputs, dst_type, return_type=tuple)


def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.

    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def is_list_of(seq, expected_type):
    """Check whether it is a list of some type.

    A partial method of :func:`is_seq_of`.
    """
    return is_seq_of(seq, expected_type, seq_type=list)


def is_tuple_of(seq, expected_type):
    """Check whether it is a tuple of some type.

    A partial method of :func:`is_seq_of`.
    """
    return is_seq_of(seq, expected_type, seq_type=tuple)


def slice_list(in_list, lens):
    """Slice a list into several sub lists by a list of given length.

    Args:
        in_list (list): The list to be sliced.
        lens(int or list): The expected length of each out list.

    Returns:
        list: A list of sliced list.
    """
    if isinstance(lens, int):
        assert len(in_list) % lens == 0
        lens = [lens] * int(len(in_list) / lens)
    if not isinstance(lens, list):
        raise TypeError('"indices" must be an integer or a list of integers')
    elif sum(lens) != len(in_list):
        raise ValueError('sum of lens and list length does not '
                         f'match: {sum(lens)} != {len(in_list)}')
    out_list = []
    idx = 0
    for i in range(len(lens)):
        out_list.append(in_list[idx:idx + lens[i]])
        idx += lens[i]
    return out_list


def concat_list(in_list):
    """Concatenate a list of list into a single list.

    Args:
        in_list (list): The list of list to be merged.

    Returns:
        list: The concatenated flat list.
    """
    return list(itertools.chain(*in_list))


def check_prerequisites(
        prerequisites,
        checker,
        msg_tmpl='Prerequisites "{}" are required in method "{}" but not '
        'found, please install them first.'):  # yapf: disable
    """A decorator factory to check if prerequisites are satisfied.

    Args:
        prerequisites (str of list[str]): Prerequisites to be checked.
        checker (callable): The checker method that returns True if a
            prerequisite is meet, False otherwise.
        msg_tmpl (str): The message template with two variables.

    Returns:
        decorator: A specific decorator.
    """

    def wrap(func):

        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            requirements = [prerequisites] if isinstance(
                prerequisites, str) else prerequisites
            missing = []
            for item in requirements:
                if not checker(item):
                    missing.append(item)
            if missing:
                print(msg_tmpl.format(', '.join(missing), func.__name__))
                raise RuntimeError('Prerequisites not meet.')
            else:
                return func(*args, **kwargs)

        return wrapped_func

    return wrap


def _check_py_package(package):
    try:
        import_module(package)
    except ImportError:
        return False
    else:
        return True


def _check_executable(cmd):
    if subprocess.call(f'which {cmd}', shell=True) != 0:
        return False
    else:
        return True


def requires_package(prerequisites):
    """A decorator to check if some python packages are installed.

    Example:
        >>> @requires_package('numpy')
        >>> func(arg1, args):
        >>>     return numpy.zeros(1)
        array([0.])
        >>> @requires_package(['numpy', 'non_package'])
        >>> func(arg1, args):
        >>>     return numpy.zeros(1)
        ImportError
    """
    return check_prerequisites(prerequisites, checker=_check_py_package)


def requires_executable(prerequisites):
    """A decorator to check if some executable files are installed.

    Example:
        >>> @requires_executable('ffmpeg')
        >>> func(arg1, args):
        >>>     print(1)
        1
    """
    return check_prerequisites(prerequisites, checker=_check_executable)


def deprecated_api_warning(name_dict, cls_name=None):
    """A decorator to check if some argments are deprecate and try to replace
    deprecate src_arg_name to dst_arg_name.

    Args:
        name_dict(dict):
            key (str): Deprecate argument names.
            val (str): Expected argument names.

    Returns:
        func: New function.
    """

    def api_warning_wrapper(old_func):

        @functools.wraps(old_func)
        def new_func(*args, **kwargs):
            # get the arg spec of the decorated method
            args_info = getfullargspec(old_func)
            # get name of the function
            func_name = old_func.__name__
            if cls_name is not None:
                func_name = f'{cls_name}.{func_name}'
            if args:
                arg_names = args_info.args[:len(args)]
                for src_arg_name, dst_arg_name in name_dict.items():
                    if src_arg_name in arg_names:
                        warnings.warn(
                            f'"{src_arg_name}" is deprecated in '
                            f'`{func_name}`, please use "{dst_arg_name}" '
                            'instead')
                        arg_names[arg_names.index(src_arg_name)] = dst_arg_name
            if kwargs:
                for src_arg_name, dst_arg_name in name_dict.items():
                    if src_arg_name in kwargs:
                        warnings.warn(
                            f'"{src_arg_name}" is deprecated in '
                            f'`{func_name}`, please use "{dst_arg_name}" '
                            'instead')
                        kwargs[dst_arg_name] = kwargs.pop(src_arg_name)

            # apply converted arguments to the decorated method
            output = old_func(*args, **kwargs)
            return output

        return new_func

    return api_warning_wrapper
