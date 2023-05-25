import re

import torch

from torch.nn.parallel import DataParallel, DistributedDataParallel

from .registry import Registry

MODULE_WRAPPERS = Registry('module wrapper')
MODULE_WRAPPERS.register_module(module=DataParallel)
MODULE_WRAPPERS.register_module(module=DistributedDataParallel)


def is_module_wrapper(module):
    """Check if a module is a module wrapper.
    The following 3 modules in MMCV (and their subclasses) are regarded as
    module wrappers: DataParallel, DistributedDataParallel,
    MMDistributedDataParallel (the deprecated version). You may add you own
    module wrapper by registering it to mmcv.parallel.MODULE_WRAPPERS.
    Args:
        module (nn.Module): The module to be checked.
    Returns:
        bool: True if the input module is a module wrapper.
    """
    module_wrappers = tuple(MODULE_WRAPPERS.module_dict.values())
    return isinstance(module, module_wrappers)

def _load_checkpoint(
        model,
        filename,
        map_location=None,
        strict=False,
        logger=None,
        revise_keys=[(r'^module\.', '')]):
    """Load checkpoint from a file or URI.
    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.
        revise_keys (list): A list of customized keywords to modify the
            state_dict in checkpoint. Each item is a (pattern, replacement)
            pair of the regular expression operations. Default: strip
            the prefix 'module.' by [(r'^module\\.', '')].
    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """

    def _load_state_dict(module, state_dict, strict=False, logger=None):
        """Load state_dict to a module.
        This method is modified from :meth:`torch.nn.Module.load_state_dict`.
        Default value for ``strict`` is set to ``False`` and the message for
        param mismatch will be shown even if strict is False.
        Args:
            module (Module): Module that receives the state_dict.
            state_dict (OrderedDict): Weights.
            strict (bool): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
            logger (:obj:`logging.Logger`, optional): Logger to log the error
                message. If not specified, print function will be used.
        """

        unexpected_keys = []
        all_missing_keys = []
        err_msg = []

        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        # use _load_from_state_dict to enable checkpoint version control
        def load(module, prefix=''):
            # recursively check parallel module in case that the model has a
            # complicated structure, e.g., nn.Module(nn.Module(DDP))
            if is_module_wrapper(module):
                module = module.module
            local_metadata = {} if metadata is None else metadata.get(
                prefix[:-1], {})
            module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                         all_missing_keys, unexpected_keys,
                                         err_msg)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(module)
        load = None  # break load->load reference cycle

        # ignore "num_batches_tracked" of BN layers
        missing_keys = [
            key for key in all_missing_keys if 'num_batches_tracked' not in key
        ]

        if unexpected_keys:
            err_msg.append('unexpected key in source '
                           f'state_dict: {", ".join(unexpected_keys)}\n')
        if missing_keys:
            err_msg.append(
                f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

        if len(err_msg) > 0:
            err_msg.insert(
                0, 'The model and loaded state dict do not match exactly\n')
            err_msg = '\n'.join(err_msg)
            if strict:
                raise RuntimeError(err_msg)
            elif logger is not None:
                logger.warning(err_msg)
            else:
                print(err_msg)

    with open(filename, 'rb', 0) as f_obj:
        checkpoint = torch.load(f_obj, map_location)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    # get state_dict from checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    # strip prefix of state_dict
    for p, r in revise_keys:
        state_dict = {re.sub(p, r, k): v for k, v in state_dict.items()}
    # load state_dict
    _load_state_dict(model, state_dict, strict, logger)
    return checkpoint