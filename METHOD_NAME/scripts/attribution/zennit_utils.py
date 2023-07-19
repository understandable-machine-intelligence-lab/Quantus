import torchvision
import torch

from zennit import canonizers as zcanon
from zennit import composites as zcomp
from zennit import torchvision as zvision
from zennit import rules as zrules
from zennit import types as ztypes

def get_zennit_canonizer(model):
    """
    Checks the type of model and selects the corresponding zennit canonizer
    """

    #ResNet
    if isinstance(model, torchvision.models.ResNet):
        return zvision.ResNetCanonizer

    #VGG
    if isinstance(model, torchvision.models.VGG):
        return zvision.VGGCanonizer

    #default fallback (only the above types have specific canonizers in zennit for now)
    return zcanon.SequentialMergeBatchNorm

class Epsilon(zcomp.LayerMapComposite):
    '''An explicit composite using the epsilon rule for all layers

    Parameters
    ----------
    epsilon: callable or float, optional
        Stabilization parameter for the ``Epsilon`` rule. If ``epsilon`` is a float, it will be added to the
        denominator with the same sign as each respective entry. If it is callable, a function ``(input: torch.Tensor)
        -> torch.Tensor`` is expected, of which the output corresponds to the stabilized denominator. Note that this is
        called ``stabilizer`` for all other rules.
    stabilizer: callable or float, optional
        Stabilization parameter for rules other than ``Epsilon``. If ``stabilizer`` is a float, it will be added to the
        denominator with the same sign as each respective entry. If it is callable, a function ``(input: torch.Tensor)
        -> torch.Tensor`` is expected, of which the output corresponds to the stabilized denominator.
    layer_map: list[tuple[tuple[torch.nn.Module, ...], Hook]]
        A mapping as a list of tuples, with a tuple of applicable module types and a Hook. This will be prepended to
        the ``layer_map`` defined by the composite.
    zero_params: list[str], optional
        A list of parameter names that shall set to zero. If `None` (default), no parameters are set to zero.
    canonizers: list[:py:class:`zennit.canonizers.Canonizer`], optional
        List of canonizer instances to be applied before applying hooks.
    '''
    def __init__(self, epsilon=1e-6, stabilizer=1e-6, layer_map=None, zero_params=None, canonizers=None):
        if layer_map is None:
            layer_map = []

        rule_kwargs = {'zero_params': zero_params}
        layer_map = layer_map + zcomp.layer_map_base(stabilizer) + [
            (ztypes.Convolution, zrules.Epsilon(epsilon=epsilon, **rule_kwargs)),
            (torch.nn.Linear, zrules.Epsilon(epsilon=epsilon, **rule_kwargs)),
        ]
        super().__init__(layer_map=layer_map, canonizers=canonizers)

class ZPlus(zcomp.LayerMapComposite):
    '''An explicit composite using the epsilon rule for all layers

    Parameters
    ----------
    stabilizer: callable or float, optional
        Stabilization parameter for rules other than ``Epsilon``. If ``stabilizer`` is a float, it will be added to the
        denominator with the same sign as each respective entry. If it is callable, a function ``(input: torch.Tensor)
        -> torch.Tensor`` is expected, of which the output corresponds to the stabilized denominator.
    layer_map: list[tuple[tuple[torch.nn.Module, ...], Hook]]
        A mapping as a list of tuples, with a tuple of applicable module types and a Hook. This will be prepended to
        the ``layer_map`` defined by the composite.
    zero_params: list[str], optional
        A list of parameter names that shall set to zero. If `None` (default), no parameters are set to zero.
    canonizers: list[:py:class:`zennit.canonizers.Canonizer`], optional
        List of canonizer instances to be applied before applying hooks.
    '''
    def __init__(self, stabilizer=1e-6, layer_map=None, zero_params=None, canonizers=None):
        if layer_map is None:
            layer_map = []

        rule_kwargs = {'zero_params': zero_params}
        layer_map = layer_map + zcomp.layer_map_base(stabilizer) + [
            (ztypes.Convolution, zrules.ZPlus(stabilizer=stabilizer, **rule_kwargs)),
            (torch.nn.Linear, zrules.ZPlus(stabilizer=stabilizer, **rule_kwargs)),
        ]
        super().__init__(layer_map=layer_map, canonizers=canonizers)