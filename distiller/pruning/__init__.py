#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
:mod:`distiller.pruning` is a package implementing various pruning algorithms.
"""

from .magnitude_pruner import MagnitudeParameterPruner
from .automated_gradual_pruner import AutomatedGradualPruner, \
                                      L1RankedStructureParameterPruner_AGP, \
                                      L2RankedStructureParameterPruner_AGP, \
                                      ActivationAPoZRankedFilterPruner_AGP, \
                                      ActivationMeanRankedFilterPruner_AGP, \
                                      GradientRankedFilterPruner_AGP, \
                                      RandomRankedFilterPruner_AGP, \
                                      BernoulliFilterPruner_AGP
from .level_pruner import SparsityLevelParameterPruner
from .sensitivity_pruner import SensitivityPruner
from .splicing_pruner import SplicingPruner
from .structure_pruner import StructureParameterPruner
from .ranked_structures_pruner import L1RankedStructureParameterPruner, \
                                      L2RankedStructureParameterPruner, \
                                      ActivationAPoZRankedFilterPruner, \
                                      ActivationMeanRankedFilterPruner, \
                                      GradientRankedFilterPruner,       \
                                      RandomRankedFilterPruner,         \
                                      RandomLevelStructureParameterPruner, \
                                      BernoulliFilterPruner,            \
                                      FMReconstructionChannelPruner
from .baidu_rnn_pruner import BaiduRNNPruner
from .greedy_filter_pruning import greedy_pruner
from .compenstation import CompensatePrunner
import torch
import torch.nn as nn

del magnitude_pruner
del automated_gradual_pruner
del level_pruner
del sensitivity_pruner
del structure_pruner
del ranked_structures_pruner


def mask_tensor(tensor, mask, inplace=True):
    """Mask the provided tensor

    Args:
        tensor - the torch-tensor to mask
        mask - binary coefficient-masking tensor.  Has the same type and shape as `tensor`
    Returns:
        tensor = tensor * mask  ;where * is the element-wise multiplication operator
    """
    assert tensor.type() == mask.type()
    assert tensor.shape == mask.shape
    if mask is not None:
        return tensor.data.mul_(mask) if inplace else tensor.data.mul(mask)
    return tensor


def create_mask_threshold_criterion(tensor, threshold):
    """Create a tensor mask using a threshold criterion.

    All values smaller or equal to the threshold will be masked-away.
    Granularity: Element-wise
    Args:
        tensor - the tensor to threshold.
        threshold - a floating-point threshold value.
    Returns:
        boolean mask tensor, having the same size as the input tensor.
    """
    with torch.no_grad():
        mask = torch.gt(torch.abs(tensor), threshold).type(tensor.type())
        return mask


def create_mask_level_criterion(tensor, desired_sparsity):
    """Create a tensor mask using a level criterion.

    A specified fraction of the input tensor will be masked.  The tensor coefficients
    are first sorted by their L1-norm (absolute value), and then the lower `desired_sparsity`
    coefficients are masked.
    Granularity: Element-wise

    WARNING: due to the implementation details (speed over correctness), this will perform
    incorrectly if "too many" of the coefficients have the same value. E.g. this will fail:
        a = torch.ones(3, 64, 32, 32)
        mask = distiller.create_mask_level_criterion(a, desired_sparsity=0.3)
        assert math.isclose(distiller.sparsity(mask), 0.3)

    Args:
        tensor - the tensor to mask.
        desired_sparsity - a floating-point value in the range (0..1) specifying what
            percentage of the tensor will be masked.
    Returns:
        boolean mask tensor, having the same size as the input tensor.
    """
    with torch.no_grad():
        # partial sort
        bottomk, _ = torch.topk(tensor.abs().view(-1),
                                int(desired_sparsity * tensor.numel()),
                                largest=False,
                                sorted=True)
        threshold = bottomk.data[-1]  # This is the largest element from the group of elements that we prune away
        mask = create_mask_threshold_criterion(tensor, threshold)
        return mask


def create_mask_sensitivity_criterion(tensor, sensitivity):
    """Create a tensor mask using a sensitivity criterion.

    Mask an input tensor based on the variance of the distribution of the tensor coefficients.
    Coefficients in the distribution's specified band around the mean will be masked (symmetrically).
    Granularity: Element-wise
    Args:
        tensor - the tensor to mask.
        sensitivity - a floating-point value specifying the sensitivity.  This is a simple
            multiplier of the standard-deviation.
    Returns:
        boolean mask tensor, having the same size as the input tensor.
    """
    if not hasattr(tensor, 'stddev'):
        tensor.stddev = torch.std(tensor).item()
    with torch.no_grad():
        threshold = tensor.stddev * sensitivity
        mask = create_mask_threshold_criterion(tensor, threshold)
        return mask

def create_mask_similarity_criterion(tensor, threshold, bn_lambda):
    """Create a tensor redirection using cosine similarity criterion.

    takes a already masked tensor and compensate it by merging masked coefficients each
    with its most similar coefficient.
    Args:
        tensor - tensor masked based on sensitivity criterion.
        threshold - a floating-point threshold value betwee 0 and 1. Enumerate masked
            coefficients and find a most similar coefficient that is kept based on cosine
            similarity. Then if the similiarity is larger or equal to the threshold, masked
            coefficients will be recovered and redirected to its similar coefficient.
        bn_lambda - a floating-point value specifying how much should differences on bias be
            considered while calculating similiarity.
    Returns:
        redirect tensor, having the same size as the input tensor. 0 for nothing changed,
        otherwise the value is the index of its most similar coefficient.
    """

def similarity_criterion_most_similar(tensor, threshold, bn_lambda):
    """Calculate similarity for creat_mask_similarity_criterion().

    Features with exactly same value have different direction after batch normalization layer.
    Similarity in different kind of layers are calculated respectively. Relationship between
    masked and non-masked coefficients are represented as redirection tensor.
    Args:
        tensor - tensor masked based on sensitivity criterion.
        threshold - a floating-point threshold value betwee 0 and 1. Enumerate masked
            coefficients and find a most similar coefficient that is kept based on cosine
            similarity. Then if the similiarity is larger or equal to the threshold, masked
            coefficients will be recovered and redirected to its similar coefficient.
        bn_lambda - a floating-point value specifying how much should differences on bias be
            considered while calculating similiarity.
    Returns:
        redirect tensor, having the same size as the input tensor. 0 for nothing changed,
        otherwise the value is the index of its most similar coefficient.
    """
    # similarity is defined differently in batch normalization layer
    pass


def most_similar(weights_column: torch.Tensor, neurons: torch.Tensor) -> Tuple(torch.Tensor, float, float):
    neurons_T = neurons.transpose(0, 1)
    cos = nn.CosineSimilarity(0)
    similarities = [cos(neuron, weights_column).item() for neuron in neurons_T]
    similarities = torch.Tensor(similarities)
    max_similarity, max_neuron_index = torch.max(similarities, 0)
    scale = torch.norm(weights_column) / torch.norm(neurons_T[max_neuron_index])

    return neurons_T[max_neuron_index], max_similarity, scale


def get_important_weights(layer_name: str, parameter_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    return parameter_dict['weight']


def decompose(original_weights: torch.Tensor, important_weights: torch.Tensor, threshould: float) -> torch.Tensor:
    """
    [Inputs]
    original_weights: (N[i], N[i+1]) 
    important_weights: (N[i], P[i+1])

    [Outputs]
    scaling_matrix: (P[i+1], N[i+1])
    """

    scaling_matrix = torch.zeros(important_weights.size()[-1], original_weights.size()[-1])

    for weight in original_weights.transpose(0, -1):
        if weight in important_weights.transpose(0, -1):
            scaling_matrix[important_weights.transpose(0, -1) == weight] = 1
        else:
            most_similar_neuron, similarity, scale = most_similar(weight, important_weights)
            most_similar_neuron_index_in_important_weights = important_weights == most_similar_neuron
            if similarity >= threshould:
                scaling_matrix[most_similar_neuron_index_in_important_weights] = scale

    return scaling_matrix


def compensation(module_name: str, original_weights_2: torch.Tensor, scaling_matrix: torch.Tensor) -> torch.Tensor:
    """
    [Inputs]
    original_weights_2: (N[i+2], N[i+1], K, K)
    scaling_matrix: (P[n+1], N[i+1])
    [Outputs]
    new_weights_2: (N[i+2], P[i+1])
    """
    if module_name.startswith('conv'):
        # 2-mode product
        new_weights_2 = torch.tensordot(original_weights_2, scaling_matrix, dims=([1], [1]))
    else:
        # Note that when multiplying the next layer, it needs to be transposed
        new_weights_2 = torch.matmul(scaling_matrix, original_weights_2)
    return new_weights_2


def reload_weights(model: nn.Module, compensated_weight_list: List[torch.Tensor]) -> nn.Module:
    for i, layer in enumerate(model.state_dict()):
        model.state_dict()[layer].copy_(compensated_weight_list[i])

    return model


def merge_pruning_compensation(model: nn.Module, threshould: float) -> nn.Module:
    last_name = ""
    last_layer: torch.Tensor = None
    compensated_weight_list = []

    for name, layer in model.named_modules():
        if name:  # otherwise it's the parent node nad we don't need it
            if last_name and last_layer:
                last_parameter_dict = dict(last_layer.named_parameters())

                last_important_weights = get_important_weights(last_name, last_parameter_dict)
                scaling_matrix = decompose(last_parameter_dict['weight'], last_important_weights, threshould)

                parameter_dict = dict(layer.named_parameters())
                new_weights = compensation(name, parameter_dict['weight'], scaling_matrix)
            else:
                parameter_dict = dict(layer.named_parameters())
                # Just return this layer's original weight
                new_weights = parameter_dict['weight']

            last_name, last_layer = name, layer
            compensated_weight_list.append(new_weights)

    new_model = reload_weights(model, compensated_weight_list)

    return new_model
