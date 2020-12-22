

import logging
from typing import Dict, List, Tuple
import torch
from torch import nn

msglogger = logging.getLogger()

def most_similar(weights_column: torch.Tensor, neurons: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
    """
    Get the most similar column, it's similarity, and the scale (to be inserted into t he scaling matrix)
    """
    neurons_T = neurons.transpose(0, 1)

    msglogger.info('neurons_T', neurons_T.size())
    msglogger.info('weights_column', weights_column.size())

    cos = nn.CosineSimilarity(0)
    similarities = [cos(neuron, weights_column).item() for neuron in neurons_T]
    similarities = torch.Tensor(similarities)
    max_similarity, max_neuron_index = torch.max(similarities, 0)
    scale = torch.norm(weights_column) / torch.norm(neurons_T[max_neuron_index])

    return neurons_T[max_neuron_index], max_similarity, scale

def convert_to_important_weights(original_weights: torch.Tensor, mask)->torch.Tensor:
    important_selector = mask.mask.logical_not()
    return original_weights[important_selector] 

def decompose(original_weights: torch.Tensor,  mask, threshould: float) -> torch.Tensor:
    """
    Calculate the scaling matrix. Use before pruning the current layer.

    [Inputs]
    original_weights: (N[i], N[i+1]) 
    important_weights: (N[i], P[i+1])

    [Outputs]
    scaling_matrix: (P[i+1], N[i+1])
    """
    important_weights = convert_to_important_weights(original_weights, mask)
    msglogger.info("important_weights", important_weights.size())
    scaling_matrix = torch.zeros(important_weights.size()[-1], original_weights.size()[-1])
    msglogger.info("scaling_matrix", scaling_matrix.size())

    msglogger.info("original_weights", original_weights.size())

    for i, weight in enumerate(original_weights.transpose(0, -1)):
        if weight in important_weights.transpose(0, -1):
            scaling_matrix[important_weights.transpose(0, -1) == weight][i] = 1
        else:
            most_similar_neuron, similarity, scale = most_similar(weight, important_weights)
            most_similar_neuron_index_in_important_weights = important_weights == most_similar_neuron
            if similarity >= threshould:
                scaling_matrix[most_similar_neuron_index_in_important_weights][i] = scale

    return scaling_matrix


def compensation(layer_name: str, original_weights_next_layer: torch.Tensor, scaling_matrix: torch.Tensor) -> torch.Tensor:
    """
    Calculate the new compensated weights of the next layer according to the scaling matrix and the layer name.

    [Inputs]
    original_weights_2: (N[i+2], N[i+1], K, K)
    scaling_matrix: (P[n+1], N[i+1])
    [Outputs]
    new_weights_2: (N[i+2], P[i+1])
    """
    msglogger.info("original_weights_next_layer", original_weights_next_layer.size())
    msglogger.info("scaling_matrix", scaling_matrix.size())
    if layer_name.index('conv') >= 0:
        # 2-mode product
        new_weights_2 = torch.tensordot(original_weights_next_layer, scaling_matrix, dims=([1], [1]))
    else:
        # Note that when multiplying the next layer, it needs to be transposed
        new_weights_2 = torch.matmul(scaling_matrix, original_weights_next_layer)
    return new_weights_2


def reload_weights(model: nn.Module, layer: str, compensated_weight: torch.Tensor) -> nn.Module:
    """
    Reload weights from the `compensated_weight_list` to `model`.
    """
    model.state_dict()[layer].copy_(compensated_weight)



# def apply_merge_pruning_compensation(model: nn.Module, threshould: float, last_layer_name: str, last_layer: torch.Tensor,
#                                      layer_name: str, layer: torch.Tensor, zero_mask_dict: Dict[str, torch.Tensor]):
#     """
#     Give compensation of the current layer to the next layer
#     """
#     last_parameter_dict = dict(last_layer.named_parameters())

#     last_important_weights = get_important_weights(last_zero_mask)
#     scaling_matrix = decompose(last_parameter_dict['weight'], last_important_weights, threshould)

#     parameter_dict = dict(layer.named_parameters())
#     new_weights = compensation(name, parameter_dict['weight'], scaling_matrix)

#     reload_weights(model,  new_weights)
