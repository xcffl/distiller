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


import distiller
from .ranked_structures_pruner import *


class SensitivityPruner(object):
    """Use algorithm from "Learning both Weights and Connections for Efficient
    Neural Networks" - https://arxiv.org/pdf/1506.02626v3.pdf

    I.e.: "The pruning threshold is chosen as a quality parameter multiplied
    by the standard deviation of a layers weights."
    In this code, the "quality parameter" is referred to as "sensitivity" and
    is based on the values learned from performing sensitivity analysis.

    Note that this implementation deviates slightly from the algorithm Song Han
    describes in his PhD dissertation, in that the threshold value is set only
    once.  In his PhD dissertation, Song Han describes a growing threshold, at
    each iteration.  This requires n+1 hyper-parameters (n being the number of
    pruning iterations we use): the threshold and the threshold increase (delta)
    at each pruning iteration.
    The implementation that follows, takes advantage of the fact that as pruning
    progresses, more weights are pulled toward zero, and therefore the threshold
    "traps" more weights.  Thus, we can use less hyper-parameters and achieve the
    same results.
    """

    def __init__(self, name, sensitivities, **kwargs):
        self.name = name
        self.sensitivities = sensitivities

    def set_param_mask(self, param, param_name, zeros_mask_dict, meta):
        if param_name not in self.sensitivities:
            if '*' not in self.sensitivities:
                return
            else:
                sensitivity = self.sensitivities['*']
        else:
            sensitivity = self.sensitivities[param_name]

        zeros_mask_dict[param_name].mask = distiller.create_mask_sensitivity_criterion(param, sensitivity)

class L1RankedStructureParameterPruner_SP(SensitivityPruner):
    def __init__(self, name, sensitivities, group_type, weights, group_dependency=None, kwargs=None):
        super().__init__(name, sensitivities)
        self.pruner = L1RankedStructureParameterPruner(name, group_type, desired_sparsity=0, weights=weights,
                                                       group_dependency=group_dependency, kwargs=kwargs)

