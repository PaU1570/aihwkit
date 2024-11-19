# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""aihwkit example 33: simple network with one layer using Mixed Precision learning rule and asymmetric update.

Mixed precision is based on the paper Nandakumar et al (2020) (see
https://www.frontiersin.org/articles/10.3389/fnins.2020.00406/full).
"""
# pylint: disable=invalid-name

# Imports from PyTorch.
from torch import Tensor
from torch.nn.functional import mse_loss

# Imports from aihwkit.
from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import (
    DigitalRankUpdateRPUConfig,
    MixedPrecisionCompound,
    PiecewiseStepDevice
)
from aihwkit.simulator.rpu_base import cuda
from aihwkit.simulator.parameters.enums import AsymmetricPulseType

# Prepare the datasets (input and expected output).
x = Tensor([[0.1, 0.2, 0.4, 0.3], [0.2, 0.1, 0.1, 0.3]])
y = Tensor([[1.0, 0.5], [0.7, 0.3]])

# Select the device model to use in the training. While one can use a
# presets as well, we here build up the RPU config from more basic
# devices. We use the relevant RPU config for using a digital rank
# update and transfer to analog device (like in mixed precision) and
# set it to a mixed precision compound which in turn uses a
# PiecewiseStep analog device:
rpu_config = DigitalRankUpdateRPUConfig(device=MixedPrecisionCompound(
                                                asymmetric_pulsing_dir=AsymmetricPulseType.UP,
                                                asymmetric_pulsing_up=2,
                                                asymmetric_pulsing_down=1,
                                                device=PiecewiseStepDevice(
                                                construction_seed=2024,
                                                dw_min=0.10365738941762762,
                                                dw_min_dtod=0.0,
                                                dw_min_std=0.0,
                                                reset_std=0.0,
                                                up_down=-0.030147609463787584,
                                                w_max=1.1522620505409014,
                                                w_min=-1.1035877799602885,
                                                piecewise_up=[
                                                    3.8804523058203038,
                                                    2.1328001458243673,
                                                    0.7776331245664537,
                                                    0.24540853347730893,
                                                    0.990787512183888,
                                                    1.5073448914301681,
                                                    1.83857563473977,
                                                    2.0229040769299007,
                                                    2.093959411758182,
                                                    2.080851179569486,
                                                    2.0084447549427673,
                                                    1.897636834337899,
                                                    1.7656309237425045,
                                                    1.6262128263187947,
                                                    1.4900261300503985,
                                                    1.3648476953892,
                                                    1.2558631429021712,
                                                    1.165942340918205,
                                                    1.0959148931749518,
                                                    1.0448456264656512,
                                                    1.0103100782859673,
                                                    0.9886699844808228,
                                                    0.9753487668912323,
                                                    0.9651070210011373,
                                                    0.9523180035842397,
                                                    0.9312431203508356,
                                                    0.8963074135946504,
                                                    0.8423750498396714,
                                                    0.7650248074869839,
                                                    0.6608255644616033,
                                                    0.5276117858593091,
                                                    0.36475901159348245
                                                ],
                                                piecewise_down=[
                                                    0.3981662621946436,
                                                    0.12774412983584799,
                                                    0.09783842322796237,
                                                    0.2841996751367432,
                                                    0.436585630327541,
                                                    0.5598602136024065,
                                                    0.658495464196305,
                                                    0.7365617298450304,
                                                    0.7977178608531161,
                                                    0.8452014041617488,
                                                    0.8818187974166799,
                                                    0.909935563036138,
                                                    0.9314665022787415,
                                                    0.9478658893114107,
                                                    0.9601176652772808,
                                                    0.9687256323636132,
                                                    0.9737036478697085,
                                                    0.974565818274819,
                                                    0.9703166933060607,
                                                    0.9594414600063258,
                                                    0.939896136802195,
                                                    0.9090977675718499,
                                                    0.8639146157129854,
                                                    0.8006563582107222,
                                                    0.7150642797055187,
                                                    0.6023014665610836,
                                                    0.45694300093228885,
                                                    0.2729661548330802,
                                                    0.04374058420439246,
                                                    0.2379814770179418,
                                                    0.5800750228352759,
                                                    0.9910520811182386
                                                ]
                                            ),
                                            ))

# print the config (default values are omitted)
print("\nPretty-print of non-default settings:\n")
print(rpu_config)

model = AnalogLinear(4, 2, bias=True, rpu_config=rpu_config)

# print module structure
print("\nModule structure:\n")
print(model)

# a more detailed printout of the instantiated
print("\nC++ RPUCudaTile information:\n")
print(next(model.analog_tiles()).tile)

# Move the model and tensors to cuda if it is available.
if cuda.is_compiled():
    x = x.cuda()
    y = y.cuda()
    model.cuda()

# Define an analog-aware optimizer, preparing it for using the layers.
opt = AnalogSGD(model.parameters(), lr=0.1)
opt.regroup_param_groups(model)

for epoch in range(500):
    # Delete old gradients
    opt.zero_grad()
    # Add the training Tensor to the model (input).
    pred = model(x)
    # Add the expected output Tensor.
    loss = mse_loss(pred, y)
    # Run training (backward propagation).
    loss.backward()

    opt.step()
    print("{}: Loss error: {:.16f}".format(epoch, loss), end="\r" if epoch % 50 else "\n")

print(model.get_weights())