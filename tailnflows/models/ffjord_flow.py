import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import Transform

from ffjord.layers.cnf import CNF
from ffjord.layers.odefunc import ODEnet, ODEfunc


class FfjordLayer(Transform):
    """
    A cts time layer in the nflows framework
    """

    def __init__(
        self,
        dim,
        hidden_dims,
        T=1.0,
        layer_type="concat",
        nonlinearity="softplus",
        divergence_fn="approximate",
        residual=False,
        rademacher=False,
        regularization_fns=None,
    ):
        super().__init__()
        diffeq = ODEnet(
            hidden_dims=hidden_dims,
            input_shape=(dim,),
            strides=None,
            conv=False,
            layer_type=layer_type,
            nonlinearity=nonlinearity,
        )
        odefunc = ODEfunc(
            diffeq=diffeq,
            divergence_fn=divergence_fn,
            residual=residual,
            rademacher=rademacher,
        )
        self.ffjord_cnf = CNF(
            odefunc=odefunc, T=T, regularization_fns=regularization_fns
        )

    def forward(self, inputs, context=None):
        startlad = torch.zeros(inputs.shape[0], 1).to(inputs)
        z, lad = self.ffjord_cnf.forward(inputs, logpz=startlad)
        return z, -lad.reshape(-1)

    def inverse(self, inputs, context=None):
        startlad = torch.zeros(inputs.shape[0], 1).to(inputs)
        x, lad = self.ffjord_cnf.forward(inputs, logpz=startlad, reverse=True)
        return x, lad.reshape(-1)


## FLOW DEFS
# def ffjord_flow(dim):
#     transforms = [
#         TailAffineMarginalTransform(
#             features=dim,
#             pos_tail_init=0.2 * torch.ones([dim]),
#             neg_tail_init=0.2 * torch.ones([dim]),
#         ),
#         FfjordLayer(dim, (64, 64, 64)),
#     ]
#     transforms = [InverseTransform(transform) for transform in transforms]
#     return Flow(
#         transform=CompositeTransform(transforms),
#         distribution=StandardNormal([dim]),
#     )


# def raw_ffjord_flow(dim):
#     transforms = [FfjordLayer(dim, (64, 64, 64))]
#     transforms = [InverseTransform(transform) for transform in transforms]
#     return Flow(
#         transform=CompositeTransform(transforms),
#         distribution=StandardNormal([dim]),
#     )
