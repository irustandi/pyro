from __future__ import absolute_import, division, print_function

import math

import torch
from torch.distributions import constraints
from torch.nn import Parameter

from .kernel import Kernel


class DotProduct(Kernel):
    """
    Base class for kernels which are functions of :math:`x \cdot z`.
    """

    def __init__(self, input_dim, variance=None, active_dims=None, name=None):
        super(DotProduct, self).__init__(input_dim, active_dims, name)

        if variance is None:
            variance = torch.tensor(1.)
        self.variance = Parameter(variance)
        self.set_constraint("variance", constraints.positive)

    def _dot_product(self, X, Z=None, diag=False):
        """
        Returns :math:`X \cdot Z`.
        """
        if Z is None:
            Z = X
        X = self._slice_input(X)
        if diag:
            return (X ** 2).sum(-1)

        Z = self._slice_input(Z)
        if X.shape[1] != Z.shape[1]:
            raise ValueError("Inputs must have the same number of features.")

        return X.matmul(Z.t())


class Linear(DotProduct):
    """
    Implementation of Linear kernel:

        :math:`k(x, z) = \sigma^2 x \cdot z.`

    Doing Gaussian Process regression with linear kernel is equivalent to doing a
    linear regression.

    .. note:: Here we implement the homogeneous version. To use the inhomogeneous
        version, consider using :class:`Polynomial` kernel with ``degree=1`` or making
        a :class:`.Sum` with a :class:`.Bias` kernel.
    """

    def __init__(self, input_dim, variance=None, active_dims=None, name="Linear"):
        super(Linear, self).__init__(input_dim, variance, active_dims, name)

    def forward(self, X, Z=None, diag=False):
        variance = self.get_param("variance")
        return variance * self._dot_product(X, Z, diag)


class Polynomial(DotProduct):
    r"""
    Implementation of Polynomial kernel:

        :math:`k(x, z) = \sigma^2(\text{bias} + x \cdot z)^d.`

    :param torch.Tensor bias: Bias parameter of this kernel. Should be positive.
    :param int degree: Degree :math:`d` of the polynomial.
    """

    def __init__(self, input_dim, variance=None, bias=None, degree=1, active_dims=None,
                 name="Polynomial"):
        super(Polynomial, self).__init__(input_dim, variance, active_dims, name)

        if bias is None:
            bias = torch.tensor(1.)
        self.bias = Parameter(bias)
        self.set_constraint("bias", constraints.positive)

        if not isinstance(degree, int) or degree < 1:
            raise ValueError("Degree for Polynomial kernel should be a positive "
                             "integer.")
        self.degree = degree

    def forward(self, X, Z=None, diag=False):
        variance = self.get_param("variance")
        bias = self.get_param("bias")
        return variance * ((bias + self._dot_product(X, Z, diag)) ** self.degree)


class ArcCosine(DotProduct):
    r"""
    Implementation of ArcCosine kernel:

        :math:`k(x, z) = \sigma^2(\text{bias} + x \cdot z)^d.`

    :param torch.Tensor bias: Bias parameter of this kernel. Should be positive.
    :param int degree: Degree :math:`d` of the polynomial.
    """

    def __init__(self, input_dim, degrees, variance=None, active_dims=None,
                 name="ArcCosine"):
        super(ArcCosine, self).__init__(input_dim, variance, active_dims, name)

        self.degrees = degrees

    def _compute_J(self, degree, theta):
        if 0 == degree:
            return math.pi - theta
        if 1 == degree:
            return torch.sin(theta) + (math.pi - theta) * torch.cos(theta)
        if 2 == degree:
            return 3 * torch.sin(theta) * torch.cos(theta) + (math.pi - theta) * (1 + 2 * torch.pow(torch.cos(theta), 2))
        if 3 == degree:
            return 4 * torch.pow(torch.sin(theta), 3) + 15 * torch.sin(theta) * torch.pow(torch.cos(theta), 2) + (math.pi - theta) * (
                        9 * torch.pow(torch.sin(theta), 2) * torch.cos(theta) + 15 * torch.pow(torch.cos(theta), 3));
        raise ValueError

    def forward(self, X, Z=None, diag=False):
        if Z is None:
            Z = X

        num_levels = len(self.degrees)

        k_xx_l = self._dot_product(X, X, False)
        k_zz_l = self._dot_product(Z, Z, False)
        k_xz_l = self._dot_product(X, Z, False)

        for l in range(num_levels):
            denom_term = torch.matmul(torch.unsqueeze(torch.diag(k_xx_l), dim=1), torch.unsqueeze(torch.diag(k_zz_l), dim=0))
            theta_l = torch.acos(torch.clamp(k_xz_l / torch.sqrt(denom_term), -1.0, 1.0))
            k_xz_l = torch.pow(denom_term, self.degrees[l] / 2) / math.pi * self._compute_J(self.degrees[l], theta_l)

            if num_levels > l + 1:
                k_xx_l = torch.pow(k_xx_l, self.degrees[l]) / math.pi * self._compute_J(self.degrees[l], torch.zeros_like(k_xx_l))
                k_zz_l = torch.pow(k_zz_l, self.degrees[l]) / math.pi * self._compute_J(self.degrees[l], torch.zeros_like(k_zz_l))

        return k_xz_l