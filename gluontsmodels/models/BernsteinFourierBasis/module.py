import torch
import torch.nn as nn
import torch.nn.functional as F
from math import comb

from typing import Optional

from gluonts.torch.scaler import StdScaler
from gluonts.torch.distributions import StudentTOutput
from gluonts.torch.util import weighted_average
from gluonts.model import Input, InputSpec

# Define a generic MLP class
class GenericMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, activation_fn=nn.GELU):
        super(GenericMLP, self).__init__()
        layers = []
        for i in range(num_layers - 1):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(activation_fn())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

# Define the Fourier basis function
def fourier_basis(x, v, max_period=1,device = None ):
    """
    Generates the Fourier basis functions for degrees 0 through v evaluated at points x with a specified maximum period.

    Parameters:
        x (torch.Tensor): The values to evaluate the Fourier basis functions at.
        v (int): The maximum index of the Fourier basis functions to generate.
        max_period (float): The maximum period of the Fourier basis functions. Defaults to 1.

    Returns:
        torch.Tensor: A tensor where each row corresponds to a different degree of the Fourier basis function evaluated at x.
    """
    if v < 0:
        raise ValueError("v must be non-negative.")

    k = torch.arange(v + 1,device=device).unsqueeze(0)
    fourier_values = torch.concat([
        torch.sin(2 * torch.pi * k * x / max_period),
        torch.cos(2 * torch.pi * k * x / max_period)
    ], dim=-1)

    return fourier_values

# Define the Bernstein basis function
def bernstein_polynomial(x, n, v, device=None):
    """
    Generates the Bernstein polynomials for degrees 0 through v evaluated at points x.

    Parameters:
        x (torch.Tensor): The values to evaluate the polynomial at.
        n (int): The degree of the polynomial.
        v (int): The maximum index of the Bernstein polynomial basis to generate.

    Returns:
        torch.Tensor: A tensor where each row corresponds to a different degree of the Bernstein polynomial evaluated at x.
    """
    if not (0 <= v <= n):
        raise ValueError("v must be between 0 and n inclusive.")

    binom_coeffs = torch.tensor([comb(n, i) for i in range(v + 1)], dtype=torch.float32, device=device).unsqueeze(0)
    x_powers = x ** torch.arange(v + 1,device=device).unsqueeze(0)
    one_minus_x_powers = (1 - x) ** (n - torch.arange(v + 1,device=device).unsqueeze(0))
    bernstein_values = binom_coeffs * x_powers * one_minus_x_powers

    return bernstein_values

# Define the neural network model
class BasisModel(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 fourier_v,
                 max_period,
                 bernstein_v, 
                 prediction_length,
                 context_length,
                 num_layers=2, 
                 activation_fn=nn.GELU,
                 distr_output=StudentTOutput(),
                 device=None
                ):
        super(BasisModel, self).__init__()
        self.input_dim = input_dim
        self.max_period = max_period
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.distr_output = distr_output
        self.device=device
                    
        self.scaler = StdScaler(dim=1, keepdim=True)
        
        self.input_proj = nn.Linear(1, input_dim)
        
        self.fourier_mlp = GenericMLP(context_length, hidden_dim, 2 * (fourier_v + 1), num_layers, activation_fn)
        self.bernstein_mlp = GenericMLP(context_length, hidden_dim, bernstein_v + 1 + 1, num_layers, activation_fn)  # Extra output for sign
        self.register_buffer('t', torch.linspace(0, 1, prediction_length,device=device).unsqueeze(-1).unsqueeze(0))

        self.fourier_v = fourier_v
        self.bernstein_n = bernstein_v
        self.bernstein_v = bernstein_v

        self.args_proj = distr_output.get_args_proj(input_dim)

        #self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def describe_inputs(self, batch_size=1) -> InputSpec:
        return InputSpec(
            {
                "past_target": Input(
                    shape=(batch_size, self.context_length), dtype=torch.float
                ),
                "past_observed_values": Input(
                    shape=(batch_size, self.context_length), dtype=torch.float
                ),
                "past_feat_dynamic_real": Input(
                    shape=(batch_size, self.context_length, 1), dtype=torch.float
                ),
            },
            torch.zeros,
        )

    def forward(self,
                past_target: torch.tensor,
                past_observed_values: torch.tensor,
                past_feat_dynamic_real: torch.tensor,
               ):
        B, T = past_target.shape

        x, loc, scale = self.scaler(past_target, past_observed_values)

        x = self.input_proj(x.unsqueeze(-1)).transpose(-2,-1)

        # Reshape input to [B * C, T]
        x_reshaped = x.reshape(-1, T)

        # Generate Fourier coefficients
        fourier_coeffs = self.fourier_mlp(x_reshaped)

        # Generate Bernstein coefficients and sign
        bernstein_output = self.bernstein_mlp(x_reshaped)
        bernstein_coeffs = F.relu(bernstein_output[:, :-1])
        sign = torch.nn.functional.softsign(bernstein_output[:, -1]).unsqueeze(-1)
        bernstein_coeffs = torch.cumsum(bernstein_coeffs, dim=1) * sign

        # Evaluate Fourier basis functions
        fourier_basis_values = fourier_basis(
            self.t * self.prediction_length + 1 + past_feat_dynamic_real[:,-1:,:].repeat(1,1,self.input_dim).transpose(-2,-1).reshape(B*self.input_dim,1,1), 
            self.fourier_v, 
            self.max_period,
            device=self.device
        ).expand(B * self.input_dim, -1, -1)

        # Evaluate Bernstein basis functions
        bernstein_basis_values = bernstein_polynomial(self.t, self.bernstein_n, self.bernstein_v,device=self.device).expand(B * self.input_dim, -1,-1)

        # Combine the coefficients with the basis functions
        fourier_result = torch.sum(fourier_coeffs.unsqueeze(1) * fourier_basis_values, dim=2)
        bernstein_result = torch.sum(bernstein_coeffs.unsqueeze(1) * bernstein_basis_values, dim=2)

        result = (bernstein_result + fourier_result).view(B,-1,self.prediction_length).transpose(-2,-1)
        distr_args = self.args_proj(result)

        return distr_args, loc, scale

    def loss(
        self,
        past_target: torch.tensor,
        past_observed_values: torch.tensor,
        past_feat_dynamic_real: torch.tensor,
        future_target: torch.tensor,
        future_observed_values: torch.tensor,
    ):
        distr_args, loc, scale = self(
            past_target=past_target, past_observed_values=past_observed_values,past_feat_dynamic_real=past_feat_dynamic_real
        )
        loss = self.distr_output.loss(
            target=future_target, distr_args=distr_args, loc=loc, scale=scale
        )
        return weighted_average(loss, weights=future_observed_values, dim=-1)
        