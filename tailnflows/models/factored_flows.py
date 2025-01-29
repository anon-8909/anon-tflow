import torch
from nflows.flows import Flow


class FactoredFlow(torch.nn.Module):
    def __init__(self, marginal_flow: Flow, conditional_flow: Flow, marginal_dim: int):
        super().__init__()
        self.marginal_dim = marginal_dim
        self.marginal_flow = marginal_flow
        self.conditional_flow = conditional_flow

    def sample_and_log_prob(self, num_samples):
        z_y, log_p_z_y = self.marginal_flow._distribution.sample_and_log_prob(
            num_samples
        )
        y, lad_y = self.marginal_flow._transform.inverse(z_y)

        z_x, log_p_z_x = self.conditional_flow._distribution.sample_and_log_prob(
            num_samples
        )
        x_g_y, lad_x = self.conditional_flow._transform.inverse(z_x, context=z_y)

        log_p_y = log_p_z_y - lad_y  # p(y)
        log_p_x_g_y = log_p_z_x - lad_x  # p(x | y)
        log_p_joint = log_p_y + log_p_x_g_y  # p(y, x)

        return torch.hstack([y, x_g_y]), log_p_joint

    def marginal_log_prob(self, y):
        return self.marginal_flow.log_prob(y)

    def log_prob(self, y_x):
        y, x_g_y = torch.tensor_split(y_x, (self.marginal_dim,), 1)
        z_y, lad_y = self.marginal_flow._transform(y)
        log_p_joint = self.marginal_flow._distribution.log_prob(z_y)
        log_p_joint += lad_y
        log_p_joint += self.conditional_flow.log_prob(x_g_y, context=z_y)
        return log_p_joint
