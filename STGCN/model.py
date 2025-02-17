import math
import numpy as np
import torch
from torch import distributions, nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal


class TimeBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features_out=out_channels)
        """
        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 3, 1, 2)
        temp = self.conv1(X) * torch.sigmoid(self.conv2(X))
        out = F.relu(temp + self.conv3(X))
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out


class STGCNBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution on each node in
    isolation, followed by a graph convolution, followed by another temporal
    convolution on each node.
    """

    def __init__(self, in_channels, spatial_channels, out_channels,
                 num_nodes):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param spatial_channels: Number of output channels of the graph
        convolutional, spatial sub-block.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        """
        super(STGCNBlock, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels, out_channels=out_channels)
        self.Theta1 = nn.Parameter(torch.Tensor(out_channels, spatial_channels))
        self.temporal2 = TimeBlock(in_channels=spatial_channels, out_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        t = self.temporal1(X)
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        # t2 = F.relu(torch.einsum("ijkl,lp->ijkp", [lfs, self.Theta1]))
        t2 = F.relu(torch.matmul(lfs, self.Theta1))
        t3 = self.temporal2(t2)
        return self.batch_norm(t3)
        # return t3


class STGCN(nn.Module):
    """
    Spatio-temporal graph convolutional network as described in
    https://arxiv.org/abs/1709.04875v3 by Yu et al.
    Input should have shape (batch_size, num_nodes, num_input_time_steps,
    num_features).
    """

    def __init__(self, A, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output):
        """
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(STGCN, self).__init__()
        self.A = A
        self.block1 = STGCNBlock(in_channels=num_features, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes)
        self.block2 = STGCNBlock(in_channels=64, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes)
        self.last_temporal = TimeBlock(in_channels=64, out_channels=64)
        self.fully = nn.Linear((num_timesteps_input - 2 * 5) * 64,
                               num_timesteps_output)

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
        out1 = self.block1(X, self.A)
        out2 = self.block2(out1, self.A)
        out3 = self.last_temporal(out2)
        out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1)))
        return out4.transpose(1, 2)


class mar_regressor(nn.Module):
    def __init__(self, num_nodes, seq_length, init='diagonal', scaling=0.001):
        super(mar_regressor,self).__init__()

        self.num_nodes = num_nodes
        self.seq_length = seq_length

        self.alpha = 1/seq_length
        self.beta = 1/num_nodes

        if init == 'random':
            self.A = nn.parameter.Parameter(torch.randn(seq_length, seq_length)*scaling, requires_grad=True)
            self.B = nn.parameter.Parameter(torch.randn(num_nodes, num_nodes)*scaling, requires_grad=True)
        elif init == 'zeros':
            self.A = nn.parameter.Parameter(torch.zeros(seq_length, seq_length), requires_grad=True)
            self.B = nn.parameter.Parameter(torch.zeros(num_nodes, num_nodes), requires_grad=True)
        elif init == 'diagonal':
            self.A = nn.parameter.Parameter(torch.eye(seq_length, seq_length)*scaling, requires_grad=True)
            self.B = nn.parameter.Parameter(torch.eye(num_nodes, num_nodes)*scaling, requires_grad=True)
        else:
            self.A = nn.parameter.Parameter(torch.randn(seq_length, seq_length)*scaling, requires_grad=True)
            self.B = nn.parameter.Parameter(torch.eye(num_nodes, num_nodes)*scaling, requires_grad=True)
        
    def getRegulizer(self, ):
        return self.alpha**2*self.A.abs().sum() + self.beta**2*self.B.abs().sum()

    def forward(self, res_t_s):
        res_t = self.A@res_t_s@self.B
        return res_t


class MGD_loss_eye(nn.Module):
    """
    Multivariate normal distribution loss.
    """

    distribution_class = distributions.MultivariateNormal

    def __init__(self, num_nodes, seq_length):
        super(MGD_loss_eye, self).__init__()

        self.num_nodes = num_nodes
        self.out_dim = seq_length
        self.sigma = nn.Parameter(torch.randn(1), requires_grad=True)

    def sample(self, x, n_sample=100):
        y_shape = x.shape
        dist = self.map_x_to_distribution(x)

        eps = dist.sample((y_shape[0], n_sample)).reshape(y_shape[0], n_sample, *y_shape[1:])

        return eps

    def map_x_to_distribution(self, x: torch.Tensor) -> distributions.Normal:
        """
        x: (B, Q, N)
        """
        loc = torch.zeros(self.num_nodes*self.out_dim, device=x.device)
        scale_tril = torch.sqrt(F.softplus(self.sigma))*torch.eye(self.num_nodes*self.out_dim, device=self.sigma.device)

        distr = self.distribution_class(
            loc=loc,
            scale_tril=scale_tril,
        )

        return distr

    def forward(self, eps_t, y_t):
        """
        eps_t: scaled epsilon, (B, Q, N)
        y_t: unscaled ground truth, (B, Q, N)
        """
        mask = (y_t != 0.0).float()
        eps_t = eps_t * mask

        dist = self.map_x_to_distribution(eps_t)
        loss = -dist.log_prob(eps_t.flatten(start_dim=1))

        return loss.mean()


def softplus_inv(y):
    finfo = torch.finfo(y.dtype)
    return y.where(y > 20.0, y + (y + finfo.eps).neg().expm1().neg().log())


class MGD_loss_full_eye(nn.Module):
    """
    Multivariate low-rank normal distribution loss.
    """

    distribution_class = distributions.LowRankMultivariateNormal

    def __init__(
            self, 
            num_nodes: int, 
            seq_length: int, 
            rank_n: int = 30, 
            rank_q: int = 12, 
            scaling: float = 1.0, 
            beta: float = 1.0, 
            sigma_init: float = 1.0, 
            sigma_minimum: float = 1e-3
            ):
        super(MGD_loss_full_eye, self).__init__()

        self.num_nodes = num_nodes
        self.out_dim = seq_length
        self.rank_n = rank_n
        self.rank_q = rank_q
        self.sigma_minimum = sigma_minimum
        self.sigma_init = sigma_init

        # determine bias
        self._diag_bias: float = (
            softplus_inv(torch.tensor(self.sigma_init) ** 2).item() if self.sigma_init > 0.0 else 0.0
        )
        # determine normalizer to bring unscaled diagonal close to 1.0
        self._cov_factor_scale_n: float = np.sqrt(self.rank_n)
        self._cov_factor_scale_q: float = np.sqrt(self.rank_q)

        # self.L_n = nn.Parameter(torch.diag(torch.randn(num_nodes))*scaling, requires_grad=False)
        # self.L_q = nn.Parameter(torch.diag(torch.randn(seq_length))*scaling, requires_grad=False)

        self.L_n = nn.Parameter(torch.randn(num_nodes, self.rank_n)*scaling, requires_grad=True)
        self.L_q = nn.Parameter(torch.randn(seq_length, self.rank_q)*scaling, requires_grad=True)

        # self.L_n = nn.Parameter(torch.diag(torch.rand(num_nodes))*scaling, requires_grad=False)
        # self.L_q = nn.Parameter(torch.diag(torch.rand(seq_length))*scaling, requires_grad=False)

        self.sigma = nn.Parameter(torch.randn(1), requires_grad=True)

        # self.sigma = nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def sample(self, x, n_sample=100):
        y_shape = x.shape
        dist = self.map_x_to_distribution(x)

        eps = dist.sample((y_shape[0], n_sample)).reshape(y_shape[0], n_sample, *y_shape[1:])

        return eps

    def map_x_to_distribution(self, x: torch.Tensor) -> distributions.Normal:
        """
        x: (B, Q, N)
        """
        loc = torch.zeros(self.num_nodes*self.out_dim, device=x.device)

        #2 with scaling
        L_q = self.L_q / self._cov_factor_scale_q
        L_n = self.L_n / self._cov_factor_scale_n
    
        cov_factor = torch.kron(L_q, L_n)

        #2 with diag bias and jitter
        cov_diag = F.softplus(self.sigma + self._diag_bias) + self.sigma_minimum**2
        cov_diag = cov_diag.repeat(cov_factor.shape[0])

        distr = self.distribution_class(
            loc=loc,
            cov_factor=cov_factor,
            cov_diag=cov_diag,
        )

        return distr

    def forward(self, eps_t, y_t):
        """
        eps_t: (B, Q, N)
        y_t: (B, Q, N)
        """
        mask = (y_t != 0.0).float()
        eps_t = eps_t * mask

        dist = self.map_x_to_distribution(eps_t)
        loss = -dist.log_prob(eps_t.flatten(start_dim=1))

        return loss.mean()


class dynamic_reg(nn.Module):
    def __init__(self, args, adj_mx):
        super(dynamic_reg, self).__init__()

        self.scaler = args.scaler
        self.out_dim = args.out_dim
        self.dr = args.dr
        self.rho = args.rho
        self.loss = args.loss_type
        self.rank_n = args.rank_n
        self.rank_q = args.rank_q

        self.mean_model = STGCN(A=adj_mx, num_nodes=args.num_nodes, num_features=args.in_dim, num_timesteps_input=args.seq_len, num_timesteps_output=args.out_dim)

        if self.dr:
            self.res_model = mar_regressor(num_nodes=args.num_nodes, seq_length=args.seq_len, init=args.dr_init)
        
        if self.rho > 0:
            if self.loss == "full":
                self.cov_model = MGD_loss_full_eye(num_nodes=args.num_nodes, seq_length=args.seq_len, rank_n=args.rank_n, rank_q=args.rank_q, scaling=args.cov_scaling, beta=args.beta)
            else:
                self.cov_model = MGD_loss_eye(num_nodes=args.num_nodes, seq_length=args.seq_len)

    def forward(self, x, y):
        y_t = y[:,:self.out_dim,:,0]

        if self.dr:
            x_t = torch.cat([x[...,:1], x[...,2:]], dim=-1)
            x_t_s, y_t_s = x[...,1:], y[:,:self.out_dim,:,1]
            y_t_hat = self.mean_model(x_t.transpose(1, 2))
            with torch.no_grad():
                y_t_s_hat = self.mean_model(x_t_s.transpose(1, 2))
                eta_t_s = self.scaler.transform(y_t_s) - y_t_s_hat
                mask = (y_t_s != 0.0).float()
                eta_t_s = eta_t_s * mask

            eta_t = self.res_model(eta_t_s)
            y_t_hat += eta_t
        else:
            y_t_hat = self.mean_model(x.transpose(1, 2))

        if self.rho > 0:
            eps_t = self.scaler.transform(y_t) - y_t_hat
            nll = self.cov_model(eps_t, y_t)
        else:
            nll = 0.0

        return y_t_hat, nll
