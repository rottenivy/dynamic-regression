# -*- coding:utf-8 -*-
import numpy as np
import math
from scipy.sparse.linalg import eigs
import torch
from torch import distributions, nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from scipy import io


def scaled_Laplacian(W):
    '''
    compute \tilde{L}
    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices
    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)
    '''
    assert W.shape[0] == W.shape[1]
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    lambda_max = eigs(L, k=1, which='LR')[0].real
    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}
    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)
    K: the maximum order of chebyshev polynomials
    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}
    '''
    N = L_tilde.shape[0]

    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials


class Spatial_Attention_layer(nn.Module):
    '''
    compute spatial attention scores
    '''
    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Spatial_Attention_layer, self).__init__()
        self.W1 = nn.parameter.Parameter(torch.FloatTensor(num_of_timesteps).to(DEVICE))
        self.W2 = nn.parameter.Parameter(torch.FloatTensor(in_channels, num_of_timesteps).to(DEVICE))
        self.W3 = nn.parameter.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
        self.bs = nn.parameter.Parameter(torch.FloatTensor(1, num_of_vertices, num_of_vertices).to(DEVICE))
        self.Vs = nn.parameter.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices).to(DEVICE))


    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B,N,N)
        '''

        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)  # (b,N,F,T)(T)->(b,N,F)(F,T)->(b,N,T)

        rhs = torch.matmul(self.W3, x).transpose(-1, -2)  # (F)(b,N,F,T)->(b,N,T)->(b,T,N)

        product = torch.matmul(lhs, rhs)  # (b,N,T)(b,T,N) -> (B, N, N)

        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))  # (N,N)(B, N, N)->(B,N,N)

        S_normalized = F.softmax(S, dim=1)

        return S_normalized


class cheb_conv_withSAt(nn.Module):
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv_withSAt, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])

    def forward(self, x, spatial_attention):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (b, N, F_out)

            for k in range(self.K):

                T_k = self.cheb_polynomials[k]  # (N,N)

                T_k_with_at = T_k.mul(spatial_attention)   # (N,N)*(N,N) = (N,N) 多行和为1, 按着列进行归一化

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = T_k_with_at.permute(0, 2, 1).matmul(graph_signal)  # (N, N)(b, N, F_in) = (b, N, F_in) 因为是左乘，所以多行和为1变为多列和为1，即一行之和为1，进行左乘

                output = output + rhs.matmul(theta_k)  # (b, N, F_in)(F_in, F_out) = (b, N, F_out)

            outputs.append(output.unsqueeze(-1))  # (b, N, F_out, 1)

        return F.relu(torch.cat(outputs, dim=-1))  # (b, N, F_out, T)


class Temporal_Attention_layer(nn.Module):
    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Temporal_Attention_layer, self).__init__()
        self.U1 = nn.Parameter(torch.FloatTensor(num_of_vertices).to(DEVICE))
        self.U2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_vertices).to(DEVICE))
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
        self.be = nn.Parameter(torch.FloatTensor(1, num_of_timesteps, num_of_timesteps).to(DEVICE))
        self.Ve = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps).to(DEVICE))

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B, T, T)
        '''
        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)
        # x:(B, N, F_in, T) -> (B, T, F_in, N)
        # (B, T, F_in, N)(N) -> (B,T,F_in)
        # (B,T,F_in)(F_in,N)->(B,T,N)

        rhs = torch.matmul(self.U3, x)  # (F)(B,N,F,T)->(B, N, T)

        product = torch.matmul(lhs, rhs)  # (B,T,N)(B,N,T)->(B,T,T)

        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))  # (B, T, T)

        E_normalized = F.softmax(E, dim=1)

        return E_normalized


class cheb_conv(nn.Module):
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])

    def forward(self, x):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (b, N, F_out)

            for k in range(self.K):

                T_k = self.cheb_polynomials[k]  # (N,N)

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = graph_signal.permute(0, 2, 1).matmul(T_k).permute(0, 2, 1)

                output = output + rhs.matmul(theta_k)

            outputs.append(output.unsqueeze(-1))

        return F.relu(torch.cat(outputs, dim=-1))


class ASTGCN_block(nn.Module):

    def __init__(self, DEVICE, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials, num_of_vertices, num_of_timesteps):
        super(ASTGCN_block, self).__init__()
        self.TAt = Temporal_Attention_layer(DEVICE, in_channels, num_of_vertices, num_of_timesteps)
        self.SAt = Spatial_Attention_layer(DEVICE, in_channels, num_of_vertices, num_of_timesteps)
        self.cheb_conv_SAt = cheb_conv_withSAt(K, cheb_polynomials, in_channels, nb_chev_filter)
        self.time_conv = nn.Conv2d(nb_chev_filter, nb_time_filter, kernel_size=(1, 3), stride=(1, time_strides), padding=(0, 1))
        self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))
        self.ln = nn.LayerNorm(nb_time_filter)  #需要将channel放到最后一个维度上

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, nb_time_filter, T)
        '''
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        # TAt
        temporal_At = self.TAt(x)  # (b, T, T)

        x_TAt = torch.matmul(x.reshape(batch_size, -1, num_of_timesteps), temporal_At).reshape(batch_size, num_of_vertices, num_of_features, num_of_timesteps)

        # SAt
        spatial_At = self.SAt(x_TAt)

        # cheb gcn
        spatial_gcn = self.cheb_conv_SAt(x, spatial_At)  # (b,N,F,T)
        # spatial_gcn = self.cheb_conv(x)

        # convolution along the time axis
        time_conv_output = self.time_conv(spatial_gcn.permute(0, 2, 1, 3))  # (b,N,F,T)->(b,F,N,T) 用(1,3)的卷积核去做->(b,F,N,T)

        # residual shortcut
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3))  # (b,N,F,T)->(b,F,N,T) 用(1,1)的卷积核去做->(b,F,N,T)

        x_residual = self.ln(F.relu(x_residual + time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)
        # (b,F,N,T)->(b,T,N,F) -ln-> (b,T,N,F)->(b,N,F,T)

        return x_residual


class ASTGCN_submodule(nn.Module):

    def __init__(self, DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials, num_for_predict, len_input, num_of_vertices):
        '''
        :param nb_block:
        :param in_channels:
        :param K:
        :param nb_chev_filter:
        :param nb_time_filter:
        :param time_strides:
        :param cheb_polynomials:
        :param nb_predict_step:
        '''
        super(ASTGCN_submodule, self).__init__()
        
        self.BlockList = nn.ModuleList([ASTGCN_block(DEVICE, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials, num_of_vertices, len_input)])
        self.BlockList.extend([ASTGCN_block(DEVICE, nb_time_filter, K, nb_chev_filter, nb_time_filter, 1, cheb_polynomials, num_of_vertices, len_input//time_strides) for _ in range(nb_block-1)])
        self.final_conv = nn.Conv2d(int(len_input/time_strides), num_for_predict, kernel_size=(1, nb_time_filter))

    def forward(self, x):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        x = x.permute(0, 2, 3, 1)

        for block in self.BlockList:
            x = block(x)

        output = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)
        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)

        return output.transpose(1, 2)


def ASTGCN(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, adj_mx, num_for_predict, len_input, num_of_vertices):
    '''
    :param DEVICE:
    :param nb_block:
    :param in_channels:
    :param K:
    :param nb_chev_filter:
    :param nb_time_filter:
    :param time_strides:
    :param cheb_polynomials:
    :param nb_predict_step:
    :param len_input
    :return:
    '''
    L_tilde = scaled_Laplacian(adj_mx)
    cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(DEVICE) for i in cheb_polynomial(L_tilde, K)]
    model = ASTGCN_submodule(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials, num_for_predict, len_input, num_of_vertices)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    return model


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
        # res_t = F.tanh(res_t)
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

        self.L_n = nn.Parameter(torch.randn(num_nodes, self.rank_n)*scaling, requires_grad=True)
        self.L_q = nn.Parameter(torch.randn(seq_length, self.rank_q)*scaling, requires_grad=True)

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
    def __init__(self, args, in_channels, adj_mx, seq_len, num_nodes, scaler):
        super(dynamic_reg, self).__init__()

        self.scaler = scaler
        self.out_dim = args.out_dim
        self.dr = args.dr
        self.rho = args.rho
        self.loss = args.loss_type
        self.rank_n = args.rank_n
        self.rank_q = args.rank_q

        self.mean_model = ASTGCN(DEVICE=args.device, nb_block=args.nb_block, in_channels=in_channels, K=args.K, nb_chev_filter=args.nb_chev_filter, nb_time_filter=args.nb_time_filter, time_strides=args.time_strides, adj_mx=adj_mx, num_for_predict=args.out_dim, len_input=seq_len, num_of_vertices=num_nodes)

        if self.dr:
            self.res_model = mar_regressor(num_nodes=num_nodes, seq_length=args.out_dim, init=args.dr_init)
        
        if self.rho > 0:
            if self.loss == "full":
                self.cov_model = MGD_loss_full_eye(num_nodes=num_nodes, seq_length=args.out_dim, rank_n=args.rank_n, rank_q=args.rank_q, scaling=args.cov_scaling, beta=args.beta)
            else:
                self.cov_model = MGD_loss_eye(num_nodes=num_nodes, seq_length=args.out_dim)

    def forward(self, x, y):
        y_t = y[:,:self.out_dim,:,0]

        if self.dr:
            x_t = torch.cat([x[...,:1], x[...,2:]], dim=-1)
            x_t_s, y_t_s = x[...,1:], y[:,:self.out_dim,:,1]
            y_t_hat = self.mean_model(x_t)
            with torch.no_grad():
                y_t_s_hat = self.mean_model(x_t_s)
                eta_t_s = self.scaler.transform(y_t_s) - y_t_s_hat
                mask = (y_t_s != 0.0).float()
                eta_t_s = eta_t_s * mask

            eta_t = self.res_model(eta_t_s)
            y_t_hat += eta_t
        else:
            y_t_hat = self.mean_model(x)

        if self.rho > 0:
            eps_t = self.scaler.transform(y_t) - y_t_hat
            nll = self.cov_model(eps_t, y_t)
        else:
            nll = 0.0

        return y_t_hat, nll