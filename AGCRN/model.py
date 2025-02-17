import math
import numpy as np
import torch
from torch import distributions, nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal


class AVWGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.Tensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.Tensor(embed_dim, dim_out))
    def forward(self, x, node_embeddings):
        #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        #output shape [B, N, C]
        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num).to(supports.device), supports]
        #default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)                       #N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, x)      #B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     #b, N, dim_out
        return x_gconv


class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AVWGCN(dim_in+self.hidden_dim, 2*dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(dim_in+self.hidden_dim, dim_out, cheb_k, embed_dim)

    def forward(self, x, state, node_embeddings):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)


class AVWDCRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)


class AGCRN(nn.Module):
    def __init__(self, args):
        super(AGCRN, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = 1
        self.horizon = args.horizon
        self.num_layers = args.num_layers

        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)

        self.encoder = AVWDCRNN(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
                                args.embed_dim, args.num_layers)

        #predictor
        self.end_conv = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

    def forward(self, source, targets=None, teacher_forcing_ratio=0.5):
        #source: B, P, N, D
        #target: B, Q, N, D
        #supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)

        init_state = self.encoder.init_hidden(source.shape[0])
        output, _ = self.encoder(source, init_state, self.node_embeddings)      #B, T, N, hidden
        output = output[:, -1:, :, :]                                   #B, 1, N, hidden

        #CNN based predictor
        output = self.end_conv((output))                         #B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2)                             #B, T, N, C

        return output.squeeze(-1)


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

        # self.L_n = nn.Parameter(torch.diag(torch.randn(num_nodes))*scaling, requires_grad=True)
        # self.L_q = nn.Parameter(torch.diag(torch.randn(seq_length))*scaling, requires_grad=True)

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
    def __init__(self, args, scaler):
        super(dynamic_reg, self).__init__()

        self.scaler = scaler
        self.out_dim = args.horizon
        self.dr = args.dr
        self.rho = args.rho
        self.loss = args.loss_type
        self.rank_n = args.rank_n
        self.rank_q = args.rank_q

        self.mean_model = AGCRN(args)

        if self.dr:
            self.res_model = mar_regressor(num_nodes=args.num_nodes, seq_length=args.horizon, init=args.dr_init)

        if self.rho > 0:
            if self.loss == "full":
                self.cov_model = MGD_loss_full_eye(num_nodes=args.num_nodes, seq_length=self.out_dim, rank_n=args.rank_n, rank_q=args.rank_q, scaling=args.cov_scaling, beta=args.beta)
            else:
                self.cov_model = MGD_loss_eye(num_nodes=args.num_nodes, seq_length=self.out_dim)

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