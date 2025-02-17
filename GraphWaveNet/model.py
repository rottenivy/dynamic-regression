import math
import numpy as np
import torch
from torch import distributions, nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class gwnet(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2,out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2):
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len +=1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2  # 1->2
                receptive_field += additional_scope  # 1->2->4->5->7->....
                additional_scope *= 2  # 1->2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field



    def forward(self, x):
        x = x.transpose(1, 3)
        in_len = x.shape[3]
        if in_len < self.receptive_field:
            x = nn.functional.pad(x, (self.receptive_field-in_len,0,0,0))
            
        x = self.start_conv(x)  # [64, 1, 325, 13] -> [64, 32, 325, 13]
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):
            residual = x  # [64, 32, 325, 13]
            # dilated convolution
            filter = self.filter_convs[i](residual)  # [64, 32, 325, 13] -> [64, 32, 325, 12]
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)  # [64, 32, 325, 13] -> [64, 32, 325, 12]
            gate = torch.sigmoid(gate)
            x = filter * gate
            # parametrized skip connection
            s = x
            s = self.skip_convs[i](s)  # [64, 32, 325, 13] -> [64, 256, 325, 12]
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x,self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]

            x = self.bn[i](x)  # [64, 32, 325, 1] in last layer of last block

        h = F.relu(skip)  # [B, 256, N, 1]
        h = F.relu(self.end_conv_1(h))  # [B, 512, N, 1]
        output = self.end_conv_2(h)  # [B, 512, N, 1] -> [B, Q, N, 1]
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

        # U_n, _, _ = torch.linalg.svd(torch.randn(num_nodes, rank_n), full_matrices=False)
        # U_q, _, _ = torch.linalg.svd(torch.randn(seq_length, rank_q), full_matrices=False)

        # self.L_n = nn.Parameter(U_n, requires_grad=True)
        # self.L_q = nn.Parameter(U_q, requires_grad=True)

        # self.L_n = nn.Parameter(torch.diag(torch.randn(num_nodes))*scaling, requires_grad=True)
        # self.L_q = nn.Parameter(torch.diag(torch.randn(seq_length))*scaling, requires_grad=True)

        self.L_n = nn.Parameter(torch.randn(num_nodes, self.rank_n)*scaling, requires_grad=True)
        self.L_q = nn.Parameter(torch.randn(seq_length, self.rank_q)*scaling, requires_grad=True)

        # self.L_n_0 = nn.Parameter(torch.zeros(num_nodes, self.rank_n), requires_grad=False)
        # self.L_q_0 = nn.Parameter(torch.zeros(seq_length, self.rank_q), requires_grad=False)

        self.sigma = nn.Parameter(torch.randn(1), requires_grad=True)
        # self.sigma = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        # self.full_cov = False

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

        #1 without scaling 
        # cov_factor = torch.kron(self.L_q, self.L_n)

        #2 with scaling
        L_q = self.L_q / self._cov_factor_scale_q
        L_n = self.L_n / self._cov_factor_scale_n
    
        # if self.full_cov:
        #     L_q = self.L_q / self._cov_factor_scale_q
        #     L_n = self.L_n / self._cov_factor_scale_n
        # else:
        #     L_q = self.L_q_0
        #     L_n = self.L_n_0

        cov_factor = torch.kron(L_q, L_n)

        #1 without diag bias and jitter
        # cov_diag = F.softplus(self.sigma).repeat(cov_factor.shape[0])
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
    def __init__(self, device, dr, dr_init, rho, loss_type, rank_n, rank_q, cov_scaling, beta, num_nodes, seq_length, in_dim, dropout, supports, gcn_bool, addaptadj, adjinit, nhid, scaler):
        super(dynamic_reg, self).__init__()

        self.scaler = scaler
        self.num_nodes = num_nodes
        self.out_dim = seq_length
        self.dr = dr
        self.rho = rho
        self.loss = loss_type
        self.rank_n = rank_n
        self.rank_q = rank_q

        self.mean_model = gwnet(device=device, num_nodes=num_nodes, dropout=dropout, supports=supports, gcn_bool= gcn_bool, addaptadj=addaptadj, aptinit=adjinit, in_dim=in_dim, out_dim=seq_length, residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid*8, end_channels=nhid*16)

        if self.dr:
            self.res_model = mar_regressor(num_nodes=num_nodes, seq_length=seq_length, init=dr_init)
        
        if self.rho > 0:
            if self.loss == "full":
                self.cov_model = MGD_loss_full_eye(num_nodes=num_nodes, seq_length=seq_length, rank_n=rank_n, rank_q=rank_q, scaling=cov_scaling, beta=beta)
            else:
                self.cov_model = MGD_loss_eye(num_nodes=num_nodes, seq_length=seq_length)

    def forward(self, x, y):
        y_t = y[:,:self.out_dim,:,0]

        if self.dr:
            x_t = torch.cat([x[...,:1], x[...,2:]], dim=-1)
            x_t_s, y_t_s = x[...,1:], y[:,:self.out_dim,:,1]
            y_t_hat = self.mean_model(x_t)
            with torch.no_grad():
                y_t_s_hat = self.mean_model(x_t_s)
                # eta_t_s = y_t_s - self.scaler.inverse_transform(y_t_s_hat)
                eta_t_s = self.scaler.transform(y_t_s) - y_t_s_hat
                mask = (y_t_s != 0.0).float()
                eta_t_s = eta_t_s * mask

            eta_t = self.res_model(eta_t_s)
            y_t_hat += eta_t
        else:
            y_t_hat = self.mean_model(x)

        if self.rho > 0:
            # eps_t = y_t - self.scaler.inverse_transform(y_t_hat)
            # use eps
            eps_t = self.scaler.transform(y_t) - y_t_hat
            nll = self.cov_model(eps_t, y_t)
            # use y_t_hat
            # nll = self.cov_model(y_t_hat, self.scaler.transform(y_t))
        else:
            nll = 0.0

        return y_t_hat, nll


# class MGD_loss_full_eye(nn.Module):
#     def __init__(self, num_nodes, seq_length, scaling=1.0, beta=1.0, rank_n=10, rank_q=3):
#         super(MGD_loss_full_eye, self).__init__()

#         self.num_nodes = num_nodes
#         self.out_dim = seq_length
#         self.scaling = scaling

#         self.L_n = nn.Parameter(torch.randn(num_nodes, rank_n), requires_grad=True)
#         self.L_q = nn.Parameter(torch.randn(seq_length, rank_q), requires_grad=True)

#         self.sigma = nn.Parameter(torch.randn(1), requires_grad=True)


#     def log_prob(self, e_t):
#         sigma_pos = F.softplus(self.sigma)
#         sigma_pos_inv = 1/sigma_pos
#         E_inv_diag = sigma_pos_inv.repeat(e_t.shape[1]).unsqueeze(0)

#         A = torch.kron(self.L_q, self.L_n)
#         At_Einv_A = A.T*E_inv_diag@A
#         m = At_Einv_A.size(-1)
#         K.view(-1, m * m)[:, ::m + 1] += 1

#         return torch.mean(-0.5 * (log_det + mahalanobis.squeeze() + e_t.shape[1]*math.log(2*math.pi)))


#     def forward(self, eps_t, y_t):
#         """
#         eps_t: (B, Q, N)
#         y_t: (B, Q, N)
#         """
#         mask = (y_t != 0.0).float()
#         eps_t = eps_t * mask

#         nll = -self.log_prob(eps_t.flatten(start_dim=1))

#         L_s, L_t = self.get_L()
#         K_s = (L_s@L_s.T)*self.scaling
#         K_t = (L_t@L_t.T)*self.scaling
#         D_t, U_t = torch.linalg.eigh(K_t)
#         D_s, U_s = torch.linalg.eigh(K_s)
#         # D_t, U_t = torch.linalg.eig(K_t)
#         # D_s, U_s = torch.linalg.eig(K_s)
#         # D_t, U_t, D_s, U_s = D_t.real, U_t.real, D_s.real, U_s.real

#         capacitance_mat = torch.kron(D_t, D_s) + self.sigma**2
#         H = (U_t.T@eps_t@U_s).flatten(start_dim=1).unsqueeze(-1)
#         mahalanobis = H.mT@torch.diag_embed(1/capacitance_mat)@H

#         log_det = capacitance_mat.log().sum()

#         return -self.log_prob(eps_t.reshape(eps_t.shape[0], eps_t.shape[1]*eps_t.shape[2], -1), mahalanobis, log_det)


# class MGD_loss_full_eye(nn.Module):
#     def __init__(self, num_nodes, seq_length, scaling=1.0, beta=1.0):
#         super(MGD_loss_full_eye, self).__init__()

#         self.num_nodes = num_nodes
#         self.out_dim = seq_length
#         self.scaling = scaling

#         self.L1 = nn.Parameter(torch.diag(torch.rand(num_nodes)), requires_grad=False)
#         self.L2 = nn.Parameter(torch.diag(torch.rand(seq_length)), requires_grad=False)
#         # self.L1 = nn.Parameter(torch.rand(num_nodes, num_nodes)*scaling, requires_grad=False)
#         # self.L2 = nn.Parameter(torch.rand(seq_length, seq_length)*scaling, requires_grad=False)

#         # self.sigma = nn.Parameter(torch.randn(1), requires_grad=False)
#         self.sigma = nn.Parameter(torch.tensor(1.0), requires_grad=False)

#         # self.act = nn.Softplus(beta=beta, threshold=0)
#         self.act = nn.Softplus(beta=beta)

#     def get_L(self):
#         L_spatial, L_temporal = torch.tril(self.L1), torch.tril(self.L2)

#         L_spatial[torch.arange(self.num_nodes), torch.arange(self.num_nodes)] = self.act(L_spatial[torch.arange(self.num_nodes), torch.arange(self.num_nodes)])
#         L_temporal[torch.arange(self.out_dim), torch.arange(self.out_dim)] = self.act(L_temporal[torch.arange(self.out_dim), torch.arange(self.out_dim)])

#         # L_spatial[torch.arange(self.num_nodes), torch.arange(self.num_nodes)] = torch.clamp(L_spatial[torch.arange(self.num_nodes), torch.arange(self.num_nodes)], min=1e-5)
#         # L_temporal[torch.arange(self.out_dim), torch.arange(self.out_dim)] = torch.clamp(L_temporal[torch.arange(self.out_dim), torch.arange(self.out_dim)], min=1e-5)

#         return L_spatial, L_temporal

#     def sample(self, y_shape, n_sample=100):
#         L_s, L_t = self.get_L()
#         K_s = (L_s@L_s.T)*self.scaling
#         K_t = (L_t@L_t.T)*self.scaling
#         cov = torch.kron(K_t, K_s) + self.sigma**2*torch.eye(self.num_nodes*self.out_dim, device=K_t.device)

#         dist = MultivariateNormal(loc=torch.zeros(cov.shape[0], device=cov.device), covariance_matrix=cov)

#         eps = dist.sample((y_shape[0], n_sample)).reshape(y_shape[0], n_sample, *y_shape[1:])

#         return eps
    
#     def log_prob(self, e_t, mahalanobis, log_det):
#         return torch.mean(-0.5 * (log_det + mahalanobis.squeeze() + e_t.shape[1]*math.log(2*math.pi)))

#     def forward(self, eps_t, y_t):
#         mask = (y_t != 0.0).float()
#         eps_t = eps_t * mask

#         L_s, L_t = self.get_L()
#         K_s = (L_s@L_s.T)*self.scaling
#         K_t = (L_t@L_t.T)*self.scaling
#         D_t, U_t = torch.linalg.eigh(K_t)
#         D_s, U_s = torch.linalg.eigh(K_s)
#         # D_t, U_t = torch.linalg.eig(K_t)
#         # D_s, U_s = torch.linalg.eig(K_s)
#         # D_t, U_t, D_s, U_s = D_t.real, U_t.real, D_s.real, U_s.real

#         capacitance_mat = torch.kron(D_t, D_s) + self.sigma**2
#         H = (U_t.T@eps_t@U_s).flatten(start_dim=1).unsqueeze(-1)
#         mahalanobis = H.mT@torch.diag_embed(1/capacitance_mat)@H

#         log_det = capacitance_mat.log().sum()

#         return -self.log_prob(eps_t.reshape(eps_t.shape[0], eps_t.shape[1]*eps_t.shape[2], -1), mahalanobis, log_det)


# class MGD_loss_full_eye(nn.Module):
#     def __init__(self, num_nodes, seq_length, scaling=1.0, beta=1.0):
#         super(MGD_loss_full_eye, self).__init__()

#         self.num_nodes = num_nodes
#         self.out_dim = seq_length

#         self.L1 = nn.Parameter(torch.randn(num_nodes, num_nodes)*scaling, requires_grad=True)
#         self.L2 = nn.Parameter(torch.randn(seq_length, seq_length)*scaling, requires_grad=True)

#         self.sigma = nn.Parameter(torch.randn(1), requires_grad=True)

#         self.act = nn.Softplus(beta=beta)

#     def get_L(self):
#         L_spatial, L_temporal = torch.tril(self.L1), torch.tril(self.L2)

#         L_spatial[torch.arange(self.num_nodes), torch.arange(self.num_nodes)] = self.act(L_spatial[torch.arange(self.num_nodes), torch.arange(self.num_nodes)])
#         L_temporal[torch.arange(self.out_dim), torch.arange(self.out_dim)] = self.act(L_temporal[torch.arange(self.out_dim), torch.arange(self.out_dim)])

#         return L_spatial, L_temporal

#     def sample(self, y_shape, n_sample=100):
#         L_s, L_t = self.get_L()
#         K_s = L_s@L_s.T
#         K_t = L_t@L_t.T
#         cov = torch.kron(K_t, K_s) + self.sigma**2*torch.eye(self.num_nodes*self.out_dim, device=K_t.device)

#         dist = MultivariateNormal(loc=torch.zeros(cov.shape[0], device=cov.device), covariance_matrix=cov)

#         eps = dist.sample((y_shape[0], n_sample)).reshape(y_shape[0], n_sample, *y_shape[1:])

#         return eps
    
#     def log_prob(self, e_t, mahalanobis, log_det):
#         return torch.mean(-0.5 * (log_det + mahalanobis.squeeze() + e_t.shape[1]*math.log(2*math.pi)))

#     def forward(self, eps_t, y_t):
#         mask = (y_t != 0.0).float()
#         eps_t = eps_t * mask

#         L_s, L_t = self.get_L()
#         K_s = L_s@L_s.T
#         K_t = L_t@L_t.T
#         cov_mat = torch.kron(K_t, K_s) + self.sigma**2*torch.eye(self.num_nodes*self.out_dim, device=eps_t.device)

#         dist = MultivariateNormal(torch.zeros(self.num_nodes*self.out_dim, device=eps_t.device), cov_mat)

#         return torch.mean(-dist.log_prob(eps_t.flatten(start_dim=1)))


# class MGD_loss_full_eye(nn.Module):
#     """
#     parameterize the Orthogonal matrix directly
#     """
#     def __init__(self, num_nodes, seq_length, scaling=1.0, beta=1.0):
#         super(MGD_loss_full_eye, self).__init__()

#         self.num_nodes = num_nodes
#         self.out_dim = seq_length

#         self.U_t = nn.utils.parametrizations.orthogonal(nn.Linear(seq_length, seq_length))
#         self.U_s = nn.utils.parametrizations.orthogonal(nn.Linear(num_nodes, num_nodes))

#         self.D_t = nn.Parameter(torch.randn(seq_length), requires_grad=True)
#         self.D_s = nn.Parameter(torch.randn(num_nodes), requires_grad=True)

#         self.sigma = nn.Parameter(torch.randn(1), requires_grad=True)

#         self.act = nn.Softplus(beta=beta)
#         # self.act = lambda x: nn.ELU()(x) + 1

#     def get_params(self):
#         return self.U_t.weight, self.U_s.weight, self.act(self.D_t), self.act(self.D_s)

#     def log_prob(self, e_t, mahalanobis, log_det):
#         return torch.mean(-0.5 * (log_det + mahalanobis.squeeze() + e_t.shape[1]*math.log(2*math.pi)))

#     def forward(self, eps_t, y_t):
#         mask = (y_t != 0.0).float()
#         eps_t = eps_t * mask

#         U_t, U_s, D_t, D_s = self.get_params()

#         capacitance_mat = torch.kron(D_s, D_t) + self.sigma**2
#         H = (U_t.T@eps_t@U_s).flatten(start_dim=1).unsqueeze(-1)
#         mahalanobis = H.mT@torch.diag_embed(1/capacitance_mat)@H

#         log_det = capacitance_mat.log().sum()

#         return -self.log_prob(eps_t.reshape(eps_t.shape[0], eps_t.shape[1]*eps_t.shape[2], -1), mahalanobis, log_det)
    
