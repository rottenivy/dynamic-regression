import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions, nn
from abc import abstractmethod

class BaseModel(nn.Module):
    def __init__(self, node_num, input_dim, output_dim, seq_len=12, horizon=12):
        super(BaseModel, self).__init__()
        self.node_num = node_num
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.horizon = horizon


    @abstractmethod
    def forward(self):
        raise NotImplementedError


    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])


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
    def __init__(self, device, dr, dr_init, rho, loss_type, rank_n, rank_q, cov_scaling, beta, num_nodes, seq_length, in_dim, dropout, nhid, scaler, model_args):
        super(dynamic_reg, self).__init__()
        self.scaler = scaler
        self.num_nodes = num_nodes
        self.out_dim = seq_length
        self.dr = dr
        self.rho = rho
        self.loss = loss_type
        self.rank_n = rank_n
        self.rank_q = rank_q

        self.mean_model = D2STGNN(node_num=num_nodes,
                    input_dim=in_dim,
                    output_dim=1,
                    model_args=model_args
                    )

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
            x_t, x_t_s = x.split(x.size(-1)//2, dim=-1)
            y_t_s = y[:,:self.out_dim,:,x.size(-1)//2]
            y_t_hat = self.mean_model(x_t).squeeze(-1)
            with torch.no_grad():
                y_t_s_hat = self.mean_model(x_t_s).squeeze(-1)
                eta_t_s = self.scaler.transform(y_t_s) - y_t_s_hat
                mask = (y_t_s != 0.0).float()
                eta_t_s = eta_t_s * mask

            eta_t = self.res_model(eta_t_s)
            y_t_hat += eta_t
        else:
            y_t_hat = self.mean_model(x).squeeze(-1)

        if self.rho > 0:
            eps_t = self.scaler.transform(y_t) - y_t_hat
            nll = self.cov_model(eps_t, y_t)
        else:
            nll = 0.0

        return y_t_hat, nll


class D2STGNN(BaseModel):
    '''
    Reference code: https://github.com/zezhishao/D2STGNN
    '''
    def __init__(self, model_args, **args):
        super(D2STGNN, self).__init__(**args)
        self._in_feat = model_args['num_feat']
        self._hidden_dim = model_args['num_hidden']
        self._node_dim = model_args['node_hidden']
        self._forecast_dim = 256
        self._output_hidden = 512

        self._node_num = self.node_num
        self._k_s = model_args['k_s']
        self._k_t = model_args['k_t']
        self._num_layers = model_args['layer']

        model_args['use_pre'] = False
        model_args['dy_graph'] = True
        model_args['sta_graph'] = True

        self._model_args = model_args

        self.embedding = nn.Linear(self._in_feat, self._hidden_dim)

        self.node_emb_u = nn.Parameter(torch.empty(self._node_num, self._node_dim))
        self.node_emb_d = nn.Parameter(torch.empty(self._node_num, self._node_dim))
        self.T_i_D_emb = nn.Parameter(torch.empty(model_args['tpd'], model_args['time_emb_dim']))
        self.D_i_W_emb = nn.Parameter(torch.empty(7, model_args['time_emb_dim']))

        self.layers = nn.ModuleList([DecoupleLayer(self._hidden_dim, fk_dim=self._forecast_dim, **model_args)])
        for _ in range(self._num_layers - 1):
            self.layers.append(DecoupleLayer(self._hidden_dim, fk_dim=self._forecast_dim, **model_args))

        if model_args['dy_graph']:
            self.dynamic_graph_constructor = DynamicGraphConstructor(**model_args)

        self.out_fc_1 = nn.Linear(self._forecast_dim, self._output_hidden)
        self.out_fc_2 = nn.Linear(self._output_hidden, model_args['gap'])

        self.reset_parameter()


    def reset_parameter(self):
        nn.init.xavier_uniform_(self.node_emb_u)
        nn.init.xavier_uniform_(self.node_emb_d)
        nn.init.xavier_uniform_(self.T_i_D_emb)
        nn.init.xavier_uniform_(self.D_i_W_emb)


    def _graph_constructor(self, **inputs):
        E_d = inputs['node_embedding_u']
        E_u = inputs['node_embedding_d']
        if self._model_args['sta_graph']:
            static_graph = [F.softmax(F.relu(torch.mm(E_d, E_u.T)), dim=1)]
        else:
            static_graph = []
        if self._model_args['dy_graph']:
            dynamic_graph = self.dynamic_graph_constructor(**inputs)
        else:
            dynamic_graph = []
        return static_graph, dynamic_graph


    def _prepare_inputs(self, history_data):
        num_feat = self._model_args['num_feat']

        node_emb_u = self.node_emb_u
        node_emb_d = self.node_emb_d

        time_in_day_feat = self.T_i_D_emb[(history_data[:, :, :, num_feat] * self._model_args['tpd']).type(torch.LongTensor)]
        day_in_week_feat = self.D_i_W_emb[(history_data[:, :, :, num_feat+1] * 7).type(torch.LongTensor)]

        history_data = history_data[:, :, :, :num_feat]
        return history_data, node_emb_u, node_emb_d, time_in_day_feat, day_in_week_feat


    def forward(self, history_data, label=None):  # (b, t, n, f)
        history_data, node_embedding_u, node_embedding_d, time_in_day_feat, day_in_week_feat   = self._prepare_inputs(history_data)

        static_graph, dynamic_graph = self._graph_constructor(node_embedding_u=node_embedding_u, node_embedding_d=node_embedding_d, history_data=history_data, time_in_day_feat=time_in_day_feat, day_in_week_feat=day_in_week_feat)

        history_data = self.embedding(history_data)

        dif_forecast_hidden_list = []
        inh_forecast_hidden_list = []

        inh_backcast_seq_res = history_data
        for _, layer in enumerate(self.layers):
            inh_backcast_seq_res, dif_forecast_hidden, inh_forecast_hidden = layer(inh_backcast_seq_res, dynamic_graph, static_graph, node_embedding_u, node_embedding_d, time_in_day_feat, day_in_week_feat)
            dif_forecast_hidden_list.append(dif_forecast_hidden)
            inh_forecast_hidden_list.append(inh_forecast_hidden)

        dif_forecast_hidden = sum(dif_forecast_hidden_list)
        inh_forecast_hidden = sum(inh_forecast_hidden_list)
        forecast_hidden = dif_forecast_hidden + inh_forecast_hidden

        forecast = self.out_fc_2(F.relu(self.out_fc_1(F.relu(forecast_hidden))))
        forecast = forecast.transpose(1,2).contiguous().view(forecast.shape[0], forecast.shape[2], -1)
        return forecast.transpose(1, 2).unsqueeze(-1)


class DecoupleLayer(nn.Module):
    def __init__(self, hidden_dim, fk_dim=256, **model_args):
        super().__init__()
        self.estimation_gate = EstimationGate(node_emb_dim=model_args['node_hidden'], time_emb_dim=model_args['time_emb_dim'], hidden_dim=64)
        self.dif_layer = DifBlock(hidden_dim, forecast_hidden_dim=fk_dim, **model_args)
        self.inh_layer = InhBlock(hidden_dim, forecast_hidden_dim=fk_dim, **model_args)


    def forward(self, history_data, dynamic_graph, static_graph, node_embedding_u, node_embedding_d, time_in_day_feat, day_in_week_feat):
        gated_history_data = self.estimation_gate(node_embedding_u, node_embedding_d, time_in_day_feat, day_in_week_feat, history_data)

        dif_backcast_seq_res, dif_forecast_hidden = self.dif_layer(history_data=history_data, gated_history_data=gated_history_data, dynamic_graph=dynamic_graph, static_graph=static_graph)

        inh_backcast_seq_res, inh_forecast_hidden = self.inh_layer(dif_backcast_seq_res)
        return inh_backcast_seq_res, dif_forecast_hidden, inh_forecast_hidden


class EstimationGate(nn.Module):
    def __init__(self, node_emb_dim, time_emb_dim, hidden_dim):
        super().__init__()
        self.fully_connected_layer_1 = nn.Linear(2 * node_emb_dim + time_emb_dim * 2, hidden_dim)
        self.activation = nn.ReLU()
        self.fully_connected_layer_2 = nn.Linear(hidden_dim, 1)


    def forward(self, node_embedding_u, node_embedding_d, time_in_day_feat, day_in_week_feat, history_data):
        batch_size, seq_length, _, _ = time_in_day_feat.shape
        estimation_gate_feat = torch.cat([time_in_day_feat, day_in_week_feat, node_embedding_u.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_length,  -1, -1), node_embedding_d.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_length,  -1, -1)], dim=-1)
        hidden = self.fully_connected_layer_1(estimation_gate_feat)
        hidden = self.activation(hidden)

        estimation_gate = torch.sigmoid(self.fully_connected_layer_2(hidden))[:, -history_data.shape[1]:, :, :]
        history_data = history_data * estimation_gate
        return history_data


class ResidualDecomp(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.ln = nn.LayerNorm(input_shape[-1])
        self.ac = nn.ReLU()


    def forward(self, x, y):
        u = x - self.ac(y)
        u = self.ln(u)
        return u


class DifBlock(nn.Module):
    def __init__(self, hidden_dim, forecast_hidden_dim=256, use_pre=None, dy_graph=None, sta_graph=None, **model_args):
        super().__init__()
        self.pre_defined_graph = model_args['adjs']

        self.localized_st_conv = STLocalizedConv(hidden_dim, pre_defined_graph=self.pre_defined_graph, use_pre=use_pre, dy_graph=dy_graph, sta_graph=sta_graph, **model_args)

        self.forecast_branch = DifForecast(hidden_dim, forecast_hidden_dim=forecast_hidden_dim, **model_args)
        self.backcast_branch = nn.Linear(hidden_dim, hidden_dim)
        self.residual_decompose = ResidualDecomp([-1, -1, -1, hidden_dim])


    def forward(self, history_data, gated_history_data, dynamic_graph, static_graph):
        hidden_states_dif = self.localized_st_conv(gated_history_data, dynamic_graph, static_graph)

        forecast_hidden = self.forecast_branch(gated_history_data, hidden_states_dif, self.localized_st_conv, dynamic_graph, static_graph)
        backcast_seq = self.backcast_branch(hidden_states_dif)

        history_data = history_data[:, -backcast_seq.shape[1]:, :, :]
        backcast_seq_res = self.residual_decompose(history_data, backcast_seq)

        return backcast_seq_res, forecast_hidden


class STLocalizedConv(nn.Module):
    def __init__(self, hidden_dim, pre_defined_graph=None, use_pre=None, dy_graph=None, sta_graph=None, **model_args):
        super().__init__()
        self.k_s = model_args['k_s']
        self.k_t = model_args['k_t']
        self.hidden_dim = hidden_dim

        self.pre_defined_graph = pre_defined_graph
        self.use_predefined_graph = use_pre
        self.use_dynamic_hidden_graph = dy_graph
        self.use_static_hidden_graph = sta_graph

        self.support_len = len(self.pre_defined_graph) + \
            int(dy_graph) + int(sta_graph)
        self.num_matric = (int(use_pre) * len(self.pre_defined_graph) + len(
            self.pre_defined_graph) * int(dy_graph) + int(sta_graph)) * self.k_s + 1
        self.dropout = nn.Dropout(model_args['dropout'])
        self.pre_defined_graph = self.get_graph(self.pre_defined_graph)

        self.fc_list_updt = nn.Linear(
            self.k_t * hidden_dim, self.k_t * hidden_dim, bias=False)
        self.gcn_updt = nn.Linear(
            self.hidden_dim * self.num_matric, self.hidden_dim)

        self.bn = nn.BatchNorm2d(self.hidden_dim)
        self.activation = nn.ReLU()


    def gconv(self, support, X_k, X_0):
        out = [X_0]
        for graph in support:
            if len(graph.shape) == 2:
                pass
            else:
                graph = graph.unsqueeze(1)
            H_k = torch.matmul(graph, X_k)
            out.append(H_k)
        out = torch.cat(out, dim=-1)
        out = self.gcn_updt(out)
        out = self.dropout(out)
        return out


    def get_graph(self, support):
        graph_ordered = []
        mask = 1 - torch.eye(support[0].shape[0]).to(support[0].device)
        for graph in support:
            k_1_order = graph
            graph_ordered.append(k_1_order * mask)

            for k in range(2, self.k_s + 1):
                k_1_order = torch.matmul(graph, k_1_order)
                graph_ordered.append(k_1_order * mask)

        st_local_graph = []
        for graph in graph_ordered:
            graph = graph.unsqueeze(-2).expand(-1, self.k_t, -1)
            graph = graph.reshape(
                graph.shape[0], graph.shape[1] * graph.shape[2])
            st_local_graph.append(graph)

        return st_local_graph


    def forward(self, X, dynamic_graph, static_graph):
        X = X.unfold(1, self.k_t, 1).permute(0, 1, 2, 4, 3)
        batch_size, seq_len, node_num, kernel_size, num_feat = X.shape

        support = []
        if self.use_predefined_graph:
            support = support + self.pre_defined_graph
        if self.use_dynamic_hidden_graph:
            support = support + dynamic_graph
        if self.use_static_hidden_graph:
            support = support + self.get_graph(static_graph)

        # parallelize
        X = X.reshape(batch_size, seq_len, node_num, kernel_size * num_feat)
        out = self.fc_list_updt(X)
        out = self.activation(out)
        out = out.view(batch_size, seq_len, node_num, kernel_size, num_feat)
        X_0 = torch.mean(out, dim=-2)
        X_k = out.transpose(-3, -2).reshape(batch_size, seq_len, kernel_size*node_num, num_feat)
        hidden = self.gconv(support, X_k, X_0)
        return hidden


class DifForecast(nn.Module):
    def __init__(self, hidden_dim, forecast_hidden_dim=None, **model_args):
        super().__init__()
        self.k_t = model_args['k_t']
        self.output_seq_len = model_args['seq_len']
        self.forecast_fc = nn.Linear(hidden_dim, forecast_hidden_dim)
        self.model_args = model_args


    def forward(self, gated_history_data, hidden_states_dif, localized_st_conv, dynamic_graph, static_graph):
        predict = []
        history = gated_history_data
        predict.append(hidden_states_dif[:, -1, :, :].unsqueeze(1))
        for _ in range(int(self.output_seq_len / self.model_args['gap'])-1):
            _1 = predict[-self.k_t:]
            if len(_1) < self.k_t:
                sub = self.k_t - len(_1)
                _2 = history[:, -sub:, :, :]
                _1 = torch.cat([_2] + _1, dim=1)
            else:
                _1 = torch.cat(_1, dim=1)
            predict.append(localized_st_conv(_1, dynamic_graph, static_graph))
        predict = torch.cat(predict, dim=1)
        predict = self.forecast_fc(predict)
        return predict


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=None, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)


    def forward(self, X):
        X = X + self.pe[:X.size(0)]
        X = self.dropout(X)
        return X


class InhBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, bias=True, forecast_hidden_dim=256, **model_args):
        super().__init__()
        self.num_feat = hidden_dim
        self.hidden_dim = hidden_dim

        self.pos_encoder = PositionalEncoding(hidden_dim, model_args['dropout'])
        self.rnn_layer = RNNLayer(hidden_dim, model_args['dropout'])
        self.transformer_layer = TransformerLayer(hidden_dim, num_heads, model_args['dropout'], bias)

        self.forecast_block = InhForecast(hidden_dim, forecast_hidden_dim, **model_args)
        self.backcast_fc = nn.Linear(hidden_dim, hidden_dim)
        self.residual_decompose = ResidualDecomp([-1, -1, -1, hidden_dim])


    def forward(self, hidden_inherent_signal):
        [batch_size, seq_len, node_num, num_feat] = hidden_inherent_signal.shape

        hidden_states_rnn = self.rnn_layer(hidden_inherent_signal)
        hidden_states_rnn = self.pos_encoder(hidden_states_rnn)
        hidden_states_inh = self.transformer_layer(hidden_states_rnn, hidden_states_rnn, hidden_states_rnn)

        forecast_hidden = self.forecast_block(hidden_inherent_signal, hidden_states_rnn, hidden_states_inh, self.transformer_layer, self.rnn_layer, self.pos_encoder)
        hidden_states_inh = hidden_states_inh.reshape(seq_len, batch_size, node_num, num_feat)
        hidden_states_inh = hidden_states_inh.transpose(0, 1)
        backcast_seq = self.backcast_fc(hidden_states_inh)
        backcast_seq_res= self.residual_decompose(hidden_inherent_signal, backcast_seq)
        return backcast_seq_res, forecast_hidden


class RNNLayer(nn.Module):
    def __init__(self, hidden_dim, dropout=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self, X):
        [batch_size, seq_len, node_num, hidden_dim] = X.shape
        X = X.transpose(1, 2).reshape(batch_size * node_num, seq_len, hidden_dim)
        hx = torch.zeros_like(X[:, 0, :])
        output  = []
        for _ in range(X.shape[1]):
            hx = self.gru_cell(X[:, _, :], hx)
            output.append(hx)
        output = torch.stack(output, dim=0)
        output = self.dropout(output)
        return output


class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=None, bias=True):
        super().__init__()
        self.multi_head_self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, bias=bias)
        self.dropout = nn.Dropout(dropout)


    def forward(self, X, K, V):
        hidden_states_MSA = self.multi_head_self_attention(X, K, V)[0]
        hidden_states_MSA = self.dropout(hidden_states_MSA)
        return hidden_states_MSA


class InhForecast(nn.Module):
    def __init__(self, hidden_dim, fk_dim, **model_args):
        super().__init__()
        self.output_seq_len = model_args['seq_len']
        self.model_args = model_args

        self.forecast_fc = nn.Linear(hidden_dim, fk_dim)


    def forward(self, X, RNN_H, Z, transformer_layer, rnn_layer, pe):
        [batch_size, _, node_num, num_feat] = X.shape

        predict = [Z[-1, :, :].unsqueeze(0)]
        for _ in range(int(self.output_seq_len / self.model_args['gap'])-1):
            _gru = rnn_layer.gru_cell(predict[-1][0], RNN_H[-1]).unsqueeze(0)
            RNN_H = torch.cat([RNN_H, _gru], dim=0)

            if pe is not None:
                RNN_H = pe(RNN_H)

            _Z = transformer_layer(_gru, K=RNN_H, V=RNN_H)
            predict.append(_Z)

        predict = torch.cat(predict, dim=0)
        predict = predict.reshape(-1, batch_size, node_num, num_feat)
        predict = predict.transpose(0, 1)
        predict = self.forecast_fc(predict)
        return predict


class DynamicGraphConstructor(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.k_s = model_args['k_s']
        self.k_t = model_args['k_t']
        self.hidden_dim = model_args['num_hidden']
        self.node_dim = model_args['node_hidden']

        self.distance_function = DistanceFunction(**model_args)
        self.mask = Mask(**model_args)
        self.normalizer = Normalizer()
        self.multi_order = MultiOrder(order=self.k_s)


    def st_localization(self, graph_ordered):
        st_local_graph = []
        for modality_i in graph_ordered:
            for k_order_graph in modality_i:
                k_order_graph = k_order_graph.unsqueeze(
                    -2).expand(-1, -1, self.k_t, -1)
                k_order_graph = k_order_graph.reshape(
                    k_order_graph.shape[0], k_order_graph.shape[1], k_order_graph.shape[2] * k_order_graph.shape[3])
                st_local_graph.append(k_order_graph)
        return st_local_graph


    def forward(self, **inputs):
        X = inputs['history_data']
        E_d = inputs['node_embedding_d']
        E_u = inputs['node_embedding_u']
        T_D = inputs['time_in_day_feat']
        D_W = inputs['day_in_week_feat']

        dist_mx = self.distance_function(X, E_d, E_u, T_D, D_W)
        dist_mx = self.mask(dist_mx)
        dist_mx = self.normalizer(dist_mx)
        mul_mx = self.multi_order(dist_mx)
        dynamic_graphs = self.st_localization(mul_mx)
        return dynamic_graphs


class DistanceFunction(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.hidden_dim = model_args['num_hidden']
        self.node_dim = model_args['node_hidden']
        self.time_slot_emb_dim = self.hidden_dim
        self.input_seq_len = model_args['seq_len']
        self.dropout = nn.Dropout(model_args['dropout'])
        self.fc_ts_emb1 = nn.Linear(self.input_seq_len, self.hidden_dim * 2)
        self.fc_ts_emb2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.time_slot_embedding = nn.Linear(model_args['time_emb_dim'], self.time_slot_emb_dim)

        self.ts_feat_dim = self.hidden_dim
        self.all_feat_dim = self.ts_feat_dim + self.node_dim + model_args['time_emb_dim'] * 2
        self.WQ = nn.Linear(self.all_feat_dim, self.hidden_dim, bias=False)
        self.WK = nn.Linear(self.all_feat_dim, self.hidden_dim, bias=False)
        self.bn = nn.BatchNorm1d(self.hidden_dim * 2)


    def reset_parameters(self):
        for q_vec in self.q_vecs:
            nn.init.xavier_normal_(q_vec.data)
        for bias in self.biases:
            nn.init.zeros_(bias.data)


    def forward(self, X, E_d, E_u, T_D, D_W):
        T_D = T_D[:, -1, :, :]
        D_W = D_W[:, -1, :, :]

        X = X[:, :, :, 0].transpose(1, 2).contiguous()
        [batch_size, node_num, seq_len] = X.shape
        X = X.view(batch_size * node_num, seq_len)
        dy_feat = self.fc_ts_emb2(self.dropout(self.bn(F.relu(self.fc_ts_emb1(X)))))
        dy_feat = dy_feat.view(batch_size, node_num, -1)

        emb1 = E_d.unsqueeze(0).expand(batch_size, -1, -1)
        emb2 = E_u.unsqueeze(0).expand(batch_size, -1, -1)

        X1 = torch.cat([dy_feat, T_D, D_W, emb1], dim=-1)
        X2 = torch.cat([dy_feat, T_D, D_W, emb2], dim=-1)
        X = [X1, X2]
        adjacent_list = []
        for _ in X:
            Q = self.WQ(_)
            K = self.WK(_)
            QKT = torch.bmm(Q, K.transpose(-1, -2)) / math.sqrt(self.hidden_dim)
            W = torch.softmax(QKT, dim=-1)
            adjacent_list.append(W)
        return adjacent_list


class Mask(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.mask = model_args['adjs']


    def _mask(self, index, adj):
        mask = self.mask[index] + torch.ones_like(self.mask[index]) * 1e-7
        return mask.to(adj.device) * adj


    def forward(self, adj):
        result = []
        for index, _ in enumerate(adj):
            result.append(self._mask(index, _))
        return result


class Normalizer(nn.Module):
    def __init__(self):
        super().__init__()


    def _remove_nan_inf(self, tensor):
        tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
        tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
        return tensor


    def _norm(self, graph):
        degree = torch.sum(graph, dim=2)
        degree = self._remove_nan_inf(1 / degree)
        degree = torch.diag_embed(degree)
        normed_graph = torch.bmm(degree, graph)
        return normed_graph


    def forward(self, adj):
        return [self._norm(_) for _ in adj]


class MultiOrder(nn.Module):
    def __init__(self, order=2):
        super().__init__()
        self.order = order


    def _multi_order(self, graph):
        graph_ordered = []
        k_1_order = graph
        mask = torch.eye(graph.shape[1]).to(graph.device)
        mask = 1 - mask
        graph_ordered.append(k_1_order * mask)
        for k in range(2, self.order + 1):
            k_1_order = torch.matmul(k_1_order, graph)
            graph_ordered.append(k_1_order * mask)
        return graph_ordered


    def forward(self, adj):
        return [self._multi_order(_) for _ in adj]
