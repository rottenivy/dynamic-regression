import pickle
import numpy as np
import pandas as pd
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from torch.utils.data import Dataset
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave

    
def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)
    

class StandardScaler():
    def __init__(self, mean, std):
        """
        (B, T, N)
        """
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class getRawTrafficData():
    def __init__(self, file_dir, filename, train_ratio=0.7, test_ratio=0.2):
        """
        data: (T, N)
        """
        df = pd.read_hdf(os.path.join(file_dir, f"{filename}.h5"))
        self.data_ts = torch.from_numpy(df.values).float()

        self.num_timesteps = self.data_ts.shape[0]
        self.num_sensors = self.data_ts.shape[1]

        self.train_steps = round(train_ratio * self.num_timesteps)
        self.test_steps = round(test_ratio * self.num_timesteps)
        self.val_steps = self.num_timesteps - self.train_steps - self.test_steps

    def __mean__(self, split):
        if split == 'train':
            mean = self.data_ts[:self.train_steps].mean(0, keepdim=True).unsqueeze(0)
        elif split == 'val':
            mean = self.data_ts[self.train_steps:self.train_steps+self.val_steps].mean(0, keepdim=True).unsqueeze(0)
        else:
            mean = self.data_ts[-self.test_steps:].mean(0, keepdim=True).unsqueeze(0)
        return mean

    def __std__(self, split):
        if split == 'train':
            std = self.data_ts[:self.train_steps].std(0, keepdim=True).unsqueeze(0)
        elif split == 'val':
            std = self.data_ts[self.train_steps:self.train_steps+self.val_steps].std(0, keepdim=True).unsqueeze(0)
        else:
            std = self.data_ts[-self.test_steps:].std(0, keepdim=True).unsqueeze(0)
        return std


class TrafficDataset(Dataset):
    def __init__(self, debug_mode, device, file_dir, split, transform=True, target_transform=False, scaler=None):
        """
        x: (B, P, N, 1+C)
        y: (B, Q, N, 1+C)
        """
        self.debug_mode = debug_mode
        self.file_dir = file_dir
        self.split = split
        self.x, self.y = self.__read__(split)
        self.x, self.y = self.x.to(device), self.y.to(device)
        self.in_dim, self.seq_len, self.num_nodes = self.x.shape[-1], self.x.shape[1], self.x.shape[2]

        self.scaler = scaler if scaler else StandardScaler(mean=self.x[..., 0].mean(), std=self.x[..., 0].std())

        if transform:
            self.x[...,0] = self.scaler.transform(self.x[...,0])
        if target_transform:
            self.y[...,0] = self.scaler.transform(self.y[...,0])

    def __getScaler__(self,):
        return self.scaler

    def __read__(self, split):
        with np.load(os.path.join(self.file_dir, f"{split}.npz")) as f:
            if self.debug_mode:
                x = f["x"][...,:15,:]
                y = f["y"][...,:15,:]
            else:
                x = f["x"]
                y = f["y"]
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, output_dir, patience, model_name='best_model', best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        self.output_dir = output_dir
        self.model_name = model_name
        self.val_increase = 0
        self.patience = patience

    def __call__(
        self, current_valid_loss, 
        epoch, model, optimizer, criterion
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, '%s/%s.pth'%(self.output_dir, self.model_name))
            self.val_increase = 0
            return False
        else:
            self.val_increase += 1
            if self.val_increase > self.patience:
                self.val_increase = 0
                return True
            else:
                return False


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_adj(debug_mode, pkl_filename, adjtype):
    # if pkl_filename[-3:] == 'pkl':
    #     sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    # else:
    adj_mx = np.load(pkl_filename)
    if debug_mode:
        adj_mx = adj_mx[:15,:15]
    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return adj


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rrmse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    # MSE
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    # Nomalizer
    loss_norm = (labels.mean()-labels)**2
    loss_norm = loss_norm * mask
    loss_norm = torch.where(torch.isnan(loss_norm), torch.zeros_like(loss_norm), loss_norm)
    return torch.sqrt(torch.sum(loss))/torch.sqrt(torch.sum(loss_norm))


def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    mse = masked_mse(pred,real,0.0).item()
    rrmse = masked_rrmse(pred,real,0.0).item()
    return mae,mape,rmse,mse,rrmse


class quantile_loss():
    def __init__(self, quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9]):
        super().__init__()
        self.quantiles = quantiles

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor):
        quantiles = torch.quantile(y_pred, torch.tensor(self.quantiles, device=y_pred.device), dim=0)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - quantiles[i]
            losses.append(torch.max((q-1)*errors, q*errors).unsqueeze(-1))
        losses = 2*torch.cat(losses, dim=-1)

        return losses