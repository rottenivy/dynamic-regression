import numpy as np
import torch
import utils


def train_epoch(dataloader, model, loss_fn, optimizer, out_dim, rho=0.0, scheduler_L=None, scaler=None, clip=None):
    model.train()
    log_loss, log_mape, log_rmse  = [], [], []
    for batch, (x, y) in enumerate(dataloader):
        y_t = y[:,:out_dim,:,0]
        y_t_hat, nll = model(x, y)

        if scaler:
            y_t_hat = scaler.inverse_transform(y_t_hat)

        loss = (1-rho)*loss_fn(y_t_hat, y_t, 0.0) + rho*nll + model.res_model.getRegulizer() if model.dr else (1-rho)*loss_fn(y_t_hat, y_t, 0.0) + rho*nll

        optimizer.zero_grad()
        loss.backward()

        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.cov_model.parameters(), clip)
            
        optimizer.step()

        if scheduler_L is not None:
            scheduler_L.step()

        mape = utils.masked_mape(y_t_hat, y_t, 0.0).item()
        rmse = utils.masked_rmse(y_t_hat, y_t, 0.0).item()

        log_loss.append(loss.item())
        log_mape.append(mape)
        log_rmse.append(rmse)

    return np.mean(log_loss), np.mean(log_mape), np.mean(log_rmse)


def eval_epoch(dataloader, model, loss_fn, out_dim, scaler=None, rho=0.0):
    model.eval()
    with torch.no_grad():
        log_loss, log_mape, log_rmse  = [], [], []
        for batch, (x, y) in enumerate(dataloader):
            y_t = y[:,:out_dim,:,0]
            y_t_hat, nll = model(x, y)

            if scaler:
                y_t_hat = scaler.inverse_transform(y_t_hat)

            loss = (1-rho)*loss_fn(y_t_hat, y_t, 0.0) + rho*nll + model.res_model.getRegulizer() if model.dr else (1-rho)*loss_fn(y_t_hat, y_t, 0.0) + rho*nll

            mape = utils.masked_mape(y_t_hat, y_t, 0.0).item()
            rmse = utils.masked_rmse(y_t_hat, y_t, 0.0).item()

            log_loss.append(loss.item())
            log_mape.append(mape)
            log_rmse.append(rmse)

    return np.mean(log_loss), np.mean(log_mape), np.mean(log_rmse)


def test_epoch(dataloader, model, scaler=None, n_sample=None):
    model.eval()
    with torch.no_grad():
        y_hat_list = []
        for batch, (x, y) in enumerate(dataloader):
            y_t_hat, _ = model(x, y)

            if n_sample is not None:
                eps = model.cov_model.sample(y_t_hat, n_sample=n_sample)
                y_t_hat = y_t_hat.unsqueeze(1).repeat(1, n_sample, 1, 1)
                y_t_hat += eps

            if scaler:
                y_t_hat = scaler.inverse_transform(y_t_hat)
                
            y_hat_list.append(y_t_hat)

    return torch.cat(y_hat_list, dim=0)

