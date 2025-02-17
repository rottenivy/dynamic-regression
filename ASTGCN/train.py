import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import shutil
import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse
import time
import utils
from engine import *
from model import *
from pyro.ops.stats import crps_empirical
from torch.utils.tensorboard import SummaryWriter

# import wandb
# wandb.login()

parser = argparse.ArgumentParser()
parser.add_argument('--debug_mode',action='store_true',help='whether in debug mode')  # set store_true to disable
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data_path',type=str,default='../../data/traffic',help='data path')
parser.add_argument('--adj_path',type=str,default='../../data/traffic/sensor_graph/npy',help='data path')
parser.add_argument('--data',type=str,default='pemsd7m',help='data path')
parser.add_argument('--covariates',type=str,default='spd',help='data path')
####################### model parameters #######################
parser.add_argument('--nb_block',type=int,default=2,help='')
parser.add_argument('--K',type=int,default=3,help='')
parser.add_argument('--nb_chev_filter',type=int,default=64,help='')
parser.add_argument('--nb_time_filter',type=int,default=64,help='')
parser.add_argument('--time_strides',type=int,default=1,help='')
####################### model parameters #######################
parser.add_argument('--repeats',type=int,default=3,help='repeat experiments')
parser.add_argument('--out_dim',type=int,default=12,help='')
parser.add_argument('--f_step',type=int,default=12,help='')
parser.add_argument('--epochs',type=int,default=200,help='')
parser.add_argument('--patience',type=int,default=15,help='experiment note')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--seed',type=int,default=99,help='random seed')
parser.add_argument('--save',type=str,default='outputs',help='save path')
####################### dynamic regression parameters #######################
parser.add_argument('--dr',action='store_true',help='whether to add res regression')
parser.add_argument('--dr_init',type=str,default='diagonal',help='data path')
parser.add_argument('--rho',type=float,default=1.0,help='data path')
parser.add_argument('--loss_type',type=str,default='eye',help='data path')
parser.add_argument('--rank_n',type=int,default=None,help='')
parser.add_argument('--rank_q',type=int,default=None,help='')
parser.add_argument('--cov_scaling',type=float,default=1.0,help='data path')
parser.add_argument('--beta',type=float,default=1.0,help='data path')
####################### dynamic regression parameters #######################
args = parser.parse_args()

clip = 5 if args.loss_type == 'full' else None

####################### debug #######################
# args.epochs = 3
# args.data = "pemsd7m"
# args.covariates = "spd_delay12"
# args.data = "pems08_flow"
# args.covariates = "flow_delay2016"
# args.dr = True
# args.dr_init = "zeros"
# args.rho = 1.0
# args.loss_type = "full"
####################### debug #######################


#set seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
print(torch.__version__)
print('Using {} device'.format(device))

rank_n_disp = "full" if args.rank_n is None else args.rank_n
rank_q_disp = "full" if args.rank_q is None else args.rank_q

note = '%s_%s_dr_%s_drinit_%s_rho_%s_loss_%s_Rn_%s_Rq_%s_Cs_%s'%(args.data, args.covariates, args.dr, args.dr_init, args.rho, args.loss_type, rank_n_disp, rank_q_disp, args.cov_scaling)

if os.path.isdir(os.path.join(args.save, note)):
    shutil.rmtree(os.path.join(args.save, note), ignore_errors=True)
os.makedirs(os.path.join(args.save, note))

if os.path.isdir(os.path.join(args.save, note, 'best_model')):
    shutil.rmtree(os.path.join(args.save, note, 'best_model'), ignore_errors=True)
os.makedirs(os.path.join(args.save, note, 'best_model'))


def main():
    log_file = open(os.path.join(args.save, note, 'best_model', 'log'), 'w')

    debug_mode = args.debug_mode
    #load data
    training_data = utils.TrafficDataset(debug_mode, device=device, file_dir=os.path.join(args.data_path, args.data, args.covariates), split='train')
    scaler = training_data.__getScaler__()
    val_data = utils.TrafficDataset(debug_mode, device=device, file_dir=os.path.join(args.data_path, args.data, args.covariates), split='val', scaler=scaler)
    testing_data = utils.TrafficDataset(debug_mode, device=device, file_dir=os.path.join(args.data_path, args.data, args.covariates), split='test', scaler=scaler)

    if args.dr:
        in_channels, seq_len, num_nodes = training_data.in_dim-1, training_data.seq_len, training_data.num_nodes
    else:
        in_channels, seq_len, num_nodes = training_data.in_dim, training_data.seq_len, training_data.num_nodes

    if args.rank_n is None:
        args.rank_n = num_nodes
    if args.rank_q is None:
        args.rank_q = args.out_dim

    log = 'No. of features: %s, No. of nodes: %s'%(in_channels, num_nodes)
    utils.log_string(log_file, log)
    
    train_dataloader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(testing_data, batch_size=args.batch_size, shuffle=False)

    adj_mx = np.load(os.path.join(args.adj_path, 'adj_mx_%s.npy'%(args.data)))

    print(args)

    best_model_repeat_id = np.nan
    best_test_loss = np.inf
    mae, rmse, mape, mse, rrmse = [], [], [], [], []
    best_val_loss = []
    print_stp = [2, 5, 8, 11]
    for repeat in range(args.repeats):
        k1 = time.time()
        if os.path.isdir(os.path.join(args.save, note, 'repeat%s'%(repeat+1))):
            shutil.rmtree(os.path.join(args.save, note, 'repeat%s'%(repeat+1)), ignore_errors=True)
        os.makedirs(os.path.join(args.save, note, 'repeat%s'%(repeat+1)))
        utils.log_string(log_file, "Start exp.%s"%(repeat+1))

        save_best_model = utils.SaveBestModel(output_dir=os.path.join(args.save, note, 'repeat%s'%(repeat+1)), patience=args.patience)

        model = dynamic_reg(args, in_channels, adj_mx, seq_len, num_nodes, scaler).to(device)

        loss_fn = utils.masked_mse

        if args.loss_type == 'full':
            cov_fac_params = [p for name, p in model.named_parameters() if 'L' in name]
            others = [p for name, p in model.named_parameters() if 'L' not in name]
            grouped_parameters = [
                {"params": cov_fac_params, 'lr': args.learning_rate, 'weight_decay': args.weight_decay*100},  # TODO add mixture projecter to this
                {"params": others, 'lr': args.learning_rate, 'weight_decay': args.weight_decay}]

            optimizer = torch.optim.Adam(grouped_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)


        writer = SummaryWriter(os.path.join(args.save, note, 'repeat%s'%(repeat+1)), comment=note)
        utils.log_string(log_file, 'Start training')

        for e in range(args.epochs):
            t1 = time.time()
            mtrain_loss, mtrain_mape, mtrain_rmse = train_epoch(dataloader=train_dataloader, model=model, loss_fn=loss_fn, optimizer=optimizer, out_dim=args.out_dim, scaler=scaler, rho=args.rho, clip=clip)
            t2 = time.time()

            s1 = time.time()
            mvalid_loss, mvalid_mape, mvalid_rmse = eval_epoch(dataloader=val_dataloader, model=model, loss_fn=loss_fn, out_dim=args.out_dim, scaler=scaler, rho=args.rho)
            s2 = time.time()

            early_stop = save_best_model(mvalid_loss, e, model, optimizer, loss_fn)

            log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f} secs/epoch, Inference Time: {:.4f} secs'
            utils.log_string(log_file, log.format(e+1, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1), (s2-s1)))

            writer.add_scalars('Loss/train', {'train_loss':mtrain_loss, 'val_loss':mvalid_loss}, e)

            # if early_stop:
            #     if args.loss_type == 'full' and not model.cov_model.L1.requires_grad:
            #         model.cov_model.L1.requires_grad = True
            #         model.cov_model.L2.requires_grad = True
            #         model.cov_model.sigma.requires_grad = True
            #     else:
            #         break

            if early_stop:
                break

        writer.close()
        k2 = time.time()
        utils.log_string(log_file, 'Training finished, training time: %.1fs' % (k2 - k1))

        # load best model
        utils.log_string(log_file, 'Load best model')
        model.load_state_dict(torch.load(os.path.join(args.save, note, 'repeat%s'%(repeat+1), 'best_model.pth'))['model_state_dict'])

        utils.log_string(log_file, 'Start testing')
        y_test_hat = test_epoch(test_dataloader, model, scaler=scaler)
            
        utils.log_string(log_file, 'Performance in each prediction step:')
        utils.log_string(log_file, '\t\tMAE\t\tRMSE\t\tMAPE')

        amae, amape, armse, amse, arrmse = [], [], [], [], []
        for i in range(args.f_step):
            metrics = utils.metric(y_test_hat[:,i,:], testing_data.y[:,i,:,0].to(device))
            log = 'Step {:d}\t\t{:.2f}\t\t{:.2f}\t\t{:.2f}%\t\t{:.2f}\t\t{:.4f}'
            utils.log_string(log_file, log.format(i+1, metrics[0], metrics[2], metrics[1]*100, metrics[3], metrics[4]))
            amae.append(metrics[0])
            amape.append(metrics[1])
            armse.append(metrics[2])
            amse.append(metrics[3])
            arrmse.append(metrics[4])

        log = 'Average:\t{:.2f}\t\t{:.2f}\t\t{:.2f}%\t\t{:.2f}\t\t{:.4f}'
        utils.log_string(log_file, log.format(np.mean(amae), np.mean(armse), np.mean(amape)*100, np.mean(amse), np.mean(arrmse)))

        if save_best_model.best_valid_loss < best_test_loss:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': save_best_model.best_valid_loss,
                }, '%s/%s.pth'%(os.path.join(args.save, note, 'best_model'), 'best_model'))
            best_model_repeat_id = repeat + 1
            best_test_loss = save_best_model.best_valid_loss
        best_val_loss.append(save_best_model.best_valid_loss)

        mae.append(list(np.array(amae)[print_stp])+[np.mean(amae)])
        rmse.append(list(np.array(armse)[print_stp])+[np.mean(armse)])
        mape.append(list(np.array(amape)[print_stp])+[np.mean(amape)])
        mse.append(list(np.array(amse)[print_stp])+[np.mean(amse)])
        rrmse.append(list(np.array(arrmse)[print_stp])+[np.mean(arrmse)])

    mae, rmse, mape, mse, rrmse = np.array(mae), np.array(rmse), np.array(mape), np.array(mse), np.array(rrmse)
    m_mae, m_rmse, m_mape, m_mse, m_rrmse = mae.mean(0), rmse.mean(0), mape.mean(0), mse.mean(0), rrmse.mean(0)

    log = 'The best model is repeat %s'%(best_model_repeat_id)
    utils.log_string(log_file, log)

    log = 'The average val loss is %.4f'%(np.mean(best_val_loss))
    utils.log_string(log_file, log)

    log = 'Average MAE, RMSE, MAPE, MSE, RRMSE of Step 3, 6, 9, 12, and avg for %s runs:'%(args.repeats)
    utils.log_string(log_file, log)

    log = '{:.2f}\t{:.2f}\t{:.2f}%\t{:.2f}\t{:.4f}\t{:.2f}\t{:.2f}\t{:.2f}%\t{:.2f}\t{:.4f}\t{:.2f}\t{:.2f}\t{:.2f}%\t{:.2f}\t{:.4f}\t{:.2f}\t{:.2f}\t{:.2f}%\t{:.2f}\t{:.4f}\t{:.2f}\t{:.2f}\t{:.2f}%\t{:.2f}\t{:.4f}'.format(m_mae[0], m_rmse[0], m_mape[0]*100, m_mse[0], m_rrmse[0], m_mae[1], m_rmse[1], m_mape[1]*100, m_mse[1], m_rrmse[1], m_mae[2], m_rmse[2], m_mape[2]*100, m_mse[2], m_rrmse[2], m_mae[3], m_rmse[3], m_mape[3]*100, m_mse[3], m_rrmse[3], m_mae[4], m_rmse[4], m_mape[4]*100, m_mse[4], m_rrmse[4])
    utils.log_string(log_file, log)
    log_file.close()

    ################# probabilistic #################
    # log_file = open(os.path.join(args.save, note, 'best_model', 'test.txt'), 'w')

    # actuals = testing_data.y[...,0].to(device)

    # model.load_state_dict(torch.load(os.path.join(args.save, note, 'repeat%s'%(repeat+1), 'best_model.pth'), map_location="cuda")['model_state_dict'])
    # y_test_hat = test_epoch(test_dataloader, model, n_sample=100, scaler=args.scaler)  # B, n_sample, Q, N

    # q_loss = utils.quantile_loss()
    # print_metrics = []
    # print_stp = [2, 5, 8, 11]
    # for repeat in range(args.repeats):
    #     utils.log_string(log_file, "Load Model %s"%(repeat+1))
    #     utils.log_string(log_file, 'Performance in each prediction step:')
    #     utils.log_string(log_file, '\tCRPS_mean\tCRPS_sum')

    #     y_test_hat = test_epoch(test_dataloader, model, n_sample=100, scaler=args.scaler)  # B, n_sample, Q, N

    #     metrics = []
    #     for i in range(args.f_step):
    #         crps = crps_empirical(y_test_hat[..., i, :].permute(1, 0, 2), actuals[:, i])
    #         crps_mean = crps.mean().item()
    #         crps_sum = (crps.sum()/actuals[:, i].sum()).item()

    #         q_losses = q_loss.loss(y_test_hat[..., i, :].permute(1, 0, 2), actuals[:, i])
            
    #         risks = []
    #         for j in range(q_losses.shape[-1]):
    #             risks.append(q_losses[..., j].mean().item())

    #         log = 'Step {:d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
    #         utils.log_string(log_file, log.format(i+1, crps_mean, crps_sum, *risks))

    #         metrics.append([crps_mean, crps_sum] + risks)

    #     metrics = np.array(metrics)

    #     log = 'Average:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
    #     utils.log_string(log_file, log.format(*metrics.mean(0)))

    #     print_metrics.append(np.concatenate([metrics[print_stp], np.mean(metrics, axis=0, keepdims=True)]))

    # print_metrics = np.array(print_metrics).mean(0)

    # log = 'Average CRPS_mean, CRPS_sum, q_risks of Step 3, 6, 9, 12, and avg for %s runs:'%(args.repeats)
    # utils.log_string(log_file, log)

    # for i in range(print_metrics.shape[0]):
    #     try:
    #         log = 'Step {:d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
    #         utils.log_string(log_file, log.format(print_stp[i]+1, *print_metrics[i]))
    #     except:
    #         log = 'Avg: \t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
    #         utils.log_string(log_file, log.format(*print_metrics[i]))

    # log_file.close()

    return best_val_loss


if __name__ == "__main__":
    # wandb.init(config=args)
    t1 = time.time()
    score = main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
    # wandb.log({'val_loss': score})
    # wandb.finish()
