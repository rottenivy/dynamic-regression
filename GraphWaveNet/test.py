import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse
import time
import utils
from engine import *
from model import *
from pyro.ops.stats import crps_empirical

parser = argparse.ArgumentParser()
parser.add_argument('--debug_mode',action='store_true',help='whether in debug mode')  # set store_true to disable
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data_path',type=str,default='../../data/traffic',help='data path')
parser.add_argument('--adj_path',type=str,default='../../data/traffic/sensor_graph/npy',help='data path')
parser.add_argument('--data',type=str,default='pemsd7m',help='data path')
parser.add_argument('--covariates',type=str,default='spd',help='data path')

parser.add_argument('--repeats',type=int,default=3,help='repeat experiments')
parser.add_argument('--out_dim',type=int,default=12,help='')
parser.add_argument('--f_step',type=int,default=12,help='')
parser.add_argument('--epochs',type=int,default=200,help='')
parser.add_argument('--patience',type=int,default=15,help='experiment note')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
parser.add_argument('--gcn_bool',action='store_false',help='whether to add graph convolution layer')
parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
parser.add_argument('--addaptadj',action='store_false',help='whether add adaptive adj')
parser.add_argument('--randomadj',action='store_false',help='whether random initialize adaptive adj')
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
parser.add_argument('--data_split',type=str,default='test',help='data path')
parser.add_argument('--n_sample',type=int,default=100,help='')
args = parser.parse_args()


####################### debug #######################
# args.data = "pems08_flow"
# args.covariates = "flow_delay2016"
# args.dr = True
# args.dr_init = "diagonal"
# args.rho = 1.0
# args.loss_type = "full"
# args.cov_scaling = 0.001
# args.data_split = "test"
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

def main():
    log_file = open(os.path.join(args.save, note, 'best_model', '%s.txt'%(args.data_split)), 'w')

    debug_mode = args.debug_mode
    # load data
    training_data = utils.TrafficDataset(debug_mode, device=device, file_dir=os.path.join(args.data_path, args.data, args.covariates), split='train')
    scaler = training_data.__getScaler__()
    testing_data = utils.TrafficDataset(debug_mode, device=device, file_dir=os.path.join(args.data_path, args.data, args.covariates), split=args.data_split, scaler=scaler)

    if args.dr:
        in_dim, seq_len, num_nodes = training_data.in_dim-1, training_data.seq_len, training_data.num_nodes
    else:
        in_dim, seq_len, num_nodes = training_data.in_dim, training_data.seq_len, training_data.num_nodes

    if args.rank_n is None:
        args.rank_n = num_nodes
    if args.rank_q is None:
        args.rank_q = args.out_dim
        
    log = 'No. of features: %s, No. of nodes: %s'%(in_dim, num_nodes)
    utils.log_string(log_file, log)
    
    test_dataloader = DataLoader(testing_data, batch_size=args.batch_size, shuffle=False)

    actuals = testing_data.y[...,0].to(device)

    supports = None if args.aptonly else [torch.tensor(i).to(device) for i in utils.load_adj(debug_mode, os.path.join(args.adj_path, 'adj_mx_%s.npy'%(args.data)), args.adjtype)]
    adjinit = None if args.randomadj else supports[0]

    model = dynamic_reg(device=device, dr=args.dr, dr_init=args.dr_init, rho=args.rho, loss_type=args.loss_type, rank_n=args.rank_n, rank_q=args.rank_q, cov_scaling=args.cov_scaling, beta=args.beta, num_nodes=num_nodes, seq_length=seq_len, in_dim=in_dim, dropout=args.dropout, supports=supports, gcn_bool=args.gcn_bool, addaptadj=args.addaptadj, adjinit=adjinit, nhid=args.nhid, scaler=scaler).to(device)

    q_loss = utils.quantile_loss()
    print_metrics = []
    print_stp = [2, 5, 8, 11]
    for repeat in range(args.repeats):
        utils.log_string(log_file, "Load Model %s"%(repeat+1))
        utils.log_string(log_file, 'Performance in each prediction step:')
        utils.log_string(log_file, '\tRRMSE\tCRPS_mean\tCRPS_sum\tq-0.1\tq-0.25\tq-0.5\tq-0.75\tq-0.9')

        model.load_state_dict(torch.load(os.path.join(args.save, note, 'repeat%s'%(repeat+1), 'best_model.pth'), map_location=device)['model_state_dict'])
        
        y_test_hat = test_epoch(test_dataloader, model, n_sample=args.n_sample, scaler=scaler)  # B, n_sample, Q, N

        metrics = []
        for i in range(args.f_step):
            rrmse = utils.masked_rrmse(y_test_hat.mean(1)[:, i], actuals[:, i]).item()

            crps = crps_empirical(y_test_hat[..., i, :].permute(1, 0, 2), actuals[:, i])
            crps_mean = crps.mean().item()
            crps_sum = (crps.sum()/actuals[:, i].sum()).item()

            q_losses = q_loss.loss(y_test_hat[..., i, :].permute(1, 0, 2), actuals[:, i])
            
            risks = []
            for j in range(q_losses.shape[-1]):
                risks.append((q_losses[..., j].sum()/actuals[:, i].sum()).item())

            log = 'Step {:d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
            utils.log_string(log_file, log.format(i+1, rrmse, crps_mean, crps_sum, *risks))

            metrics.append([rrmse, crps_mean, crps_sum] + risks)

        metrics = np.array(metrics)

        log = 'Average:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
        utils.log_string(log_file, log.format(*metrics.mean(0)))

        print_metrics.append(np.concatenate([metrics[print_stp], np.mean(metrics, axis=0, keepdims=True)]))

    print_metrics = np.array(print_metrics).mean(0)

    log = 'Average RRMSE, CRPS_mean, CRPS_sum, p_risks of Step 3, 6, 9, 12, and avg for %s runs:'%(args.repeats)
    utils.log_string(log_file, log)

    for i in range(print_metrics.shape[0]):
        try:
            log = 'Step {:d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
            utils.log_string(log_file, log.format(print_stp[i]+1, *print_metrics[i]))
        except:
            log = 'Avg: \t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
            utils.log_string(log_file, log.format(*print_metrics[i]))

    log_file.close()


if __name__ == "__main__":
    t1 = time.time()
    score = main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
