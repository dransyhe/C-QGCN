from __future__ import division
from __future__ import print_function

import datetime
import json
import logging
import os
import pickle
import time

import numpy as np
import optimizers
import torch
from config import parser
from models.base_models import NCModel, LPModel, MDModel
from utils.data_utils import load_data, sparse_mx_to_torch_sparse_tensor
from utils.train_utils import get_dir_name, format_metrics
from utils.distortions import compute_f_score

# torch.autograd.set_detect_anomaly(True)

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if int(args.double_precision):
        torch.set_default_dtype(torch.float64)
    if int(args.cuda) >= 0:
        torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    args.patience = args.epochs if not args.patience else  int(args.patience)
    logging.getLogger().setLevel(logging.INFO)
    if args.save:
        if not args.save_dir:
            dt = datetime.datetime.now()
            date = f"{dt.year}_{dt.month}_{dt.day}"
            # models_dir = os.path.join(os.environ['LOG_DIR'], args.task, date)
            models_dir = os.path.join(os.getcwd() + "/logs/", args.task, date)
            save_dir = get_dir_name(models_dir)
        else:
            save_dir = args.save_dir
        logging.basicConfig(level=logging.INFO,
                            handlers=[
                                logging.FileHandler(os.path.join(save_dir, 'log.txt')),
                                logging.StreamHandler()
                            ])

    logging.info(f'Using: {args.device}')
    logging.info("Using seed {}.".format(args.seed))

    # Load data
    # data = load_data(args, os.path.join(os.environ['DATAPATH'], args.dataset))
    data = load_data(args, os.path.join(os.getcwd() + "/data/", args.dataset))
    args.n_nodes, args.feat_dim = data['features'].shape
    if args.task == 'nc':
        Model = NCModel
        args.n_classes = int(data['labels'].max() + 1)
        logging.info(f'Num classes: {args.n_classes}')
    elif args.task == 'md':
        Model = MDModel
        args.eval_freq = args.epochs + 1
    else:
        args.nb_false_edges = len(data['train_edges_false'])
        args.nb_edges = len(data['train_edges'])
        if args.task == 'lp':
            Model = LPModel
        else:
            Model = RECModel
            # No validation for reconstruction task
            args.eval_freq = args.epochs + 1

    if not args.lr_reduce_freq:
        args.lr_reduce_freq = args.epochs

    # Model and optimizer
    model = Model(args)
    logging.info(str(model))
    # print(model.parameters())
    curvature_lr = 1e-4
    if args.model == 'HGCN' and args.task == 'md':
        if args.curv_aware:
            pararms = [{'params': model.encoder.layers[0].linear.weight},
                       {'params': model.encoder.layers[0].linear.bias},
                       {'params': model.encoder.layers[0].c_in, 'lr':curvature_lr},
                       {'params': model.encoder.layers[0].c_out, 'lr':curvature_lr},
                       {'params': model.encoder.r_map.weight},
                       {'params': model.encoder.r_map.bias},
                       {'params': model.decoder.curv_alpha}]
        else:
            pararms = [{'params': model.encoder.layers[0].linear.weight},
                       {'params': model.encoder.layers[0].linear.bias},
                       {'params': model.encoder.layers[0].c_in, 'lr': curvature_lr},
                       {'params': model.encoder.layers[0].c_out, 'lr': curvature_lr}]
        optimizer = getattr(optimizers, args.optimizer)(pararms, lr=args.lr,weight_decay=args.weight_decay)
    else:
        optimizer = getattr(optimizers, args.optimizer)(params=model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    # for g in optimizer

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(args.lr_reduce_freq),
        gamma=float(args.gamma)
    )
    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    logging.info(f"Total number of parameters: {tot_params}")
    if args.cuda is not None and int(args.cuda) >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
        model = model.to(args.device)
        for x, val in data.items():
            if torch.is_tensor(data[x]):
                data[x] = data[x].to(args.device)

    # pre-compute f_scores
    data_path = os.path.join(os.getcwd() + "/data/", args.dataset)
    filename = os.path.join(data_path, args.dataset + '_f_score.p')
    if not args.load_f:
        data['f_score'] = compute_f_score(data['adj_train'])
        pickle.dump(data['f_score'], open(filename, 'wb'))
    else:
        data['f_score'] = pickle.load(open(filename, 'rb'))
    if args.cuda > -1:
        data['f_score'] = data['f_score'].to(args.cuda)

    # Train model
    t_total = time.time()
    counter = 0
    best_val_metrics = model.init_metric_dict()
    best_test_metrics = None
    best_emb = None
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        embeddings = model.encode(data['features'], data['adj_train_norm'])
        train_metrics = model.compute_metrics(embeddings, data, 'train')
        train_metrics['loss'].backward()
        if args.grad_clip is not None:
            max_norm = float(args.grad_clip)
            all_params = list(model.parameters())
            for param in all_params:
                torch.nn.utils.clip_grad_norm_(param, max_norm)
        optimizer.step()
        lr_scheduler.step()
        if (epoch + 1) % args.log_freq == 0:
            logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1),
                                   'lr: {}'.format(lr_scheduler.get_lr()[0]),
                                   format_metrics(train_metrics, 'train'),
                                   'time: {:.4f}s'.format(time.time() - t)
                                   ]))
        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            embeddings = model.encode(data['features'], data['adj_train_norm'])
            val_metrics = model.compute_metrics(embeddings, data, 'val')
            if (epoch + 1) % args.log_freq == 0:
                logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1), format_metrics(val_metrics, 'val')]))
            if model.has_improved(best_val_metrics, val_metrics):
                best_test_metrics = model.compute_metrics(embeddings, data, 'test')
                best_emb = embeddings.cpu()
                if args.save:
                    np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.detach().numpy())
                best_val_metrics = val_metrics
                counter = 0
            else:
                counter += 1
                if counter == args.patience and epoch > args.min_epochs:
                    logging.info("Early stopping")
                    break

    logging.info("Optimization Finished!")
    logging.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    if not best_test_metrics:
        model.eval()
        best_emb = model.encode(data['features'], data['adj_train_norm'])
        best_test_metrics = model.compute_metrics(best_emb, data, 'test')
    logging.info(" ".join(["Val set results:", format_metrics(best_val_metrics, 'val')]))
    logging.info(" ".join(["Test set results:", format_metrics(best_test_metrics, 'test')]))
    if args.save:
        np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.cpu().detach().numpy())
        if hasattr(model.encoder, 'att_adj'):
            filename = os.path.join(save_dir, args.dataset + '_att_adj.p')
            pickle.dump(model.encoder.att_adj.cpu().to_dense(), open(filename, 'wb'))
            print('Dumped attention adj: ' + filename)

        json.dump(vars(args), open(os.path.join(save_dir, 'config.json'), 'w'))
        torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
        logging.info(f"Saved model in {save_dir}")

if __name__ == '__main__':
    # import pdb; pdb.set_trace()
    args = parser.parse_args()
    train(args)
