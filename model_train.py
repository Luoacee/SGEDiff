import os.path

from easydict import EasyDict
from utils.transforms import FeaturizeLigandAtom
from utils.transforms import FeaturizeProteinAtom
from utils.transforms import FeaturizeLigandBond
from utils.transforms import FeaturizeProteinBond
from utils.transforms import NormalizeVina
from utils.transforms import InteractionEx
from utils.transforms import ImportantEx
from utils.transforms import NormlizeProperty
from utils.data_tools import DataCollate
import pickle
import numpy as np
import pandas as pd
import yaml

from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
# from utils.check import backward_hook
import torch.utils.tensorboard
from torch_geometric.transforms import Compose
from torch.nn.utils import clip_grad_norm_

import random
import torch
from utils import misc
import utils.transforms as trans
from utils.train import get_optimizer
from utils.train import get_scheduler
from utils.train import inf_iterator
from tqdm import tqdm
from scipy import stats
from sklearn.metrics import roc_auc_score
import seaborn as sns
import time
from functools import partial
import sys

# models loading
from models.SGEDiff.molopt_score_model import ScorePosNet3D


class ARGS:
    def __init__(self):
        self.ckpt = "1000001.pt"
        self.logdir = "./logs_diffusion"
        self.config = "./configs/training.yml"
        self.tag = ""
        self.cuda_id = 0
        self.value_only = False
        self.train_report_iter = 200
        self.start_epoch = 0


def load_config(path):
    with open(path, 'r') as f:
        return EasyDict(yaml.safe_load(f))


exclude_keys = ["name", "p_idx", "l_idx", "dataset",
                "protein_atom_aromatic", "protein_atom_aromatic_h",
                "protein_atom_class", "protein_atom_is_backbone", "interaction",
                "protein_atom_res_name", "protein_bond_class", "ligand_atom_class",
                "ligand_atom_aromatic",
                "ligand_atom_hybridization", "ligand_bond_class"
                ]

FOLLOW_BATCH = ["ligand_atom_feature", "ligand_bond_feature",
                "protein_atom_feature", "protein_bond_feature",
                "imp_batch_idx",
                ]


def main():
    config = load_config("configs/training.yml")
    args = ARGS()
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    log_dir = misc.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = misc.get_logger('train', log_dir)
    logger.info(args)
    logger.info(config)
    transform = [
        FeaturizeLigandAtom("add_aromatic"),
        FeaturizeProteinAtom(),
        FeaturizeLigandBond(),
        FeaturizeProteinBond(),
        NormalizeVina(),
        InteractionEx(),
        ImportantEx(),
        NormlizeProperty()
    ]

    transform = Compose(transform)
    single_model(args.cuda_id,  config, args, transform,
                                                    log_dir, ckpt_dir, logger)


def single_model(rank, config, args, transform, log_dir, ckpt_dir, logger):
    device = "cuda:%s" % rank if torch.cuda.is_available() else "cpu"

    # logger
    misc.seed_all(config.train.seed)

    train_path = "data/crossdock_train.pdata"
    valid_path = "data/crossdock_valid.pdata"

    with open(train_path, "rb") as f:
        train_data = pickle.load(f)

    with open(valid_path, "rb") as f:
        valid_data = pickle.load(f)

    train_dataset = DataCollate(train_data, transform=transform)
    valid_dataset = DataCollate(valid_data, transform=transform)

    train_iterator = DataLoader(train_dataset, batch_size=8, follow_batch=FOLLOW_BATCH,
                                exclude_keys=exclude_keys, shuffle=True)
    train_iterator = inf_iterator(train_iterator)

    val_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, follow_batch=FOLLOW_BATCH,
                            exclude_keys=exclude_keys)

    model = ScorePosNet3D(
        config.model,
        protein_atom_feature_dim=FeaturizeProteinAtom().feature_dim,
        ligand_atom_feature_dim=FeaturizeLigandAtom("add_aromatic").feature_dim
    ).to(device)

    logger.info("Building models %s..." % rank)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)

    if args.ckpt != "":
        ckpt_state = torch.load(args.ckpt, map_location="cuda:%d" % rank, weights_only=False)
        model.load_state_dict(ckpt_state['model'])

    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)

    if args.ckpt != "":
        scheduler.load_state_dict(ckpt_state['scheduler'])


    start_it = args.start_epoch

    def train(it):

        model.train()
        optimizer.zero_grad()
        random_box = list(range(100))
        for _ in range(config.train.n_acc_batch):
            batch = next(train_iterator).to(device)
            protein_noise = torch.randn_like(batch.protein_atom_coords) * config.train.pos_noise_std
            gt_protein_pos = batch.protein_atom_coords + protein_noise

            random.shuffle(random_box)
            if random_box[0] < 30:
                results = model.get_diffusion_loss(
                    protein_pos=gt_protein_pos.float(),
                    protein_v=batch.protein_atom_feature.float(),
                    affinity=batch.score.float(),
                    batch_protein=batch.protein_atom_feature_batch,
                    subgraph_info=None,
                    ligand_pos=batch.ligand_atom_coords.float(),
                    ligand_v=batch.ligand_atom_feature,
                    batch_ligand=batch.ligand_atom_feature_batch,
                )
            else:
                results = model.get_diffusion_loss(
                    protein_pos=gt_protein_pos.float(),
                    protein_v=batch.protein_atom_feature.float(),
                    affinity=batch.score.float(),
                    batch_protein=batch.protein_atom_feature_batch,
                    subgraph_info=dict(
                        subgraph_pidx=[torch.tensor(i).to(device) for i in batch.imp_batch_idx],
                        subgraph_act =[torch.tensor(i).to(device) for i in batch.new_interaction],
                        subgraph_cls =[torch.tensor(i).to(device) for i in batch.imp_bond_cls],
                        subgraph_ligand_mask = [torch.tensor(i).to(device) for i in batch.imp_ligand_mask]
                    ),
                    ligand_pos=batch.ligand_atom_coords.float(),
                    ligand_v=batch.ligand_atom_feature,
                    batch_ligand=batch.ligand_atom_feature_batch,
                )

            if args.value_only:
                results['loss'] = results['loss_exp']

            loss, loss_pos, loss_v, loss_exp = results['loss'], results['loss_pos'], results['loss_v'], results[
                'loss_exp']
            loss = loss / config.train.n_acc_batch
            loss.backward()


        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)  # 模型梯度norm

        optimizer.step()  # 反向传播

        if it % args.train_report_iter == 0:
            logger.info(
                '[Train] Iter %d | Loss %.6f (pos %.6f | v %.6f | exp %.6f) | Lr: %.6f | Grad Norm: %.6f' % (
                    it, loss, loss_pos, loss_v, loss_exp, optimizer.param_groups[0]['lr'], orig_grad_norm  # 修改
                )
            )

            print('Time: %s GPU: %s [Train] Iter %d | Loss %.6f (pos %.6f | v %.6f | exp %.6f) | Lr: %.6f | Grad Norm: %.6f' % (
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), rank,
                    it, loss, loss_pos, loss_v, loss_exp, optimizer.param_groups[0]['lr'], orig_grad_norm  # 修改
                ))

            for k, v in results.items():
                if torch.is_tensor(v) and v.squeeze().ndim == 0:
                    writer.add_scalar(f'train/{k}', v, it)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
            writer.add_scalar('train/grad', orig_grad_norm, it)
            writer.flush()

    def validate(it):
        # fix time steps
        sum_loss, sum_loss_pos, sum_loss_v, sum_loss_exp, sum_n = 0, 0, 0, 0, 0
        all_pred_v, all_true_v, all_pred_exp, all_true_exp = [], [], [], []
        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_loader, desc='Validate'):
                batch = batch.to(device)
                batch_size = batch.num_graphs
                for t in np.linspace(0, model.num_timesteps - 1, 10).astype(int):
                    time_step = torch.tensor([t] * batch_size).to(device)

                    results = model.get_diffusion_loss(
                        protein_pos=batch.protein_atom_coords.float(),
                        protein_v=batch.protein_atom_feature.float(),
                        affinity=batch.score.float(),
                        batch_protein=batch.protein_atom_feature_batch,

                        # imp_info=None,
                        subgraph_info=dict(
                            subgraph_pidx=[torch.tensor(i).to(device) for i in batch.imp_batch_idx],
                            subgraph_act=[torch.tensor(i).to(device) for i in batch.new_interaction],
                            subgraph_cls=[torch.tensor(i).to(device) for i in batch.imp_bond_cls],
                            subgraph_ligand_mask=[torch.tensor(i).to(device) for i in batch.imp_ligand_mask]
                        ),

                        ligand_pos=batch.ligand_atom_coords.float(),
                        ligand_v=batch.ligand_atom_feature,  #
                        batch_ligand=batch.ligand_atom_feature_batch,
                        time_step=time_step
                    )
                    loss, loss_pos, loss_v, loss_exp, pred_exp = results['loss'], results['loss_pos'], results[
                        'loss_v'], results['loss_exp'], results['pred_exp']

                    sum_loss += float(loss) * batch_size
                    sum_loss_pos += float(loss_pos) * batch_size
                    sum_loss_v += float(loss_v) * batch_size
                    sum_loss_exp += float(loss_exp) * batch_size
                    sum_n += batch_size
                    all_pred_v.append(results['ligand_v_recon'].detach().cpu().numpy())
                    all_true_v.append(batch.ligand_atom_feature.detach().cpu().numpy())
                    all_pred_exp.append(pred_exp.detach().cpu().numpy())
                    all_true_exp.append(batch.score.float().detach().cpu().numpy())

        avg_loss = sum_loss / sum_n
        avg_loss_pos = sum_loss_pos / sum_n
        avg_loss_v = sum_loss_v / sum_n
        avg_loss_exp = sum_loss_exp / sum_n
        atom_auroc = get_auroc(np.concatenate(all_true_v), np.concatenate(all_pred_v, axis=0),
                               feat_mode=config.data.transform.ligand_atom_mode)

        exp_pearsonr = get_pearsonr(np.concatenate(all_true_exp, axis=0), np.concatenate(all_pred_exp, axis=0))

        if config.train.scheduler.type == 'plateau':
            scheduler.step(avg_loss)
        elif config.train.scheduler.type == 'warmup_plateau':
            scheduler.step_ReduceLROnPlateau(avg_loss)
        else:
            scheduler.step()

        logger.info(
            '[Validate] Iter %05d | Loss %.6f | Loss pos %.6f | Loss v %.6f e-3 | Loss exp %.6f e-3 | Avg atom auroc %.6f' % (
                it, avg_loss, avg_loss_pos, avg_loss_v * 1000, avg_loss_exp * 1000, atom_auroc
            )
        )
        print('Time: %s GPU: %s [Validate] Iter %05d | Loss %.6f | Loss pos %.6f | Loss v %.6f e-3 | Loss exp %.6f e-3 | Avg atom auroc %.6f' % (
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), rank,
                it, avg_loss, avg_loss_pos, avg_loss_v * 1000, avg_loss_exp * 1000, atom_auroc))
        writer.add_scalar('val/loss', avg_loss, it)
        writer.add_scalar('val/loss_pos', avg_loss_pos, it)
        writer.add_scalar('val/loss_v', avg_loss_v, it)
        writer.add_scalar('val/loss_exp', avg_loss_exp, it)
        writer.add_scalar('val/atom_auroc', atom_auroc, it)
        writer.add_scalar('val/pcc', exp_pearsonr[0], it)
        writer.add_scalar('val/pvalue', exp_pearsonr[1], it)
        # fig = plt.figure(figsize=(12,12))

        writer.add_figure('val/pcc_fig', sns.lmplot(data=pd.DataFrame({
            'pred': np.concatenate(all_pred_exp, axis=0),
            'true': np.concatenate(all_true_exp, axis=0)
        }), x='pred', y='true').set(title='pcc %.6f | pvalue %.6f' % (exp_pearsonr[0], exp_pearsonr[1])).fig, it)
        writer.flush()

        if args.value_only:
            return avg_loss_exp

        return avg_loss

    # def test(it):
    #     # fix time steps
    #     sum_loss, sum_loss_pos, sum_loss_v, sum_loss_exp, sum_n = 0, 0, 0, 0, 0
    #     all_pred_v, all_true_v, all_pred_exp, all_true_exp = [], [], [], []
    #     with torch.no_grad():
    #         model.eval()
    #         for batch in tqdm(test_loader, desc='Test'):
    #             batch = batch.to(device)
    #             batch_size = batch.num_graphs
    #             for t in np.linspace(0, model.num_timesteps - 1, 10).astype(int):
    #                 time_step = torch.tensor([t] * batch_size).to(device)
    #                 results = model.get_diffusion_loss(
    #                     protein_pos=batch.protein_atom_coords.float(),
    #                     protein_v=batch.protein_atom_feature.float(),
    #                     affinity=batch.score.float(),
    #                     batch_protein=batch.protein_atom_feature_batch,
    #
    #                     ligand_pos=batch.ligand_atom_coords.float(),
    #                     ligand_v=batch.ligand_atom_feature,
    #                     batch_ligand=batch.ligand_atom_feature_batch,
    #                     time_step=time_step
    #                 )
    #                 loss, loss_pos, loss_v, loss_exp, pred_exp = results['loss'], results['loss_pos'], results[
    #                     'loss_v'], results['loss_exp'], results['pred_exp']
    #
    #                 sum_loss += float(loss) * batch_size
    #                 sum_loss_pos += float(loss_pos) * batch_size
    #                 sum_loss_v += float(loss_v) * batch_size
    #                 sum_loss_exp += float(loss_exp) * batch_size
    #                 sum_n += batch_size
    #                 all_pred_v.append(results['ligand_v_recon'].detach().cpu().numpy())
    #                 all_true_v.append(batch.ligand_atom_feature.detach().cpu().numpy())
    #                 all_pred_exp.append(pred_exp.detach().cpu().numpy())
    #                 all_true_exp.append(batch.score.float().detach().cpu().numpy())
    #
    #     avg_loss = sum_loss / sum_n
    #     avg_loss_pos = sum_loss_pos / sum_n
    #     avg_loss_v = sum_loss_v / sum_n
    #     avg_loss_exp = sum_loss_exp / sum_n
    #     atom_auroc = get_auroc(np.concatenate(all_true_v), np.concatenate(all_pred_v, axis=0),
    #                            feat_mode=config.data.transform.ligand_atom_mode)
    #
    #     exp_pearsonr = get_pearsonr(np.concatenate(all_true_exp, axis=0), np.concatenate(all_pred_exp, axis=0))
    #
    #     logger.info(
    #         '[Test] Iter %05d | Loss %.6f | Loss pos %.6f | Loss v %.6f e-3 | Loss exp %.6f e-3 | Avg atom auroc %.6f' % (
    #             it, avg_loss, avg_loss_pos, avg_loss_v * 1000, avg_loss_exp * 1000, atom_auroc
    #         )
    #     )
    #     writer.add_scalar('test/loss', avg_loss, it)
    #     writer.add_scalar('test/loss_pos', avg_loss_pos, it)
    #     writer.add_scalar('test/loss_v', avg_loss_v, it)
    #     writer.add_scalar('test/loss_exp', avg_loss_exp, it)
    #     writer.add_scalar('test/atom_auroc', atom_auroc, it)
    #     writer.add_scalar('test/pcc', exp_pearsonr[0], it)
    #     writer.add_scalar('test/pvalue', exp_pearsonr[1], it)
    #     # fig = plt.figure(figsize=(12,12))
    #
    #     writer.add_figure('test/pcc_fig', sns.lmplot(data=pd.DataFrame({
    #         'pred': np.concatenate(all_pred_exp, axis=0),
    #         'true': np.concatenate(all_true_exp, axis=0)
    #     }), x='pred', y='true').set(title='pcc %.6f | pvalue %.6f' % (exp_pearsonr[0], exp_pearsonr[1])).fig, it)
    #     writer.flush()
    #
    #     if args.value_only:
    #         return avg_loss_exp
    #
    #     return avg_loss

    best_loss, best_iter = None, None
    for it in range(start_it, config.train.max_iters):
        train(it)
        if (rank == 0) and (it % config.train.val_freq == 0 or it == config.train.max_iters):
            val_loss = validate(it)
            # if config.data.name == 'pdbbind':
            #     _ = test(it)
            if best_loss is None or val_loss < best_loss:
                logger.info(f'[Validate] Best val loss achieved: {val_loss:.6f}')
                best_loss, best_iter = val_loss, it
                ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                torch.save({
                    'config': config,
                    'models': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iteration': it,
                }, ckpt_path)
            else:
                logger.info(f'[Validate] Val loss is not improved. '
                            f'Best val loss: {best_loss:.6f} at iter {best_iter}')

def get_auroc(y_true, y_pred, feat_mode):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    avg_auroc = 0.
    possible_classes = set(y_true)
    for c in possible_classes:
        auroc = roc_auc_score(y_true == c, y_pred[:, c])
        avg_auroc += auroc * np.sum(y_true == c)
        mapping = {
            'basic': trans.MAP_INDEX_TO_ATOM_TYPE_ONLY,
            'add_aromatic': trans.MAP_INDEX_TO_ATOM_TYPE_AROMATIC,
            'full': trans.MAP_INDEX_TO_ATOM_TYPE_FULL
        }
        print(f'atom: {mapping[feat_mode][c]} \t auc roc: {auroc:.4f}')
    return avg_auroc / len(y_true)


def get_pearsonr(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return stats.pearsonr(y_true, y_pred)





if __name__ == "__main__":
    main()

