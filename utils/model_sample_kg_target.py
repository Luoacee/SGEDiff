import argparse
import os
import shutil
import time
import sys
import importlib
import pickle

from utils.transforms import FeaturizeLigandAtom
from utils.transforms import FeaturizeProteinAtom
from utils.transforms import FeaturizeLigandBond
from utils.transforms import FeaturizeProteinBond
from utils.transforms import NormalizeVina
from utils.transforms import InteractionEx
from utils.transforms import ImportantEx
from datasets.DataProcessingMDB import DataCollate
from utils.external_data_process import TrainingCollate, pre_process

sys.path.append(os.path.abspath('./'))

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose
from torch_scatter import scatter_sum, scatter_mean
from tqdm.auto import tqdm

import models

import utils.misc as misc
import utils.transforms as trans
from datasets import get_dataset
# from datasets.pl_data import FOLLOW_BATCH
from models.TargetDiff.molopt_score_model import log_sample_categorical
from models.TargetDiff.molopt_score_model import ScorePosNet3D as Ta
from models.KGDiff.molopt_score_model import ScorePosNet3D as Kg
from models.CGDiff.molopt_score_model import ScorePosNet3D as Cg
from utils.evaluation import atom_num

FOLLOW_BATCH = ["ligand_atom_feature", "ligand_bond_feature",
                "protein_atom_feature", "protein_bond_feature",
                "imp_batch_idx",
                ]

exclude_keys = ["name", "p_idx", "l_idx", "dataset",
                "protein_atom_aromatic", "protein_atom_aromatic_h",
                "protein_atom_class", "protein_atom_is_backbone", "interaction",
                "protein_atom_res_name", "protein_bond_class", "ligand_atom_class",
                "ligand_atom_aromatic",
                "ligand_atom_hybridization", "ligand_bond_class"
                ]


def unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms):
    all_step_v = [[] for _ in range(n_data)]
    for v in ligand_v_traj:  # step_i
        v_array = v.cpu().numpy()
        for k in range(n_data):
            all_step_v[k].append(v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
    all_step_v = [np.stack(step_v) for step_v in all_step_v]  # num_samples * [num_steps, num_atoms_i]
    return all_step_v


def sample_diffusion_ligand(model, data, num_samples, batch_size=16, device='cuda',
                            num_steps=None, center_pos_mode='protein',
                            sample_num_atoms='prior', guide_mode='joint',
                            sampling_model = None ,
                            value_model=None,
                            type_grad_weight=1., pos_grad_weight=1.):
    all_pred_pos, all_pred_v, all_pred_exp = [], [], []
    all_pred_pos_traj, all_pred_v_traj,all_pred_exp_atom_traj = [], [], []
    all_pred_exp_traj = torch.tensor([])
    all_pred_v0_traj, all_pred_vt_traj = [], []
    time_list = []
    num_batch = int(np.ceil(num_samples / batch_size))
    current_i = 0
    for i in tqdm(range(num_batch)):
        n_data = batch_size if i < num_batch - 1 else num_samples - batch_size * (
                    num_batch - 1)
        batch = Batch.from_data_list([data.clone() for _ in range(n_data)], follow_batch=FOLLOW_BATCH).to(
            device)
        t1 = time.time()
        with torch.no_grad():
            batch_protein = batch.protein_atom_feature_batch
            if sample_num_atoms == 'prior':
                pocket_size = atom_num.get_space_size(batch.protein_atom_coords.detach().cpu().numpy())
                ligand_num_atoms = [atom_num.sample_atom_num(pocket_size).astype(int) for _ in
                                    range(n_data)]
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(
                    device)

            elif sample_num_atoms == 'range':
                ligand_num_atoms = list(range(current_i + 1, current_i + n_data + 1))
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(
                    device)
            elif sample_num_atoms == 'ref':
                batch_ligand = batch.ligand_element_batch
                ligand_num_atoms = scatter_sum(torch.ones_like(batch_ligand), batch_ligand,
                                               dim=0).tolist()
            else:
                raise ValueError

            center_pos = scatter_mean(batch.protein_atom_coords, batch_protein, dim=0)
            center_pos = batch.ligand_center
            batch_center_pos = center_pos[batch_ligand]

            init_ligand_pos = batch_center_pos + torch.randn_like(batch_center_pos)

            uniform_logits = torch.zeros(len(batch_ligand), model.num_classes).to(device)
            init_ligand_v_prob = log_sample_categorical(uniform_logits)
            init_ligand_v = init_ligand_v_prob.argmax(dim=-1)

            # -> 模型采样，模型得到采样后的预测值
            if sampling_model == "cg_diff":

                r = model.sample_diffusion(
                    guide_mode=guide_mode,
                    value_model=value_model,
                    type_grad_weight=type_grad_weight,
                    pos_grad_weight=pos_grad_weight,
                    protein_pos=batch.protein_atom_coords.float(),
                    protein_v=batch.protein_atom_feature.float(),
                    batch_protein=batch_protein,
                    imp_info= dict(
                        imp_pidx=[torch.tensor(i).to(init_ligand_pos.device) for i in batch.imp_batch_idx],
                        imp_act =[torch.tensor(i).to(init_ligand_pos.device) for i in batch.new_interaction],
                        imp_cls =[torch.tensor(i).to(init_ligand_pos.device) for i in batch.imp_bond_cls],
                        imp_ligand_mask=[torch.tensor(i).to(init_ligand_pos.device) for i in batch.imp_ligand_mask]
                    ),
                    init_ligand_pos=init_ligand_pos.float(),
                    init_ligand_v=init_ligand_v,
                    batch_ligand=batch_ligand,
                    num_steps=num_steps,
                    center_pos_mode=center_pos_mode
                )

            else:
                r = model.sample_diffusion(
                    guide_mode=guide_mode,
                    value_model=value_model,
                    type_grad_weight=type_grad_weight,
                    pos_grad_weight=pos_grad_weight,
                    protein_pos=batch.protein_atom_coords.float(),
                    protein_v=batch.protein_atom_feature.float(),
                    batch_protein=batch_protein,

                    init_ligand_pos=init_ligand_pos.float(),
                    init_ligand_v=init_ligand_v,
                    batch_ligand=batch_ligand,
                    num_steps=num_steps,
                    center_pos_mode=center_pos_mode
                )

            ligand_pos, ligand_v, ligand_pos_traj, ligand_v_traj = r['pos'], r['v'], r['pos_traj'], r['v_traj']
            ligand_v0_traj, ligand_vt_traj = r['v0_traj'], r['vt_traj']
            exp_traj = r['exp_traj']
            exp_atom_traj = r['exp_atom_traj']


            # unbatch pos
            ligand_cum_atoms = np.cumsum([0] + ligand_num_atoms)
            ligand_pos_array = ligand_pos.cpu().numpy().astype(np.float64)
            all_pred_pos += [ligand_pos_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in
                             range(n_data)]  # num_samples * [num_atoms_i, 3]
            # unbatch exp
            if guide_mode == 'joint' or guide_mode == 'pdbbind_random' or guide_mode == 'valuenet':
                all_pred_exp += exp_traj[-1]
                all_pred_exp_traj = torch.concat([all_pred_exp_traj, torch.stack(exp_traj, dim=0)],
                                                 dim=-1)

                all_step_exp_atom = unbatch_v_traj(exp_atom_traj, n_data, ligand_cum_atoms)
                all_pred_exp_atom_traj += [v for v in all_step_exp_atom]
            all_step_pos = [[] for _ in range(n_data)]
            for p in ligand_pos_traj:  # step_i
                p_array = p.cpu().numpy().astype(np.float64)
                for k in range(n_data):
                    all_step_pos[k].append(p_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
            all_step_pos = [np.stack(step_pos) for step_pos in
                            all_step_pos]  # num_samples * [num_steps, num_atoms_i, 3]
            all_pred_pos_traj += [p for p in all_step_pos]

            ligand_v_array = ligand_v.cpu().numpy()
            all_pred_v += [ligand_v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in range(n_data)]

            all_step_v = unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms)
            all_pred_v_traj += [v for v in all_step_v]
            all_step_v0 = unbatch_v_traj(ligand_v0_traj, n_data, ligand_cum_atoms)
            all_pred_v0_traj += [v for v in all_step_v0]
            all_step_vt = unbatch_v_traj(ligand_vt_traj, n_data, ligand_cum_atoms)
            all_pred_vt_traj += [v for v in all_step_vt]

        t2 = time.time()
        time_list.append(t2 - t1)
        current_i += n_data

    if guide_mode == 'joint' or guide_mode == 'pdbbind_random' or guide_mode == 'valuenet':
        all_pred_exp = torch.stack(all_pred_exp, dim=0).numpy()

    return all_pred_pos, all_pred_v, all_pred_exp, all_pred_pos_traj, all_pred_v_traj, all_pred_exp_traj, all_pred_v0_traj, all_pred_vt_traj, all_pred_exp_atom_traj, time_list


def main(protein_id, sampling_model):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/sampling.yml')
    parser.add_argument('-i', '--data_id', type=int, default=protein_id)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=6)
    # parser.add_argument('--guide_mode', type=str, default='joint',
    #                     choices=['joint', 'pdbbind_random', 'vina', 'valuenet', 'wo'])


    ###############
    parser.add_argument('--type_grad_weight', type=float, default=100)
    parser.add_argument('--pos_grad_weight', type=float, default=1)
    parser.add_argument('--result_path', type=str, default='model_results')

    args = parser.parse_args()
    sampling_model = sampling_model

    if sampling_model == "kg_diff_1":
        guide_mode = "joint"
        ScorePosNet3D = Kg
        # mdu = importlib.import_module("models.KGDiff.molopt_score_model", "molopt_score_model")
    elif sampling_model == "target_diff":
        guide_mode = "wo"
        ScorePosNet3D = Ta
        # mdu = importlib.import_module("models.TargetDiff", "molopt_score_model")
    elif sampling_model == "cg_diff":
        ScorePosNet3D = Cg
        guide_mode = "cross_graph"
    else:
        raise ValueError

    result_path = os.path.join(args.result_path, sampling_model)
    os.makedirs(result_path, exist_ok=True)  #
    shutil.copyfile(args.config, os.path.join(result_path, 'sample.yml'))
    logger = misc.get_logger('sampling', log_dir=result_path)

    # Load config
    config = misc.load_config(args.config)
    logger.info(config)
    misc.seed_all(config.sample.seed)


    ckpt = torch.load(config.model[sampling_model], map_location=args.device)

    # value模型存在use_classifier_guide: True
    logger.info(f"Training Config: {ckpt['config']}")
    logger.info(f"args: {args}")

    # Transforms
    ligand_atom_mode = ckpt['config'].data.transform.ligand_atom_mode

    transform = [
        FeaturizeLigandAtom("add_aromatic"),
        FeaturizeProteinAtom(),
        FeaturizeLigandBond(),
        FeaturizeProteinBond(),
        NormalizeVina(),
        InteractionEx(),
        ImportantEx()
    ]
    transform = Compose(transform)

    # Load dataset
    # dataset, subsets = get_dataset(  #
    #     config=ckpt['config'].data,
    #     transform=transform
    # )
    # if ckpt['config'].data.name == 'pl':
    #     test_set = subsets['test']
    # elif ckpt['config'].data.name == 'pdbbind':
    #     test_set = subsets['test']
    # else:
    #     raise ValueError
    # loading test ##############
    logger.info("Dataset Loading...")
    # train_iterator = torch.load("data/TargetDiff/training_dataloader.d", weights_only=False)
    with open("../DataCenter/target_real_sar3.pdata", "rb") as f:
        val_loader = pickle.load(f)
    # val_loader = torch.load("temp_test_save.pt", weights_only=False)
    valid_dataset = DataCollate(val_loader, transform=transform)
    logger.info(f'Test: {len(valid_dataset)}')

    # Load model
    model = ScorePosNet3D(
        ckpt['config'].model,
        protein_atom_feature_dim=FeaturizeProteinAtom().feature_dim,
        ligand_atom_feature_dim=FeaturizeLigandAtom(("add_aromatic")).feature_dim,
    ).to(args.device)

    model.load_state_dict(ckpt['model'])

    data = valid_dataset[args.data_id]

    # data = val_loader
    pred_pos, pred_v, pred_exp, pred_pos_traj, pred_v_traj, pred_exp_traj, pred_v0_traj, pred_vt_traj, pred_exp_atom_traj, time_list = sample_diffusion_ligand(
        model, data, config.sample.num_samples,
        batch_size=args.batch_size, device=args.device,
        num_steps=config.sample.num_steps,
        center_pos_mode=config.sample.center_pos_mode,
        sample_num_atoms=config.sample.sample_num_atoms,
        guide_mode=guide_mode,
        # value_model=value_model,
        value_model=None,
        sampling_model=sampling_model,
        type_grad_weight=args.type_grad_weight,
        pos_grad_weight=args.pos_grad_weight
    )

    result = {
        'data': data,
        'pred_ligand_pos': pred_pos,
        'pred_ligand_v': pred_v,
        'pred_exp': pred_exp,
        'pred_ligand_pos_traj': pred_pos_traj,  # ？
        'pred_ligand_v_traj': pred_v_traj,  # ？
        'pred_exp_traj': pred_exp_traj,  # ？
        'pred_exp_atom_traj': pred_exp_atom_traj,  # ？
        'time': time_list
    }
    logger.info('Sample done!')

    torch.save(result, os.path.join(result_path, f'result_{args.data_id}.pt'))

def process(protein_path):
    all_data = dict()
    for p_name in os.listdir(protein_path):
        protein_path = "%s/%s/%s_protein.pdb" % (protein_path, p_name, p_name)
        ligand_path = "%s/%s/%s_ligand.sdf" % (protein_path, p_name, p_name)
        collate_data = pre_process(protein_path, ligand_path)
        if isinstance(collate_data, list):
            continue
        collate_data["name"] = p_name
        all_data[p_name] = collate_data

    get_t = TrainingCollate()
    get_t.data_get(all_data)
    print("valid number %s / pass rate: %s" % (len(get_t.data), int(len(get_t.data)/len(os.listdir('TestProtein'))*100)))
    torch.save(get_t.data, "temp_test_save.pt")


if __name__ == '__main__':
    # process('TestProtein')
    for id in range(0, 16):
        main(id, "target_diff")
        main(id, "kg_diff")
