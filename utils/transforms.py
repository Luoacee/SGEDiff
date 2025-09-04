import torch
import torch.nn.functional as F
import numpy as np

from utils.pl_data import ProteinLigandData
from utils import data as utils_data

AROMATIC_FEAT_MAP_IDX = utils_data.ATOM_FAMILIES_ID['Aromatic']

# only atomic number 1, 6, 7, 8, 9, 15, 16, 17 exist
MAP_ATOM_TYPE_FULL_TO_INDEX = {
    (1, 'S', False): 0,
    (6, 'SP', False): 1,
    (6, 'SP2', False): 2,
    (6, 'SP2', True): 3,
    (6, 'SP3', False): 4,
    (7, 'SP', False): 5,
    (7, 'SP2', False): 6,
    (7, 'SP2', True): 7,
    (7, 'SP3', False): 8,
    (8, 'SP2', False): 9,
    (8, 'SP2', True): 10,
    (8, 'SP3', False): 11,
    (9, 'SP3', False): 12,
    (15, 'SP2', False): 13,
    (15, 'SP2', True): 14,
    (15, 'SP3', False): 15,
    (15, 'SP3D', False): 16,
    (16, 'SP2', False): 17,
    (16, 'SP2', True): 18,
    (16, 'SP3', False): 19,
    (16, 'SP3D', False): 20,
    (16, 'SP3D2', False): 21,
    (17, 'SP3', False): 22
}

MAP_ATOM_TYPE_ONLY_TO_INDEX = {
    1: 0,
    6: 1,
    7: 2,
    8: 3,
    9: 4,
    15: 5,
    16: 6,
    17: 7,
}

MAP_ATOM_TYPE_AROMATIC_TO_INDEX = {
    (1, False): 0,
    (6, False): 1,
    (6, True): 2,
    (7, False): 3,
    (7, True): 4,
    (8, False): 5,
    (8, True): 6,
    (9, False): 7,
    (15, False): 8,
    (15, True): 9,
    (16, False): 10,
    (16, True): 11,
    (17, False): 12
}

MAP_INDEX_TO_ATOM_TYPE_ONLY = {v: k for k, v in MAP_ATOM_TYPE_ONLY_TO_INDEX.items()}
MAP_INDEX_TO_ATOM_TYPE_AROMATIC = {v: k for k, v in MAP_ATOM_TYPE_AROMATIC_TO_INDEX.items()}
MAP_INDEX_TO_ATOM_TYPE_FULL = {v: k for k, v in MAP_ATOM_TYPE_FULL_TO_INDEX.items()}


def get_atomic_number_from_index(index, mode):
    if mode == 'basic':
        atomic_number = [MAP_INDEX_TO_ATOM_TYPE_ONLY[i] for i in index.tolist()]
    elif mode == 'add_aromatic':
        atomic_number = [MAP_INDEX_TO_ATOM_TYPE_AROMATIC[i][0] for i in index.tolist()]
    elif mode == 'full':
        atomic_number = [MAP_INDEX_TO_ATOM_TYPE_FULL[i][0] for i in index.tolist()]
    else:
        raise ValueError
    return atomic_number

def get_protein_atomic_number_from_index(index):
    atomic_numbers = torch.LongTensor([1, 6, 7, 8, 16, 34])  # H, C, N, O, S, Se
    return atomic_numbers[index].tolist()

def is_aromatic_from_index(index, mode):
    if mode == 'add_aromatic':
        is_aromatic = [MAP_INDEX_TO_ATOM_TYPE_AROMATIC[i][1] for i in index.tolist()]
    elif mode == 'full':
        is_aromatic = [MAP_INDEX_TO_ATOM_TYPE_FULL[i][2] for i in index.tolist()]
    elif mode == 'basic':
        is_aromatic = None
    else:
        raise ValueError
    return is_aromatic


def get_hybridization_from_index(index, mode):
    if mode == 'full':
        hybridization = [MAP_INDEX_TO_ATOM_TYPE_AROMATIC[i][1] for i in index.tolist()]
    else:
        raise ValueError
    return hybridization


def get_index(atom_num, hybridization, is_aromatic, mode):
    if mode == 'basic':
        return MAP_ATOM_TYPE_ONLY_TO_INDEX[int(atom_num)]
    elif mode == 'add_aromatic':
        # self.atomic_numbers = torch.LongTensor([1, 6, 7, 8, 9, 15, 16, 17])  # H, C, N, O, F, P, S, Cl
        if (int(atom_num), bool(is_aromatic)) in MAP_ATOM_TYPE_AROMATIC_TO_INDEX:
            return MAP_ATOM_TYPE_AROMATIC_TO_INDEX[int(atom_num), bool(is_aromatic)]
        else:
            return MAP_ATOM_TYPE_AROMATIC_TO_INDEX[(1, False)]
    else:
        return MAP_ATOM_TYPE_FULL_TO_INDEX[(int(atom_num), str(hybridization), bool(is_aromatic))]


class FeaturizeProteinAtom(object):

    def __init__(self):
        super().__init__()
        self.atomic_numbers = torch.LongTensor([1, 6, 7, 8, 16])  # H, C, N, O, S
        self.max_num_aa = 20
    @property
    def feature_dim(self):
        return self.atomic_numbers.size(0) + self.max_num_aa + 1

    def __call__(self, data):  #
        element = data.protein_atom_class.view(-1, 1) == self.atomic_numbers.view(1, -1)  # (N_atoms, N_elements)
        amino_acid = F.one_hot(data.protein_atom_res_name.long(), num_classes=self.max_num_aa)
        is_backbone = data.protein_atom_is_backbone.view(-1, 1).long()
        x = torch.cat([element, amino_acid, is_backbone], dim=-1)
        data.protein_atom_feature = x
        return data


class FeaturizeLigandAtom(object):

    def __init__(self, mode='basic'):
        super().__init__()
        assert mode in ['basic', 'add_aromatic', 'full']
        self.mode = mode

    @property
    def feature_dim(self):
        if self.mode == 'basic':
            return len(MAP_ATOM_TYPE_ONLY_TO_INDEX)
        elif self.mode == 'add_aromatic':  # √
            return len(MAP_ATOM_TYPE_AROMATIC_TO_INDEX)
        else:
            return len(MAP_ATOM_TYPE_FULL_TO_INDEX)

    def __call__(self, data):

        element_list = data.ligand_atom_class
        hybridization_list = data.ligand_atom_hybridization
        # aromatic_list = [v[AROMATIC_FEAT_MAP_IDX] for v in data.ligand_atom_feature]
        aromatic_list = data.ligand_atom_aromatic

        x = [get_index(e, h, a, self.mode) for e, h, a in zip(element_list, hybridization_list, aromatic_list)]
        x = torch.tensor(x)
        data.ligand_atom_feature = x
        data.ligand_bond_idx = np.concatenate([data.ligand_bond_idx,
                              np.concatenate([data.ligand_bond_idx[:, 1].unsqueeze(1),
                                              data.ligand_bond_idx[:, 0].unsqueeze(1)], axis=1)], axis=0)
        data.ligand_bond_idx = np.array(data.ligand_bond_idx)
        return data


class FeaturizeLigandBond(object):

    def __init__(self):
        super().__init__()

    def __call__(self, data):
        data.ligand_bond_feature = F.one_hot(data.ligand_bond_class.long() - 1, num_classes=len(utils_data.BOND_TYPES))
        return data


class RandomRotation(object):
    def __init__(self):
        super().__init__()

    def __call__(self,  data: ProteinLigandData):
        M = np.random.randn(3, 3)
        Q, __ = np.linalg.qr(M)
        Q = torch.from_numpy(Q.astype(np.float32))
        data.ligand_pos = data.ligand_pos @ Q
        data.protein_pos = data.protein_pos @ Q
        return data
    
class NormalizeVina(object):

    def __init__(self, mode='pl'):
        super().__init__()
        self.mode = mode
        if mode == 'pl':
            self.max_v = 0
            self.min_v = -16
        elif mode == 'pdbbind':
            self.max_v = 16
            self.min_v = 0
        else:
            raise ValueError
    
    def _trans(self, vina_score):
        if self.mode == 'pl':
            return (self.max_v - np.clip(vina_score, self.min_v, self.max_v)) / (self.max_v - self.min_v)
        elif self.mode == 'pdbbind':
            return np.clip(vina_score, self.min_v, self.max_v) / (self.max_v - self.min_v)
        else:
            raise ValueError

    def __call__(self,  data):
        data.score = self._trans(data.score)
        return data


class NormlizeProperty(object):
    def __init__(self, mode="A-B-C", weight="1,1,1"):
        # A: Vina
        # B: Value: QED SA
        # C: Prop: Tox
        self.A = ["vina_score"]
        self.B = ["qed", "sa"]
        self.C = ["dili", "herg"]
        self.mode = mode.split("-")
        self.weight = weight.split(",")
        assert len(self.mode) == len(self.weight)

    @staticmethod
    def norm_tools(values):
        alpha = 0.35
        beta = 0.15

        min_val = min(values)
        geo_mean = np.prod(values) ** (1 / 5)
        ad_mean = np.mean(values) ** 2

        score = (geo_mean ** (1 - alpha - beta)) * \
                (min_val ** alpha) * \
                (ad_mean ** beta)

        score = 0.9 * (1.9 * score / (0.9 + score)) + 0.1
        return score

    @staticmethod
    def correct_value(data, attrs, reverse=False):
        scores = []
        for attr in attrs:
            if reverse:
                scores.append(1-getattr(data, attr))
            else:
                scores.append(getattr(data, attr))
        return scores


    def __call__(self, data):
        data.vina_score = data.score
        data.score = 1
        scores = []
        for mode in self.mode:
            prop_name = getattr(self, mode)
            if mode == "C":
                reverse=True
            else:
                reverse=False
            scores += self.correct_value(data, prop_name, reverse)
        data.score = torch.tensor(self.norm_tools(scores))
        return data



class FeaturizeProteinBond(object):

    def __init__(self):
        super().__init__()

    def __call__(self, data):
        data.protein_bond_feature = F.one_hot(data.protein_bond_class.long() - 1, num_classes=len(utils_data.BOND_TYPES))  # 直接生成onehot
        return data


class InteractionEx(object):

    def __init__(self):
        super().__init__()

    def _interaction_ext(self, data):
        start_atom = []
        end_atom = []
        interaction_class = []

        vt_start_atom = []
        vt_end_atom = []
        vt_interaction_class = []

        vt_atom_idx = []
        # vt_atom_coords = []

        interaction = data.interaction

        for itn in interaction:
            s_atom = itn[0][0]
            e_atom = itn[0][1]

            if len(s_atom) == 1 and len(e_atom) == 1:  # normal
                start_atom.append(itn[0][0][0])
                end_atom.append(itn[0][1][0])
                interaction_class.append(int(itn[1]))

            else:
                continue


        return start_atom, end_atom, interaction_class

    def __call__(self, data):
        # interaction
        start_atom, end_atom, interaction_class = self._interaction_ext(data)
        assert len(start_atom) == len(end_atom)
        if len(start_atom) == 0:
            data.interaction_idx = [None]
            data.interaction_class = [None]
        else:
            data.interaction_idx = np.array([(i, j) for i, j in zip(start_atom, end_atom)]).transpose(1, 0)  # 包括虚拟节点
            data.interaction_class = interaction_class  # 包括虚拟节点位置
        return data


class ImportantEx(object):

    def __init__(self):
        super().__init__()

    def __call__(self, data):
        data = self.get_important_ex(data)
        return data

    @staticmethod
    def get_important_ex(data):

        # 正常状态
        try:
            pures_data_map = np.array(data.pures_idx)
            act_data_map = np.array([[start, end, cls] for start, end, cls in zip(np.array(data.interaction_idx[0]),
                                                                                  np.array(data.interaction_idx[1]),
                                                                                  data.interaction_class) if
                                     (start >= 0) and (end >= 0)])
            update_protein = list(set(list([*list(pures_data_map), *list(act_data_map[:, 0])])))
            update_protein_map = {update_protein[i]:i for i in range(len(update_protein))}
            update_ligand_map = {i:i+len(update_protein) for i in range(len(data.ligand_atom_coords))}

            batch_idx = [*update_protein, *list(np.array(range(len(data.ligand_atom_coords)))+len(data.protein_atom_coords))]

            new_interaction = [[update_protein_map[p], update_ligand_map[l]] for p, l in zip(data.interaction_idx[0],
                                                          data.interaction_idx[1])]
            # print(act_data_map[:, :-1])
            # print(new_interaction)
            bond_cls = act_data_map[:, 2]

            ligand_mask = [*list([False for _ in range(len(update_protein))]),
                           *list([True for _ in range(len(data.ligand_atom_coords))])]

            data.imp_batch_idx = np.array(batch_idx)
            data.new_interaction = np.array(new_interaction)
            data.imp_bond_cls = np.array(bond_cls)
            data.imp_ligand_mask = np.array(ligand_mask)
        except:

            data.imp_batch_idx = np.array([*list(range(len(data.protein_atom_coords))),
                                  *list(np.array(range(len(data.ligand_atom_coords))) + len(data.protein_atom_coords))])
            data.new_interaction = np.array([])
            data.imp_bond_cls = np.array([])

            data.imp_ligand_mask = [*list([False for _ in range(len(data.protein_atom_coords))]),
                           *list([True for _ in range(len(data.ligand_atom_coords))])]

        return data
#




