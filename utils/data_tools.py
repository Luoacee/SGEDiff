from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import numpy as np
from torch_geometric.data import Data

class TrainingCollate:
    def __init__(self, exclude_keys=None):
        if exclude_keys is None:
            self.exclude_keys = [""]
        else:
            self.exclude_keys = exclude_keys

        self.data = None

    def data_get(self, data_dict):
        prop_names = data_dict.keys()

        data_pocket = []

        for key in tqdm(prop_names, total=len(prop_names)):
            data = {}
            read_item = data_dict[key]

            # data_point = BasedData()
            for itm in read_item:
                if itm == "name":
                    # self.__getattribute__(itm).append(key)
                    # setattr(data_point, itm, key)
                    data[itm] = key
                elif itm in self.exclude_keys:
                    continue
                elif itm in ["interaction", "datasets", "ligand_atom_hybridization", "smiles", "pn", 'ln']:
                    # setattr(data_point, itm, read_item[itm])
                    data[itm] = read_item[itm]
                elif itm in ["qed", "logp", "tpsa"]:
                    rd = np.around(read_item[itm], decimals=4)
                    data[itm] = rd
                elif itm == "ligand_center":
                    data[itm] = torch.from_numpy(np.array([read_item["ligand_center"]]))  # BATCH
                else:
                    # self.__getattribute__(itm).append(re ad_item[itm])
                    rd = torch.from_numpy(np.array(read_item[itm]))
                    # setattr(data_point, itm, torch.from_numpy(rd))
                    data[itm] = rd
            data_pocket.append(BasedData(**data))
        self.data = data_pocket

    def __call__(self, *args, **kwargs):
        return self.data

class BasedData(Data):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DataCollate(Dataset):
    def __init__(self, dataset, transform=None):
        super().__init__()
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.transform is not None:
            return self.transform(self.dataset[index])
        else:
            return self.dataset[index]


