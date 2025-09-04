from tqdm import tqdm
import pickle
from utils.data_process import in_per_process
from utils.data_tools import TrainingCollate
from utils.sample import single_sample

protein_path = "data/Test/SIR3_HUMAN_117_398_0/XXX.pdb"
ligand_path = "data/Test/SIR3_HUMAN_117_398_0/XXX.mol2"

### Ligand-Protein
data_collate = in_per_process(protein_path, ligand_path)
### Only Protein
# data_collate = in_per_process(protein_path, None)


get_t = TrainingCollate()
get_data = get_t.data_get({"0": data_collate})

# sample_model
single_sample(get_t.data, "SGEDiff")   # SGEDiff-NG