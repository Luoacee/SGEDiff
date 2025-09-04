
from vina import Vina
from pymol import cmd
import os
import numpy as np
import subprocess
from lxml import etree
from collections import defaultdict
from torch_geometric.data import Data
import pickle
import warnings
import shutil
import time
import torch
import lmdb
# process tools
import gzip
from AutoDockTools.MoleculePreparation import AD4ReceptorPreparation
from MolKit import Read
from rdkit.Chem import SDMolSupplier, SDWriter
from rdkit.Chem import MolFromPDBFile
from rdkit.Chem import BondType
from rdkit.Chem import HybridizationType
from rdkit import Chem
from rdkit import RDLogger
from meeko import PDBQTMolecule
from meeko import MoleculePreparation
from meeko import PDBQTWriterLegacy
from meeko import RDKitMolCreate
from puresnet import predict
from openbabel import pybel
from rdkit.Chem import BondType
from rdkit.Chem import HybridizationType
from rdkit.Chem import QED
from utils.evaluation import sascorer
from rdkit.ML.Descriptors import MoleculeDescriptors
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "8"
prop_name = ['TPSA', 'MolLogP', 'NumHAcceptors', 'NumHDonors']
InteractionsIdx2Class = {0: "hydrophobic_interactions", 1:"hydrogen_bonds", 2:"salt_bridges", 3:"pi_stacks",
                 4:"pi_cation_interactions", 5:"halogen_bonds"}
InteractionClass2Idx = {v: k for k, v in InteractionsIdx2Class.items()}
validInteractions = list(InteractionsIdx2Class.values())  # only 6
AA_NAME_SYM = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
    'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
    'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
}
AA_NAME_NUMBER = {
    k: i for i, k in enumerate(AA_NAME_SYM.keys())
}
BACKBONE_ATOM = ["N", "CA", "C", "O"]
BONDTYPE = {BondType.SINGLE: 1, BondType.DOUBLE: 2, BondType.TRIPLE: 3, BondType.AROMATIC: 4}
HYBRIDTYPE = {HybridizationType.UNSPECIFIED:"S",
              HybridizationType.S:"S", HybridizationType.SP:"SP", HybridizationType.SP2:"SP2",
              HybridizationType.SP3:"SP3", HybridizationType.SP3D:"SP3D", HybridizationType.SP3D2:"SP3D2",
              }
des_cal = MoleculeDescriptors.MolecularDescriptorCalculator(prop_name)

class OSReading:
    @staticmethod
    def read_protein(path, remove_Hs=False, remove_explict_Hs=False):
        if remove_Hs:
            cmd.load(path)
            if remove_explict_Hs:
                cmd.remove("not visible and elem H")
            else:
                cmd.remove("elem H")
            cmd.save("protein_tmp.pdb")
            cmd.delete("all")

        # pdb
            protein = MolFromPDBFile("protein_tmp.pdb", sanitize=False, removeHs=False)
        else:
            protein = MolFromPDBFile(path, sanitize=False, removeHs=False)
        return protein

    @staticmethod
    def read_ligand(path, removeHs=False):
        # sdf
        ligand_list = list()
        ligand = SDMolSupplier(path, removeHs=removeHs)
        for mol in ligand:
            ligand_list.append(mol)
        return ligand_list

    @staticmethod
    def pickle_load(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def file_path(*args):
        return os.path.join(*args)


# Init tools
osr = OSReading()


class BaseProcess:
    def __init__(self, protein_path, ligand_path=None):
        self.protein_atom_remove = ["Zn", "Co", "Fe", "Au", "Hg",
                                    "Ni", "Cd", "Li", "Cu", "Ag", "In"]
        self.tmp_path = "tmp"
        self.protein_path = protein_path
        self.ligand_path = ligand_path
        self.protein_name =  protein_path.split('/')[-1].split('.')[0]
        if not os.path.exists(protein_path):
            raise FileNotFoundError
        if ligand_path:
            if not os.path.exists(ligand_path):
                raise FileNotFoundError
            self.protein_name = ligand_path.split('/')[-1].split('.')[0]

        self.tmp_path += "/%s" % self.protein_name

        if not os.path.exists(self.tmp_path):
            os.makedirs(self.tmp_path)

        self.pro_p, self.pro_pdb_qt_prep = self.pro_prep()
        self.lig_p, self.lig_pdb_qt_prep, self.ligand_coords = None, None, None
        if ligand_path:
            self.lig_p, self.lig_pdb_qt_prep, self.ligand_coords = self.lig_prep()


    def pro_prep(self):
        # remove some atoms
        prep_pro_path = "%s/%s_pro_processed.pdb" % (self.tmp_path, self.protein_name)
        prep_pdbqt_path = "%s/%s_pro_processed.pdbqt" % (self.tmp_path, self.protein_name)

        cmd.load(self.protein_path)
        for atom in self.protein_atom_remove:
            cmd.remove('element %s' % atom)
        cmd.remove("solvent")
        cmd.h_add()
        cmd.save(prep_pro_path)
        cmd.delete("all")

        # pdb qt
        # pro_mol = Read(prep_pro_path)[0]

        # AD4ReceptorPreparation(pro_mol, outputfilename=prep_pdbqt_path,
        #                        cleanup='nphs_lps_waters_nonstdres')
        return prep_pro_path, prep_pdbqt_path

    def lig_prep(self):
        prep_lig_path = "%s/%s_lig_processed.sdf" % (self.tmp_path, self.protein_name)
        prep_pdbqt_path = "%s/%s_lig_processed.pdbqt" % (self.tmp_path, self.protein_name)

        if self.ligand_path.split('.')[-1] == "mol2":
            file_time = time.time()
            ob_mol = next(pybel.readfile("mol2", self.ligand_path))
            ob_mol.write("sdf", "tmp_%s.sdf" % file_time)
            read_mol = Chem.SDMolSupplier("tmp_%s.sdf" % file_time, removeHs=False)[0]
            os.remove("tmp_%s.sdf" % file_time)
        elif self.ligand_path.split('.')[-1] == "sdf":
            read_mol = SDMolSupplier(self.ligand_path, removeHs=False)[0]
        elif self.ligand_path.split('.')[-1] == "gz":
            with gzip.open(self.ligand_path, 'rt') as f:
                sdf_file = f.read()
            read_mol = Chem.SDMolSupplier()
            read_mol.SetData(sdf_file, removeHs=False)
            read_mol = read_mol[0]
        else:
            raise ValueError

        mol = read_mol
        mol = Chem.AddHs(mol,addCoords=True)

        sdf_writer = SDWriter(prep_lig_path)
        sdf_writer.write(mol)
        sdf_writer.close()
        coords = np.array(mol.GetConformer().GetPositions())
        ligand_coords = np.mean(coords, axis=0)

        return prep_lig_path, prep_pdbqt_path, ligand_coords


class VinaScoring:
    def __init__(self, box_size=30):
        self.protein_atom_remove = ["Zn", "Co", "Fe", "Au", "Hg",
                               "Ni", "Cd", "Li", "Cu", "Ag", "In"]
        self.box_size = [box_size, box_size, box_size]

    def __call__(self, pro_pdb_qt_prep, lig_pdb_qt_prep, ligand_coords): # protein:pdb, ligand:sdf
        vina_score = self.vina_score(pro_pdb_qt_prep, lig_pdb_qt_prep, ligand_coords)
        return vina_score

    def vina_score(self, receptor_path, ligand_path, box_position):
        if os.stat(receptor_path).st_size == 0:
            return 100
        if os.stat(ligand_path).st_size == 0:
            return 100

        vina = Vina(seed=712, verbosity=0, cpu=0)
        vina.set_receptor(receptor_path)
        vina.set_ligand_from_file(ligand_path)
        vina.compute_vina_maps(center=box_position, box_size=self.box_size)
        # clean
        os.remove(receptor_path)
        os.remove(ligand_path)
        return vina.score()[0]


class PLIPAnalysis:
    def __init__(self, protein_path, ligand_path, file_name, output_dir, stdout=False):
        self.protein_path = protein_path
        self.file_name = file_name  # protein name
        self.ligand_path = ligand_path
        self.output_dir = output_dir
        self.stdout = stdout


    def pl_combine(self):
        protein = Chem.MolFromPDBFile(self.protein_path, removeHs=False, sanitize=False)
        ligand = Chem.SDMolSupplier(self.ligand_path, removeHs=False)[0]

        pl = Chem.CombineMols(protein, ligand)
        Chem.MolToPDBFile(pl, self.output_dir + "/%s_combine.pdb" % self.file_name)
        return self.output_dir + "/%s_combine.pdb" % self.file_name

    def analysis(self):
        path = self.pl_combine()
        command = r""" plip -f {} --name {} --nofixfile --nohydro -o {} -x""".format(path,
                                                                                          "plip_analysis",
                                                                                          self.output_dir)
        results = subprocess.run(
            f"{command}",
            shell=True, capture_output=True, text=True, encoding="utf-8")
        plip_path = self.output_dir + "/plip_analysis"

        if self.stdout is True:
            print("\n" * 2 + results.stderr)

        with open(self.output_dir + "/plipcmd_stdout.txt", "w") as F:
            F.write(results.stderr)
        collate_value = self.collate_interaction(plip_path)
        return collate_value

    def collate_interaction(self, dirs):
        root = etree.parse("%s.xml" % dirs)
        os.remove("%s.xml" % dirs)
        root_name = [i.text for i in root.findall(".//longname")]
        unl_idx = root_name.index("UNL")
        record_results = dict()
        smiles = root.findall(".//smiles")[unl_idx].text
        record_results["name"] = self.file_name
        record_results["smiles"] = smiles

        interactions = root.findall('.//interactions')
        interaction = interactions[unl_idx]

        valid_tags = list()
        for subI in interaction:
            sub_tag = subI.tag
            if len(interaction.findall(f".//{sub_tag}")[0].getchildren()) > 0:
                valid_tags.append(sub_tag)

        for tag in valid_tags:
            sub_ = interaction.findall(f".//{tag}")[0].getchildren()
            head = ["resnr", "restype", "reschain", "ligcoo", "protcoo"]
            record_table = [["RESNR", "RESTYPE", "RESCHAIN", "LIGCOO X", "LIGCOO Y",
                             "COOLIDGE Z", "PROTCOO X", "PROTCOO Y", "PROTCOO Z"]]
            inter_numbers = len(sub_)
            inter_dict = dict()
            inter_dict["numbers"] = inter_numbers

            for s in sub_:
                resnr = s.findall(f".//{head[0]}")[0].text
                restype = s.findall(f".//{head[1]}")[0].text
                reschain = s.findall(f".//{head[2]}")[0].text

                ligcoo_x = s.findall(f".//{head[3]}")[0].findall(f".//x")[0].text
                ligcoo_y = s.findall(f".//{head[3]}")[0].findall(f".//y")[0].text
                ligcoo_z = s.findall(f".//{head[3]}")[0].findall(f".//z")[0].text

                protcoo_x = s.findall(f".//{head[4]}")[0].findall(f".//x")[0].text
                protcoo_y = s.findall(f".//{head[4]}")[0].findall(f".//y")[0].text
                protcoo_z = s.findall(f".//{head[4]}")[0].findall(f".//z")[0].text

                # protcoo = s.findall(f".//{head[4].lower()}")[0].text
                record_table.append(
                    [resnr, restype, reschain, ligcoo_x, ligcoo_y, ligcoo_z, protcoo_x, protcoo_y, protcoo_z])
            record_table = np.vstack(record_table)
            inter_dict["table"] = record_table
            record_results[tag] = inter_dict
        return record_results


class PuresPredict:
    def __init__(self, protein_path, molecule=None):
        self.protein_path = protein_path
        self.molecule = molecule

    def run(self):
        try:
            predict.make_prediction(self.protein_path, device=0, mode="A")

        except Exception as e:
            print("Pures error: %s" % e)

    def analysis(self):
        out_file = os.listdir("results")
        puresCombinePath = "PuresResults"
        if os.path.exists(puresCombinePath) is False:
            os.mkdir(puresCombinePath)

        sub_paths = []
        for sub_file in os.listdir(os.path.join("results", out_file[0])):
            sub_paths.append(os.path.join("results", out_file[0], sub_file))
        for sub_file in sub_paths:
            cmd.load(sub_file)

        cmd.save(os.path.join("PuresResults", "%s_pu_com.pdb" % out_file[0].split("_")[0]))
        cmd.delete("all")

        pocket_mol = self.molecule
        pures_mol = osr.read_protein(osr.file_path(puresCombinePath, "%s_pu_com.pdb" % out_file[0].split("_")[0]))

        if (pocket_mol is None) or (pures_mol is None):
            print("No pures pocket mol found")
            return [], None

        limit_atom = [""]

        paring_idx = self.pures_idx_matching(pocket_mol, pures_mol, limit_atom)
        assert pocket_mol.GetNumAtoms() >= len(paring_idx), "Pocket Error"

        try:
            shutil.rmtree(puresCombinePath)
        except Exception as e:
            _ = e
        try:
            shutil.rmtree("results")
        except Exception as e:
            _ = e

        return paring_idx, pures_mol


    @staticmethod
    def pures_idx_matching(mol, matching_mol, limit, remove_h=True):
        idx_table = []
        mol_coords = mol.GetConformer().GetPositions()

        matching_coords = matching_mol.GetConformer().GetPositions()
        atom_type = [a.GetSymbol() for a in matching_mol.GetAtoms()]

        for idx, (a_coords, a_type) in enumerate(zip(matching_coords, atom_type)):
            if a_type in limit:
                continue

            position_distance = []
            for jdx, p_coords in enumerate(mol_coords):
                position_distance.append(np.sum((a_coords - p_coords) ** 2) ** 0.5)

            min_idx = int(np.argmin(position_distance))

            if np.min(position_distance) > 1e-4:
                continue

            else:

                idx_table.append((min_idx, mol_coords[min_idx]))
                if not remove_h:
                    coo_atom = mol.GetAtomWithIdx(min_idx)
                    for nei in coo_atom.GetNeighbors():
                        if nei.GetSymbol() == "H":
                            idx_table.append((nei.GetIdx(), mol_coords[nei.GetIdx()]))

        return idx_table


class PIPLINE:
    @staticmethod
    def plip_refine(plip_results):
        ligand_dict = dict()
        res_info = []
        interactions = list(plip_results.keys())[2:]
        valid_interactions = sorted(list(set(interactions) & set(validInteractions)))

        if len(valid_interactions) == 0:
            return None

        res_idx, res_name, res_chain, lig, pro, inter_cls = [], [], [], [], [], []

        for inc in valid_interactions:
            interaction_table = plip_results[inc]["table"]
            interaction_res_idx = interaction_table[1:, 0]
            interaction_res_name = interaction_table[1:, 1]
            interaction_res_chain = interaction_table[1:, 2]
            lig_coords = interaction_table[1:, 3: 6]
            pro_coords = interaction_table[1:, 6: 9]
            interaction_mark = [InteractionClass2Idx[inc] for _ in range(len(interaction_res_name))]

            res_idx.append(interaction_res_idx)
            res_name.append(interaction_res_name)
            res_chain.append(interaction_res_chain)
            lig.append(lig_coords)
            pro.append(pro_coords)
            inter_cls += [interaction_mark]

        res_idx = np.concatenate(res_idx).astype(int)  # (15,) #
        res_name = np.concatenate(res_name)  # (15,)
        res_chain = np.concatenate(res_chain)  # (15,)
        lig = np.concatenate(lig).astype(float)  # (15,3)
        pro = np.concatenate(pro).astype(float)  # (15,3)
        inter_cls = np.concatenate(inter_cls).astype(int)  # (15,)

        for rdx, (res_c, res_id) in enumerate(zip(res_chain, res_idx)):
            res_info += [
                [res_name[rdx], res_id, res_c, str(list(pro[rdx])), inter_cls[rdx], str(list(lig[rdx]))]]


        if len(res_info) > 0:
            res_info = np.vstack(res_info)
            ligand_dict["res_name"] = res_info[:, 0]
            ligand_dict["res_id"] = res_info[:, 1].astype(int)
            ligand_dict["res_chain"] = res_info[:, 2]
            ligand_dict["res_interaction_coords"] = np.vstack([eval(i) for i in res_info[:, 3]])
            ligand_dict["interaction_cls"] = res_info[:, 4]
            ligand_dict["ligand_interaction_coords"] = np.vstack([eval(i) for i in res_info[:, 5]])
        else:
            None
        return ligand_dict

    @staticmethod
    def distance_pocket(center, protein_path, distance, file_dir):
        cmd.load(protein_path)
        cmd.pseudoatom("ligand_center", pos=list(center))


        cmd.select("within_distance", f"br. (all within {distance} of ligand_center)")
        cmd.remove("ligand_center")

        if cmd.count_atoms("within_distance") > 0:
            cmd.save(file_dir + "/pocket_%sA.pdb" % distance, "within_distance")
        cmd.delete("all")
        pocket_loading = osr.read_protein(file_dir + "/pocket_%sA.pdb" % distance, remove_Hs=True)
        # os.remove("pocket_%s_A.pdb" % distance)
        return pocket_loading

    @staticmethod
    def interaction_pocket(plip_results, path):
        p_name = path.split("/")[-1]
        res_idx, res_chain = plip_results["res_id"], plip_results["res_chain"]

        if res_idx is not None:
            res_dict = defaultdict(list)
            command = defaultdict(str)
            pymol_cmd = []
            for ri, rc in zip(res_idx, res_chain):
                res_dict[rc].append(ri)
            for k in res_dict.keys():
                res_dict[k] = list(set(np.array(res_dict[k]).astype(str)))
                command[k] = "chain %s and resi " % k + "+".join(res_dict[k])
            for k in command.keys():
                pymol_cmd.append(command[k])
            pymol_cmd = " or ".join(pymol_cmd)

            cmd.load(osr.file_path(path, "%s_pro_processed.pdb" % p_name))
            cmd.select("res", pymol_cmd)
            cmd.save(path + "/pocket_interaction.pdb", "res")

            cmd.delete("all")
            mol_read = osr.read_protein(path + "/pocket_interaction.pdb", remove_Hs=True)
            return mol_read
        else:
            return None


    def atom_mapping(self, plip_results, pocket_mol, ligand_mol, operate_dir):  # 10A pocket
        res_coords = plip_results["res_interaction_coords"]
        lig_coords = plip_results["ligand_interaction_coords"]

        interaction_cls = plip_results["interaction_cls"]
        protein_matching = self.matching(pocket_mol, res_coords, 3, interaction_cls, operate_dir)

        # print(res_coords)
        ligand_matching = self.matching(ligand_mol, lig_coords, 2, interaction_cls, operate_dir)
        # p-l

        valid_coords_idx, interaction_map = [], []
        for idx, (i, j) in enumerate(zip(protein_matching, ligand_matching)):
            if (i is not None) and (j is not None):
                interaction_map += [[i, j]]
                valid_coords_idx.append(idx)
        return interaction_map, res_coords[valid_coords_idx], lig_coords[valid_coords_idx]

    def matching(self, mol, matching_coords, distance, interactions, operate_dir):

        idx_table = []
        mol_coords = np.around(mol.GetConformer().GetPositions(), 3)  # 坐标


        for idx, i in enumerate(matching_coords):
            position_distance = []
            for jdx, j in enumerate(mol_coords):
                position_distance.append(np.sum((i - j) ** 2) ** 0.5)
            min_idx = int(np.argmin(position_distance))

            if np.min(position_distance) > 0.1:

                assert distance is not None, "Distance error: %s" % np.mean(position_distance)

                if interactions[idx] == '1':
                    idx_table.append(None)
                    continue

                Chem.MolToPDBFile(mol, operate_dir + "/matching.pdb")
                cmd.load(operate_dir + "/matching.pdb")
                cmd.pseudoatom("center_", pos=list(i))
                cmd.select("in_range_atoms", "all within %s of center_" % distance)

                cmd.remove("center_")
                range_atom_coords = cmd.get_coords("in_range_atoms")
                cmd.delete("all")
                os.remove(operate_dir + "/matching.pdb")

                if range_atom_coords is None or len(range_atom_coords) == 1:
                    idx_table.append(None)
                    continue

                assert len(range_atom_coords) > 0, "no range atoms"

                range_bond_type = [interactions[idx] for _ in range(len(range_atom_coords))]
                range_atom_idx = self.matching(mol, range_atom_coords, None, range_bond_type, None)
                range_atom_idx = [atm[0] for atm in range_atom_idx]

                idx_table += [range_atom_idx]
            else:
                idx_table += [[min_idx]]
        return idx_table

    # 开发入口
    def __call__(self, protein_path, center, distance=10, file_dir=None, ligand_path=None, plip_results=None):
        XA_pocket_mol = self.distance_pocket(center, protein_path, distance, file_dir) # 10A_pocket
        ligand_mol = SDMolSupplier(ligand_path)[0]
        plip_results = self.plip_refine(plip_results)  # plip_results
        if plip_results is not None:
            plip_interaction_cls = plip_results["interaction_cls"]

            interaction_pocket_mol = self.interaction_pocket(plip_results, file_dir)  # interaction_pocket
            interaction_mapping, new_res_coords, new_lig_coords = self.atom_mapping(
                plip_results, XA_pocket_mol, ligand_mol, file_dir)
        else:
            interaction_pocket_mol, interaction_mapping, new_res_coords, new_lig_coords = None, None, None, None
            plip_interaction_cls = None

        return {
            "XA_pocket_mol": XA_pocket_mol,
            "ligand_mol": ligand_mol,
            "interaction_pocket_mol": interaction_pocket_mol,
            "interaction_mapping": interaction_mapping,
            "interaction_cls": plip_interaction_cls,
            "new_res_coords": new_res_coords,
            "new_lig_coords": new_lig_coords
        }


class ProteinProcess:
    def data_collate(self, p_info):
        protein_pocket = p_info["XA_pocket_mol"]

        # pures_info
        pures_res_idx = p_info["pures_mapping"]
        assert protein_pocket is not None

        # pures check
        protein_coords = protein_pocket.GetConformer().GetPositions()

        pures_idx = []

        for pr in pures_res_idx:  # 检测pures
            pr_atom = pr[0]
            pr_coords = pr[1]
            assert np.sum((protein_coords[pr_atom] - pr_coords) ** 2) ** 0.5 < 1e-4, "Pures coords error!"
            pures_idx.append(int(pr_atom))



        info_process = self.info_process(p_info["pn"], p_info["ln"], protein_pocket, pures_idx)
        return info_process

    def info_process(self, pn, ln, protein_mol, pures):
        wt_info = dict()
        # print(wt_key)
        wt_info["pn"] = pn
        wt_info["ln"] = ln

        valid_interaction = []  # S -> E, cls, S_coords, E_coords

        protein_mol_coords = protein_mol.GetConformer().GetPositions()

        # Chem.MolToPDBFile(protein_mol, "Test.pdb")

        pro_atom_class, pro_backbone, pro_res_names, pro_hybridization = [], [], [], []
        # pro_aromaticity_h = []
        pro_aromaticity = []
        pro_bond_class, pro_bond_idx = [], []
        pro_atom_coords = []

        error_marker = False
        for adx, atom in enumerate(protein_mol.GetAtoms()):
            # if atom.GetAtomicNum() == 1:
            #     continue
            # else:
            atom_symbol = atom.GetAtomicNum()

            atom_res_name = atom.GetPDBResidueInfo().GetResidueName()

            if atom_res_name not in list(AA_NAME_SYM.keys()):
                error_marker = True
                break

            pro_res_names.append(AA_NAME_NUMBER[atom_res_name])

            if atom.GetPDBResidueInfo().GetName().rstrip(" ").lstrip(" ") in BACKBONE_ATOM:
                pro_backbone.append(1)
            else:
                pro_backbone.append(0)

            pro_atom_class.append(atom_symbol)

            atom_aromatic = atom.GetIsAromatic()
            pro_aromaticity.append(atom_aromatic)
            # pro_aromaticity_h.append(atom_aromatic)

            pro_atom_coords.append(protein_mol_coords[adx])

        if error_marker:
            return [1, "ErrorResName"]
        try:
            assert len(pro_backbone) == len(pro_atom_class), "%s, %s" % (len(pro_backbone), len(pro_atom_class))
            assert len(pro_backbone) == len(set([str(i) for i in np.array(pro_atom_coords).tolist()]))
            assert len(pro_backbone) == len(pro_atom_coords)
        except Exception as e:
            return [1, str(e)]

        for bond in protein_mol.GetBonds():
            pro_bond_class.append(BONDTYPE[bond.GetBondType()])
            pro_bond_idx.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))

        wt_info["protein_atom_class"] = pro_atom_class
        wt_info["protein_atom_is_backbone"] = pro_backbone
        wt_info["protein_atom_aromatic"] = pro_aromaticity
        wt_info["protein_atom_coords"] = pro_atom_coords
        wt_info["protein_atom_res_name"] = pro_res_names
        wt_info["protein_bond_class"] = pro_bond_class
        wt_info["protein_bond_idx"] = pro_bond_idx


        wt_info["pures_idx"] = pures
        return wt_info

class ProteinLigandProcess:
    def data_collate(self, p_info):
        protein_pocket = p_info["XA_pocket_mol"]
        ligand_center_coord = p_info["ligand_center"]
        ligand_mol = p_info["ligand_mol"]

        desc = des_cal.CalcDescriptors(ligand_mol)
        mol_qed = QED.qed(ligand_mol)
        mol_sa = sascorer.compute_sa_score(ligand_mol)
        mol_ha = desc[2]
        mol_hd = desc[3]
        mol_tpsa = desc[0]
        mol_logp = desc[1]
        smiles = Chem.MolToSmiles(ligand_mol)

        # intact_info
        interaction_cls = p_info["interaction_cls"]
        # ligand_split = [int(i) for i in p_info["ligand_idx"]]
        # f_intact_type = self.split_list(intact_type, ligand_split)
        interaction_mapping = p_info["interaction_mapping"]
        interaction_ligand_coords = p_info["ligand_interaction_coords"]
        interaction_res_coords = p_info["res_interaction_coords"]

        # pures_info
        pures_res_idx = p_info["pures_mapping"]
        assert protein_pocket is not None

        # value
        score_value = p_info["score"]


        # pures check
        protein_coords = protein_pocket.GetConformer().GetPositions()

        pures_idx = []

        for pr in pures_res_idx:  # 检测pures
            pr_atom = pr[0]
            pr_coords = pr[1]
            assert np.sum((protein_coords[pr_atom] - pr_coords) ** 2) ** 0.5 < 1e-4, "Pures coords error!"
            pures_idx.append(int(pr_atom))



        info_process = self.info_process(p_info["pn"], p_info["ln"], protein_pocket,
                                         ligand_mol, ligand_center_coord, interaction_cls,
                                         interaction_mapping, interaction_ligand_coords, interaction_res_coords,
                                         pures_idx, score_value, smiles,
                                         mol_qed, mol_sa, mol_ha, mol_hd, mol_logp, mol_tpsa)
        return info_process

    def info_process(self, pn, ln, protein_mol, ligand_mol, ligand_center, interaction_cls,
                     interaction_mapping, interaction_ligand_coords, interaction_res_coords, pures, vina_score,smiles,
                     mol_qed, mol_sa, mol_ha, mol_hd, mol_logp, mol_tpsa):
        wt_info = dict()
        # print(wt_key)
        wt_info["pn"] = pn
        wt_info["ln"] = ln
        wt_info["ligand_center"] = ligand_center
        wt_info["score"] = vina_score

        wt_info["qed"] = mol_qed
        wt_info["sa"] = mol_sa
        wt_info["ha"] = mol_ha
        wt_info["hd"] = mol_hd
        wt_info["logp"] = mol_logp
        wt_info["tpsa"] = mol_tpsa

        wt_info["smiles"] = smiles

        valid_interaction = []  # S -> E, cls, S_coords, E_coords

        protein_mol_coords = protein_mol.GetConformer().GetPositions()
        ligand_mol_coords = ligand_mol.GetConformer().GetPositions()


        if interaction_mapping is not None:
            for int_idx in range(len(interaction_mapping)):
                intact = interaction_mapping[int_idx]

                # check protein
                if len(intact[0]) == 1:
                    assert np.sum(
                        (interaction_res_coords[int_idx] - protein_mol_coords[intact[0][0]]) ** 2) ** 0.5 < 1e-1

                # check ligand
                if len(intact[1]) == 1:
                    assert np.sum(
                        (interaction_ligand_coords[int_idx] - ligand_mol_coords[intact[1][0]]) ** 2) ** 0.5 < 1e-1

                valid_interaction.append((intact, interaction_cls[int_idx],
                                          interaction_res_coords[int_idx], interaction_ligand_coords[int_idx]))
        # except Exception as e:
        #     return [1, str(e)]
        else:
            valid_interaction = None
        wt_info["interaction"] = valid_interaction  # 保留有效的相互作用


        pro_atom_class, pro_backbone, pro_res_names, pro_hybridization = [], [], [], []
        # pro_aromaticity_h = []
        pro_aromaticity = []
        pro_bond_class, pro_bond_idx = [], []
        pro_atom_coords = []

        error_marker = False
        for adx, atom in enumerate(protein_mol.GetAtoms()):
            # if atom.GetAtomicNum() == 1:
            #     continue
            # else:
            atom_symbol = atom.GetAtomicNum()

            atom_res_name = atom.GetPDBResidueInfo().GetResidueName()

            if atom_res_name not in list(AA_NAME_SYM.keys()):
                error_marker = True
                break

            pro_res_names.append(AA_NAME_NUMBER[atom_res_name])

            if atom.GetPDBResidueInfo().GetName().rstrip(" ").lstrip(" ") in BACKBONE_ATOM:
                pro_backbone.append(1)
            else:
                pro_backbone.append(0)

            pro_atom_class.append(atom_symbol)

            atom_aromatic = atom.GetIsAromatic()
            pro_aromaticity.append(atom_aromatic)
            # pro_aromaticity_h.append(atom_aromatic)

            pro_atom_coords.append(protein_mol_coords[adx])


        if error_marker:
            return [1, "ErrorResName"]
        try:
            assert len(pro_backbone) == len(pro_atom_class), "%s, %s" % (len(pro_backbone), len(pro_atom_class))
            assert len(pro_backbone) == len(set([str(i) for i in np.array(pro_atom_coords).tolist()]))
            assert len(pro_backbone) == len(pro_atom_coords)
        except Exception as e:
            return [1, str(e)]

        for bond in protein_mol.GetBonds():
            pro_bond_class.append(BONDTYPE[bond.GetBondType()])
            pro_bond_idx.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))

        wt_info["protein_atom_class"] = pro_atom_class
        wt_info["protein_atom_is_backbone"] = pro_backbone
        wt_info["protein_atom_aromatic"] = pro_aromaticity
        # wt_info["protein_atom_aromatic_h"] = pro_aromaticity_h
        wt_info["protein_atom_coords"] = pro_atom_coords
        # wt_info["protein_atom_hybridization"] = pro_hybridization
        wt_info["protein_atom_res_name"] = pro_res_names
        wt_info["protein_bond_class"] = pro_bond_class
        wt_info["protein_bond_idx"] = pro_bond_idx

        lig_atom_class, lig_aromaticity, lig_hybridization = [], [], []
        lig_bond_class, lig_bond_idx = [], []
        lig_atom_coords = []

        for adx, atom in enumerate(ligand_mol.GetAtoms()):
            atom_symbol = atom.GetAtomicNum()
            lig_atom_class.append(atom_symbol)

            atom_aromatic = atom.GetIsAromatic()
            lig_aromaticity.append(atom_aromatic)

            lig_atom_coords.append(ligand_mol_coords[adx])

            atom_sp = atom.GetHybridization()
            lig_hybridization.append(HYBRIDTYPE[atom_sp])

        for bond in ligand_mol.GetBonds():
            lig_bond_class.append(BONDTYPE[bond.GetBondType()])
            lig_bond_idx.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))

        try:
            assert len(lig_atom_class) == len(lig_aromaticity), "%s, %s" % (
            len(lig_atom_class), len(lig_aromaticity))
            assert len(lig_hybridization) == len(lig_aromaticity), "%s, %s" % (
            len(lig_atom_class), len(lig_aromaticity))

        except Exception as e:
            return [1, str(e)]

        wt_info["ligand_atom_class"] = lig_atom_class
        wt_info["ligand_atom_aromatic"] = lig_aromaticity
        wt_info["ligand_atom_coords"] = lig_atom_coords
        wt_info["ligand_atom_hybridization"] = lig_hybridization
        wt_info["ligand_bond_class"] = lig_bond_class
        wt_info["ligand_bond_idx"] = lig_bond_idx
        wt_info["pures_idx"] = pures

        return wt_info

class BasedData(Data):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)



def vina_metric(pro_path, lig_path):
    container = BaseProcess(pro_path, lig_path)
    score = VinaScoring()
    vina_score = score(container.pro_pdb_qt_prep, container.lig_pdb_qt_prep, container.ligand_coords)
    return vina_score


def pre_process(pn, ln=None, delete_tmp=False):
    container = BaseProcess(pn, ln)

    affinity=100
    if ln:
        print("ProteinLigand")
        plip_analysis = PLIPAnalysis(container.pro_p, container.lig_p,
                                     container.protein_name, container.tmp_path)
        plip_results = plip_analysis.analysis()
        pipline = PIPLINE()
        pipline_results = pipline(container.pro_p, container.ligand_coords, 10,
                                  container.tmp_path, ligand_path=container.lig_p,
                                  plip_results=plip_results)


        pures = PuresPredict(container.pro_p, pipline_results["XA_pocket_mol"])
        pures.run()
        pures_res_results, pures_mol = pures.analysis()

        data_collate = {
            "ln": ln,
            "pn": pn,
            # based info
            "ligand_center": container.ligand_coords,

            # mol
            "XA_pocket_mol": pipline_results["XA_pocket_mol"],
            "ligand_mol": pipline_results["ligand_mol"],

            # vina
            "score": affinity,
            "interaction_pocket_mol": pipline_results["interaction_pocket_mol"],
            "res_interaction_coords": pipline_results["new_res_coords"],
            "ligand_interaction_coords": pipline_results["new_lig_coords"],
            "interaction_cls": pipline_results["interaction_cls"],
            "interaction_mapping": pipline_results["interaction_mapping"],

            # puresnet
            "pures_mol": pures_mol,
            "pures_mapping": pures_res_results
        }

        process_tools = ProteinLigandProcess()
        collate_data = process_tools.data_collate(data_collate)
    else:
        print("OnlyProtein")
        pocket_loading = osr.read_protein(container.pro_p, remove_Hs=True)
        pures = PuresPredict(container.pro_p, pocket_loading)
        pures.run()
        pures_res_results, pures_mol = pures.analysis()

        data_collate = {
            "ln": "not_input",
            "pn": pn,
            # based info
            "ligand_center": container.ligand_coords,

            # mol
            "XA_pocket_mol": pocket_loading,
            "ligand_mol": None,

            # puresnet
            "pures_mol": pures_mol,
            "pures_mapping": pures_res_results
        }
        process_tools = ProteinProcess()
        collate_data = process_tools.data_collate(data_collate)

    if delete_tmp:
        try:
            shutil.rmtree(container.tmp_path)
        except Exception as e:
            print(e)

    return collate_data


def in_per_process(protein_path, ligand_path):
    try:
        return pre_process(protein_path, ligand_path, True)
    except:
        raise NotImplementedError

if __name__ == "__main__":
    from tqdm import tqdm
    import pickle
    from utils import data_tools

    # 提供蛋白和配体，增加先验知识，interaction
    filePath = "/data/crossdock2020v1.1/"



    cross_dock_train = "/data/it2_tt_0_lowrmsd_mols_train0_fixed.types"
    cross_dock_test = "/data/it2_tt_0_lowrmsd_mols_test0_fixed.types"

    protein_path = "data/Test/SIR3_HUMAN_117_398_0/5d7n_D_rec.pdb"
    ligand_path = "data/Test/SIR3_HUMAN_117_398_0/5d7n_D_rec_4jt9_1ns_lig_tt_min_0.mol2"
    data_collate = in_per_process(protein_path, ligand_path)
    get_t = data_tools.TrainingCollate()
    get_t.data_get({"0": data_collate})
    print(get_t.data[0])

    exit()
    with open(cross_dock_test, "r") as F:
        train_read = F.readlines()

    record_data = dict()
    start = 155000
    end = 160000

    for idx in tqdm(list(range(start, end)), total=end-start):  # 遍历每一条数据process
        if idx != start and idx % 1000 == 0:
            with open("cross/%s.pickle" % idx, "wb") as F:
                pickle.dump(record_data, F)
            record_data = dict()

        k = train_read[idx].split(" ")
        record_data[idx] = in_pre_process(filePath, k)
        if record_data[idx] is None:
            with open("error.txt", "a") as F:
                F.write(str(idx) + "\n")

    with open("cross/%s.pickle" % (idx+1), "wb") as F:
        pickle.dump(record_data, F)



