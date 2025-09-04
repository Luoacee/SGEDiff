import argparse
from pathlib import Path
import torch
from openbabel import openbabel
openbabel.obErrorLog.StopLogging()  # suppress OpenBabel messages

import utils
from lightning_modules import LigandPocketDDPM


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--pdbfile', type=str)
    parser.add_argument('--resi_list', type=str, nargs='+', default=None)
    parser.add_argument('--ref_ligand', type=str, default=None)
    parser.add_argument('--outfile', type=Path)
    parser.add_argument('--n_samples', type=int, default=120)
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--num_nodes_lig', type=int, default=None)
    parser.add_argument('--all_frags', action='store_true')
    parser.add_argument('--sanitize', action='store_true')
    parser.add_argument('--relax', action='store_true')
    parser.add_argument('--resamplings', type=int, default=10)
    parser.add_argument('--jump_length', type=int, default=1)
    parser.add_argument('--timesteps', type=int, default=None)
    args = parser.parse_args()

    # pdb_id = Path(args.pdbfile).stem

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    model = LigandPocketDDPM.load_from_checkpoint(
        args.checkpoint, map_location=device)
    model = model.to(device)

    if args.batch_size is None:
        args.batch_size = args.n_samples
    assert args.n_samples % args.batch_size == 0

    if args.num_nodes_lig is not None:
        num_nodes_lig = torch.ones(args.n_samples, dtype=int) * \
                        args.num_nodes_lig
    else:
        num_nodes_lig = None

    import os

    input_folder_1 = 'sample/2_2'
    input_folder_2 = 'sample/2_2'


    output_folder = 'model_sample'

    # 遍历1_1文件夹
    for folder_idx, input_folder in enumerate([input_folder_1, input_folder_2], 1):
        idx = 0
        for root, dirs, files in os.walk(input_folder):
            for dir_name in dirs:
                protein_folder = os.path.join(root, dir_name)

                protein_path = None
                ligand_path = None

                for file_name in os.listdir(protein_folder):
                    file_path = os.path.join(protein_folder, file_name)
                    if file_name.endswith('.pdb'):
                        protein_path = file_path
                    elif file_name.endswith('.sdf'):
                        ligand_path = file_path

                #################
                #################
                #################

                molecules = []
                for i in range(args.n_samples // args.batch_size):
                    molecules_batch = model.generate_ligands(
                        protein_path, args.batch_size, args.resi_list, ligand_path,
                        num_nodes_lig, args.sanitize, largest_frag=not args.all_frags,
                        relax_iter=(200 if args.relax else 0),
                        resamplings=args.resamplings, jump_length=args.jump_length,
                        timesteps=args.timesteps)
                    molecules.append(molecules_batch)

                # Make SDF files
                # utils.write_sdf_file(args.outfile, molecules)

                #################
                #################
                #################

                output_dir = os.path.join(output_folder, str(folder_idx))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                output_file_name = f"{idx}-{dir_name}-{os.path.basename(ligand_path)}.pt"

                output_file_path = os.path.join(output_dir, output_file_name)
                torch.save(molecules, output_file_path)

                idx += 1

