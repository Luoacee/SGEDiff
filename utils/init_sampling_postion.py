from torch_scatter import scatter_mean

def init_position(data):
    try:
        center_pos = data.ligand_center
        print("\nInitializing center position method: Ligand")
        return center_pos
    except:
        pass
    try:
        center_pos = scatter_mean(data["protein_atom_coords"][data["pures_idx"]], data.pures_idx_batch, dim=0)
        print("\nInitializing center position method: Predicted Pocket")
        return center_pos
    except:
        pass
    try:
        center_pos = scatter_mean(data.protein_atom_coords, data.protein_atom_feature_batch, dim=0)
        print("\nInitializing center position method: Full Protein")
        return center_pos
    except:
        pass

    raise NotImplementedError
