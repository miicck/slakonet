import argparse
import os.path
import traceback

import torch
from jarvis.core.atoms import Atoms, ase_to_atoms
from ase.io import read as ase_read
from jarvis.core.kpoints import Kpoints3D as Kpoints
from tqdm import tqdm

from slakonet.atoms import Geometry
from slakonet.main import generate_shell_dict_upto_Z65
from slakonet.optim import kpts_to_klines, default_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("structures", nargs="*")
    args = parser.parse_args()

    for f in args.structures:
        assert os.path.isfile(f)

    model = default_model()
    shell_dict = generate_shell_dict_upto_Z65()

    n_success = 0
    n_fail = 0

    with torch.no_grad():
        for f in tqdm(args.structures, desc="Predicting"):

            atoms = ase_to_atoms(ase_read(f))
            geometry = Geometry.from_ase_atoms([atoms.ase_converter()])
            kpoints = Kpoints().kpath(atoms, line_density=20)
            klines = kpts_to_klines(kpoints.kpts, default_points=2)

            try:

                properties, success = model.compute_multi_element_properties(
                    geometry=geometry,
                    shell_dict=shell_dict,
                    klines=klines,
                    get_fermi=True,
                    with_eigenvectors=True,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )

                if not success:
                    raise RuntimeError("Failed to compute properties")
                n_success += 1

            except Exception as e:

                n_fail += 1
                print(traceback.format_exc())

    print(f"{n_success}/{n_success + n_fail} successful")
