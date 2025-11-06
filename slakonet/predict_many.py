import argparse
import os.path
import time

import numpy as np
import torch
from ase.io import read as ase_read
from jarvis.core.atoms import ase_to_atoms
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

    results = []

    with torch.no_grad():
        for f in tqdm(args.structures, desc="Predicting"):

            atoms = ase_to_atoms(ase_read(f))
            geometry = Geometry.from_ase_atoms([atoms.ase_converter()])
            kpoints = Kpoints().kpath(atoms, line_density=20)
            klines = kpts_to_klines(kpoints.kpts, default_points=2)

            t_start = time.time()
            try:

                properties, success = model.compute_multi_element_properties(
                    geometry=geometry,
                    shell_dict=shell_dict,
                    klines=klines,
                    get_fermi=True,
                    with_eigenvectors=True,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )

                results.append({
                    "success": success,
                    "reason": "Prediction failed",
                    "time": time.time() - t_start,
                })

            except Exception as e:

                results.append({
                    "success": False,
                    "reason": f"Exception: {e}",
                    "time": time.time() - t_start,
                })

    n_success = sum(r["success"] for r in results)
    n_fail = sum(not r["success"] for r in results)
    print(f"{n_success}/{n_success + n_fail} successful")
    av_time = np.mean([r["time"] for r in results if r["success"]])
    print(f"    Average successful time: {av_time}")
    av_time = np.mean([r["time"] for r in results if not r["success"]])
    print(f"    Average fail time: {av_time}")
