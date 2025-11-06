import argparse
import json
import os.path
import time
import traceback

import numpy as np
import torch
from ase import Atoms as AseAtoms
from ase.io import read as ase_read
from tqdm import tqdm

from slakonet.atoms import Geometry
from slakonet.main import generate_shell_dict_upto_Z65, SimpleDftb
from slakonet.optim import default_model


class BatchedSlakonet:

    def __init__(self):
        self.model = default_model()
        self.shell_dict = generate_shell_dict_upto_Z65()
        self.updated_skfs = self.model.get_updated_skfs()
        self.h_feed = self.model._create_comprehensive_feed(self.updated_skfs, self.shell_dict, "H")
        self.s_feed = self.model._create_comprehensive_feed(self.updated_skfs, self.shell_dict, "S")

    def predict(
            self,
            structure: AseAtoms,
            k_grid: tuple[int, int, int],
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> dict:
        geometry = Geometry.from_ase_atoms([structure])

        t_start = time.time()
        calc = SimpleDftb(
            geometry=geometry,
            shell_dict=self.shell_dict,
            kpoints=torch.tensor(k_grid),
            h_feed=self.h_feed,
            s_feed=self.s_feed,
            nelectron=self.model._calculate_system_electrons(geometry, self.updated_skfs),
            device=device,
            with_eigenvectors=False,
        )

        def tensor_to_list(t: torch.Tensor) -> list:
            return t.detach().tolist()

        try:
            properties = {
                "eigenvalues": tensor_to_list(calc()),
                "success": True,
                "reason": "Success"
            }
        except Exception as e:
            properties = {
                "success": False,
                "reason": str(e)
            }

        properties["time"] = time.time() - t_start
        return properties


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("structures", nargs="*")
    args = parser.parse_args()

    for f in args.structures:
        assert os.path.isfile(f)

    model = BatchedSlakonet()
    results = []

    with torch.no_grad():
        for f in tqdm(args.structures, desc="Predicting"):

            atoms = ase_read(f)
            try:
                results.append(model.predict(
                    structure=atoms,
                    k_grid=(6, 6, 6),
                ))
            except Exception as e:
                print(traceback.format_exc())

            with open(f + ".slakonet.json", "w") as f:
                json.dump(results[-1], f)

    print(f"Result status:")
    for r in results:
        print(f"    {r['reason']}")

    n_success = sum(r["success"] for r in results)
    n_fail = sum(not r["success"] for r in results)
    print(f"{n_success}/{n_success + n_fail} successful")
    av_time = np.mean([r["time"] for r in results if r["success"]])
    print(f"    Average successful time: {av_time}")
    av_time = np.mean([r["time"] for r in results if not r["success"]])
    print(f"    Average fail time: {av_time}")
