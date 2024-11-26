import os

import torch
import numpy as np
from xfold.alphafold3 import AlphaFold3
from xfold.params import import_jax_weights_


def main():
    debug_input_path = "./debug_input"

    batch = {}

    for f_name in os.listdir(debug_input_path):
        if f_name.endswith(".npy"):
            f_path = os.path.join(debug_input_path, f_name)
            feature_name = f_name.split(".")[0]

            data = np.load(f_path)
            data = torch.from_numpy(data)
            batch[feature_name] = data.cuda()

    print(batch.keys())

    model = AlphaFold3().cuda()
    model.eval()
    import_jax_weights_(
        model, "/home/svu/e0917621/scratch/Protenix/af3.bin.zst")

    with torch.inference_mode():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            model(batch)


if __name__ == "__main__":
    main()
