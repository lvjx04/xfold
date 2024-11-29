import os

import torch
import torch.utils._pytree as pytree
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
            result = model(batch)
            result['__identifier__'] = model.__identifier__.numpy().tobytes()

        result = pytree.tree_map_only(
            torch.Tensor,
            lambda x: x.to(
                dtype=torch.float32) if x.dtype == torch.bfloat16 else x,
            result,
        )
        result = pytree.tree_map_only(
            torch.Tensor, lambda x: x.cpu().detach().numpy(), result)

    import pdb
    pdb.set_trace()


if __name__ == "__main__":
    main()
