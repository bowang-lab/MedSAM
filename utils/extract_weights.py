# %%
import torch
import argparse
import os
join = os.path.join

# %%
parser = argparse.ArgumentParser()
parser.add_argument("-from_pth", type=str,
                    help="Path to the .pth file from which the weights will be extracted")
parser.add_argument("-to_pth", type=str,
                    help="Path to the .pth file to which the weights will be saved")
args = parser.parse_args()

# %%
from_pth = args.from_pth
to_pth = args.to_pth

# %%
from_pth = torch.load(from_pth, map_location='cpu')
assert "model" in from_pth.keys(), "The .pth file does not contain the model weights"
weights = from_pth["model"]
torch.save(weights, to_pth)
print("Weights are saved to {}".format(to_pth))