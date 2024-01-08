import argparse
import os
import re
import sys

import torch
from tqdm import tqdm

sys.path.append(".")
import internlm  # noqa: E402,F401 # pylint: disable=W0611,C0413


def load(fp):
    with open(fp, "rb") as f:
        pt_data = torch.load(f, map_location="cpu")
    return pt_data


def revert(filename, src, tgt, layers_per_stage, num_experts):
    matched = re.match("model_tp([0-9]+)_pp([0-9]+).pt", filename)
    tp, pp = int(matched.group(1)), int(matched.group(2))
    global_layer_id = pp * layers_per_stage
    print(f"Reverting checkpoints to MoE from {filename}...")

    state = load(os.path.join(src, filename))
    for layer_i in range(layers_per_stage):
        w1s = state.pop((f"model.blocks.{layer_i}.mlp.w1.weight"))
        w2s = state.pop((f"model.blocks.{layer_i}.mlp.w2.weight"))
        w3s = state.pop((f"model.blocks.{layer_i}.mlp.w3.weight"))

        for expert_id in range(num_experts):
            moe_state = {}
            moe_state[f"model.blocks.{layer_i}.mlp.moe_layer.experts.experts.{expert_id}.w1.weight"] = w1s.clone()
            moe_state[f"model.blocks.{layer_i}.mlp.moe_layer.experts.experts.{expert_id}.w2.weight"] = w2s.clone()
            moe_state[f"model.blocks.{layer_i}.mlp.moe_layer.experts.experts.{expert_id}.w3.weight"] = w3s.clone()

            torch.save(
                moe_state,
                os.path.join(tgt, f"model_moe_layer{global_layer_id}_expert{expert_id}_tp{tp}.pt"),
            )

        global_layer_id += 1

    torch.save(state, os.path.join(tgt, filename))


def print_args(args):
    print("-------------- Arguments --------------")
    print(f"Source Path: {args.src}")
    print(f"Target Path: {args.tgt}")
    print(f"Expert Number: {args.num_experts}")
    print(f"TopK experts choice: {args.topk}")
    print("---------------------------------------")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src", type=str, help="Input folder")
    parser.add_argument("--tgt", type=str, help="Output folder")
    parser.add_argument("--num-experts", type=int, help="Number of experts")
    parser.add_argument("--topk", type=int, help="TopK experts choice")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    print_args(args)

    model_config = load(os.path.join(args.src, "model_config.pt"))

    fns = list(os.listdir(args.src))
    model_fns = []
    for fn in fns:
        if fn.startswith("model_t") and not fn.endswith("md5"):
            model_fns.append(fn)
    max_tp, max_pp = -1, -1
    for fn in model_fns:
        _, tp, pp = os.path.splitext(fn)[0].split("_")
        max_pp = max(max_pp, int(pp[2:]) + 1)
        max_tp = max(max_tp, int(tp[2:]) + 1)
    layers_per_stage = model_config["num_layers"] // max_pp

    for fn in tqdm(model_fns):
        revert(fn, args.src, args.tgt, layers_per_stage, args.num_experts)

    model_config["num_experts"] = args.num_experts
    model_config["moe_gate_k"] = args.topk
    torch.save(model_config, os.path.join(args.tgt, "model_config.pt"))
