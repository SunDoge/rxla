"""Run the official 7B Qwen-Image-2.1 transformer on a tiny real topology.

The short 2x2 latent grid keeps CPU reference computation tractable while
exercising every official layer and checkpoint tensor. The resulting fixture
is consumed by RXLA's grouped-W8 full-model oracle.
"""

import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Annotated

import torch
import diffusers.utils.peft_utils as peft_utils
from safetensors import safe_open
from safetensors.torch import save_file
from typed_args import Arg, TypedArgs


class Args(TypedArgs):
    source: Annotated[Path, Arg(help="official transformer_qwenimage21.py")]
    transformer: Annotated[Path, Arg(help="downloaded official transformer directory")]
    output: Annotated[Path, Arg(help="oracle safetensors output")]
    w8: Annotated[Path | None, Arg("--w8", help="optional RXLA grouped-W8 checkpoint")] = None
    f32: Annotated[bool, Arg("--f32", help="run the numerical diagnostic in float32")] = False


def load_official_class(source: Path):
    # Diffusers 0.35 predates this inference-only convenience decorator. LoRA
    # scaling is not involved in the checkpoint oracle, so its identity
    # behavior is the exact specialization needed here.
    if not hasattr(peft_utils, "apply_lora_scale"):
        def apply_lora_scale(_argument: str):
            return lambda function: function

        peft_utils.apply_lora_scale = apply_lora_scale
    module_name = "diffusers.models.transformers.transformer_qwenimage21_oracle"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import official source {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    # The published 2.1 source forwards Diffusers' newer distributed-attention
    # argument. Our local 0.35 runtime has identical single-process attention
    # semantics but predates that keyword.
    if "parallel_config" not in inspect.signature(module.dispatch_attention_fn).parameters:
        dispatch_attention_fn = module.dispatch_attention_fn

        def compatible_dispatch_attention_fn(*args, parallel_config=None, **kwargs):
            del parallel_config
            return dispatch_attention_fn(*args, **kwargs)

        module.dispatch_attention_fn = compatible_dispatch_attention_fn
    return module.QwenImage21Transformer2DModel


def main() -> None:
    args = Args.parse_args()
    cls = load_official_class(args.source)
    model = cls.from_pretrained(
        args.transformer,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval()
    if args.f32:
        model.float()
    if args.w8 is not None:
        with safe_open(args.w8, framework="pt", device="cpu") as checkpoint:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    value = checkpoint.get_tensor(name)
                    if value.ndim == 2:
                        scale_name = name.removesuffix(".weight") + ".scale"
                        scale = checkpoint.get_tensor(scale_name)
                        value = (value.float() - 128).reshape(*scale.shape, 64)
                        value = value * scale.unsqueeze(-1)
                        value = value.reshape(parameter.shape)
                    parameter.copy_(value)
    generator = torch.Generator().manual_seed(20260929)
    dtype = torch.float32 if args.f32 else torch.bfloat16
    image = torch.randn(1, 4, 64, generator=generator, dtype=dtype)
    context = torch.randn(1, 8, 4096, generator=generator, dtype=dtype)
    timestep = torch.tensor([0.75], dtype=dtype)
    # Eight text positions followed by one VLM image slot; the transformer
    # expands that final slot to the target's 2x2 latent tokens.
    image_mask = torch.zeros(1, 9, dtype=torch.bool)
    image_mask[:, -1] = True
    with torch.inference_mode():
        joint_output = model(
            hidden_states=image,
            encoder_hidden_states=context,
            timestep=timestep,
            img_shapes=[[(1, 2, 2)]],
            img_mask=image_mask,
            return_dict=False,
        )[0]
        # The official module returns the complete text/image joint stream;
        # the pipeline and RXLA model expose only the target latent prediction.
        output = joint_output[:, -image.shape[1] :]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "image": image,
            "context": context,
            "timestep": timestep,
            "output": output,
        },
        args.output,
        metadata={"format": "qwen-image-2.1-official-transformer-oracle-v1"},
    )
    print(
        f"shape={tuple(output.shape)} checksum={output.float().sum().item():.9f} "
        f"minimum={output.float().min().item():.9f} maximum={output.float().max().item():.9f}"
    )


if __name__ == "__main__":
    main()
