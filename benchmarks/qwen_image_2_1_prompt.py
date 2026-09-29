"""Encode and cache Qwen-Image-2.1 text conditioning with Qwen3-VL."""

from pathlib import Path
from typing import Annotated

import torch
from safetensors.torch import save_file
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from typed_args import Arg, TypedArgs

SYSTEM_PROMPT = "Comprehend and analyze the provided prompt."
TEMPLATE = (
    f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


class Args(TypedArgs):
    model: Annotated[Path, Arg(help="downloaded Qwen-Image-2.1 directory")]
    prompt: Annotated[str, Arg(help="text prompt")]
    output: Annotated[Path, Arg(help="conditioning safetensors output")]
    device_map: Annotated[str, Arg("--device-map")] = "auto"


def main() -> None:
    args = Args.parse_args()
    processor = Qwen3VLProcessor.from_pretrained(args.model / "processor")
    encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model / "text_encoder",
        dtype=torch.bfloat16,
        device_map=args.device_map,
        low_cpu_mem_usage=True,
    ).eval()
    system_message = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}
    ]
    system_ids = processor.apply_chat_template(
        system_message, tokenize=True, return_dict=False
    )
    drop_tokens = len(system_ids[0])
    inputs = processor(
        text=[TEMPLATE.format(args.prompt or " ")],
        padding=True,
        padding_side="left",
        return_tensors="pt",
    ).to(encoder.device)
    language_model = getattr(encoder.model, "language_model", encoder.model)
    handle = language_model.norm.register_forward_hook(
        lambda _module, hook_inputs, _output: hook_inputs[0]
    )
    try:
        with torch.inference_mode():
            outputs = encoder(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_hidden_states=True,
            )
    finally:
        handle.remove()
    hidden = outputs.hidden_states[-1][inputs.attention_mask.bool()]
    hidden = hidden[drop_tokens:].unsqueeze(0).contiguous()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"prompt_embeds": hidden},
        args.output,
        metadata={
            "format": "rxla-qwen-image-2.1-conditioning-v1",
            "prompt": args.prompt,
        },
    )
    print(f"shape={tuple(hidden.shape)} dtype={hidden.dtype}")


if __name__ == "__main__":
    main()
