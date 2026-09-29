"""Generate a deterministic tiny Qwen-Image-2.1 transformer oracle."""

from pathlib import Path
from typing import Annotated

import torch
from safetensors.torch import save_file
from torch import nn
from torch.nn import functional as F
from typed_args import Arg, TypedArgs


class Args(TypedArgs):
    output_dir: Annotated[Path, Arg(help="directory for oracle safetensors")]


class ZeroCenterRMSNorm(nn.Module):
    def __init__(self, width: int, epsilon: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(width))
        self.epsilon = epsilon

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value_f32 = value.float()
        normalized = value_f32 * torch.rsqrt(
            value_f32.square().mean(-1, keepdim=True) + self.epsilon
        )
        return (normalized * (self.weight.float() + 1)).to(value.dtype)


class TextProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_norm = ZeroCenterRMSNorm(24)
        self.in_layer = nn.Linear(24, 32, bias=False)
        self.out_layer = nn.Linear(32, 32, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.out_layer(F.gelu(self.in_layer(self.text_norm(value)), approximate="tanh"))


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(32, 32, bias=False)
        self.to_k = nn.Linear(32, 32, bias=False)
        self.to_v = nn.Linear(32, 32, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(32, 32, bias=False), nn.Dropout(0.0)])
        self.norm_q = nn.RMSNorm(8, eps=1e-6)
        self.norm_k = nn.RMSNorm(8, eps=1e-6)

    def forward(
        self, value: torch.Tensor, angles: torch.Tensor, allowed: torch.Tensor
    ) -> torch.Tensor:
        shape = (*value.shape[:2], 4, 8)
        query = self.norm_q(self.to_q(value).reshape(shape)).transpose(1, 2)
        key = self.norm_k(self.to_k(value).reshape(shape)).transpose(1, 2)
        projected_value = self.to_v(value).reshape(shape).transpose(1, 2)

        def rotate(tensor: torch.Tensor) -> torch.Tensor:
            paired = torch.view_as_complex(tensor.float().reshape(*tensor.shape[:-1], -1, 2))
            return torch.view_as_real(paired * torch.polar(torch.ones_like(angles), angles)).flatten(3)

        attended = F.scaled_dot_product_attention(
            rotate(query), rotate(key), projected_value, attn_mask=allowed
        )
        return self.to_out[0](attended.transpose(1, 2).flatten(2))


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(32, 64, bias=False)
        self.out = nn.Linear(64, 32, bias=False)
        self.gate_layer = nn.Linear(32, 64, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.out(F.silu(self.gate_layer(value)) * self.proj(value))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Attention()
        self.img_mlp = FeedForward()

    def forward(
        self,
        value: torch.Tensor,
        modulation: torch.Tensor,
        angles: torch.Tensor,
        allowed: torch.Tensor,
    ) -> torch.Tensor:
        mod1, mod2 = modulation.chunk(2, -1)
        for mod, operation in ((mod1, self.attn), (mod2, self.img_mlp)):
            scale, gate = mod.chunk(2, -1)
            normalized = F.layer_norm(value, (32,), eps=1e-6) * (1 + scale)
            delta = operation(normalized, angles, allowed) if operation is self.attn else operation(normalized)
            value = value + gate.tanh() * delta
        return value


class TimestepEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(256, 32, bias=False)
        self.linear_2 = nn.Linear(32, 32, bias=False)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        frequency = torch.exp(-torch.log(torch.tensor(10000.0)) * torch.arange(128) / 128)
        phase = timestep[:, None] * 1000 * frequency[None]
        return self.linear_2(F.silu(self.linear_1(torch.cat([phase.cos(), phase.sin()], -1))))


class TimeTextEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.timestep_embedder = TimestepEmbedder()

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        return self.timestep_embedder(timestep)


class NormOut(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(32, 32, bias=False)


class TinyQwenImage21(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_text_embed = TimeTextEmbed()
        self.txt_in = TextProjection()
        self.img_in = nn.Linear(8, 32, bias=False)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(32, 128, bias=False))
        self.transformer_blocks = nn.ModuleList([Block(), Block()])
        self.norm_out = NormOut()
        self.proj_out = nn.Linear(32, 8, bias=False)

    @staticmethod
    def angles(text_tokens: int) -> torch.Tensor:
        phases: list[float] = []
        for token in range(text_tokens + 6):
            if token < text_tokens:
                positions = (token, token, token)
            else:
                image_index = token - text_tokens
                positions = (text_tokens, image_index // 3 - 1, image_index % 3 - 2)
            for position, width in zip(positions, (2, 2, 4), strict=True):
                phases.extend(position * 10000 ** (-2 * pair / width) for pair in range(width // 2))
        return torch.tensor(phases).reshape(1, 1, text_tokens + 6, 4)

    def forward(
        self, image: torch.Tensor, timestep: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        text_tokens = context.shape[1]
        value = torch.cat([self.txt_in(context), self.img_in(image)], 1)
        tokens = value.shape[1]
        query = torch.arange(tokens)[:, None]
        key = torch.arange(tokens)[None, :]
        allowed = ((key <= query) | ((query >= text_tokens) & (key >= text_tokens)))[None, None]
        time = self.time_text_embed(torch.cat([timestep, timestep.new_zeros(1)]))
        raw_modulation = self.modulation(time)
        modulation = torch.cat(
            [
                raw_modulation[-1:].expand(image.shape[0], text_tokens, -1),
                raw_modulation[:-1, None].expand(-1, 6, -1),
            ],
            1,
        )
        angles = self.angles(text_tokens)
        for block in self.transformer_blocks:
            value = block(value, modulation, angles, allowed)
        value = value[:, text_tokens:]
        scale = self.norm_out.linear(F.silu(time[:-1]))[:, None]
        return self.proj_out(F.layer_norm(value, (32,), eps=1e-6) * (1 + scale))


def main() -> None:
    args = Args.parse_args()
    torch.manual_seed(20260929)
    model = TinyQwenImage21().float().eval()
    image = torch.randn(1, 6, 8)
    timestep = torch.tensor([0.625], dtype=torch.float32)
    context = torch.randn(1, 5, 24)
    with torch.inference_mode():
        output = model(image, timestep, context)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: value.detach().contiguous() for name, value in model.state_dict().items()},
        args.output_dir / "weights.safetensors",
    )
    save_file(
        {"image": image, "timestep": timestep, "context": context, "output": output},
        args.output_dir / "oracle.safetensors",
    )
    print(f"parameters={sum(p.numel() for p in model.parameters())} checksum={output.sum().item():.9f}")


if __name__ == "__main__":
    main()
