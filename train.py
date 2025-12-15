#!/usr/bin/env python3

import json
import signal
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map


@dataclass
class TrainingConfig:
    model_type: str = "Image Classification"
    base_model: str = "ResNet-50"
    model_path: str = ""
    data_path: str = ""
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    image_size: int = 224
    num_classes: int = 10
    augmentation: bool = True
    pretrained: bool = True
    max_seq_len: int = 512


@dataclass
class TrainingMetrics:
    epoch: int
    step: int
    loss: float
    accuracy: float
    iter_per_sec: float
    memory_gb: float
    learning_rate: float


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm(out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm(channels)
        self.downsample = (
            nn.Conv2d(channels, channels, 1, stride) if stride > 1 else None
        )

    def __call__(self, x: mx.array) -> mx.array:
        identity = x
        out = nn.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        return nn.relu(out + identity)


class ResNet(nn.Module):
    def __init__(self, num_classes: int, layers: list = [2, 2, 2, 2]):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3)
        self.bn1 = nn.BatchNorm(64)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.fc = nn.Linear(512, num_classes)
        self.num_classes = num_classes

    def _make_layer(self, channels: int, blocks: int, stride: int = 1) -> list:
        layers = [ResidualBlock(channels, stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(channels))
        return layers

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.bn1(self.conv1(x)))
        x = nn.max_pool2d(x, 3, 2, 1)
        for layer in self.layer1:
            x = layer(x)
        for layer in self.layer2:
            x = layer(x)
        for layer in self.layer3:
            x = layer(x)
        for layer in self.layer4:
            x = layer(x)
        x = mx.mean(x, axis=(1, 2))
        return self.fc(x)


class VGGBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_convs: int):
        super().__init__()
        self.convs = []
        for i in range(num_convs):
            self.convs.append(
                nn.Conv2d(
                    in_channels if i == 0 else out_channels, out_channels, 3, 1, 1
                )
            )
            self.convs.append(nn.BatchNorm(out_channels))

    def __call__(self, x: mx.array) -> mx.array:
        for i in range(0, len(self.convs), 2):
            x = nn.relu(self.convs[i + 1](self.convs[i](x)))
        return nn.max_pool2d(x, 2, 2)


class VGG16(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = [
            VGGBlock(3, 64, 2),
            VGGBlock(64, 128, 2),
            VGGBlock(128, 256, 3),
            VGGBlock(256, 512, 3),
            VGGBlock(512, 512, 3),
        ]
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.Linear(4096, 4096),
            nn.Linear(4096, num_classes),
        )
        self.num_classes = num_classes

    def __call__(self, x: mx.array) -> mx.array:
        for block in self.features:
            x = block(x)
        x = x.reshape(x.shape[0], -1)
        return self.classifier(x)


class MBConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: int = 4,
        stride: int = 1,
    ):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.expand = (
            nn.Conv2d(in_channels, hidden_dim, 1) if expand_ratio != 1 else None
        )
        self.bn1 = nn.BatchNorm(hidden_dim) if expand_ratio != 1 else None
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1)
        self.bn2 = nn.BatchNorm(hidden_dim)
        self.project = nn.Conv2d(hidden_dim, out_channels, 1)
        self.bn3 = nn.BatchNorm(out_channels)
        self.use_residual = stride == 1 and in_channels == out_channels

    def __call__(self, x: mx.array) -> mx.array:
        identity = x
        if self.expand:
            x = nn.relu(self.bn1(self.expand(x)))
        x = nn.relu(self.bn2(self.dwconv(x)))
        x = self.bn3(self.project(x))
        if self.use_residual:
            x = x + identity
        return x


class EfficientNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.stem = ConvBlock(3, 32, 3, 2, 1)
        self.blocks = [
            MBConvBlock(32, 16, 1, 1),
            MBConvBlock(16, 24, 6, 2),
            MBConvBlock(24, 40, 6, 2),
            MBConvBlock(40, 80, 6, 2),
            MBConvBlock(80, 112, 6, 1),
            MBConvBlock(112, 192, 6, 2),
            MBConvBlock(192, 320, 6, 1),
        ]
        self.head = ConvBlock(320, 1280, 1, 1, 0)
        self.fc = nn.Linear(1280, num_classes)
        self.num_classes = num_classes

    def __call__(self, x: mx.array) -> mx.array:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        x = mx.mean(x, axis=(1, 2))
        return self.fc(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.stem = ConvBlock(3, 32, 3, 2, 1)
        self.blocks = [
            MBConvBlock(32, 16, 1, 1),
            MBConvBlock(16, 24, 6, 2),
            MBConvBlock(24, 32, 6, 2),
            MBConvBlock(32, 64, 6, 2),
            MBConvBlock(64, 96, 6, 1),
            MBConvBlock(96, 160, 6, 2),
            MBConvBlock(160, 320, 6, 1),
        ]
        self.head = ConvBlock(320, 1280, 1, 1, 0)
        self.fc = nn.Linear(1280, num_classes)
        self.num_classes = num_classes

    def __call__(self, x: mx.array) -> mx.array:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        x = mx.mean(x, axis=(1, 2))
        return self.fc(x)


class PatchEmbedding(nn.Module):
    def __init__(
        self, img_size: int, patch_size: int, in_channels: int, embed_dim: int
    ):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, patch_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.proj(x)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def __call__(self, x: mx.array) -> mx.array:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .transpose(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(0, 1, 3, 2)) * (self.head_dim**-0.5)
        attn = mx.softmax(attn, axis=-1)
        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        return self.proj(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = mx.zeros((1, 1, embed_dim))
        self.pos_embed = mx.zeros((1, num_patches + 1, embed_dim))
        self.blocks = [TransformerBlock(embed_dim, num_heads) for _ in range(depth)]
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.num_classes = num_classes

    def __call__(self, x: mx.array) -> mx.array:
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = mx.broadcast_to(self.cls_token, (B, 1, self.cls_token.shape[-1]))
        x = mx.concatenate([cls_tokens, x], axis=1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x[:, 0])


class CSPBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        mid_channels = out_channels // 2
        self.conv1 = ConvBlock(in_channels, mid_channels, 1, 1, 0)
        self.conv2 = ConvBlock(in_channels, mid_channels, 1, 1, 0)
        self.conv3 = ConvBlock(mid_channels, mid_channels, 3, 1, 1)
        self.conv4 = ConvBlock(mid_channels * 2, out_channels, 1, 1, 0)

    def __call__(self, x: mx.array) -> mx.array:
        x1 = self.conv1(x)
        x2 = self.conv3(self.conv2(x))
        return self.conv4(mx.concatenate([x1, x2], axis=-1))


class YOLOv8(nn.Module):
    def __init__(self, num_classes: int, width_mult: float = 0.25):
        super().__init__()
        base = int(64 * width_mult)
        self.stem = ConvBlock(3, base, 3, 2, 1)
        self.stage1 = nn.Sequential(
            ConvBlock(base, base * 2, 3, 2, 1), CSPBlock(base * 2, base * 2)
        )
        self.stage2 = nn.Sequential(
            ConvBlock(base * 2, base * 4, 3, 2, 1), CSPBlock(base * 4, base * 4)
        )
        self.stage3 = nn.Sequential(
            ConvBlock(base * 4, base * 8, 3, 2, 1), CSPBlock(base * 8, base * 8)
        )
        self.stage4 = nn.Sequential(
            ConvBlock(base * 8, base * 16, 3, 2, 1), CSPBlock(base * 16, base * 16)
        )
        self.head = nn.Linear(base * 16, num_classes + 5)
        self.num_classes = num_classes

    def __call__(self, x: mx.array) -> mx.array:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = mx.mean(x, axis=(1, 2))
        return self.head(x)


class LLMModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dims: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dims)
        self.pos_embedding = nn.Embedding(max_seq_len, dims)
        self.layers = [TransformerBlock(dims, num_heads) for _ in range(num_layers)]
        self.norm = nn.LayerNorm(dims)
        self.lm_head = nn.Linear(dims, vocab_size)
        self.max_seq_len = max_seq_len
        self.num_classes = vocab_size

    def __call__(self, input_ids: mx.array) -> mx.array:
        B, L = input_ids.shape
        positions = mx.arange(L)
        x = self.embedding(input_ids) + self.pos_embedding(positions)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)


class ImageDataLoader:
    def __init__(
        self,
        data_path: str,
        batch_size: int,
        image_size: int,
        augmentation: bool = True,
    ):
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.image_size = image_size
        self.augmentation = augmentation
        self.classes = sorted([d.name for d in self.data_path.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = self._load_samples()
        self.current_idx = 0

    def _load_samples(self) -> list:
        samples = []
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for class_name in self.classes:
            class_dir = self.data_path / class_name
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in extensions:
                    samples.append((str(img_path), self.class_to_idx[class_name]))
        return samples

    def __len__(self) -> int:
        return len(self.samples) // self.batch_size

    def __iter__(self):
        import random

        random.shuffle(self.samples)
        self.current_idx = 0
        return self

    def __next__(self) -> tuple:
        if self.current_idx >= len(self.samples):
            raise StopIteration

        batch_samples = self.samples[
            self.current_idx : self.current_idx + self.batch_size
        ]
        self.current_idx += self.batch_size

        if len(batch_samples) < self.batch_size:
            raise StopIteration

        images = []
        labels = []

        for img_path, label in batch_samples:
            img = self._load_image(img_path)
            images.append(img)
            labels.append(label)

        return mx.array(images), mx.array(labels)

    def _load_image(self, path: str) -> list:
        try:
            from PIL import Image

            img = Image.open(path).convert("RGB")
            img = img.resize((self.image_size, self.image_size))

            if self.augmentation:
                import random

                if random.random() > 0.5:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)

            img_array = list(img.getdata())
            img_array = [[p[0] / 255.0, p[1] / 255.0, p[2] / 255.0] for p in img_array]
            img_array = [
                img_array[i : i + self.image_size]
                for i in range(0, len(img_array), self.image_size)
            ]
            return img_array
        except:
            return [
                [[0.5, 0.5, 0.5] for _ in range(self.image_size)]
                for _ in range(self.image_size)
            ]


class TextDataLoader:
    def __init__(self, data_path: str, batch_size: int, max_seq_len: int):
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.data = self._load_data()
        self.current_idx = 0

    def _load_data(self) -> mx.array:
        all_tokens = []
        if self.data_path.is_dir():
            files = (
                list(self.data_path.glob("**/*.bin"))
                + list(self.data_path.glob("**/*.json"))
                + list(self.data_path.glob("**/*.txt"))
                + list(self.data_path.glob("**/*.jsonl"))
            )
            for f in sorted(files):
                all_tokens.extend(self._load_file(f))
        else:
            all_tokens = self._load_file(self.data_path)
        return mx.array(all_tokens, dtype=mx.int32)

    def _load_file(self, path: Path) -> list:
        tokens = []
        if path.suffix == ".bin":
            import numpy as np

            tokens = np.fromfile(path, dtype=np.int32).tolist()
        elif path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                if isinstance(item, dict) and "tokens" in item:
                    tokens.extend(item["tokens"])
                elif isinstance(item, dict) and "text" in item:
                    tokens.extend([ord(c) for c in item["text"]])
                elif isinstance(item, list):
                    tokens.extend(item)
        elif path.suffix == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        if "tokens" in item:
                            tokens.extend(item["tokens"])
                        elif "text" in item:
                            tokens.extend([ord(c) for c in item["text"]])
        else:
            with open(path, "r", encoding="utf-8") as f:
                tokens = [ord(c) for c in f.read()]
        return tokens

    def __len__(self) -> int:
        return max(1, (len(self.data) - 1) // (self.batch_size * self.max_seq_len))

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self) -> tuple:
        if self.current_idx >= len(self.data) - self.batch_size * self.max_seq_len - 1:
            raise StopIteration
        batch_tokens, batch_targets = [], []
        for _ in range(self.batch_size):
            start = self.current_idx
            end = start + self.max_seq_len
            if end + 1 > len(self.data):
                self.current_idx = 0
                start, end = 0, self.max_seq_len
            batch_tokens.append(self.data[start:end])
            batch_targets.append(self.data[start + 1 : end + 1])
            self.current_idx = end
        return mx.stack(batch_tokens), mx.stack(batch_targets)


class Trainer:
    def __init__(
        self,
        config: TrainingConfig,
        log_callback: Optional[Callable] = None,
        metrics_callback: Optional[Callable] = None,
    ):
        self.config = config
        self.log_callback = log_callback or print
        self.metrics_callback = metrics_callback
        self.stop_requested = False
        self.model = None
        self.optimizer = None

    def log(self, message: str):
        self.log_callback(message)

    def report_metrics(self, metrics: TrainingMetrics):
        if self.metrics_callback:
            self.metrics_callback(asdict(metrics))

    def create_model(self):
        model_type = self.config.model_type
        base_model = self.config.base_model
        num_classes = self.config.num_classes

        if model_type == "LLM":
            vocab_size = 32000
            self.log(f"Creating LLM model (vocab={vocab_size})")
            return LLMModel(vocab_size, max_seq_len=self.config.max_seq_len)

        self.log(f"Creating {base_model} for {num_classes} classes")

        if "ResNet-50" in base_model:
            return ResNet(num_classes, [3, 4, 6, 3])
        elif "ResNet-101" in base_model:
            return ResNet(num_classes, [3, 4, 23, 3])
        elif "VGG" in base_model:
            return VGG16(num_classes)
        elif "EfficientNet" in base_model:
            return EfficientNet(num_classes)
        elif "MobileNet" in base_model:
            return MobileNetV2(num_classes)
        elif "ViT" in base_model:
            return VisionTransformer(num_classes, self.config.image_size)
        elif "YOLOv8n" in base_model:
            return YOLOv8(num_classes, 0.25)
        elif "YOLOv8s" in base_model:
            return YOLOv8(num_classes, 0.5)
        elif "YOLOv8m" in base_model:
            return YOLOv8(num_classes, 0.75)
        else:
            return ResNet(num_classes)

    def loss_fn_classification(
        self, model: nn.Module, images: mx.array, labels: mx.array
    ) -> tuple:
        logits = model(images)
        loss = nn.losses.cross_entropy(logits, labels, reduction="mean")
        predictions = mx.argmax(logits, axis=-1)
        accuracy = mx.mean(predictions == labels)
        return loss, accuracy

    def loss_fn_llm(
        self, model: nn.Module, inputs: mx.array, targets: mx.array
    ) -> tuple:
        logits = model(inputs)
        logits = logits.reshape(-1, logits.shape[-1])
        targets = targets.reshape(-1)
        loss = nn.losses.cross_entropy(logits, targets, reduction="mean")
        return loss, mx.array(0.0)

    def train(self):
        self.log("Initializing training...")
        self.log(f"Model: {self.config.base_model} | Type: {self.config.model_type}")

        is_llm = self.config.model_type == "LLM"

        if is_llm:
            data_loader = TextDataLoader(
                self.config.data_path, self.config.batch_size, self.config.max_seq_len
            )
            vocab_size = int(mx.max(data_loader.data).item()) + 1
            self.config.num_classes = max(vocab_size, 256)
        else:
            data_loader = ImageDataLoader(
                self.config.data_path,
                self.config.batch_size,
                self.config.image_size,
                self.config.augmentation,
            )
            self.config.num_classes = len(data_loader.classes)
            self.log(
                f"Found {len(data_loader.classes)} classes: {data_loader.classes[:5]}..."
            )

        self.log(f"Dataset: {len(data_loader)} batches")

        self.model = self.create_model()
        num_params = sum(p.size for _, p in tree_flatten(self.model.parameters()))
        self.log(f"Parameters: {num_params:,}")

        lr_schedule = optim.cosine_decay(
            self.config.learning_rate,
            self.config.epochs * len(data_loader),
            self.config.learning_rate * 0.1,
        )
        self.optimizer = optim.AdamW(learning_rate=lr_schedule, weight_decay=0.01)

        loss_fn = self.loss_fn_llm if is_llm else self.loss_fn_classification
        loss_and_grad_fn = nn.value_and_grad(self.model, loss_fn)

        global_step = 0

        for epoch in range(self.config.epochs):
            if self.stop_requested:
                break

            self.log(f"Epoch {epoch + 1}/{self.config.epochs}")
            epoch_loss, epoch_acc, num_batches = 0.0, 0.0, 0

            for batch_data in data_loader:
                if self.stop_requested:
                    break

                start_time = time.time()

                if is_llm:
                    inputs, targets = batch_data
                    (loss, acc), grads = loss_and_grad_fn(self.model, inputs, targets)
                else:
                    images, labels = batch_data
                    (loss, acc), grads = loss_and_grad_fn(self.model, images, labels)

                grads = tree_map(lambda g: mx.clip(g, -1.0, 1.0), grads)
                self.optimizer.update(self.model, grads)
                mx.eval(self.model.parameters(), self.optimizer.state)

                elapsed = time.time() - start_time
                iter_per_sec = 1.0 / elapsed if elapsed > 0 else 0

                loss_val = loss.item()
                acc_val = acc.item() if not is_llm else 0.0
                epoch_loss += loss_val
                epoch_acc += acc_val
                num_batches += 1
                global_step += 1

                memory_gb = (
                    mx.metal.get_active_memory() / (1024**3)
                    if hasattr(mx, "metal")
                    else 0
                )
                current_lr = (
                    self.optimizer.learning_rate.item()
                    if hasattr(self.optimizer.learning_rate, "item")
                    else self.config.learning_rate
                )

                metrics = TrainingMetrics(
                    epoch=epoch + 1,
                    step=global_step,
                    loss=loss_val,
                    accuracy=acc_val,
                    iter_per_sec=iter_per_sec,
                    memory_gb=memory_gb,
                    learning_rate=current_lr,
                )
                self.report_metrics(metrics)

                if global_step % 10 == 0:
                    self.log(
                        f"Step {global_step} | Loss: {loss_val:.4f} | Acc: {acc_val * 100:.1f}% | {iter_per_sec:.1f} it/s"
                    )

            avg_loss = epoch_loss / max(num_batches, 1)
            avg_acc = epoch_acc / max(num_batches, 1)
            self.log(
                f"Epoch {epoch + 1} | Loss: {avg_loss:.4f} | Acc: {avg_acc * 100:.1f}%"
            )

        if not self.stop_requested:
            self.save_checkpoint("model_final.safetensors")
            self.log("Training completed")
            return True

        return False

    def save_checkpoint(self, filename: str):
        output_dir = Path(self.config.data_path).parent / "checkpoints"
        output_dir.mkdir(parents=True, exist_ok=True)
        weights = dict(tree_flatten(self.model.parameters()))
        mx.save_safetensors(str(output_dir / filename), weights)
        self.log(f"Saved: {filename}")

    def stop(self):
        self.stop_requested = True

    def export(self, output_path: str, formats: list) -> bool:
        if self.model is None:
            self.log("No model to export")
            return False

        output = Path(output_path)
        output.mkdir(parents=True, exist_ok=True)

        weights = dict(tree_flatten(self.model.parameters()))

        for fmt in formats:
            try:
                if "safetensors" in fmt.lower() or "mlx" in fmt.lower():
                    mx.save_safetensors(str(output / "model.safetensors"), weights)
                    self.log(f"Exported: model.safetensors")

                if "mlmodel" in fmt.lower() or "coreml" in fmt.lower():
                    self._export_coreml(output)

                if ".pt" in fmt.lower() or "pytorch" in fmt.lower():
                    self._export_pytorch(output, weights)

                if ".onnx" in fmt.lower():
                    self._export_onnx(output, weights)

                if ".h5" in fmt.lower() or "keras" in fmt.lower():
                    self._export_keras(output, weights)

                if "gguf" in fmt.lower():
                    mx.save_safetensors(
                        str(output / "model_for_gguf.safetensors"), weights
                    )
                    self.log(
                        f"Exported: model_for_gguf.safetensors (convert with llama.cpp)"
                    )

            except Exception as e:
                self.log(f"[WARN] {fmt} export failed: {str(e)}")

        config = {
            "model_type": self.config.model_type,
            "base_model": self.config.base_model,
            "num_classes": self.model.num_classes,
            "image_size": self.config.image_size,
        }
        with open(output / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        self.log(f"Export completed: {output}")
        return True

    def _export_coreml(self, output: Path):
        try:
            import coremltools as ct

            self.log("CoreML export requires traced model, saving config...")
            config = {
                "model_type": self.config.model_type,
                "base_model": self.config.base_model,
                "num_classes": self.model.num_classes,
            }
            with open(output / "coreml_config.json", "w") as f:
                json.dump(config, f)
            self.log("Exported: coreml_config.json (use coremltools to convert)")
        except ImportError:
            self.log("[WARN] coremltools not installed")

    def _export_pytorch(self, output: Path, weights: dict):
        try:
            import numpy as np
            import torch

            state_dict = {}
            for k, v in weights.items():
                arr = np.array(v.tolist())
                state_dict[k] = torch.from_numpy(arr)
            torch.save(state_dict, str(output / "model.pt"))
            self.log("Exported: model.pt")
        except ImportError:
            self.log("[WARN] PyTorch not installed")

    def _export_onnx(self, output: Path, weights: dict):
        try:
            import numpy as np

            np.savez(
                str(output / "model_weights.npz"),
                **{k: np.array(v.tolist()) for k, v in weights.items()},
            )
            self.log("Exported: model_weights.npz (convert to ONNX with torch.onnx)")
        except Exception as e:
            self.log(f"[WARN] ONNX export: {e}")

    def _export_keras(self, output: Path, weights: dict):
        try:
            import h5py
            import numpy as np

            with h5py.File(str(output / "model.h5"), "w") as f:
                for k, v in weights.items():
                    f.create_dataset(k, data=np.array(v.tolist()))
            self.log("Exported: model.h5")
        except ImportError:
            self.log("[WARN] h5py not installed")


class IPCServer:
    def __init__(self):
        self.trainer = None
        self.training_thread = None

    def handle_command(self, cmd: dict) -> dict:
        action = cmd.get("action")

        if action == "train":
            config = TrainingConfig(
                model_type=cmd.get("model_type", "Image Classification"),
                base_model=cmd.get("base_model", "ResNet-50"),
                model_path=cmd.get("model_path", ""),
                data_path=cmd.get("data_path", ""),
                epochs=cmd.get("epochs", 10),
                batch_size=cmd.get("batch_size", 32),
                learning_rate=float(cmd.get("learning_rate", "1e-4")),
                image_size=cmd.get("image_size", 224),
                num_classes=cmd.get("num_classes", 10),
                augmentation=cmd.get("augmentation", True),
                pretrained=cmd.get("pretrained", True),
            )
            self.trainer = Trainer(
                config, log_callback=self.send_log, metrics_callback=self.send_metrics
            )
            self.training_thread = threading.Thread(target=self._run_training)
            self.training_thread.start()
            return {"status": "started"}

        elif action == "stop":
            if self.trainer:
                self.trainer.stop()
            return {"status": "stopping"}

        elif action == "export":
            if self.trainer:
                success = self.trainer.export(
                    cmd.get("output_path", "./export"), cmd.get("formats", ["MLX"])
                )
                return {"status": "success" if success else "failed"}
            return {"status": "no_model"}

        return {"status": "unknown"}

    def _run_training(self):
        try:
            success = self.trainer.train()
            self.send_response({"type": "completed", "success": success})
        except Exception as e:
            self.send_log(f"[ERROR] {str(e)}")
            self.send_response({"type": "completed", "success": False})

    def send_log(self, message: str):
        self.send_response({"type": "log", "message": message})

    def send_metrics(self, metrics: dict):
        self.send_response({"type": "metrics", **metrics})

    def send_response(self, data: dict):
        print(json.dumps(data), flush=True)

    def run(self):
        signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
        self.send_response({"type": "ready"})
        for line in sys.stdin:
            try:
                cmd = json.loads(line.strip())
                response = self.handle_command(cmd)
                self.send_response({"type": "response", **response})
            except json.JSONDecodeError:
                self.send_response({"type": "error", "message": "Invalid JSON"})
            except Exception as e:
                self.send_response({"type": "error", "message": str(e)})


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--server":
        IPCServer().run()
    else:
        print("Usage: python train.py --server")
        print("       For standalone: use IPCServer with JSON commands")


if __name__ == "__main__":
    main()
