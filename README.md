# LLMX by SPU AI CLUB

![LLMX](2.png)

## Overview

LLMX is a native macOS application for training ML models using Apple Silicon and MLX framework. Supports LLM training, Image Classification (ResNet, VGG, EfficientNet, MobileNet, ViT), and Object Detection (YOLOv8).

## Features

| Feature | Description |
|---------|-------------|
| Image Classification | ResNet-50/101, VGG-16, EfficientNet, MobileNet, Vision Transformer |
| Object Detection | YOLOv8n, YOLOv8s, YOLOv8m |
| LLM Training | Custom transformer models |
| Real-time Monitor | Loss, Accuracy, Speed, Memory charts |
| Multi-format Export | CoreML, PyTorch, ONNX, Keras, GGUF |

## Requirements

- macOS 13.0+
- Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- MLX framework

## Installation

```bash
git clone https://github.com/SPUAICLUB-OSS/LLMX.git
cd llmx
pip install mlx pillow h5py
chmod +x build.sh
./build.sh
```

## Architecture

```
LLMX/
├── Sources/
│   ├── main.swift      # SwiftUI Application
│   ├── linker.swift    # Swift-Python IPC
│   └── train.py        # MLX Training Backend
├── Resources/
│   └── AppIcon.png
├── build.sh
└── README.md
```

## Supported Models

### Image Classification

| Model | Parameters | Input Size |
|-------|------------|------------|
| ResNet-50 | 25M | 224x224 |
| ResNet-101 | 44M | 224x224 |
| VGG-16 | 138M | 224x224 |
| EfficientNet | 5M | 224x224 |
| MobileNetV2 | 3.4M | 224x224 |
| Vision Transformer | 86M | 224x224 |

### Object Detection

| Model | Parameters | Input Size |
|-------|------------|------------|
| YOLOv8n | 3M | 640x640 |
| YOLOv8s | 11M | 640x640 |
| YOLOv8m | 25M | 640x640 |

## Dataset Format

### Image Classification

```
dataset/
├── class1/
│   ├── image1.jpg
│   └── image2.png
├── class2/
│   └── image3.jpg
└── class3/
    └── image4.png
```

### LLM Training

| Format | Description |
|--------|-------------|
| Folder | Directory with data files |
| .bin | Raw token IDs (int32) |
| .json | JSON with tokens/text |
| .jsonl | JSON Lines format |
| .txt | Plain text |

## Export Formats

| Format | Extension | Use Case |
|--------|-----------|----------|
| CoreML | .mlmodel | iOS/macOS deployment |
| MLX | .safetensors | MLX inference |
| PyTorch | .pt | PyTorch ecosystem |
| ONNX | .onnx | Cross-platform |
| Keras | .h5 | TensorFlow/Keras |
| GGUF | .gguf | llama.cpp |

## Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| epochs | 10 | Training iterations |
| batch_size | 32 | Samples per batch |
| learning_rate | 1e-4 | Initial LR |
| image_size | 224 | Input resolution |
| augmentation | true | Data augmentation |
| pretrained | true | Use pretrained weights |

## Real-time Monitoring

- Loss curve
- Accuracy curve
- Learning rate schedule
- GPU memory usage
- Training speed (it/s)
- Training log

## Building

```bash
./build.sh
```

Output:
- `build/LLMX.app`
- `build/LLMX-1.0.0.dmg`

## Python Dependencies

```bash
pip install mlx pillow h5py numpy
pip install torch  # for .pt export
pip install onnx   # for .onnx export
```

## License

MIT License

## Credits

SPU AI CLUB (AIPRENEUR) | Dotmini Software
Sripatum University
