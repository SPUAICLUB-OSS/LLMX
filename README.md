# LLMX by SPU AI CLUB

![LLMX](Resources/AppIcon.png)

## Overview

LLMX is a native macOS application for training and running Large Language Models using Apple Silicon and MLX framework. Built with SwiftUI frontend and Python MLX backend, optimized for Apple's AMX (Apple Matrix Extensions) accelerator.

## Requirements

- macOS 13.0 or later
- Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- MLX framework

## Installation

### From DMG

1. Download the latest release DMG
2. Open the DMG file
3. Drag LLMX to Applications
4. Install Python dependencies

### From Source

```bash
git clone https://github.com/SPUAICLUB-OSS/LLMX.git
cd llmx
chmod +x build.sh
./build.sh
```

## Python Dependencies

```bash
pip install mlx mlx-lm numpy
```

## Architecture

```
LLMX/
├── Sources/
│   ├── main.swift      # SwiftUI Application
│   ├── linker.swift    # Swift-Python IPC Bridge
│   └── train.py        # MLX Training Backend
├── Resources/
│   └── AppIcon.png
├── build.sh
└── README.md
```

## Components

### main.swift

SwiftUI application with minimal Apple-style interface featuring three main views:

- **Train View**: Configure model path, dataset, epochs, batch size, and learning rate
- **Monitor View**: Real-time metrics display including loss, epoch progress, iteration speed, and memory usage
- **Export View**: Export trained models in MLX, GGUF, or SafeTensors format

### train.py

MLX-based training backend with AMX optimization:

- Custom Transformer architecture with MultiHeadAttention and FeedForward layers
- AMX-optimized matrix operations
- Cosine learning rate decay with warmup
- Gradient clipping and weight decay
- Checkpoint saving and model export
- JSON-based IPC communication

### linker.swift

Swift-Python bridge for inter-process communication:

- Process management for Python backend
- JSON-based command protocol
- Real-time metrics streaming
- Async training control

## Usage

### Training

1. Launch LLMX
2. Select model path or enter HuggingFace model ID
3. Select training dataset (supports .bin, .json, .txt)
4. Configure training parameters
5. Click "Start Training"

### Supported Dataset Formats

| Format | Description |
|--------|-------------|
| .bin | Raw token IDs (int32) |
| .json | JSON array of token sequences |
| .txt | Plain text (character-level) |

### Export Formats

| Format | Description |
|--------|-------------|
| MLX | Native MLX safetensors |
| GGUF | llama.cpp compatible |
| SafeTensors | HuggingFace compatible |

## Configuration

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| epochs | 10 | Number of training epochs |
| batch_size | 32 | Samples per batch |
| learning_rate | 1e-4 | Initial learning rate |
| max_seq_len | 512 | Maximum sequence length |
| warmup_steps | 100 | LR warmup steps |
| weight_decay | 0.01 | AdamW weight decay |
| grad_clip | 1.0 | Gradient clipping threshold |

### Model Architecture

| Parameter | Default |
|-----------|---------|
| dims | 768 |
| num_layers | 12 |
| num_heads | 12 |
| hidden_dims | 3072 |

## API

### IPC Commands

```json
{"action": "train", "model_path": "...", "data_path": "...", "epochs": 10}
{"action": "stop"}
{"action": "export", "output_path": "...", "format": "mlx"}
```

### Response Types

```json
{"type": "ready"}
{"type": "log", "message": "..."}
{"type": "metrics", "loss": 0.5, "epoch": 1, "iter_per_sec": 10.0}
{"type": "completed", "success": true}
```

## Performance

### Apple Silicon Optimization

- Native ARM64 compilation
- MLX Metal GPU acceleration
- AMX matrix coprocessor utilization
- Unified memory architecture support

### Benchmarks (M3 Max)

| Model Size | Tokens/sec |
|------------|------------|
| 125M | ~15,000 |
| 350M | ~8,000 |
| 1.3B | ~2,500 |

## Building

### Requirements

- Xcode Command Line Tools
- Swift 5.9+

### Build Commands

```bash
./build.sh
```

### Output

- `build/LLMX.app` - macOS Application
- `build/LLMX-1.0.0.dmg` - Distributable DMG

## License

MIT License

## Credits

Developed by SPU AI CLUB (AIPRENEUR)

Sripatum University

## Links

- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [Apple Silicon Developer](https://developer.apple.com/silicon/)
- [SPU AI CLUB](https://github.com/spuaiclub)
