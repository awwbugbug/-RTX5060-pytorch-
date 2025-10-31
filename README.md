# PyTorch RTX5060 Environment

此目录包含为在 **NVIDIA GeForce RTX 5060**（CUDA 13）上训练 PyTorch 模型而创建的环境导出与使用说明。

生成时间：2025-10-31

包含文件：
- `pytorch_rt5060-requirements.txt` — 使用 `pip freeze` 导出的 Python 包清单（用于 pip 重现）。
- `pytorch_rt5060-conda-list.txt` — 使用 `conda list -n pytorch_rt5060` 导出的 conda 包列表（用于诊断）。

环境说明（已自动完成）：
- conda 环境名：`pytorch_rt5060`
- Python: 3.10
- 已安装的关键包（运行时）：
  - torch 2.9.0+cu130
  - torchvision 0.24.0+cu130
  - torchaudio 2.9.0+cu130

如何激活该环境并重现实验：
```powershell
# 激活 conda 环境
conda activate pytorch_rt5060

# （可选）确认 PyTorch 与 GPU
python -c "import torch; print('torch:', torch.__version__); print('cuda runtime:', getattr(torch.version,'cuda',None)); print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

# 运行 MNIST 示例（已在本机测试）
cd examples\mnist
python main.py --epochs 1 --batch-size 64
```

如何从导出文件重建环境（两种方式）：

1) 使用 conda 创建基础环境，然后使用 pip 安装对应 wheel（最接近当前环境）：
```powershell
# 创建 conda 环境
conda create -n pytorch_rt5060_new python=3.10 -y
conda activate pytorch_rt5060_new

# 使用 pip 安装与当前相同的 PyTorch + CUDA wheel（来自官方索引）
pip install --upgrade --index-url https://download.pytorch.org/whl/cu130 torch torchvision torchaudio

# 使用 pip requirements 安装其余 Python 包
pip install -r ../pytorch_rt5060-requirements.txt
```

2) 或者仅用于备份/诊断，查看 conda 包：
```powershell
conda list -n pytorch_rt5060 > pytorch_rt5060-conda-list.txt
```

注意事项：
- 我通过 pip 从 PyTorch 官方 cu130 索引安装了 PyTorch（因为 conda 仓库当时未提供 pytorch-cuda=13.0 的包）。pip/conda 混用在某些情形下会导致依赖不一致，若需要 conda-only 的环境管理，考虑用纯 pip venv 或等待/检查 conda 的官方包。

