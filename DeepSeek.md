# DeepSeek API 配置说明

## 配置概述

本项目已配置为使用 **DeepSeek API** 进行智能问答，配置如下：

- **聊天模型**: DeepSeek API (`deepseek-chat`)
- **向量模型**: 本地 HuggingFace (`paraphrase-multilingual-MiniLM-L12-v2`)

> 注意：DeepSeek 不支持 embeddings API，因此使用本地 HuggingFace 模型生成文本向量。

## 配置文件 (.env)

```env
# AI 配置
USE_OLLAMA=false

# DeepSeek API 配置
OPENAI_API_KEY=sk-847931b3c2ac459ebc9e298ae950f2bf
OPENAI_API_BASE=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
```

## 依赖安装

确保已安装以下依赖：

```bash
pip install sentence-transformers torch
```

如果有 GPU，会自动使用 GPU 加速向量生成。

## 测试配置

运行测试脚本验证配置：

```bash
python test_deepseek.py
```

预期输出：
```
✓ DeepSeek 聊天 API: 正常
✓ 本地 Embeddings: 正常
🎉 所有测试通过！
```

## 使用步骤

### 1. 启动 GUI

```bash
python 原来的gui.py
```

### 2. 构建知识库

1. 切换到 **"📚 历史知识库"** 标签页
2. 点击 **"添加 PDF/TXT 文件"** 选择文件
3. 点击 **"🚀 构建历史知识库"**
4. 等待处理完成

### 3. 智能问答

1. 切换到 **"💬 智能问答"** 标签页
2. 输入问题并按回车
3. 系统会：
   - 使用本地 HuggingFace 模型检索相关文档
   - 使用 DeepSeek API 生成回答

## 工作流程

```
用户问题
    ↓
[本地 HuggingFace] 生成问题向量
    ↓
[FAISS 向量库] 检索相关文档
    ↓
[DeepSeek API] 基于文档生成回答
    ↓
返回答案
```

## 优势

1. **成本优化**: DeepSeek API 价格低廉
2. **隐私保护**: 向量生成在本地完成
3. **性能优秀**: GPU 加速向量生成
4. **中文支持**: 多语言模型支持中文

## 常见问题

### Q: 首次运行很慢？
A: 首次运行会下载 HuggingFace 模型（约 400MB），下载到 `~/.cache/huggingface/`

### Q: 如何使用 GPU？
A: 安装 CUDA 版本的 PyTorch：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Q: DeepSeek API 配额用完了？
A: 更换 API Key 或切换到其他兼容 OpenAI 格式的 API

## 技术细节

- **向量维度**: 384 (MiniLM-L12)
- **向量库**: FAISS
- **分块大小**: 500 字符
- **检索数量**: 6 个相关文档片段
- **温度参数**: 0.3 (更确定性的回答)

## 支持

如有问题，请检查：
1. `.env` 文件配置是否正确
2. 依赖是否完整安装
3. 运行 `test_deepseek.py` 查看详细错误
