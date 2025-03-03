# video-analyzer-cn
使用视觉模型分析视频的工具（汉化）
# video-analyzer-cn
使用视觉模型分析视频的工具（汉化版）

## 重要更新说明

### 音频处理模块更新 (2024-03-27)

为了提高工具的易用性和降低系统要求，我们对音频处理模块进行了以下更新：

1. **更换音频识别引擎**
   - 原系统：使用 `faster-whisper` 进行音频识别
   - 现系统：改用 `SpeechRecognition` 库
   - 更换原因：
     - `faster-whisper` 需要较大的系统资源和 GPU 支持
     - 安装过程复杂，依赖较多
     - 对普通用户不够友好

2. **主要改动**
   - 移除了 `torch` 和 `faster-whisper` 依赖
   - 添加了 `SpeechRecognition` 库支持
   - 简化了音频处理流程
   - 优化了错误处理和日志输出
   - 所有注释和文档已翻译成中文

3. **新增特性**
   - 支持多种音频格式
   - 更好的跨平台兼容性
   - 降低了系统资源要求
   - 提供了更友好的错误提示

## 系统要求

- Python 3.8 或更高版本
- FFmpeg（用于音频提取）
- 网络连接（用于语音识别服务）

## 安装说明

1. 克隆仓库：
```bash
git clone https://github.com/xhnoob/video-analyzer-cn.git
cd video-analyzer-cn
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 安装 FFmpeg（如果尚未安装）：
   - Windows: `choco install ffmpeg`
   - MacOS: `brew install ffmpeg`
   - Ubuntu/Debian: `sudo apt-get install ffmpeg`

## 使用方法

基本用法：
```bash
python -m video_analyzer 视频文件路径 [选项]
```

常用选项：
- `--language`: 设置音频识别的语言（默认：en-US）
- `--output`: 指定输出目录
- `--keep-frames`: 保留提取的视频帧
- `--log-level`: 设置日志级别（DEBUG/INFO/WARNING/ERROR）

## 注意事项

1. 首次运行时，请确保网络连接正常，因为语音识别需要访问在线服务
2. 对于中文视频，建议设置 `--language zh-CN`
3. 如果遇到音频提取问题，请确保已正确安装 FFmpeg

## 许可证

本项目采用 MIT 许可证
