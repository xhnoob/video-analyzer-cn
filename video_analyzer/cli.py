import argparse
from pathlib import Path
import json
import logging
import shutil
import sys
from typing import Optional
import torch
import torch.backends.mps

from .config import Config, get_client, get_model
from .frame import VideoProcessor
from .prompt import PromptLoader
from .analyzer import VideoAnalyzer
from .audio_processor import AudioProcessor, AudioTranscript
from .clients.ollama import OllamaClient
from .clients.generic_openai_api import GenericOpenAIAPIClient

# 在模块级别初始化日志记录器
logger = logging.getLogger(__name__)

def get_log_level(level_str: str) -> int:
    """将字符串日志级别转换为日志记录常量。"""
    levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    return levels.get(level_str.upper(), logging.INFO)

def cleanup_files(output_dir: Path):
    """清理临时文件和目录。"""
    try:
        frames_dir = output_dir / "frames"
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
            logger.debug(f"已清理帧目录：{frames_dir}")
            
        audio_file = output_dir / "audio.wav"
        if audio_file.exists():
            audio_file.unlink()
            logger.debug(f"已清理音频文件：{audio_file}")
    except Exception as e:
        logger.error(f"清理过程中出错：{e}")

def create_client(config: Config):
    """根据配置创建适当的客户端。"""
    client_type = config.get("clients", {}).get("default", "ollama")
    client_config = get_client(config)
    
    if client_type == "ollama":
        return OllamaClient(client_config["url"])
    elif client_type == "openai_api":
        return GenericOpenAIAPIClient(client_config["api_key"], client_config["api_url"])
    else:
        raise ValueError(f"未知的客户端类型：{client_type}")

def main():
    parser = argparse.ArgumentParser(description="使用视觉模型分析视频")
    parser.add_argument("video_path", type=str, help="视频文件路径")
    parser.add_argument("--config", type=str, default="config",
                        help="配置目录路径")
    parser.add_argument("--output", type=str, help="分析结果的输出目录")
    parser.add_argument("--client", type=str, help="要使用的客户端（ollama 或 openrouter）")
    parser.add_argument("--ollama-url", type=str, help="Ollama 服务的 URL")
    parser.add_argument("--api-key", type=str, help="OpenAI 兼容服务的 API 密钥")
    parser.add_argument("--api-url", type=str, help="OpenAI 兼容 API 的 URL")
    parser.add_argument("--model", type=str, help="要使用的视觉模型名称")
    parser.add_argument("--duration", type=float, help="要处理的时长（秒）")
    parser.add_argument("--keep-frames", action="store_true", help="分析后保留提取的帧")
    parser.add_argument("--whisper-model", type=str, help="Whisper 模型大小（tiny、base、small、medium、large）或本地 Whisper 模型快照路径")
    parser.add_argument("--start-stage", type=int, default=1, help="开始处理的阶段（1-3）")
    parser.add_argument("--max-frames", type=int, default=sys.maxsize, help="要处理的最大帧数")
    parser.add_argument("--log-level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="设置日志级别（默认：INFO）")
    parser.add_argument("--prompt", type=str, default="",
                        help="关于视频的问题")
    parser.add_argument("--language", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # 使用指定的级别设置日志记录
    log_level = get_log_level(args.log_level)
    # 配置根日志记录器
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True  # 强制重新配置根日志记录器
    )
    # 确保我们的模块日志记录器具有正确的级别
    logger.setLevel(log_level)

    # 加载并更新配置
    config = Config(args.config)
    config.update_from_args(args)

    # 初始化组件
    video_path = Path(args.video_path)
    output_dir = Path(config.get("output_dir"))
    client = create_client(config)
    model = get_model(config)
    prompt_loader = PromptLoader(config.get("prompt_dir"), config.get("prompts", []))
    
    try:
        transcript = None
        frames = []
        frame_analyses = []
        video_description = None
        
        # 阶段 1：帧和音频处理
        if args.start_stage <= 1:
            # 初始化音频处理器并提取转录，AudioProcessor 接受以下可在 config.json 中设置的参数：
            # language (str)：音频转录的语言代码（默认：None）
            # whisper_model (str)：Whisper 模型大小或路径（默认："medium"）
            # device (str)：用于音频处理的设备（默认："cpu"）
            logger.debug("正在初始化音频处理...")
            audio_processor = AudioProcessor(language=config.get("audio", {}).get("language", ""), 
                                             model_size_or_path=config.get("audio", {}).get("whisper_model", "medium"),
                                             device=config.get("audio", {}).get("device", "cpu"))
            
            logger.info("正在从视频中提取音频...")
            audio_path = audio_processor.extract_audio(video_path, output_dir)
            
            if audio_path is None:
                logger.debug("视频中未找到音频 - 跳过转录")
                transcript = None
            else:
                logger.info("正在转录音频...")
                transcript = audio_processor.transcribe(audio_path)
                if transcript is None:
                    logger.warning("无法生成可靠的转录。仅继续进行视频分析。")
            
            logger.info(f"正在使用模型 {model} 从视频中提取帧...")
            processor = VideoProcessor(
                video_path, 
                output_dir / "frames", 
                model
            )
            frames = processor.extract_keyframes(
                frames_per_minute=config.get("frames", {}).get("per_minute", 60),
                duration=config.get("duration"),
                max_frames=args.max_frames
            )
            
        # 阶段 2：帧分析
        if args.start_stage <= 2:
            logger.info("正在分析帧...")
            analyzer = VideoAnalyzer(client, model, prompt_loader, config.get("prompt", ""))
            frame_analyses = []
            for frame in frames:
                analysis = analyzer.analyze_frame(frame)
                frame_analyses.append(analysis)
                
        # 阶段 3：视频重构
        if args.start_stage <= 3:
            logger.info("正在重构视频描述...")
            video_description = analyzer.reconstruct_video(
                frame_analyses, frames, transcript
            )
        
        output_dir.mkdir(parents=True, exist_ok=True)
        results = {
            "metadata": {
                "client": config.get("clients", {}).get("default"),
                "model": model,
                "whisper_model": config.get("audio", {}).get("whisper_model"),
                "frames_per_minute": config.get("frames", {}).get("per_minute"),
                "duration_processed": config.get("duration"),
                "frames_extracted": len(frames),
                "frames_processed": min(len(frames), args.max_frames),
                "start_stage": args.start_stage,
                "audio_language": transcript.language if transcript else None,
                "transcription_successful": transcript is not None
            },
            "transcript": {
                "text": transcript.text if transcript else None,
                "segments": transcript.segments if transcript else None
            } if transcript else None,
            "frame_analyses": frame_analyses,
            "video_description": video_description
        }
        
        with open(output_dir / "analysis.json", "w") as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"分析完成。结果已保存到 {output_dir / 'analysis.json'}")
        
        print("\n转录：")
        if transcript:
            print(transcript.text)
        else:
            print("无可用的可靠转录")
            
        if video_description:
            print("\n视频描述：")
            print(video_description.get("response", "未生成描述"))
        
        if not config.get("keep_frames"):
            cleanup_files(output_dir)
            
    except Exception as e:
        logger.error(f"视频分析过程中出错：{e}")
        if not config.get("keep_frames"):
            cleanup_files(output_dir)
        raise

if __name__ == "__main__":
    main()
