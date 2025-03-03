import logging
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
import subprocess
import speech_recognition as sr
from pydub import AudioSegment

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class AudioTranscript:
    text: str
    segments: List[Dict[str, Any]]
    language: str

class AudioProcessor:
    def __init__(self, 
                 language: str | None = None,
                 model_size_or_path: str = "medium",  # 保留此参数以保持兼容性
                 device: str = "cpu"):  # 保留此参数以保持兼容性
        """使用 SpeechRecognition 初始化音频处理器。"""
        try:
            self.recognizer = sr.Recognizer()
            self.language = language if language else "en-US"  # 如果未指定语言，默认使用英语
            
            # 检查是否安装了 ffmpeg
            try:
                subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
                self.has_ffmpeg = True
            except (subprocess.CalledProcessError, FileNotFoundError):
                self.has_ffmpeg = False
                logger.warning("未找到 FFmpeg。请安装 ffmpeg 以获得更好的音频提取效果。")
                
        except Exception as e:
            logger.error(f"初始化语音识别时出错：{e}")
            raise

    def extract_audio(self, video_path: Path, output_dir: Path) -> Optional[Path]:
        """从视频文件中提取音频并转换为 WAV 格式。"""
        audio_path = output_dir / "audio.wav"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 使用 ffmpeg 提取音频
            subprocess.run([
                "ffmpeg", "-i", str(video_path),
                "-vn",  # 不包含视频
                "-acodec", "pcm_s16le",  # PCM 16位小端格式
                "-ar", "16000",  # 16kHz 采样率
                "-ac", "1",  # 单声道
                "-y",  # 覆盖输出文件
                str(audio_path)
            ], check=True, capture_output=True)
            
            logger.debug("使用 ffmpeg 成功提取音频")
            return audio_path
        except subprocess.CalledProcessError as e:
            error_output = e.stderr.decode()
            logger.error(f"FFmpeg 错误：{error_output}")
            
            # 检查错误是否表明没有音频流
            if "Output file does not contain any stream" in error_output:
                logger.debug("视频中未找到音频流 - 跳过音频提取")
                return None
                
            # 如果错误不是关于缺少音频，尝试使用 pydub 作为备选方案
            logger.info("转而使用 pydub 提取音频...")
            try:
                video = AudioSegment.from_file(str(video_path))
                audio = video.set_channels(1).set_frame_rate(16000)
                audio.export(str(audio_path), format="wav")
                logger.debug("使用 pydub 成功提取音频")
                return audio_path
            except Exception as e2:
                logger.error(f"使用 pydub 提取音频时出错：{e2}")
                raise RuntimeError(
                    "提取音频失败。请使用以下命令安装 ffmpeg：\n"
                    "Ubuntu/Debian: sudo apt-get update && sudo apt-get install -y ffmpeg\n"
                    "MacOS: brew install ffmpeg\n"
                    "Windows: choco install ffmpeg"
                )

    def transcribe(self, audio_path: Path) -> Optional[AudioTranscript]:
        """使用 SpeechRecognition 转录音频文件。"""
        try:
            with sr.AudioFile(str(audio_path)) as source:
                audio = self.recognizer.record(source)
                
            # 尝试识别语音
            text = self.recognizer.recognize_google(audio, language=self.language)
            
            # 创建一个简单的片段，因为 SpeechRecognition 不提供详细的时间信息
            segment_data = [{
                "text": text,
                "start": 0.0,
                "end": None,  # 我们没有时间信息
                "words": []  # 我们没有词级别的信息
            }]
            
            return AudioTranscript(
                text=text,
                segments=segment_data,
                language=self.language
            )
            
        except sr.UnknownValueError:
            logger.warning("语音识别无法理解音频内容")
            return None
        except sr.RequestError as e:
            logger.error(f"无法从语音识别服务获取结果：{e}")
            return None
        except Exception as e:
            logger.error(f"转录音频时出错：{e}")
            logger.exception(e)
            return None
