from typing import List, Dict, Any, Optional
import logging
from .clients.llm_client import LLMClient
from .prompt import PromptLoader
from .frame import Frame
from .audio_processor import AudioTranscript

logger = logging.getLogger(__name__)

class VideoAnalyzer:
    def __init__(self, client: LLMClient, model: str, prompt_loader: PromptLoader, user_prompt: str = ""):
        """初始化视频分析器。
        
        参数：
            client: 用于进行API调用的LLM客户端
            model: 要使用的模型名称
            prompt_loader: 提示词模板加载器
            user_prompt: 可选的用户关于视频的问题，将使用{prompt}标记注入到帧分析和视频描述提示词中
        """
        self.client = client
        self.model = model
        self.prompt_loader = prompt_loader
        self.user_prompt = user_prompt  # 存储用户关于视频的问题
        self._load_prompts()
        self.previous_analyses = []
        
    def _format_user_prompt(self) -> str:
        """如果用户提示词不为空，则添加前缀进行格式化。"""
        if self.user_prompt:
            return f"I want to know {self.user_prompt}"
        return ""
        
    def _load_prompts(self):
        """从文件加载提示词。"""
        self.frame_prompt = self.prompt_loader.get_by_index(0)  # 帧分析提示词
        self.video_prompt = self.prompt_loader.get_by_index(1)  # 视频重构提示词

    def _format_previous_analyses(self) -> str:
        """格式化之前的帧分析结果以包含在提示词中。"""
        if not self.previous_analyses:
            return ""
            
        formatted_analyses = []
        for i, analysis in enumerate(self.previous_analyses):
            formatted_analysis = (
                f"Frame {i}\n"
                f"{analysis.get('response', '无可用分析')}\n"
            )
            formatted_analyses.append(formatted_analysis)
            
        return "\n".join(formatted_analyses)

    def analyze_frame(self, frame: Frame) -> Dict[str, Any]:
        """使用LLM分析单个帧。"""
        # 用格式化的之前分析结果替换{PREVIOUS_FRAMES}标记
        # 替换提示词模板中的标记
        prompt = self.frame_prompt.replace("{PREVIOUS_FRAMES}", self._format_previous_analyses())
        prompt = prompt.replace("{prompt}", self._format_user_prompt())
        prompt = f"{prompt}\n这是在 {frame.timestamp:.2f} 秒捕获的第 {frame.number} 帧。"
        
        try:
            response = self.client.generate(
                prompt=prompt,
                image_path=str(frame.path),
                model=self.model,
                num_predict=300
            )
            logger.debug(f"成功分析了第 {frame.number} 帧")
            
            # 存储分析结果供后续帧使用
            analysis_result = {k: v for k, v in response.items() if k != "context"}
            self.previous_analyses.append(analysis_result)
            
            return analysis_result
        except Exception as e:
            logger.error(f"分析第 {frame.number} 帧时出错：{e}")
            error_result = {"response": f"分析第 {frame.number} 帧时出错：{str(e)}"}
            self.previous_analyses.append(error_result)
            return error_result

    def reconstruct_video(self, frame_analyses: List[Dict[str, Any]], frames: List[Frame], 
                         transcript: Optional[AudioTranscript] = None) -> Dict[str, Any]:
        """从帧分析结果和转录内容重构视频描述。"""
        frame_notes = []
        for i, (frame, analysis) in enumerate(zip(frames, frame_analyses)):
            frame_note = (
                f"第 {i} 帧 ({frame.timestamp:.2f}秒):\n"
                f"{analysis.get('response', '无可用分析')}"
            )
            frame_notes.append(frame_note)
        
        analysis_text = "\n\n".join(frame_notes)
        
        # 获取第一帧分析
        first_frame_text = ""
        if frame_analyses and len(frame_analyses) > 0:
            first_frame_text = frame_analyses[0].get('response', '')
        
        # 如果有转录内容则包含
        transcript_text = ""
        if transcript and transcript.text.strip():
            transcript_text = transcript.text
        
        # 替换提示词模板中的标记
        prompt = self.video_prompt.replace("{prompt}", self._format_user_prompt())
        prompt = prompt.replace("{FRAME_NOTES}", analysis_text)
        prompt = prompt.replace("{FIRST_FRAME}", first_frame_text)
        prompt = prompt.replace("{TRANSCRIPT}", transcript_text)
        
        try:
            response = self.client.generate(
                prompt=prompt,
                model=self.model,
                num_predict=1000
            )
            logger.info("成功重构了视频描述")
            return {k: v for k, v in response.items() if k != "context"}
        except Exception as e:
            logger.error(f"重构视频时出错：{e}")
            return {"response": f"重构视频时出错：{str(e)}"}
