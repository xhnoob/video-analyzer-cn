from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

@dataclass
class Frame:
    number: int
    path: Path
    timestamp: float
    score: float

class VideoProcessor:
    # 类常量
    FRAME_DIFFERENCE_THRESHOLD = 10.0
    
    def __init__(self, video_path: Path, output_dir: Path, model: str):
        self.video_path = video_path
        self.output_dir = output_dir
        self.model = model
        self.frames: List[Frame] = []
        
    def _calculate_frame_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """使用绝对差值计算两帧之间的差异。"""
        if frame1 is None or frame2 is None:
            return 0.0
        
        # 转换为灰度图以简化比较
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # 计算绝对差值和平均值
        diff = cv2.absdiff(gray1, gray2)
        score = np.mean(diff)
        
        return float(score)

    def _is_keyframe(self, current_frame: np.ndarray, prev_frame: np.ndarray, threshold: float = FRAME_DIFFERENCE_THRESHOLD) -> bool:
        """判断当前帧是否与前一帧有显著差异。"""
        if prev_frame is None:
            return True
            
        score = self._calculate_frame_difference(current_frame, prev_frame)
        return score > threshold

    def extract_keyframes(self, frames_per_minute: int = 10, duration: Optional[float] = None, max_frames: Optional[int] = None) -> List[Frame]:
        """从视频中提取关键帧，目标是每分钟提取特定数量的帧。"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件：{self.video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps
        
        if duration:
            video_duration = min(duration, video_duration)
            total_frames = int(min(total_frames, duration * fps))
        
        # 计算目标帧数
        target_frames = max(1, min(
            int((video_duration / 60) * frames_per_minute),
            total_frames,
            max_frames if max_frames is not None else float('inf')
        ))
        
        # 计算自适应采样间隔
        sample_interval = max(1, total_frames // (target_frames * 2))
        
        frame_candidates = []
        prev_frame = None
        frame_count = 0
        
        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % sample_interval == 0:
                score = self._calculate_frame_difference(frame, prev_frame)
                if score > self.FRAME_DIFFERENCE_THRESHOLD:
                    frame_candidates.append((frame_count, frame, score))
                prev_frame = frame.copy()
                
            frame_count += 1
            
        cap.release()
        
        # 选择最显著的帧
        selected_candidates = sorted(frame_candidates, key=lambda x: x[2], reverse=True)[:target_frames]
        
        # 如果指定了最大帧数，在候选帧中均匀采样
        if max_frames is not None and max_frames < len(selected_candidates):
            step = len(selected_candidates) / max_frames
            selected_frames = [selected_candidates[int(i * step)] for i in range(max_frames)]
        else:
            selected_frames = selected_candidates

        self.frames = []
        for idx, (frame_num, frame, score) in enumerate(selected_frames):
            frame_path = self.output_dir / f"frame_{idx}.jpg"
            cv2.imwrite(str(frame_path), frame)
            timestamp = frame_num / fps
            self.frames.append(Frame(idx, frame_path, timestamp, score))
        
        logger.info(f"从视频中提取了 {len(self.frames)} 帧（目标帧数为 {target_frames}）")
        return self.frames
