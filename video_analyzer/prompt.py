from pathlib import Path
import logging
from typing import List, Dict
import pkg_resources

logger = logging.getLogger(__name__)

class PromptLoader:
    def __init__(self, prompt_dir: str, prompts: List[Dict[str, str]]):
        # 处理用户提供的提示词目录
        self.prompt_dir = Path(prompt_dir).expanduser() if prompt_dir else None
        self.prompts = prompts

    def _find_prompt_file(self, prompt_path: str) -> Path:
        """在包资源、包目录或用户目录中查找提示词文件。"""
        # 首先尝试包资源（适用于所有安装模式）
        try:
            package_path = pkg_resources.resource_filename('video_analyzer', f'prompts/{prompt_path}')
            if Path(package_path).exists():
                return Path(package_path)
        except Exception as e:
            logger.debug(f"无法通过 pkg_resources 找到包中的提示词：{e}")

        # 尝试包目录（用于开发模式）
        pkg_root = Path(__file__).parent
        pkg_path = pkg_root / 'prompts' / prompt_path
        if pkg_path.exists():
            return pkg_path

        # 最后尝试用户指定的目录（如果提供了的话）
        if self.prompt_dir:
            user_path = Path(self.prompt_dir).expanduser()
            # 尝试绝对路径
            if user_path.is_absolute():
                full_path = user_path / prompt_path
                if full_path.exists():
                    return full_path
            else:
                # 尝试相对于当前目录的路径
                cwd_path = Path.cwd() / self.prompt_dir / prompt_path
                if cwd_path.exists():
                    return cwd_path

        raise FileNotFoundError(
            f"在包资源、包目录或用户目录（{self.prompt_dir}）中未找到提示词文件"
        )

    def get_by_index(self, index: int) -> str:
        """通过索引加载提示词。
        
        参数：
            index: 提示词在列表中的索引
            
        返回：
            提示词文本内容
            
        异常：
            IndexError: 如果索引超出范围
            FileNotFoundError: 如果提示词文件不存在
        """
        try:
            if index < 0 or index >= len(self.prompts):
                raise IndexError(f"提示词索引 {index} 超出范围 (0-{len(self.prompts)-1})")
            
            prompt = self.prompts[index]
            prompt_path = self._find_prompt_file(prompt["path"])
                
            logger.debug(f"从 {prompt_path} 加载提示词 '{prompt['name']}'")
            with open(prompt_path) as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"加载索引为 {index} 的提示词时出错：{e}")
            raise

    def get_by_name(self, name: str) -> str:
        """通过名称加载提示词。
        
        参数：
            name: 要加载的提示词名称
            
        返回：
            提示词文本内容
            
        异常：
            ValueError: 如果未找到指定名称的提示词
            FileNotFoundError: 如果提示词文件不存在
        """
        try:
            prompt = next((p for p in self.prompts if p["name"] == name), None)
            if prompt is None:
                raise ValueError(f"未找到名为 '{name}' 的提示词")
            
            prompt_path = self._find_prompt_file(prompt["path"])
                
            logger.debug(f"从 {prompt_path} 加载提示词 '{name}'")
            with open(prompt_path) as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"加载提示词 '{name}' 时出错：{e}")
            raise
