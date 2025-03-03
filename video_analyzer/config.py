import argparse
from pathlib import Path
import json
from typing import Any
import logging
import pkg_resources

logger = logging.getLogger(__name__)

class Config:
    def __init__(self, config_dir: str = "config"):
        # 处理用户提供的配置目录
        self.config_dir = Path(config_dir)
        self.user_config = self.config_dir / "config.json"
        
        # 首先尝试在用户提供的目录中查找 default_config.json
        self.default_config = self.config_dir / "default_config.json"
        
        # 如果未找到，回退到包的默认配置
        if not self.default_config.exists():
            try:
                default_config_path = pkg_resources.resource_filename('video_analyzer', 'config/default_config.json')
                self.default_config = Path(default_config_path)
                logger.debug(f"使用包中的默认配置：{self.default_config}")
            except Exception as e:
                logger.error(f"查找默认配置时出错：{e}")
                raise
            
        self.load_config()

    def load_config(self):
        """按以下顺序加载配置：
        1. 尝试加载用户配置（config.json）
        2. 如果失败则回退到默认配置（default_config.json）
        """
        try:
            if self.user_config.exists():
                logger.debug(f"从 {self.user_config} 加载用户配置")
                with open(self.user_config) as f:
                    self.config = json.load(f)
            else:
                logger.debug(f"未找到用户配置，从 {self.default_config} 加载默认配置")
                with open(self.default_config) as f:
                    self.config = json.load(f)
                    
            # 确保 prompts 是一个列表
            if not isinstance(self.config.get("prompts", []), list):
                logger.warning("配置中的 prompts 不是列表，设置为空列表")
                self.config["prompts"] = []
                
        except Exception as e:
            logger.error(f"加载配置时出错：{e}")
            raise

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值，可设置默认值。"""
        return self.config.get(key, default)

    def update_from_args(self, args: argparse.Namespace):
        """使用命令行参数更新配置。"""
        for key, value in vars(args).items():
            if value is not None:  # 仅在提供了参数时更新
                if key == "client":
                    self.config["clients"]["default"] = value
                elif key == "ollama_url":
                    self.config["clients"]["ollama"]["url"] = value
                elif key == "api_key":
                    self.config["clients"]["openai_api"]["api_key"] = value
                    # 如果提供了密钥但未指定客户端，使用 OpenAI API
                    if not args.client:
                        self.config["clients"]["default"] = "openai_api"
                elif key == "api_url":
                    self.config["clients"]["openai_api"]["api_url"] = value
                elif key == "model":
                    client = self.config["clients"]["default"]
                    self.config["clients"][client]["model"] = value
                elif key == "prompt":
                    self.config["prompt"] = value
                # 覆盖音频配置
                elif key == "whisper_model":
                    self.config["audio"]["whisper_model"] = value  # 默认为 'medium'
                elif key == "language":
                    if value is not None:
                        self.config["audio"]["language"] = value
                elif key == "device":
                    self.config["audio"]["device"] = value
                elif key not in ["start_stage", "max_frames"]:  # 忽略这些仅用于命令行的参数
                    self.config[key] = value

    def save_user_config(self):
        """将当前配置保存到用户配置文件。"""
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            with open(self.user_config, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.debug(f"已将用户配置保存到 {self.user_config}")
        except Exception as e:
            logger.error(f"保存用户配置时出错：{e}")
            raise

def get_client(config: Config) -> dict:
    """根据配置获取适当的客户端配置。"""
    client_type = config.get("clients", {}).get("default", "ollama")
    client_config = config.get("clients", {}).get(client_type, {})
    
    if client_type == "ollama":
        return {"url": client_config.get("url", "http://localhost:11434")}
    elif client_type == "openai_api":
        api_key = client_config.get("api_key")
        api_url = client_config.get("api_url")
        if not api_key:
            raise ValueError("使用 OpenAI API 客户端时需要提供 API 密钥")
        if not api_url:
            raise ValueError("使用 OpenAI API 客户端时需要提供 API URL")
        return {
            "api_key": api_key,
            "api_url": api_url
        }
    else:
        raise ValueError(f"未知的客户端类型：{client_type}")

def get_model(config: Config) -> str:
    """根据客户端类型和配置获取适当的模型。"""
    client_type = config.get("clients", {}).get("default", "ollama")
    client_config = config.get("clients", {}).get(client_type, {})
    return client_config.get("model", "llama3.2-vision")
