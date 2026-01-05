# llm/app/config.py
from dataclasses import dataclass

@dataclass
class LLMConfig:
    MODEL: str = "Qwen/Qwen2.5-1.5B-Instruct"
    LOAD_4BIT: bool = True
    MAX_TOKENS: int = 80
    TEMPERATURE: float = 0.3

LLM_CFG = LLMConfig()
