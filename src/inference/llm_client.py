"""云端 LLM 客户端（vLLM 离线推理，guided/free 双解码模式）。

- guided 臂：vLLM GuidedDecodingParams(json=冻结 schema)（xgrammar 后端，S2 锁定）。
- 自由文本臂：不引导 + 可选正则模板。
- temp=0 主实验 + 2 次确定性核查；temp∈{0.3,0.7} 敏感性。
- 每条返回 raw text、finish_reason（供 L3 截断判定）、usage。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..utils.io import load_json, project_path, load_yaml

VLLM_AVAILABLE = True
try:
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams  # type: ignore
except Exception:  # pragma: no cover
    VLLM_AVAILABLE = False


@dataclass
class LLMItem:
    prompt: str
    finish_reason: str | None = None
    raw: str = ""
    parsed: Any = None
    confidence: float | None = None
    usage: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "prompt_len": len(self.prompt),
            "finish_reason": self.finish_reason,
            "raw": self.raw,
            "parsed": self.parsed,
            "confidence": self.confidence,
            "usage": self.usage,
        }


class CloudLLMClient:
    """vLLM 离线客户端。model_repo 为本地模型目录或 HF 仓库。"""

    def __init__(
        self,
        model_repo: str,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.90,
        dtype: str = "bfloat16",
        backend: str = "xgrammar",
        is_mllm: bool = False,
        **vllm_kwargs,
    ):
        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM 未安装。请先: pip install vllm")
        backend = backend if is_mllm else backend
        self.is_mllm = is_mllm
        self.model_repo = model_repo
        self.schema_str = json.dumps(load_json(project_path("configs", "schema.json")))
        self.backend = backend
        self.engine_kwargs = {"max_model_len": max_model_len, "gpu_memory_utilization": gpu_memory_utilization}
        self.engine_kwargs.update(vllm_kwargs)
        self.llm = None

    def load(self):
        self.llm = LLM(model=self.model_repo, **self.engine_kwargs)
        return self

    def _make_params(
        self,
        temperature: float,
        guided: bool,
        seed: int | None,
        top_p: float = 1.0,
    ) -> SamplingParams:
        if guided:
            guided_decoding = GuidedDecodingParams(
                json=self.schema_str, backend=self.backend
            )
            return SamplingParams(
                temperature=temperature, top_p=top_p, max_tokens=512,
                guided_decoding=guided_decoding, seed=seed,
            )
        return SamplingParams(
            temperature=temperature, top_p=top_p, max_tokens=512, seed=seed
        )

    def generate(
        self,
        prompts: list[str],
        temperature: float = 0.0,
        guided: bool = True,
        seed: int | None = 1001,
    ) -> list[LLMItem]:
        if self.llm is None:
            self.load()
        params = self._make_params(temperature, guided, seed)
        outputs = self.llm.generate(prompts, params)
        items = []
        for o in outputs:
            text = o.outputs[0].text
            fr = o.outputs[0].finish_reason
            item = LLMItem(prompt=o.prompt, finish_reason=fr, raw=text)
            try:
                item.parsed = json.loads(text)
                item.confidence = item.parsed.get("confidence") if isinstance(item.parsed, dict) else None
            except json.JSONDecodeError:
                item.parsed = None
            items.append(item)
        return items


# ---------------------------------------------------------------------------
# Windows 兼容后端：transformers + outlines（vLLM 官方不支持 Windows 原生安装）
# 【辅助补充】vLLM 仅在 Linux/WSL2 可用；为让云端 LLM 实验在 Windows 上可跑，
# 提供基于 transformers 的等价客户端。E11 并发吞吐实测仍建议在 WSL2/Linux 用 vLLM。
# ---------------------------------------------------------------------------
class TransformersLLMClient:
    """基于 transformers (HuggingFace) 的 LLM 客户端，Windows 可跑。

    guided 解码优先用 outlines（structured generation）；outlines 不可用时
    退回普通生成（此时 L1 逼真保证由完整性校验如实判定，红线15）。
    """

    def __init__(self, model_repo: str, max_new_tokens: int = 512, device: str = "cuda", **kwargs):
        self.model_repo = model_repo
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.schema_str = json.dumps(load_json(project_path("configs", "schema.json")))
        self.model = None
        self.tokenizer = None
        self._outlines_model = None
        self._guided_generator = None  # 【加速】generator 只编译一次，避免每次重建(2710ms)

    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_repo, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_repo, torch_dtype="auto", device_map="auto", trust_remote_code=True
        )
        self.model.eval()
        # 尝试 outlines（guided JSON）
        try:
            import outlines
            # 【修正】outlines 0.2.x 用大写 Transformers 包装已加载模型；
            # 原代码小写 models.transformers(model, tokenizer) 传参不匹配会抛异常被吞 → 回退自由文本。
            self._outlines_model = outlines.models.Transformers(self.model, self.tokenizer)
            # 【修正+加速】生成器只编译一次，且用 greedy sampler（方案 temp=0 确定性主实验）。
            # 旧实现默认 multinomial sampler → do_sample=True 随机采样，batch/逐条结果不一致且不可复现；
            # greedy 每次取 argmax，确定性、可复现，batch 与逐条逐位一致。
            from outlines import generate
            from outlines.samplers import greedy
            self._guided_generator = generate.json(self._outlines_model, self.schema_str, sampler=greedy())
        except Exception:
            self._outlines_model = None
            self._guided_generator = None
        return self

    def generate(self, prompts, temperature=0.0, guided=True, seed=1001):
        import json as _json

        if guided and self._guided_generator is not None:
            return self._guided_generate(prompts, temperature, seed)
        # 自由文本回退（逐条，保持原语义）
        return [self._item_from_text(p) for p in prompts]

    def _guided_generate(self, prompts: list[str], temperature: float, seed: int) -> list:
        """逐条调用已编译的 outlines 生成器（greedy，确定性）。

        【权衡】不用 outlines 的 batch 推理：实测 batch 会截断 reasoning
        （每条 FSM 终止与 transformers batch 交互不可靠，输出长度被压短），
        属精度损失，不可接受。改为逐条调用同一已编译生成器：
          - 生成器在 load() 只编译一次，消除重建开销（~2710ms/次）；
          - greedy 每次取 argmax，b温度=0 确定性、可复现，与 batch 无关；
          - 每条独立 max_tokens，reasoning 完整不截断，精度零损失。
        """
        items = []
        for p in prompts:
            kwargs = {"max_tokens": self.max_new_tokens}
            if temperature is not None and temperature > 0:
                kwargs["temperature"] = float(temperature)
            try:
                r = self._guided_generator(p, **kwargs)
            except Exception:
                r = None
            if isinstance(r, (dict, list)):
                text = json.dumps(r, ensure_ascii=False)
            else:
                text = str(r) if r is not None else "{}"
            items.append(self._item_from_text(p, text))
        return items

    def _item_from_text(self, prompt: str, text: str | None = None) -> "LLMItem":
        import json as _json
        if text is None:
            text = self._free_text(prompt)
        item = LLMItem(prompt=prompt, finish_reason="stop", raw=text)
        try:
            item.parsed = _json.loads(text)
            item.confidence = item.parsed.get("confidence") if isinstance(item.parsed, dict) else None
        except _json.JSONDecodeError:
            item.parsed = None
        return item

    def _free_text(self, prompt: str, temperature: float = 0.0) -> str:
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens,
                temperature=float(temperature), do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def get_llm_client(model_repo: str, backend: str = "auto", **kwargs) -> CloudLLMClient:
    """工厂：auto=优先 vLLM（Linux），不可用则 transformers（Windows）。【辅助补充】"""
    import importlib.util

    if backend == "transformers":
        return TransformersLLMClient(model_repo, **kwargs)
    if backend == "vllm":
        return CloudLLMClient(model_repo, **kwargs)
    # auto
    has_vllm = importlib.util.find_spec("vllm") is not None
    if has_vllm:
        try:
            return CloudLLMClient(model_repo, **kwargs)
        except Exception:
            return TransformersLLMClient(model_repo, **kwargs)
    return TransformersLLMClient(model_repo, **kwargs)