"""Bounded cache for prepared ICL prompt handles."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class IclPromptCacheKey:
    tts_model_path: str
    tts_model_size: int
    tts_model_mtime_ns: int
    speaker_encoder_model_path: str
    speaker_encoder_model_size: int
    speaker_encoder_model_mtime_ns: int
    codec_encoder_model_path: str
    codec_encoder_model_size: int
    codec_encoder_model_mtime_ns: int
    tokenizer_decoder_model_path: str
    tokenizer_decoder_model_size: int
    tokenizer_decoder_model_mtime_ns: int
    reference_path: str
    reference_size: int
    reference_mtime_ns: int
    reference_text_hash: str
    language_id: int


def _file_identity(path: str) -> tuple[str, int, int]:
    resolved = str(Path(path).expanduser().resolve())
    st = os.stat(resolved)
    return resolved, st.st_size, st.st_mtime_ns


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_icl_prompt_cache_key(
    model_identities: Mapping[str, Mapping[str, Any]],
    reference_path: str,
    reference_text: str,
    language_id: int,
) -> IclPromptCacheKey:
    """Build a cache key from resolved model-role identities and reference data."""

    ref_path, ref_size, ref_mtime_ns = _file_identity(reference_path)

    def role(name: str) -> Mapping[str, Any]:
        try:
            return model_identities[name]
        except KeyError as exc:
            raise KeyError(f"missing model identity role: {name}") from exc

    tts = role("tts_model")
    speaker = role("speaker_encoder_model")
    codec = role("codec_encoder_model")
    tokenizer = role("tokenizer_decoder_model")

    return IclPromptCacheKey(
        tts_model_path=str(tts["path"]),
        tts_model_size=int(tts["size"]),
        tts_model_mtime_ns=int(tts["mtime_ns"]),
        speaker_encoder_model_path=str(speaker["path"]),
        speaker_encoder_model_size=int(speaker["size"]),
        speaker_encoder_model_mtime_ns=int(speaker["mtime_ns"]),
        codec_encoder_model_path=str(codec["path"]),
        codec_encoder_model_size=int(codec["size"]),
        codec_encoder_model_mtime_ns=int(codec["mtime_ns"]),
        tokenizer_decoder_model_path=str(tokenizer["path"]),
        tokenizer_decoder_model_size=int(tokenizer["size"]),
        tokenizer_decoder_model_mtime_ns=int(tokenizer["mtime_ns"]),
        reference_path=ref_path,
        reference_size=ref_size,
        reference_mtime_ns=ref_mtime_ns,
        reference_text_hash=_text_hash(reference_text),
        language_id=int(language_id),
    )


class IclPromptCache:
    """Small LRU cache that deterministically closes evicted prompt handles."""

    def __init__(self, max_entries: int):
        self._max_entries = max(0, int(max_entries))
        self._entries: OrderedDict[IclPromptCacheKey, Any] = OrderedDict()

    @property
    def max_entries(self) -> int:
        return self._max_entries

    def get(self, key: IclPromptCacheKey):
        value = self._entries.get(key)
        if value is None:
            return None
        self._entries.move_to_end(key)
        return value

    def put(self, key: IclPromptCacheKey, value) -> None:
        if self._max_entries <= 0:
            self._close_value(value)
            return

        old = self._entries.pop(key, None)
        if old is not None and old is not value:
            self._close_value(old)

        self._entries[key] = value
        while len(self._entries) > self._max_entries:
            _, evicted = self._entries.popitem(last=False)
            self._close_value(evicted)

    def clear(self) -> None:
        while self._entries:
            _, value = self._entries.popitem(last=False)
            self._close_value(value)

    def __len__(self) -> int:
        return len(self._entries)

    @staticmethod
    def _close_value(value) -> None:
        close = getattr(value, "close", None)
        if close is not None:
            close()
