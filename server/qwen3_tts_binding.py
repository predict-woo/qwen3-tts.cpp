"""Python ctypes binding for libqwen3tts shared library."""

import ctypes
import ctypes.util
import os
import sys
import weakref
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# C struct mirrors (must match qwen3tts_c_api.h exactly)
# ---------------------------------------------------------------------------

class Qwen3TtsParams(ctypes.Structure):
    _fields_ = [
        ("max_audio_tokens", ctypes.c_int32),
        ("temperature", ctypes.c_float),
        ("top_p", ctypes.c_float),
        ("top_k", ctypes.c_int32),
        ("n_threads", ctypes.c_int32),
        ("repetition_penalty", ctypes.c_float),
        ("language_id", ctypes.c_int32),
    ]


class Qwen3TtsAudio(ctypes.Structure):
    _fields_ = [
        ("samples", ctypes.POINTER(ctypes.c_float)),
        ("n_samples", ctypes.c_int32),
        ("sample_rate", ctypes.c_int32),
    ]


# ---------------------------------------------------------------------------
# Library discovery
# ---------------------------------------------------------------------------

def _find_library() -> str:
    """Find libqwen3tts shared library."""
    # 1. Explicit env var
    env_path = os.environ.get("QWEN3TTS_LIB_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path

    # 2. Relative to this script (../build/)
    script_dir = Path(__file__).resolve().parent
    for suffix in ("dylib", "so", "dll"):
        candidate = script_dir.parent / "build" / f"libqwen3tts.{suffix}"
        if candidate.is_file():
            return str(candidate)

    # 3. System library search
    found = ctypes.util.find_library("qwen3tts")
    if found:
        return found

    raise RuntimeError(
        "Cannot find libqwen3tts. Set QWEN3TTS_LIB_PATH or build with cmake first."
    )


def _load_library() -> ctypes.CDLL:
    """Load and configure the shared library."""
    lib_path = _find_library()
    lib = ctypes.CDLL(lib_path)

    # -- qwen3_tts_default_params --
    lib.qwen3_tts_default_params.argtypes = [ctypes.POINTER(Qwen3TtsParams)]
    lib.qwen3_tts_default_params.restype = None

    # -- qwen3_tts_create --
    lib.qwen3_tts_create.argtypes = [ctypes.c_char_p, ctypes.c_int32]
    lib.qwen3_tts_create.restype = ctypes.c_void_p

    # -- qwen3_tts_is_loaded --
    lib.qwen3_tts_is_loaded.argtypes = [ctypes.c_void_p]
    lib.qwen3_tts_is_loaded.restype = ctypes.c_int

    # -- qwen3_tts_synthesize --
    lib.qwen3_tts_synthesize.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(Qwen3TtsParams),
    ]
    lib.qwen3_tts_synthesize.restype = ctypes.POINTER(Qwen3TtsAudio)

    # -- qwen3_tts_synthesize_with_voice_file --
    lib.qwen3_tts_synthesize_with_voice_file.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p,
        ctypes.POINTER(Qwen3TtsParams),
    ]
    lib.qwen3_tts_synthesize_with_voice_file.restype = ctypes.POINTER(Qwen3TtsAudio)

    # -- qwen3_tts_synthesize_with_voice_samples --
    lib.qwen3_tts_synthesize_with_voice_samples.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
        ctypes.POINTER(Qwen3TtsParams),
    ]
    lib.qwen3_tts_synthesize_with_voice_samples.restype = ctypes.POINTER(Qwen3TtsAudio)

    # -- qwen3_tts_extract_embedding_file --
    lib.qwen3_tts_extract_embedding_file.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
    ]
    lib.qwen3_tts_extract_embedding_file.restype = ctypes.c_int32

    # -- qwen3_tts_synthesize_with_embedding --
    lib.qwen3_tts_synthesize_with_embedding.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
        ctypes.POINTER(Qwen3TtsParams),
    ]
    lib.qwen3_tts_synthesize_with_embedding.restype = ctypes.POINTER(Qwen3TtsAudio)

    # -- qwen3_tts_synthesize_icl_file --
    lib.qwen3_tts_synthesize_icl_file.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
        ctypes.POINTER(Qwen3TtsParams),
    ]
    lib.qwen3_tts_synthesize_icl_file.restype = ctypes.POINTER(Qwen3TtsAudio)

    # -- qwen3_tts_prepare_icl_prompt_file --
    lib.qwen3_tts_prepare_icl_prompt_file.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p,
        ctypes.POINTER(Qwen3TtsParams),
    ]
    lib.qwen3_tts_prepare_icl_prompt_file.restype = ctypes.c_void_p

    # -- qwen3_tts_synthesize_with_icl_prompt --
    lib.qwen3_tts_synthesize_with_icl_prompt.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p,
        ctypes.POINTER(Qwen3TtsParams),
    ]
    lib.qwen3_tts_synthesize_with_icl_prompt.restype = ctypes.POINTER(Qwen3TtsAudio)

    # -- qwen3_tts_free_icl_prompt --
    lib.qwen3_tts_free_icl_prompt.argtypes = [ctypes.c_void_p]
    lib.qwen3_tts_free_icl_prompt.restype = None

    # -- qwen3_tts_sample_rate --
    lib.qwen3_tts_sample_rate.argtypes = [ctypes.c_void_p]
    lib.qwen3_tts_sample_rate.restype = ctypes.c_int32

    # -- qwen3_tts_free_audio --
    lib.qwen3_tts_free_audio.argtypes = [ctypes.POINTER(Qwen3TtsAudio)]
    lib.qwen3_tts_free_audio.restype = None

    # -- qwen3_tts_destroy --
    lib.qwen3_tts_destroy.argtypes = [ctypes.c_void_p]
    lib.qwen3_tts_destroy.restype = None

    # -- qwen3_tts_get_error --
    lib.qwen3_tts_get_error.argtypes = [ctypes.c_void_p]
    lib.qwen3_tts_get_error.restype = ctypes.c_char_p

    # -- model metadata + preset voices --
    lib.qwen3_tts_model_type.argtypes = [ctypes.c_void_p]
    lib.qwen3_tts_model_type.restype = ctypes.c_char_p

    lib.qwen3_tts_model_size.argtypes = [ctypes.c_void_p]
    lib.qwen3_tts_model_size.restype = ctypes.c_char_p

    lib.qwen3_tts_tts_model_path.argtypes = [ctypes.c_void_p]
    lib.qwen3_tts_tts_model_path.restype = ctypes.c_char_p

    lib.qwen3_tts_speaker_encoder_model_path.argtypes = [ctypes.c_void_p]
    lib.qwen3_tts_speaker_encoder_model_path.restype = ctypes.c_char_p

    lib.qwen3_tts_codec_encoder_model_path.argtypes = [ctypes.c_void_p]
    lib.qwen3_tts_codec_encoder_model_path.restype = ctypes.c_char_p

    lib.qwen3_tts_tokenizer_decoder_model_path.argtypes = [ctypes.c_void_p]
    lib.qwen3_tts_tokenizer_decoder_model_path.restype = ctypes.c_char_p

    lib.qwen3_tts_has_speaker_encoder.argtypes = [ctypes.c_void_p]
    lib.qwen3_tts_has_speaker_encoder.restype = ctypes.c_int

    lib.qwen3_tts_speaker_count.argtypes = [ctypes.c_void_p]
    lib.qwen3_tts_speaker_count.restype = ctypes.c_int32

    lib.qwen3_tts_speaker_name.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.qwen3_tts_speaker_name.restype = ctypes.c_char_p

    lib.qwen3_tts_speaker_dialect.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.qwen3_tts_speaker_dialect.restype = ctypes.c_char_p

    lib.qwen3_tts_get_speaker_embedding.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
    ]
    lib.qwen3_tts_get_speaker_embedding.restype = ctypes.c_int32

    return lib


# ---------------------------------------------------------------------------
# High-level wrapper
# ---------------------------------------------------------------------------

class _IclPromptHandle:
    """Private owner for an opaque Qwen3TtsIclPrompt pointer."""

    def __init__(self, lib: ctypes.CDLL, ptr):
        if not ptr:
            raise ValueError("null ICL prompt pointer")
        self._lib = lib
        self._ptr = ctypes.c_void_p(ptr)
        self._finalizer = weakref.finalize(
            self, _IclPromptHandle._free, self._lib, self._ptr
        )

    @staticmethod
    def _free(lib: ctypes.CDLL, ptr: ctypes.c_void_p) -> None:
        if ptr and ptr.value:
            lib.qwen3_tts_free_icl_prompt(ptr)
            ptr.value = None

    def close(self) -> None:
        self._finalizer()

    @property
    def _as_parameter_(self):
        if not self._ptr or not self._ptr.value:
            raise RuntimeError("ICL prompt handle is closed")
        return self._ptr


class QwenTTS:
    """High-level Python wrapper for the qwen3-tts C API."""

    def __init__(self, model_dir: str, n_threads: int = 4):
        self._lib = _load_library()
        self._n_threads = n_threads if n_threads > 0 else 4
        self._handle = self._lib.qwen3_tts_create(
            model_dir.encode("utf-8"), self._n_threads
        )
        if not self._handle:
            raise RuntimeError(f"Failed to load models from {model_dir}")

    def synthesize(
        self,
        text: str,
        temperature: float = 0.9,
        top_k: int = 50,
        language_id: int = 2050,
        max_audio_tokens: int = 2048,
        repetition_penalty: float = 1.05,
        n_threads: Optional[int] = None,
    ) -> tuple[list[float], int]:
        """Synthesize text to audio. Returns (samples, sample_rate)."""
        params = self._make_params(
            temperature=temperature, top_k=top_k, language_id=language_id,
            max_audio_tokens=max_audio_tokens, repetition_penalty=repetition_penalty,
            n_threads=n_threads if n_threads is not None else self._n_threads,
        )
        audio_ptr = self._lib.qwen3_tts_synthesize(
            self._handle, text.encode("utf-8"), ctypes.byref(params)
        )
        return self._extract_audio(audio_ptr)

    def synthesize_with_embedding(
        self,
        text: str,
        embedding: list[float],
        temperature: float = 0.9,
        top_k: int = 50,
        language_id: int = 2050,
        max_audio_tokens: int = 2048,
        repetition_penalty: float = 1.05,
        n_threads: Optional[int] = None,
    ) -> tuple[list[float], int]:
        """Synthesize with a pre-computed speaker embedding."""
        params = self._make_params(
            temperature=temperature, top_k=top_k, language_id=language_id,
            max_audio_tokens=max_audio_tokens, repetition_penalty=repetition_penalty,
            n_threads=n_threads if n_threads is not None else self._n_threads,
        )
        emb_arr = (ctypes.c_float * len(embedding))(*embedding)
        audio_ptr = self._lib.qwen3_tts_synthesize_with_embedding(
            self._handle, text.encode("utf-8"),
            emb_arr, len(embedding), ctypes.byref(params),
        )
        return self._extract_audio(audio_ptr)

    def extract_embedding(self, wav_path: str) -> list[float]:
        """Extract speaker embedding from a WAV file."""
        buf_size = 2048
        buf = (ctypes.c_float * buf_size)()
        result = self._lib.qwen3_tts_extract_embedding_file(
            self._handle, wav_path.encode("utf-8"), buf, buf_size
        )
        if result < 0:
            err = self._get_error()
            raise RuntimeError(f"Failed to extract embedding: {err}")
        return list(buf[:result])

    # -- Model metadata + preset voices -------------------------------------

    @property
    def model_type(self) -> str:
        """Model variant: 'base', 'custom_voice', or 'voice_design'."""
        raw = self._lib.qwen3_tts_model_type(self._handle)
        return raw.decode("utf-8") if raw else ""

    @property
    def model_size(self) -> str:
        """Model size tag ('0b6', '1b7', ...) or empty on older GGUFs."""
        raw = self._lib.qwen3_tts_model_size(self._handle)
        return raw.decode("utf-8") if raw else ""

    @property
    def has_speaker_encoder(self) -> bool:
        """True if the model ships an ECAPA-TDNN speaker encoder."""
        return bool(self._lib.qwen3_tts_has_speaker_encoder(self._handle))

    @property
    def icl_model_identities(self) -> dict[str, dict[str, object]]:
        """Resolved role-specific model identities for ICL prompt cache keys."""
        return {
            "tts_model": self._model_file_identity(self._model_path("tts")),
            "speaker_encoder_model": self._model_file_identity(self._model_path("speaker_encoder")),
            "codec_encoder_model": self._model_file_identity(self._model_path("codec_encoder")),
            "tokenizer_decoder_model": self._model_file_identity(self._model_path("tokenizer_decoder")),
        }

    def list_speakers(self) -> list[dict]:
        """List preset voices baked into the model. Empty for Base variants.

        Each entry is a dict with keys: name, dialect (str, '' if none).
        """
        n = int(self._lib.qwen3_tts_speaker_count(self._handle))
        out = []
        for i in range(n):
            name_ptr = self._lib.qwen3_tts_speaker_name(self._handle, i)
            dialect_ptr = self._lib.qwen3_tts_speaker_dialect(self._handle, i)
            if not name_ptr:
                continue
            name = name_ptr.decode("utf-8")
            dialect = dialect_ptr.decode("utf-8") if dialect_ptr else ""
            out.append({"name": name, "dialect": dialect})
        return out

    def get_speaker_embedding(self, name: str) -> list[float]:
        """Materialize a preset voice's speaker embedding (hidden_size floats).

        Raises RuntimeError if the preset name is unknown.
        """
        buf_size = 4096  # generous: hidden_size is 1024 (0.6B) or 2048 (1.7B)
        buf = (ctypes.c_float * buf_size)()
        result = self._lib.qwen3_tts_get_speaker_embedding(
            self._handle, name.encode("utf-8"), buf, buf_size,
        )
        if result < 0:
            err = self._get_error()
            raise RuntimeError(f"Failed to get speaker embedding '{name}': {err}")
        return list(buf[:result])

    def synthesize_with_preset(
        self,
        text: str,
        speaker: str,
        **synthesize_kwargs,
    ) -> tuple[list[float], int]:
        """Synthesize using a built-in preset voice by name."""
        embedding = self.get_speaker_embedding(speaker)
        return self.synthesize_with_embedding(text, embedding, **synthesize_kwargs)

    def synthesize_icl(
        self,
        text: str,
        reference_audio_path: str,
        reference_text: str,
        temperature: float = 0.9,
        top_k: int = 50,
        language_id: int = 2050,
        max_audio_tokens: int = 2048,
        repetition_penalty: float = 1.05,
        n_threads: Optional[int] = None,
    ) -> tuple[list[float], int]:
        """Synthesize with in-context-learning voice cloning.

        Encodes the reference audio with the Mimi codec and threads the
        resulting codes plus the transcript into the talker prefill.
        Intended cloning mode for Qwen3-TTS Base variants.
        """
        params = self._make_params(
            temperature=temperature, top_k=top_k, language_id=language_id,
            max_audio_tokens=max_audio_tokens, repetition_penalty=repetition_penalty,
            n_threads=n_threads if n_threads is not None else self._n_threads,
        )
        audio_ptr = self._lib.qwen3_tts_synthesize_icl_file(
            self._handle, text.encode("utf-8"),
            reference_audio_path.encode("utf-8"),
            reference_text.encode("utf-8"),
            ctypes.byref(params),
        )
        return self._extract_audio(audio_ptr)

    def prepare_icl_prompt(
        self,
        reference_audio_path: str,
        reference_text: str,
        temperature: float = 0.9,
        top_k: int = 50,
        language_id: int = 2050,
        max_audio_tokens: int = 2048,
        repetition_penalty: float = 1.05,
        n_threads: Optional[int] = None,
    ) -> _IclPromptHandle:
        """Prepare reusable ICL prompt state from reference audio and text."""
        params = self._make_params(
            temperature=temperature, top_k=top_k, language_id=language_id,
            max_audio_tokens=max_audio_tokens, repetition_penalty=repetition_penalty,
            n_threads=n_threads if n_threads is not None else self._n_threads,
        )
        ptr = self._lib.qwen3_tts_prepare_icl_prompt_file(
            self._handle,
            reference_audio_path.encode("utf-8"),
            reference_text.encode("utf-8"),
            ctypes.byref(params),
        )
        if not ptr:
            err = self._get_error()
            raise RuntimeError(f"Failed to prepare ICL prompt: {err}")
        return _IclPromptHandle(self._lib, ptr)

    def synthesize_with_icl_prompt(
        self,
        text: str,
        prompt_handle: _IclPromptHandle,
        temperature: float = 0.9,
        top_k: int = 50,
        language_id: int = 2050,
        max_audio_tokens: int = 2048,
        repetition_penalty: float = 1.05,
        n_threads: Optional[int] = None,
    ) -> tuple[list[float], int]:
        """Synthesize with a prepared ICL prompt handle."""
        params = self._make_params(
            temperature=temperature, top_k=top_k, language_id=language_id,
            max_audio_tokens=max_audio_tokens, repetition_penalty=repetition_penalty,
            n_threads=n_threads if n_threads is not None else self._n_threads,
        )
        audio_ptr = self._lib.qwen3_tts_synthesize_with_icl_prompt(
            self._handle,
            text.encode("utf-8"),
            prompt_handle,
            ctypes.byref(params),
        )
        return self._extract_audio(audio_ptr)

    def close(self):
        """Destroy the engine and release resources."""
        if self._handle:
            self._lib.qwen3_tts_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()

    # -- Private helpers --

    def _make_params(self, **kwargs) -> Qwen3TtsParams:
        params = Qwen3TtsParams()
        self._lib.qwen3_tts_default_params(ctypes.byref(params))
        params.n_threads = self._n_threads
        for key, value in kwargs.items():
            if value is None:
                continue
            setattr(params, key, value)
        return params

    def _extract_audio(self, audio_ptr) -> tuple[list[float], int]:
        if not audio_ptr:
            err = self._get_error()
            raise RuntimeError(f"Synthesis failed: {err}")
        audio = audio_ptr.contents
        n = audio.n_samples
        # Bulk copy via ctypes cast (much faster than per-element access)
        arr = ctypes.cast(audio.samples, ctypes.POINTER(ctypes.c_float * n)).contents
        samples = list(arr)
        sample_rate = audio.sample_rate
        self._lib.qwen3_tts_free_audio(audio_ptr)
        return samples, sample_rate

    def _get_error(self) -> str:
        err = self._lib.qwen3_tts_get_error(self._handle)
        return err.decode("utf-8") if err else "unknown error"

    def _model_path(self, role: str) -> str:
        funcs = {
            "tts": self._lib.qwen3_tts_tts_model_path,
            "speaker_encoder": self._lib.qwen3_tts_speaker_encoder_model_path,
            "codec_encoder": self._lib.qwen3_tts_codec_encoder_model_path,
            "tokenizer_decoder": self._lib.qwen3_tts_tokenizer_decoder_model_path,
        }
        raw = funcs[role](self._handle)
        return raw.decode("utf-8") if raw else ""

    @staticmethod
    def _model_file_identity(path: str) -> dict[str, object]:
        resolved = Path(path).expanduser().resolve()
        st = resolved.stat()
        return {
            "path": str(resolved),
            "size": st.st_size,
            "mtime_ns": st.st_mtime_ns,
        }
