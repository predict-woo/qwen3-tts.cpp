#!/usr/bin/env python3
"""Pure standard-library tests for the ICL prompt cache."""

import hashlib
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from icl_cache import IclPromptCache, IclPromptCacheKey


class CloseTrackingPrompt:
    def __init__(self, name):
        self.name = name
        self.close_count = 0

    def close(self):
        self.close_count += 1


def file_identity(path):
    st = os.stat(path)
    return str(Path(path).resolve()), st.st_size, st.st_mtime_ns


def make_key(reference_path, reference_text="hello", language_id=2050, model_suffix="a"):
    ref_path, ref_size, ref_mtime_ns = file_identity(reference_path)
    ref_hash = hashlib.sha256(reference_text.encode("utf-8")).hexdigest()
    return IclPromptCacheKey(
        tts_model_path=f"/models/tts-{model_suffix}.gguf",
        tts_model_size=100,
        tts_model_mtime_ns=1000,
        speaker_encoder_model_path=f"/models/speaker-{model_suffix}.gguf",
        speaker_encoder_model_size=101,
        speaker_encoder_model_mtime_ns=1001,
        codec_encoder_model_path=f"/models/codec-{model_suffix}.gguf",
        codec_encoder_model_size=102,
        codec_encoder_model_mtime_ns=1002,
        tokenizer_decoder_model_path=f"/models/tokenizer-{model_suffix}.gguf",
        tokenizer_decoder_model_size=103,
        tokenizer_decoder_model_mtime_ns=1003,
        reference_path=ref_path,
        reference_size=ref_size,
        reference_mtime_ns=ref_mtime_ns,
        reference_text_hash=ref_hash,
        language_id=language_id,
    )


class IclPromptCacheTests(unittest.TestCase):
    def test_same_inputs_produce_same_key(self):
        with tempfile.NamedTemporaryFile() as ref:
            ref.write(b"audio")
            ref.flush()
            key_a = make_key(ref.name)
            key_b = make_key(ref.name)

        self.assertEqual(key_a, key_b)
        self.assertEqual(hash(key_a), hash(key_b))

    def test_changed_model_identity_changes_key(self):
        with tempfile.NamedTemporaryFile() as ref:
            ref.write(b"audio")
            ref.flush()
            key_a = make_key(ref.name, model_suffix="a")
            key_b = make_key(ref.name, model_suffix="b")

        self.assertNotEqual(key_a, key_b)

    def test_changed_reference_text_changes_key(self):
        with tempfile.NamedTemporaryFile() as ref:
            ref.write(b"audio")
            ref.flush()
            key_a = make_key(ref.name, reference_text="hello")
            key_b = make_key(ref.name, reference_text="different")

        self.assertNotEqual(key_a, key_b)

    def test_changed_reference_file_mtime_or_size_changes_key(self):
        with tempfile.NamedTemporaryFile(delete=False) as ref:
            ref.write(b"audio")
            ref.flush()
            ref_path = ref.name

        try:
            key_a = make_key(ref_path)
            time.sleep(0.001)
            with open(ref_path, "ab") as f:
                f.write(b"-changed")
            os.utime(ref_path, None)
            key_b = make_key(ref_path)
        finally:
            os.unlink(ref_path)

        self.assertNotEqual(key_a, key_b)
        self.assertNotEqual(key_a.reference_size, key_b.reference_size)
        self.assertNotEqual(key_a.reference_mtime_ns, key_b.reference_mtime_ns)

    def test_lru_eviction_removes_oldest_entry(self):
        cache = IclPromptCache(max_entries=2)
        with tempfile.NamedTemporaryFile() as ref:
            ref.write(b"audio")
            ref.flush()
            key_a = make_key(ref.name, reference_text="a")
            key_b = make_key(ref.name, reference_text="b")
            key_c = make_key(ref.name, reference_text="c")

        prompt_a = CloseTrackingPrompt("a")
        prompt_b = CloseTrackingPrompt("b")
        prompt_c = CloseTrackingPrompt("c")
        cache.put(key_a, prompt_a)
        cache.put(key_b, prompt_b)
        self.assertIs(cache.get(key_a), prompt_a)
        cache.put(key_c, prompt_c)

        self.assertIs(cache.get(key_a), prompt_a)
        self.assertIsNone(cache.get(key_b))
        self.assertIs(cache.get(key_c), prompt_c)
        self.assertEqual(prompt_b.close_count, 1)
        self.assertEqual(prompt_a.close_count, 0)
        self.assertEqual(prompt_c.close_count, 0)

    def test_clear_explicitly_closes_cached_values(self):
        cache = IclPromptCache(max_entries=4)
        with tempfile.NamedTemporaryFile() as ref:
            ref.write(b"audio")
            ref.flush()
            key_a = make_key(ref.name, reference_text="a")
            key_b = make_key(ref.name, reference_text="b")

        prompt_a = CloseTrackingPrompt("a")
        prompt_b = CloseTrackingPrompt("b")
        cache.put(key_a, prompt_a)
        cache.put(key_b, prompt_b)

        cache.clear()

        self.assertIsNone(cache.get(key_a))
        self.assertIsNone(cache.get(key_b))
        self.assertEqual(prompt_a.close_count, 1)
        self.assertEqual(prompt_b.close_count, 1)


if __name__ == "__main__":
    unittest.main()
