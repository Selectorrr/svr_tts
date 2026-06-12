"""
Copyright 2025 synthvoice.ru

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import base64
import logging
import os
import re
import traceback
import queue
import threading
from itertools import zip_longest
from pathlib import Path

import resampy
from huggingface_hub import hf_hub_download, HfApi
from numpy.typing import NDArray
from onnxruntime import SessionOptions
from tqdm import tqdm

from svr_tts.utils import split_text, split_audio, _crossfade, prepare_prosody, mute_fade, istft_ola

from typing import NamedTuple, List, Any, Optional, Sequence, Dict, Callable, Generator
import numpy as np
# noinspection PyPackageRequirements
import onnxruntime as ort
import requests
from appdirs import user_cache_dir

# Длина перекрытия для кроссфейда между аудио сегментами
OVERLAP_LENGTH = 4096
EPS = 1e-8
INPUT_SR = 24_000


class SynthesisInput(NamedTuple):
    text: str
    stress: bool
    timbre_wave_24k: np.ndarray
    prosody_wave_24k: np.ndarray


VcFn = Callable[
    [NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]],
    NDArray[np.float32],
]

PostprocessFn = Callable[
    [NDArray[np.float32], SynthesisInput, NDArray[np.float32]],
    NDArray[np.float32],
]


def _prefetch_generator(generator, buffer_size=1):
    q = queue.Queue(maxsize=buffer_size)
    sentinel = object()
    stop_event = threading.Event()

    def worker():
        try:
            for item in generator:
                if stop_event.is_set():
                    break
                while not stop_event.is_set():
                    try:
                        q.put(item, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        except Exception as e:
            try:
                q.put(e, timeout=0.5)
            except queue.Full:
                pass
        finally:
            try:
                q.put(sentinel, timeout=0.5)
            except queue.Full:
                pass

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    try:
        while True:
            item = q.get()
            if item is sentinel:
                break
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        stop_event.set()
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break


class SVR_TTS:
    REPO_ID = "selectorrrr/svr-tts-large"

    def __init__(self, api_key,
                 tokenizer_service_url: str = "https://synthvoice.ru/tokenize_batch",
                 providers: Sequence[str | tuple[str, dict[Any, Any]]] | None = None,
                 provider_options: Sequence[dict[Any, Any]] | None = None,
                 session_options: SessionOptions | None = None,
                 timbre_cache_dir: str = 'workspace/voices/',
                 user_models_dir: str | None = None,
                 dur_norm_low: float = 5.0,
                 dur_high_t0=1.0,
                 dur_high_t1=30.0,
                 dur_high_k=15.0,
                 cps_min=14.0,
                 timbre_cond: float = 0.3,
                 prosody_cond: float = 0.6,
                 max_text_len: int = 150,
                 vc_type: str = 'native_bigvgan',
                 min_prosody_len: float = 3.0,

                # ---------- автоподбор скорости ----------
                 speed_search_attempts: int = 6,
                 speed_clip_min: float = 0.5,
                 speed_clip_max: float = 2.0,
                 speed_adjust_step_pct: float = 0.08,

                # ---------- допуск по длительности (зависит от длины реплики) ----------
                len_t_short: float = 1.0,
                len_t_long: float = 15.0,

                max_longer_pct_short: float = 0.35,
                max_longer_pct_long: float = 0.15,

                max_shorter_pct_short: float = 0.25,
                max_shorter_pct_long: float = 0.10,

                vc_func: VcFn = None,
                postprocess_func: PostprocessFn = None,
                put_yo: bool = True,
                device: str = 'cuda') -> None:
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.vc_func = vc_func
        self.postprocess_func = postprocess_func
        self._providers = providers
        self._provider_options = provider_options
        self._session_options = session_options
        self.dur_norm_low = dur_norm_low
        self.dur_high_t0 = dur_high_t0
        self.dur_high_t1 = dur_high_t1
        self.dur_high_k = dur_high_k
        self.cps_min = cps_min

        self.tokenizer_service_url = tokenizer_service_url
        self._cache_dir = self._get_cache_dir()
        os.environ["TQDM_POSITION"] = "-1"

        self._user_models_dir = Path(user_models_dir).expanduser() if user_models_dir else None

        self.vc_type = vc_type
        self._init_sessions()

        if api_key:
            api_key = base64.b64encode(api_key.encode('utf-8')).decode('utf-8')
        self.api_key = api_key

        self._timbre_cache_dir = Path(os.path.join(timbre_cache_dir, "timbre_cache"))
        self._timbre_cache_dir.mkdir(parents=True, exist_ok=True)
        self.timbre_cond = timbre_cond
        self.prosody_cond = prosody_cond
        self.max_text_len = max_text_len

        if vc_type and not vc_func:
            self.OUTPUT_SR = 22_050
        else:
            self.OUTPUT_SR = 24_000
        self.FADE_LEN = int(0.1 * self.OUTPUT_SR)
        self.min_prosody_len = min_prosody_len

        # ---------- параметры автоподбора скорости ----------
        self.speed_search_attempts = int(speed_search_attempts)
        self.speed_clip_min = float(speed_clip_min)
        self.speed_clip_max = float(speed_clip_max)
        self.speed_adjust_step_pct = float(speed_adjust_step_pct)

        # ---------- параметры допусков по длительности ----------
        self.len_t_short = float(len_t_short)
        self.len_t_long = float(len_t_long)

        self.max_longer_pct_short = float(max_longer_pct_short)
        self.max_longer_pct_long = float(max_longer_pct_long)

        self.max_shorter_pct_short = float(max_shorter_pct_short)
        self.max_shorter_pct_long = float(max_shorter_pct_long)

        self.put_yo = put_yo
        self.device = device

    def _init_sessions(self) -> None:
        cache_dir = self._cache_dir

        self.base_model = ort.InferenceSession(
            self._resolve("base", cache_dir),
            providers=self._providers,
            provider_options=self._provider_options,
            sess_options=self._session_options,
        )
        if not self.vc_func:
            self.cfe_model = ort.InferenceSession(
                self._resolve("cfe", cache_dir),
                providers=self._providers,
                provider_options=self._provider_options,
                sess_options=self._session_options,
            )
            self.semantic_model = ort.InferenceSession(
                self._resolve("semantic", cache_dir),
                providers=self._providers,
                provider_options=self._provider_options,
                sess_options=self._session_options,
            )
            self.encoder_model = ort.InferenceSession(
                self._resolve("encoder", cache_dir),
                providers=self._providers,
                provider_options=self._provider_options,
                sess_options=self._session_options,
            )
            self.style_model = ort.InferenceSession(
                self._resolve("style", cache_dir),
                providers=self._providers,
                provider_options=self._provider_options,
                sess_options=self._session_options,
            )
            self.estimator_model = ort.InferenceSession(
                self._resolve("estimator", cache_dir),
                providers=self._providers,
                provider_options=self._provider_options,
                sess_options=self._session_options,
            )

            if self.vc_type == 'native_bigvgan':
                self.vocoder_model = ort.InferenceSession(
                    self._resolve("vocoder", cache_dir),
                    providers=self._providers,
                    provider_options=self._provider_options,
                    sess_options=self._session_options,
                )
            elif self.vc_type == 'native_vocos':
                self.vocoder_model = ort.InferenceSession(
                    hf_hub_download(repo_id="BSC-LT/vocos-mel-22khz",
                                    filename="mel_spec_22khz_univ.onnx",
                                    cache_dir=cache_dir),
                    providers=self._providers,
                    provider_options=self._provider_options,
                    sess_options=self._session_options,
                )

    def _get_cache_dir(self) -> str:
        cache_dir = user_cache_dir("svr_tts", "SynthVoiceRu")
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    @staticmethod
    def _pick_best_name(key: str, names: list[str]) -> str | None:
        key_l = key.lower()
        best_ver = -1
        best_len = -1
        best_name: str | None = None

        for raw in names:
            n = raw.split("/")[-1]
            nl = n.lower()
            if key_l not in nl or not nl.endswith(".onnx"):
                continue
            m = re.search(r"_v(\d+)\.onnx$", nl)
            ver = int(m.group(1)) if m else 0
            name_len = len(nl)

            if (ver > best_ver) or (ver == best_ver and name_len > best_len):
                best_ver, best_len, best_name = ver, name_len, n

        return best_name

    def _resolve(self, key: str, cache_dir: str) -> str:
        if self._user_models_dir:
            local_names = [p.name for p in self._user_models_dir.glob("*.onnx")]
            best_local = self._pick_best_name(key, local_names)
            if best_local:
                lp = (self._user_models_dir / best_local)
                if lp.is_file():
                    resolved = str(lp.resolve())
                    return resolved
        return self._download(key, cache_dir)

    def _find_in_cache(self, key: str, cache_dir: str) -> str | None:
        cache_root = Path(cache_dir)
        if not cache_root.exists():
            return None

        candidates = []
        candidates += list(cache_root.rglob(f"*{key}*"))
        candidates = [p for p in candidates if p.is_file()]

        if not candidates:
            return None

        best_name = self._pick_best_name(key, [p.name for p in candidates])
        if best_name:
            for p in candidates:
                if p.name == best_name:
                    return str(p)
        return str(candidates[0])

    def _download(self, key: str, cache_dir: str) -> str:
        cached = self._find_in_cache(key, cache_dir)
        if cached:
            return cached

        files = HfApi().list_repo_files(self.REPO_ID)
        best = self._pick_best_name(key, files)
        if not best:
            raise FileNotFoundError(
                f"Не нашли модель '{key}' ни локально, ни в HF репозитории {self.REPO_ID}."
            )
        return hf_hub_download(repo_id=self.REPO_ID, filename=best, cache_dir=cache_dir)

    def _tokenize(self, token_inputs) -> dict:
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        response = requests.post(self.tokenizer_service_url, json=token_inputs, headers=headers)
        if response.status_code != 200:
            try:
                text = response.json()['text']
            except Exception:
                text = f"Ошибка {response.status_code}: {response.text}"
            raise AssertionError(text)
        return response.json()

    def _synthesize_segment(self, cat_conditions: np.ndarray, latent_features: np.ndarray,
                            time_span: List[float], data_length: int, prompt_features: np.ndarray,
                            speaker_style: Any, prompt_length: int) -> np.ndarray:
        encoded_input = np.expand_dims(cat_conditions[:data_length, :], axis=0)
        latent_input = np.expand_dims(np.transpose(latent_features[:data_length, :], (1, 0)), axis=0)
        prompt_input = np.expand_dims(np.transpose(prompt_features[:data_length, :], (1, 0)), axis=0)
        seg_length_arr = np.array([data_length], dtype=np.int32)
        time_span_np = np.array(time_span, dtype=np.float32)

        if not isinstance(prompt_length, np.ndarray):
            prompt_length_np = np.array(prompt_length, dtype=np.int64)
        else:
            prompt_length_np = prompt_length

        dev = self.device
        dev_id = 0
        if self._provider_options:
            for opt in self._provider_options:
                if isinstance(opt, dict) and "device_id" in opt:
                    try:
                        dev_id = int(opt["device_id"])
                        break
                    except (ValueError, TypeError):
                        pass

        encoded_ort = ort.OrtValue.ortvalue_from_numpy(encoded_input, dev, dev_id)
        prompt_ort = ort.OrtValue.ortvalue_from_numpy(prompt_input, dev, dev_id)
        style_ort = ort.OrtValue.ortvalue_from_numpy(speaker_style, dev, dev_id)
        seg_len_ort = ort.OrtValue.ortvalue_from_numpy(seg_length_arr, dev, dev_id)
        time_span_ort = ort.OrtValue.ortvalue_from_numpy(time_span_np, dev, dev_id)
        prompt_length_ort = ort.OrtValue.ortvalue_from_numpy(prompt_length_np, dev, dev_id)
        latent_ort = ort.OrtValue.ortvalue_from_numpy(latent_input, dev, dev_id)

        io_binding = self.estimator_model.io_binding()
        io_binding.bind_ortvalue_input("encoded_input", encoded_ort)
        io_binding.bind_ortvalue_input("prompt_input", prompt_ort)
        io_binding.bind_ortvalue_input("speaker_style", style_ort)
        io_binding.bind_ortvalue_input("time_span", time_span_ort)
        io_binding.bind_ortvalue_input("seg_length_arr", seg_len_ort)
        io_binding.bind_ortvalue_input("prompt_length", prompt_length_ort)

        for step in range(1, len(time_span)):
            current_time_np = np.array(time_span_np[step - 1], dtype=np.float32)
            current_time_ort = ort.OrtValue.ortvalue_from_numpy(current_time_np, dev, dev_id)

            current_step_np = np.array(step, dtype=np.int32)
            current_step_ort = ort.OrtValue.ortvalue_from_numpy(current_step_np, dev, dev_id)

            io_binding.bind_ortvalue_input("current_step", current_step_ort)
            io_binding.bind_ortvalue_input("current_time_input", current_time_ort)
            io_binding.bind_ortvalue_input("latent_input", latent_ort)

            io_binding.bind_output("latent_output", dev, dev_id)
            io_binding.bind_output("current_time_output", dev, dev_id)

            self.estimator_model.run_with_iobinding(io_binding)

            outputs = io_binding.get_outputs()
            latent_ort = outputs[0]

        latent_input = latent_ort.numpy()

        if isinstance(prompt_length, np.ndarray):
            p_len = int(prompt_length.item())
        else:
            p_len = int(prompt_length)
        latent_input = latent_input[:, :, p_len:]
        if self.vc_type == 'native_bigvgan':
            wave_22050 = self.vocoder_model.run(["wave_22050"], {
                "latent_input": latent_input
            })[0]
            return wave_22050[0]
        elif self.vc_type == 'native_vocos':
            wave_22050 = self.vocos_decode(self.vocoder_model, latent_input)
            return wave_22050
        else:
            raise NotImplementedError

    def compute_style(self, wave_24k):
        speaker_style = self.style_model.run(["speaker_style"], {
            "wave_24k": wave_24k
        })
        return speaker_style[0]

    def compute_semantic(self, wave_24k):
        feat, feat_len = self.cfe_model.run(
            ["feat", "feat_len"], {
                "wave_24k": wave_24k
            })
        semantic = self.semantic_model.run(None, {
            'input_features': feat.astype(np.float32)
        })[0][:, :feat_len]
        return semantic

    def _clip_speed(self, speed: float) -> float:
        if speed < self.speed_clip_min:
            return self.speed_clip_min
        if speed > self.speed_clip_max:
            return self.speed_clip_max
        return float(speed)

    def _synthesize_base(self,
                         cur_tokens: np.ndarray,
                         timbre_wave_24k: np.ndarray,
                         prosody_wave_24k: np.ndarray,
                         duration_or_speed: float,
                         is_speed: bool,
                         scaling_min: float,
                         scaling_max: float) -> np.ndarray:
        wave_24k, _ = self.base_model.run(
            ["wave_24k", "duration"], {
                "input_ids": np.expand_dims(cur_tokens, 0),
                "timbre_wave_24k": timbre_wave_24k,
                "prosody_wave_24k": prosody_wave_24k,
                "duration_or_speed": np.array([duration_or_speed], dtype=np.float32),
                "is_speed": np.array([is_speed], dtype=bool),
                "scaling_min": np.array([scaling_min], dtype=np.float32),
                "scaling_max": np.array([scaling_max], dtype=np.float32),
                "timbre_cond": np.array([self.timbre_cond], dtype=np.float32),
                "prosody_cond": np.array([self.prosody_cond], dtype=np.float32)
            })
        return wave_24k

    @staticmethod
    def _lerp(a: float, b: float, k: float) -> float:
        return a + (b - a) * k

    def _interp_by_len(self, t_sec: float, short_val: float, long_val: float) -> float:
        t0 = self.len_t_short
        t1 = self.len_t_long
        if t1 <= t0 + 1e-9:
            return float(long_val)
        if t_sec <= t0:
            return float(short_val)
        if t_sec >= t1:
            return float(long_val)
        k = (t_sec - t0) / (t1 - t0)
        return float(self._lerp(short_val, long_val, k))

    def _length_allowances(self, target_sec: float) -> tuple[float, float]:
        t = float(target_sec)
        allow_longer = self._interp_by_len(t, self.max_longer_pct_short, self.max_longer_pct_long)
        allow_shorter = self._interp_by_len(t, self.max_shorter_pct_short, self.max_shorter_pct_long)
        return max(0.0, allow_longer), max(0.0, allow_shorter)

    def _length_bounds(self, target_sec: float) -> tuple[float, float]:
        allow_longer, allow_shorter = self._length_allowances(target_sec)
        low = float(target_sec) * (1.0 - allow_shorter)
        high = float(target_sec) * (1.0 + allow_longer)
        return low, high

    def _synthesize_with_speed_search(self,
                                      cur_tokens: np.ndarray,
                                      timbre_wave_24k: np.ndarray,
                                      prosody_wave_24k: np.ndarray,
                                      target_sec: float,
                                      scaling_min: float,
                                      scaling_max: float) -> tuple[np.ndarray, float, float]:
        target_sec = float(target_sec)
        if target_sec <= 0:
            wave = self._synthesize_base(cur_tokens, timbre_wave_24k, prosody_wave_24k, 1.0, True, scaling_min, scaling_max)
            return wave, 1.0, 0.0

        EPS_S = 1e-6
        s_min = float(self.speed_clip_min)
        s_max = float(self.speed_clip_max)
        low_s, high_s = self._length_bounds(target_sec)

        def in_bounds(out_s: float) -> bool:
            return low_s <= out_s <= high_s

        def err_ratio(out_s: float) -> float:
            if in_bounds(out_s):
                return 0.0
            if out_s < low_s:
                return (low_s - out_s) / max(target_sec, EPS)
            return (out_s - high_s) / max(target_sec, EPS)

        def unfixable_at_bound(speed: float, out_s: float) -> bool:
            if speed >= s_max - EPS_S and out_s > high_s:
                return True
            if speed <= s_min + EPS_S and out_s < low_s:
                return True
            return False

        best_wave: Optional[np.ndarray] = None
        best_speed: float = 1.0
        best_err: float = float("inf")
        last_out_sec: float = 0.0

        def synth_eval(speed: float) -> tuple[np.ndarray, float, float]:
            nonlocal best_wave, best_speed, best_err, last_out_sec
            speed = self._clip_speed(float(speed))
            wave_24k = self._synthesize_base(
                cur_tokens, timbre_wave_24k, prosody_wave_24k, speed, True, scaling_min, scaling_max
            )
            out_sec = float(wave_24k.shape[-1]) / float(INPUT_SR)
            last_out_sec = out_sec
            err = err_ratio(out_sec)

            if err < best_err:
                best_err = err
                best_speed = speed
                best_wave = wave_24k
            return wave_24k, out_sec, err

        speed = self._clip_speed(s_max)
        _, out_sec, err = synth_eval(speed)

        if err <= 0.0:
            return best_wave, best_speed, best_err
        if unfixable_at_bound(speed, out_sec):
            return best_wave, best_speed, best_err

        base_sec_est = out_sec * speed
        speed_pred = self._clip_speed(base_sec_est / max(target_sec, EPS))

        if abs(speed_pred - best_speed) > EPS_S:
            _, out_sec, err = synth_eval(speed_pred)
            if err <= 0.0:
                return best_wave, best_speed, best_err
            if unfixable_at_bound(best_speed, out_sec):
                return best_wave, best_speed, best_err

        step = float(self.speed_adjust_step_pct)
        if step <= 0.0 or self.speed_search_attempts <= 0:
            return best_wave, best_speed, best_err

        mult = 1.0 + step
        for _ in range(self.speed_search_attempts):
            if in_bounds(last_out_sec):
                break
            if last_out_sec > high_s:
                if best_speed >= s_max - EPS_S:
                    break
                next_speed = best_speed * mult
            else:
                if best_speed <= s_min + EPS_S:
                    break
                next_speed = best_speed / mult

            next_speed = self._clip_speed(next_speed)
            if unfixable_at_bound(next_speed, last_out_sec):
                break

            _, out_sec, err = synth_eval(next_speed)
            if err <= 0.0:
                break
            if unfixable_at_bound(best_speed, out_sec):
                break

        return best_wave, best_speed, best_err

    def _generate_base_waves(self, inputs, tokens, duration_or_speed, is_speed, scaling_min, scaling_max, tqdm_kwargs):
        """
        Генератор базовых волн. Выполняет только синтез базовой модели и поиск скорости.
        """
        for current_input, cur_tokens in zip_longest(
                tqdm(inputs, desc="Базовый синтез (Base Model)", **tqdm_kwargs),
                tokens,
                fillvalue=None,
        ):
            if not cur_tokens:
                yield None
                continue

            timbre_wave = current_input.timbre_wave_24k.astype(np.float32)
            prosody_wave = current_input.prosody_wave_24k.astype(np.float32)

            target_sec = len(prosody_wave) / float(INPUT_SR)
            if target_sec < float(self.min_prosody_len):
                prosody_wave_24k = timbre_wave
            else:
                prosody_wave_24k = prosody_wave

            if duration_or_speed is not None:
                wave_24k = self._synthesize_base(
                    cur_tokens,
                    timbre_wave,
                    prosody_wave_24k,
                    float(duration_or_speed),
                    bool(is_speed),
                    scaling_min,
                    scaling_max,
                )
            else:
                wave_24k, best_speed, best_err = self._synthesize_with_speed_search(
                    cur_tokens,
                    timbre_wave,
                    prosody_wave_24k,
                    target_sec,
                    scaling_min,
                    scaling_max,
                )

            yield (wave_24k, current_input, timbre_wave, prosody_wave, prosody_wave_24k)

    def _generate_voice_conversion(self, buffered_base_generator):
        """
        Генератор конверсии голоса (вторая стадия конвейера).
        Выполняется в отдельном фоновом потоке.
        """
        for base_result in buffered_base_generator:
            try:
                if base_result is None:
                    yield None
                    continue

                wave_24k, current_input, timbre_wave, prosody_wave, prosody_wave_24k = base_result
                wave_24k_orig = wave_24k.copy()

                if not self.vc_func and self.vc_type:
                    min_len = min(len(timbre_wave), len(prosody_wave))
                    timbre_wave_concat = np.concatenate((timbre_wave[:min_len], prosody_wave[:min_len]))
                    speaker_style = self.compute_style(timbre_wave_concat)

                    cat_conditions, latent_features, time_span, data_lengths, prompt_features, prompt_length = (
                        self.encoder_model.run(
                            ["cat_conditions", "latent_features", "t_span", "data_lengths", "prompt_features",
                             "prompt_length"], {
                                "wave_24k": wave_24k,
                                "semantic_wave": self.compute_semantic(wave_24k),
                                "prosody_wave": timbre_wave_concat,
                                "semantic_timbre": self.compute_semantic(timbre_wave_concat)
                            }))

                    generated_chunks: List[np.ndarray] = []
                    prev_overlap_chunk: Optional[np.ndarray] = None

                    for seg_idx, seg_length in enumerate(data_lengths):
                        segment_wave = self._synthesize_segment(cat_conditions[seg_idx],
                                                                latent_features[seg_idx],
                                                                time_span,
                                                                int(seg_length),
                                                                prompt_features[seg_idx],
                                                                speaker_style,
                                                                prompt_length)
                        if seg_idx == 0:
                            mute_fade(segment_wave, self.OUTPUT_SR)
                            chunk = segment_wave[:-OVERLAP_LENGTH]
                            generated_chunks.append(chunk)
                            prev_overlap_chunk = segment_wave[-OVERLAP_LENGTH:]
                        elif seg_idx == len(data_lengths) - 1:
                            chunk = _crossfade(prev_overlap_chunk, segment_wave, OVERLAP_LENGTH)
                            generated_chunks.append(chunk)
                            break
                        else:
                            chunk = _crossfade(prev_overlap_chunk, segment_wave[:-OVERLAP_LENGTH], OVERLAP_LENGTH)
                            generated_chunks.append(chunk)
                            prev_overlap_chunk = segment_wave[-OVERLAP_LENGTH:]

                    wave_24k = np.concatenate(generated_chunks)
                elif self.vc_func:
                    wave_24k = self.vc_func(wave_24k, current_input.timbre_wave_24k, current_input.prosody_wave_24k)

                if self.postprocess_func:
                    wave_24k = self.postprocess_func(wave_24k, current_input, wave_24k_orig)

                yield wave_24k

            except Exception as e:
                traceback.print_exc()
                yield None
                continue

    def synthesize_batch(self, inputs: List[SynthesisInput],
                         stress_exclusions: Dict[str, Any] = {},
                         duration_or_speed: float = None,
                         is_speed: bool = False,
                         scaling_min: float = float('-inf'),
                         scaling_max: float = float('inf'), tqdm_kwargs: Dict[str, Any] = None) -> Generator[Optional[np.ndarray], None, None]:
        """
        Синтезирует аудио для каждого элемента входного списка с использованием двухстадийного конвейера (Double Prefetching).
        """
        items = [{"text": inp.text, "stress": inp.stress} for inp in inputs]
        tokenize_req = {"items": items, "exclusions": stress_exclusions, "put_yo": self.put_yo}
        tokenize_resp = self._tokenize(tokenize_req)
        tokens = tokenize_resp.get('tokens') or []
        tqdm_kwargs = tqdm_kwargs or {}

        # 1. Первая стадия: Генератор базовых волн
        base_generator = self._generate_base_waves(
            inputs, tokens, duration_or_speed, is_speed, scaling_min, scaling_max, tqdm_kwargs
        )
        buffered_base_generator = _prefetch_generator(base_generator, buffer_size=1)

        # 2. Вторая стадия: Генератор конверсии голоса (диффузия, кодирование, вокодер)
        vc_generator = self._generate_voice_conversion(buffered_base_generator)
        buffered_vc_generator = _prefetch_generator(vc_generator, buffer_size=1)

        # 3. Основной поток: Только возвращает готовые результаты, не блокируясь на вычислениях
        try:
            for result in buffered_vc_generator:
                yield result
        finally:
            self._init_sessions()

    def synthesize(self, inputs, tqdm_kwargs: Dict[str, Any] = None, rtrim_top_db=40,
                   stress_exclusions: Dict[str, Any] = {}):
        split_inputs = []
        mapping = []

        for idx, inp in enumerate(inputs):
            chunks = split_text(inp.text, self.max_text_len)
            chunks = split_audio(inp.prosody_wave_24k, chunks)

            for chunk_text, chunk_prosody in chunks:
                split_inputs.append(SynthesisInput(
                    text=chunk_text,
                    stress=inp.stress,
                    timbre_wave_24k=inp.timbre_wave_24k,
                    prosody_wave_24k=prepare_prosody(chunk_prosody, INPUT_SR, rtrim_top_db)
                ))
            mapping.append((idx, len(chunks)))

        batch_generator = self.synthesize_batch(split_inputs, stress_exclusions, tqdm_kwargs=tqdm_kwargs)

        wave_idx = 0
        OVERLAP_LEN = self.FADE_LEN

        for idx, count in mapping:
            generated_chunks = []
            prev_overlap_chunk = None
            ok = True

            for seg_idx in range(count):
                try:
                    wave = next(batch_generator)
                except StopIteration:
                    wave = None
                except Exception as e:
                    traceback.print_exc()
                    wave = None

                if not ok:
                    continue

                if wave is None:
                    ok = False
                    continue

                if seg_idx == 0:
                    if count > 1:
                        generated_chunks.append(wave[:-OVERLAP_LEN])
                    else:
                        generated_chunks.append(wave)
                    prev_overlap_chunk = wave[-OVERLAP_LEN:]
                elif seg_idx == count - 1:
                    chunk = _crossfade(prev_overlap_chunk, wave, OVERLAP_LEN)
                    generated_chunks.append(chunk)
                else:
                    chunk = _crossfade(prev_overlap_chunk, wave[:-OVERLAP_LEN], OVERLAP_LEN)
                    generated_chunks.append(chunk)
                    prev_overlap_chunk = wave[-OVERLAP_LEN:]

            if ok and generated_chunks:
                result_24k = np.concatenate(generated_chunks)
                yield result_24k
            else:
                yield None

    def vocos_decode(self, sess: ort.InferenceSession, mel: np.ndarray,
                     n_fft: int = 1024, hop: int = 256) -> np.ndarray:
        assert mel.ndim == 3 and mel.shape[1] == 80, f"mel shape {mel.shape}"

        mag, xr, yi = sess.run(None, {'mels': mel})
        spec = (mag.astype(np.float32)) * (xr.astype(np.float32) + 1j * yi.astype(np.float32))

        result = istft_ola(spec.astype(np.complex64), n_fft=n_fft, hop=hop)
        result = resampy.resample(result, INPUT_SR, self.OUTPUT_SR)
        return result