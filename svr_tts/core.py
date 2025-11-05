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
import traceback
from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm import tqdm

from svr_tts.utils import split_text, split_audio, _crossfade, prepare_prosody, mute_fade

"""
Модуль синтеза речи с использованием нескольких моделей ONNX.
В модуле реализована генерация аудио из входного текста с учетом тембра и просодии.
Основные компоненты:
- Токенизация текста с помощью REST-сервиса.
- Инференс базовой, семантической, кодирующей, оценочной и вокодерной моделей.
- Обработка сегментов аудио с применением кроссфейда для плавного соединения.

Перед запуском убедитесь, что модели находятся по указанным путям и
что сервис токенизации доступен.
"""

from typing import NamedTuple, List, Any, Optional, Sequence, Dict
import numpy as np
# noinspection PyPackageRequirements
import onnxruntime as ort
import requests
from appdirs import user_cache_dir

# Длина перекрытия для кроссфейда между аудио сегментами
OVERLAP_LENGTH = 4096
EPS = 1e-8
INPUT_SR = 24_000
OUTPUT_SR = 22_050
FADE_LEN = int(0.1 * OUTPUT_SR)

logger = logging.getLogger(__name__)


class SynthesisInput(NamedTuple):
    """
    Структура входных данных для синтеза речи.

    Атрибуты:
        text: исходный текст для синтеза.
        stress: флаг, указывающий на использование ударений в тексте.
        timbre_wave_24k: массив для модели тембра (24kHz).
        prosody_wave_24k: массив для модели просодии (24kHz).
    """
    text: str
    stress: bool
    timbre_wave_24k: np.ndarray
    prosody_wave_24k: np.ndarray




class SVR_TTS:
    """
    Класс для синтеза речи с использованием нескольких ONNX моделей.

    Методы:
        _tokenize: отправляет запрос к сервису токенизации.
        _synthesize_segment: генерирует аудио для одного сегмента.
        synthesize_batch: синтезирует аудио для каждого элемента входных данных.
    """

    REPO_ID = "selectorrrr/svr-tts-large"
    MODEL_FILES = {
        "base": "svr_base_v3.onnx",
        "semantic": "svr_semantic.onnx",
        "encoder": "svr_encoder_v1.onnx",
        "style": "svr_style.onnx",
        "estimator": "svr_estimator.onnx",
        "vocoder": "svr_vocoder.onnx",
        "cfe": "svr_cfe.onnx",
    }

    def __init__(self, api_key, tokenizer_service_url: str = "https://synthvoice.ru/tokenize_batch",
                 providers: Sequence[str | tuple[str, dict[Any, Any]]] | None = None,
                 provider_options: Sequence[dict[Any, Any]] | None = None,
                 timbre_cache_dir='workspace/voices/') -> None:
        """
        Инициализация объектов инференс-сессий для всех моделей.

        Аргументы:
            tokenizer_service_url: URL для REST-сервиса токенизации.
            providers: список провайдеров для ONNX-моделей (например, CUDA или CPU).
        """
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.tokenizer_service_url = tokenizer_service_url
        cache_dir = self._get_cache_dir()
        os.environ["TQDM_POSITION"] = "-1"
        self.base_model = ort.InferenceSession(self._download("base", cache_dir), providers=providers,
                                               provider_options=provider_options)
        self.cfe_model = ort.InferenceSession(self._download("cfe", cache_dir), providers=providers,
                                               provider_options=provider_options)
        self.semantic_model = ort.InferenceSession(self._download("semantic", cache_dir), providers=providers,
                                                   provider_options=provider_options)
        self.encoder_model = ort.InferenceSession(self._download("encoder", cache_dir), providers=providers,
                                                  provider_options=provider_options)
        self.style_model = ort.InferenceSession(self._download("style", cache_dir), providers=providers,
                                                provider_options=provider_options)
        self.estimator_model = ort.InferenceSession(self._download("estimator", cache_dir), providers=providers,
                                                    provider_options=provider_options)
        self.vocoder_model = ort.InferenceSession(self._download("vocoder", cache_dir), providers=providers,
                                                  provider_options=provider_options)
        if api_key:
            api_key = base64.b64encode(api_key.encode('utf-8')).decode('utf-8')
        self.api_key = api_key
        self._timbre_cache_dir = Path(os.path.join(timbre_cache_dir, "timbre_cache"))
        self._timbre_cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_dir(self) -> str:
        cache_dir = user_cache_dir("svr_tts", "SynthVoiceRu")
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    def _download(self, key: str, cache_dir: str) -> str:
        return hf_hub_download(repo_id=self.REPO_ID, filename=self.MODEL_FILES[key], cache_dir=cache_dir)

    def _tokenize(self, token_inputs: List[dict]) -> dict:
        """
        Отправляет данные для токенизации к REST-сервису и возвращает результат.

        Аргументы:
            token_inputs: список словарей с данными текста и флагом ударений.

        Возвращает:
            Массив токенов, полученных от сервиса.

        Генерирует:
            AssertionError, если HTTP статус запроса не 200.
        """
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
        """
        Генерирует аудио для одного сегмента после кодирования.

        Аргументы:
            cat_conditions: категориальные условия для сегмента.
            latent_features: начальные латентные признаки для сегмента.
            t_span: временные метки для оценки.
            data_length: реальная длина сегмента для обработки.
            prompt_features: признаки подсказки для сегмента.
            speaker_style: стиль дикции, переданный из кодировщика.
            prompt_length: длина подсказки.

        Возвращает:
            Сегмент аудио в виде numpy-массива.
        """
        # Подготовка входных данных для инференса сегмента
        encoded_input = np.expand_dims(cat_conditions[:data_length, :], axis=0)
        latent_input = np.expand_dims(np.transpose(latent_features[:data_length, :], (1, 0)), axis=0)
        prompt_input = np.expand_dims(np.transpose(prompt_features[:data_length, :], (1, 0)), axis=0)
        seg_length_arr = np.array([data_length], dtype=np.int32)

        # Итеративно запускаем оценочную модель
        for step in range(1, len(time_span)):
            current_time = np.array(time_span[step - 1], dtype=np.float32)
            current_step = np.array(step, dtype=np.int32)
            latent_input, current_time = self.estimator_model.run(["latent_output", "current_time_output"], {
                "encoded_input": encoded_input,
                "prompt_input": prompt_input,
                "current_step": current_step,
                "speaker_style": speaker_style,
                "current_time_input": current_time,
                "time_span": np.array(time_span, dtype=np.float32),
                "seg_length_arr": seg_length_arr,
                "latent_input": latent_input,
                "prompt_length": prompt_length,
            })

        # Генерация аудио через вокодер
        latent_input = latent_input[:, :, prompt_length:]
        wave_22050 = self.vocoder_model.run(["wave_22050"], {
            "latent_input": latent_input
        })[0]
        return wave_22050[0]

    def synthesize_batch(self, inputs: List[SynthesisInput],
                         duration_or_speed: float = None,
                         is_speed: bool = False,
                         scaling_min: float = 0.875,
                         scaling_max: float = 1.3, tqdm_kwargs: Dict[str, Any] = None) -> List[np.ndarray]:
        """
        Синтезирует аудио для каждого элемента входного списка.

        Аргументы:
            inputs: список объектов SynthesisInput с данными для синтеза.
            duration_or_speed: желаемая продолжительность или скорость (если задана).
            is_speed: True, если задается скорость речи, False если продолжительность.
            scaling_min: минимальный коэффициент масштабирования.
            scaling_max: максимальный коэффициент масштабирования.

        Возвращает:
            Список numpy-массивов, каждый из которых представляет сгенерированное аудио.
        """
        synthesized_audios: List[Optional[np.ndarray]] = []
        token_list = [{"text": inp.text, "stress": inp.stress} for inp in inputs]
        tokenize_resp = self._tokenize(token_list)
        # Обработка каждого элемента входных данных
        tqdm_kwargs = tqdm_kwargs or {}
        for idx, current_input in enumerate(
                tqdm(inputs, desc=tokenize_resp['desc'], **tqdm_kwargs)):
            if not tokenize_resp['tokens'][idx]:
                synthesized_audios.append(None)
                continue
            timbre_wave = current_input.timbre_wave_24k.astype(np.float32)
            prosody_wave = current_input.prosody_wave_24k.astype(np.float32)

            # Если не задана скорость, рассчитаем длительность
            if not is_speed and not duration_or_speed:
                duration_or_speed = len(prosody_wave) / 24000

            # Получение базовых признаков через базовую модель
            wave_24k, _ = \
                self.base_model.run(
                    ["wave_24k", "duration"], {
                        "input_ids": np.expand_dims(tokenize_resp['tokens'][idx], 0),
                        "prosody_wave_24k": prosody_wave,
                        "duration_or_speed": np.array([duration_or_speed], dtype=np.float32),
                        "is_speed": np.array([is_speed], dtype=bool),
                        "scaling_min": np.array([scaling_min], dtype=np.float32),
                        "scaling_max": np.array([scaling_max], dtype=np.float32)
                    })

            min_len = min(len(timbre_wave), len(prosody_wave))
            timbre_wave = np.concatenate((timbre_wave[:min_len], prosody_wave[:min_len]))
            speaker_style = self.compute_style(timbre_wave)

            # Получаем условия для дальнейшего кодирования и генерации
            cat_conditions, latent_features, time_span, data_lengths, prompt_features, prompt_length = (
                self.encoder_model.run(
                    ["cat_conditions", "latent_features", "t_span", "data_lengths", "prompt_features",
                     "prompt_length"], {
                        "wave_24k": wave_24k,
                        "semantic_wave": self.compute_semantic(wave_24k),
                        "prosody_wave": timbre_wave,
                        "semantic_timbre": self.compute_semantic(timbre_wave)
                    }))

            generated_chunks: List[np.ndarray] = []
            prev_overlap_chunk: Optional[np.ndarray] = None

            # Обработка каждого сегмента аудио
            for seg_idx, seg_length in enumerate(data_lengths):
                segment_wave = self._synthesize_segment(cat_conditions[seg_idx],
                                                        latent_features[seg_idx],
                                                        time_span,
                                                        int(seg_length),
                                                        prompt_features[seg_idx],
                                                        speaker_style,
                                                        prompt_length)
                # Если это первый сегмент, сохраняем начальную часть и устанавливаем перекрытие
                if seg_idx == 0:
                    mute_fade(segment_wave, OUTPUT_SR)
                    chunk = segment_wave[:-OVERLAP_LENGTH]
                    generated_chunks.append(chunk)
                    prev_overlap_chunk = segment_wave[-OVERLAP_LENGTH:]
                # Если это последний сегмент, осуществляем окончательное склеивание
                elif seg_idx == len(data_lengths) - 1:
                    chunk = _crossfade(prev_overlap_chunk, segment_wave, OVERLAP_LENGTH)
                    generated_chunks.append(chunk)
                    break
                # Для всех промежуточных сегментов
                else:
                    chunk = _crossfade(prev_overlap_chunk, segment_wave[:-OVERLAP_LENGTH], OVERLAP_LENGTH)
                    generated_chunks.append(chunk)
                    prev_overlap_chunk = segment_wave[-OVERLAP_LENGTH:]

            # Объединяем все сегменты в одно аудио
            synthesized_audios.append(np.concatenate(generated_chunks))
        return synthesized_audios

    def synthesize(self, inputs, max_text_len=150, tqdm_kwargs: Dict[str, Any] = None):
        split_inputs = []
        mapping = []

        for idx, inp in enumerate(inputs):
            chunks = split_text(inp.text, max_text_len)
            chunks = split_audio(inp.prosody_wave_24k, chunks)

            for chunk_text, chunk_prosody in chunks:
                split_inputs.append(SynthesisInput(
                    text=chunk_text,
                    stress=inp.stress,
                    timbre_wave_24k=inp.timbre_wave_24k,
                    prosody_wave_24k=prepare_prosody(chunk_prosody, INPUT_SR)
                ))
            mapping.append((idx, len(chunks)))

        try:
            all_waves = self.synthesize_batch(split_inputs, tqdm_kwargs=tqdm_kwargs)
        except Exception as e:
            traceback.print_exc()
            all_waves = [None] * len(split_inputs)

        merged = []
        wave_idx = 0
        OVERLAP_LEN = FADE_LEN

        for _, count in mapping:
            generated_chunks = []
            prev_overlap_chunk = None
            ok = True

            for seg_idx in range(count):
                wave_22050 = all_waves[wave_idx]
                wave_idx += 1

                if wave_22050 is None:
                    ok = False
                    break

                if seg_idx == 0:
                    generated_chunks.append(wave_22050[:-OVERLAP_LEN])
                    prev_overlap_chunk = wave_22050[-OVERLAP_LEN:]
                elif seg_idx == count - 1:
                    chunk = _crossfade(prev_overlap_chunk, wave_22050, OVERLAP_LEN)
                    generated_chunks.append(chunk)
                else:
                    chunk = _crossfade(prev_overlap_chunk, wave_22050[:-OVERLAP_LEN], OVERLAP_LEN)
                    generated_chunks.append(chunk)
                    prev_overlap_chunk = wave_22050[-OVERLAP_LEN:]

            merged.append(np.concatenate(generated_chunks) if ok else None)

        return merged