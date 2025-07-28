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
import hashlib
import logging
import os
import pickle
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm import tqdm

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
import onnxruntime as ort
import requests
from appdirs import user_cache_dir

# Длина перекрытия для кроссфейда между аудио сегментами
OVERLAP_LENGTH = 4096

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


def _crossfade(prev_chunk: np.ndarray, next_chunk: np.ndarray, overlap: int) -> np.ndarray:
    """
    Применяет кроссфейд (плавное смешивание) к двум аудио сегментам.

    Аргументы:
        prev_chunk: предыдущий аудио сегмент (numpy-массив).
        next_chunk: следующий аудио сегмент (numpy-массив).
        overlap: число точек перекрытия для кроссфейда.

    Возвращает:
        Обновленный next_chunk, где его начало плавно заменено данными из конца prev_chunk.
    """
    fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
    fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
    next_chunk[:overlap] = next_chunk[:overlap] * fade_in + prev_chunk[-overlap:] * fade_out
    return next_chunk


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
        "base": "svr_base_v2.onnx",
        "semantic": "svr_semantic.onnx",
        "encoder": "svr_encoder.onnx",
        "estimator": "svr_estimator.onnx",
        "vocoder": "svr_vocoder.onnx",
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
        self.semantic_model = ort.InferenceSession(self._download("semantic", cache_dir), providers=providers,
                                                   provider_options=provider_options)
        self.encoder_model = ort.InferenceSession(self._download("encoder", cache_dir), providers=providers,
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

    def _cacheable_timbre(self, speaker_style, timbre_wave):
        # создаём ключ из содержимого wav (bytes view быстрее, чем tobytes())
        key = hashlib.sha256(timbre_wave.view('uint8')).hexdigest()
        cache_file = self._timbre_cache_dir / f"{key}.pkl"

        # пытаемся загрузить кэш, ловим только отсутствие файла или кривой pkl
        try:
            return pickle.loads(cache_file.read_bytes())
        except (FileNotFoundError, pickle.UnpicklingError):
            # атомарно записываем новый кэш во временный файл, потом переименовываем
            data = pickle.dumps(speaker_style, protocol=pickle.HIGHEST_PROTOCOL)
            with tempfile.NamedTemporaryFile(dir=self._timbre_cache_dir, delete=False) as tmp:
                tmp.write(data)
                tmp.flush()
            Path(tmp.name).replace(cache_file)
            return speaker_style

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
            wave_24k, wave_feat, wave_feat_len, timbre_feat, timbre_feat_len, _ = \
                self.base_model.run(
                    ["wave_24k", "wave_feat", "wave_feat_len", "timbre_feat", "timbre_feat_len", "duration"], {
                        "input_ids": np.expand_dims(tokenize_resp['tokens'][idx], 0),
                        "timbre_wave_24k": timbre_wave,
                        "prosody_wave_24k": prosody_wave,
                        "duration_or_speed": np.array([duration_or_speed], dtype=np.float32),
                        "is_speed": np.array([is_speed], dtype=bool),
                        "scaling_min": np.array([scaling_min], dtype=np.float32),
                        "scaling_max": np.array([scaling_max], dtype=np.float32)
                    })

            # Получение семантических признаков для аудио и тембра
            semantic_wave = self.semantic_model.run(None, {
                'input_features': wave_feat.astype(np.float32)
            })[0][:, :wave_feat_len]
            semantic_timbre = self.semantic_model.run(None, {
                'input_features': timbre_feat.astype(np.float32)
            })[0][:, :timbre_feat_len]

            # Получаем условия для дальнейшего кодирования и генерации
            cat_conditions, latent_features, time_span, data_lengths, prompt_features, speaker_style, prompt_length = (
                self.encoder_model.run(
                    ["cat_conditions", "latent_features", "t_span", "data_lengths", "prompt_features",
                     "speaker_style", "prompt_length"], {
                        "wave_24k": wave_24k,
                        "prosody_wave": prosody_wave,
                        "semantic_wave": semantic_wave,
                        "semantic_timbre": semantic_timbre
                    }))

            generated_chunks: List[np.ndarray] = []
            prev_overlap_chunk: np.ndarray | None = None

            speaker_style = self._cacheable_timbre(speaker_style, timbre_wave)

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
