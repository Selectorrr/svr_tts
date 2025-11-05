import re
from typing import List, Optional, Callable, Tuple

import numpy as np


def split_text(text: str, max_text_len: int, splitter: str = r'(?<=[.!?…])\s+') -> List[str]:
    phrases = re.split(splitter, text.strip()) if text else []
    chunks, cur = [], ""
    for ph in phrases:
        if not ph:
            continue
        add = ((" " if cur else "") + ph)
        if len(cur) + len(add) <= max_text_len:
            cur += add
        else:
            if cur:
                chunks.append(cur)
            cur = ph
    if cur:
        chunks.append(cur)
    return chunks


def split_audio(prosody_wave_24k: np.ndarray,
                text_chunks: List[str],
                *, sr: int = 24000,
                fade_ms: int = 30,
                pad_right_ms: int = 100,
                char_counter: Optional[Callable[[str], int]] = None,
                ) -> List[Tuple[str, np.ndarray]]:
    if not text_chunks:
        return []

    if char_counter is None:
        def char_counter(s: str) -> int:
            return max(0, len(re.sub(r"\s+", "", s)))

    y = np.asarray(prosody_wave_24k, dtype=np.float32)
    total_samples = int(y.shape[0])
    if total_samples <= 0:
        return []

    duration_sec = total_samples / float(sr)
    weights = np.asarray([char_counter(c) for c in text_chunks], dtype=np.float64)
    if weights.sum() <= 0:
        weights[:] = 1.0

    # длительности чанков и границы в сэмплах
    durations_sec = (weights / weights.sum()) * duration_sec
    cum_times = np.cumsum(durations_sec)
    cum_times[-1] = duration_sec
    bounds = np.rint(cum_times * sr).astype(np.int64)
    starts = np.concatenate(([0], bounds[:-1]))
    ends = bounds
    ends[-1] = total_samples

    # монотонность и минимум 1 сэмпл
    for i in range(len(starts)):
        if ends[i] <= starts[i]:
            ends[i] = min(total_samples, starts[i] + 1)
        if i + 1 < len(starts):
            starts[i + 1] = max(starts[i + 1], ends[i])

    pad_right = int(round(pad_right_ms * sr / 1000.0))

    out: List[Tuple[str, np.ndarray]] = []
    for chunk_text, s, e in zip(text_chunks, starts, ends):
        seg = y[s:e].astype(np.float32, copy=True)

        # cos^2 фейды (как в _crossfade)
        seg_ms = int(1000 * (len(seg) / sr))
        eff_fade_ms = max(0, min(fade_ms, seg_ms // 2))
        fade_n = int(round(eff_fade_ms * sr / 1000.0))

        if fade_n > 0:
            # входящий фейд 0→1
            fade_in = np.cos(np.linspace(np.pi / 2, 0, fade_n)) ** 2  # [0..1]
            seg[:fade_n] *= fade_in.astype(np.float32, copy=False)

            # исходящий фейд 1→0
            fade_out = np.cos(np.linspace(0, np.pi / 2, fade_n)) ** 2  # [1..0]
            seg[-fade_n:] *= fade_out.astype(np.float32, copy=False)

        if pad_right > 0:
            seg = np.concatenate([seg, np.zeros(pad_right, dtype=np.float32)], axis=0)

        out.append((chunk_text, seg))

    return out


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
    overlap = min(overlap, len(prev_chunk), len(next_chunk))
    fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
    fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
    next_chunk[:overlap] = next_chunk[:overlap] * fade_in + prev_chunk[-overlap:] * fade_out
    return next_chunk

def prepare_prosody(wave, sr, top_db=40):
    # 1) обрезаем тишину справа
    wave = rtrim_audio(wave, sr, top_db=top_db)

    keep = int(0.1 * sr)  # 100 мс

    # 2) отрезаем последние 100 мс (может занулить — это ок)
    wave = wave[:-keep] if keep > 0 else wave
    if wave.size == 0:
        return pad_with_silent(wave, sr)

    # 3) мягкий косинусный фейд хвоста (не длиннее остатка)
    n = min(keep, len(wave))
    if n > 0:
        fade = (np.cos(np.linspace(0, np.pi/2, n, endpoint=True))**2).astype(np.float32, copy=False)
        wave[-n:] *= fade  # wave уже float32

    # 4) дополняем тишиной
    return pad_with_silent(wave, sr)


def rtrim_audio(wave: np.ndarray, sr: int, top_db: float = 40) -> np.ndarray:
    y = np.asarray(wave, dtype=np.float32).reshape(-1)

    if y.size == 0:
        return y

    # скользящий RMS ~20 мс
    win = max(1, int(0.02 * sr))
    pad = win // 2
    # conv без зависимостей
    sq = np.pad(y**2, (pad, pad - (win % 2 == 0)), mode='constant')
    rms = np.sqrt(np.convolve(sq, np.ones(win, dtype=np.float32) / win, mode='valid'))

    ref = float(rms.max())
    if ref <= 0.0:
        return y

    thr = ref * (10.0 ** (-top_db / 20.0))
    nz = np.flatnonzero(rms >= thr)
    if nz.size == 0:
        return np.zeros(0, dtype=np.float32)

    end = int(nz[-1]) + 1  # включительно
    return y[:end]

def pad_with_silent(wave: np.ndarray, sr: int,
                    min_total_sec: float = 1.0,
                    tail_silence_sec: float = 0.1) -> np.ndarray:
    y = np.asarray(wave, dtype=np.float32).reshape(-1)
    n = y.size
    min_len = int(round(min_total_sec * sr))
    tail_len = int(round(tail_silence_sec * sr))

    # если короче 1 сек — падим до 1 сек, иначе добавляем 100 мс
    pad = (min_len - n) if n < min_len else tail_len
    if pad <= 0:
        return y
    return np.concatenate([y, np.zeros(pad, dtype=np.float32)], axis=0)

def mute_fade(y, sr, mute_ms=45, fade_ms=5):
    m = int(sr * mute_ms / 1000); f = int(sr * fade_ms / 1000)
    y[:m] = 0
    if f and m < y.size:
        e = min(m + f, y.size)
        y[m:e] *= np.linspace(0.0, 1.0, e - m, dtype=y.dtype)
    return y