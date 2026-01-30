import math
import re
from typing import List
from typing import Optional, Callable, Tuple

import numpy as np


def split_text(
        text: str,
        max_text_len: int,
        splitter: str = r'(?<=[.!?…])\s+',
        *,
        soft_ratio: float = 1.25,
        hard_ratio: float = 1.60,
        min_chars: int = 70,
) -> List[str]:
    def _normalize(s: str) -> str:
        return re.sub(r"\s+", " ", s.strip())

    target = int(max_text_len)
    soft_max = max(target + 1, int(round(target * soft_ratio)))
    hard_max = max(soft_max + 1, int(round(target * hard_ratio)))

    # 1) режем на предложения
    phrases = re.split(splitter, text.strip()) if text else []
    phrases = [_normalize(p) for p in phrases if p and p.strip()]

    # 2) если "предложение" само длиннее hard_max — дробим по безопасным местам
    def _split_long_phrase(p: str) -> List[str]:
        p = _normalize(p)
        if len(p) <= hard_max:
            return [p]

        parts = []
        rest = p

        punct_seps = ["; ", ": ", " — ", " - ", ", "]
        # союзные “крайние” разрезы (перед пробелом-совсем-край)
        conj_seps = [
            " потому что ",
            " чтобы ",
            " который ", " которая ", " которые ",
            " когда ",
            " если ",
            " хотя ",
            " но ",
            " а ",
            " и ",
            " что ",
        ]

        while len(rest) > hard_max:
            cut = -1

            # 2.1) сначала пунктуация
            for sep in punct_seps:
                pos = rest.rfind(sep, 0, hard_max + 1)
                if pos != -1:
                    cut = pos + len(sep)
                    break

            # 2.2) потом союзы (крайний вариант ДО пробела)
            if cut == -1:
                low = rest.lower()
                for c in conj_seps:
                    pos = low.rfind(c, 0, hard_max + 1)
                    if pos != -1:
                        # режем ПЕРЕД союзом, чтобы союз ушел в правую часть
                        cut = pos + 1  # после ведущего пробела
                        break

            # 2.3) совсем край — пробел
            if cut == -1:
                pos = rest.rfind(" ", 0, hard_max + 1)
                if pos != -1:
                    cut = pos + 1
                else:
                    cut = hard_max  # совсем без вариантов

            left = _normalize(rest[:cut])
            if left:
                parts.append(left)
            rest = _normalize(rest[cut:])

        if rest:
            parts.append(rest)
        return parts

    expanded: List[str] = []
    for ph in phrases:
        expanded.extend(_split_long_phrase(ph))
    phrases = expanded

    # 3) упаковка в чанки (target + soft с look-ahead)
    chunks: List[str] = []
    cur = ""

    def _try_add(cur_s: str, ph: str) -> str:
        return (cur_s + " " + ph).strip() if cur_s else ph

    n = len(phrases)
    for i, ph in enumerate(phrases):
        candidate = _try_add(cur, ph)

        if len(candidate) <= target:
            cur = candidate
            continue

        # можно перелезть в soft, если это предотвращает микрочанк
        if len(candidate) <= soft_max:
            if len(ph) < min_chars:
                cur = candidate
                continue
            if i + 1 < n and len(phrases[i + 1]) < min_chars:
                cur = candidate
                continue

        # фиксируем текущий и начинаем новый
        if cur:
            chunks.append(cur)
            cur = ph
        else:
            cur = ph

        # страховка: вдруг cur > hard_max
        if len(cur) > hard_max:
            chunks.extend(_split_long_phrase(cur))
            cur = ""

    if cur:
        chunks.append(cur)

    # 4) пост-обработка: сливаем мелкие чанки с соседями
    def _merge_if_small(chs: List[str]) -> List[str]:
        out: List[str] = []
        i = 0
        while i < len(chs):
            s = chs[i]
            if len(s) < min_chars:
                if out and len(out[-1]) + 1 + len(s) <= hard_max:
                    out[-1] = (out[-1] + " " + s).strip()
                    i += 1
                    continue
                if i + 1 < len(chs) and len(s) + 1 + len(chs[i + 1]) <= hard_max:
                    chs[i + 1] = (s + " " + chs[i + 1]).strip()
                    i += 1
                    continue
            out.append(s)
            i += 1
        return out

    chunks = _merge_if_small(chunks)

    # 5) перебалансировка хвоста
    def _rebalance_tail(chs: List[str]) -> List[str]:
        if len(chs) < 2:
            return chs
        last = chs[-1]
        if len(last) >= min_chars:
            return chs

        prev = chs[-2]
        seps = [", ", "; ", ": ", " — ", " - ", " "]
        cut_positions = set()
        for sep in seps:
            start = 0
            while True:
                pos = prev.find(sep, start)
                if pos == -1:
                    break
                cut_positions.add(pos + len(sep))
                start = pos + 1

        candidates = sorted([p for p in cut_positions if 0 < p < len(prev)], reverse=True)

        best = None
        best_score = None
        for cut in candidates[:60]:
            new_prev = _normalize(prev[:cut])
            moved = _normalize(prev[cut:])
            if not new_prev or not moved:
                continue

            new_last = _normalize(moved + " " + last)
            if len(new_prev) < min_chars:
                continue
            if len(new_last) < min_chars:
                continue
            if len(new_last) > hard_max:
                continue

            score = abs(len(new_last) - target)
            if best is None or score < best_score:
                best = (new_prev, new_last)
                best_score = score

        if best:
            chs = chs[:-2] + [best[0], best[1]]
        return chs

    chunks = _rebalance_tail(chunks)
    return [_normalize(c) for c in chunks if c and c.strip()]


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
        fade = (np.cos(np.linspace(0, np.pi / 2, n, endpoint=True)) ** 2).astype(np.float32, copy=False)
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
    sq = np.pad(y ** 2, (pad, pad - (win % 2 == 0)), mode='constant')
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
    m = int(sr * mute_ms / 1000);
    f = int(sr * fade_ms / 1000)
    y[:m] = 0
    if f and m < y.size:
        e = min(m + f, y.size)
        y[m:e] *= np.linspace(0.0, 1.0, e - m, dtype=y.dtype)
    return y


def max_cps_log(dur_sec: float, cps_min=14.0, cps_max=19.0, t0=1.0, t1=30.0, k=15.0) -> float:
    # clamp dur to [t0..t1]
    if dur_sec <= t0:
        return cps_min
    if dur_sec >= t1:
        return cps_max

    # normalize to 0..1
    x = (dur_sec - t0) / (t1 - t0)

    # log-like saturating curve in 0..1
    # k controls curvature: bigger k = быстрее растёт в начале и сильнее насыщается
    y = math.log1p(k * x) / math.log1p(k)

    return cps_min + (cps_max - cps_min) * y


def target_duration(sec: float, n_chars: int, low: float = 5.0, t0=1.0, t1=30.0, k=15.0, cps_min=14.0):
    """
    sec      — исходная длительность аудио (сек)
    n_chars  — кол-во англ. букв
    low/high — допустимый диапазон букв/с
    return: (target_sec, stretch)
    """
    high = max_cps_log(sec, cps_min=cps_min, t0=t0, t1=t1, k=k)
    if sec <= 0 or n_chars <= 0:
        return sec, 1.0
    cps = n_chars / sec
    if cps < low:  # слишком медленно → укоротить
        tgt = n_chars / low
    elif cps > high:  # слишком быстро → удлинить
        tgt = n_chars / high
    else:  # ок → без изменений
        tgt = sec
    return tgt, (tgt / sec)


def extend_wave(wave: np.ndarray, duration_scale: float) -> np.ndarray:
    if duration_scale <= 0:
        return wave

    if abs(duration_scale - 1.0) < 1e-6:
        return wave

    orig_len = len(wave)
    target_len = max(int(round(orig_len * duration_scale)), 24_000)

    reps = int(np.ceil(target_len / orig_len))
    tiled = np.tile(wave, reps)

    return tiled[:target_len]


def istft_ola(spec: np.ndarray, n_fft: int = 1024, hop: int = 256) -> np.ndarray:
    """
    spec: complex64 [B, N, T] (N = n_fft//2+1)
    return: float32 [T_wav] (первый батч), 22.05 kHz
    """
    assert spec.ndim == 3, f"spec shape {spec.shape}"
    B, N, T = spec.shape
    assert N == n_fft // 2 + 1, f"N={N} != n_fft//2+1"

    win = np.hanning(n_fft).astype(np.float32)  # [n_fft]
    win_sq = (win ** 2).astype(np.float32)

    # iFFT по частотной оси N -> time n_fft
    # np.fft.irfft по последней оси частоты, нам нужно axis=1
    time_frames = np.fft.irfft(spec, n=n_fft, axis=1).astype(np.float32)  # [B, n_fft, T]
    time_frames *= win[:, None].astype(np.float32)  # window

    out_size = (T - 1) * hop + n_fft
    pad = (n_fft - hop) // 2

    # overlap-add
    y = np.zeros((B, out_size), dtype=np.float32)
    env = np.zeros((out_size,), dtype=np.float32)
    for t in range(T):
        start = t * hop
        y[:, start:start + n_fft] += time_frames[:, :, t]
        env[start:start + n_fft] += win_sq

    # обрезаем паддинг по краям
    y = y[:, pad:out_size - pad]
    env = env[pad:out_size - pad]

    # нормализация огибающей окна
    env = np.maximum(env, 1e-11)
    y = (y / env[None, :]).astype(np.float32)

    return y[0]  # первый батч

def ensure_min(prosody_wave: np.ndarray, sr: int = 24000, min_sec: float = 3.0) -> np.ndarray:
    y = np.asarray(prosody_wave, dtype=np.float32).reshape(-1)
    min_len = int(min_sec * sr)

    if y.size >= min_len:
        return y

    reps = int(np.ceil(min_len / y.size))  # сколько раз повторить
    return np.tile(y, reps)[:min_len]

