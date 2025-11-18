import numpy
import numpy as np
import resampy
import soundfile

from svr_tts import SVR_TTS
from svr_tts.core import SynthesisInput

EPS = 1e-8

def l2norm(x: np.ndarray, axis=-1):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + EPS)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + EPS))

if __name__ == '__main__':
    tts = SVR_TTS(api_key="selector:tCJdy8PR", user_models_dir='tmp/bloodlines2')

    prosody, sr = soundfile.read('tmp/cin__hejtman_t_dobra_pano_MPFs.ogg')
    if prosody.ndim > 1:
        prosody = numpy.mean(prosody, axis=1)
    prosody = resampy.resample(prosody, sr, 24_000)
    sr = 24_000

    timbre, sr = soundfile.read('tmp/hejtman_tomas.wav')
    if timbre.ndim > 1:
        timbre = numpy.mean(timbre, axis=1)
    timbre = resampy.resample(timbre, sr, 24_000)
    sr = 24_000

    test_timbre, sr = soundfile.read('tmp/Answer.wav')
    if test_timbre.ndim > 1:
        test_timbre = numpy.mean(test_timbre, axis=1)
    test_timbre = resampy.resample(test_timbre, sr, 24_000)
    sr = 24_000

    style  = tts.compute_style(test_timbre.astype('float32'))

    # a = tts.compute_style(timbre.astype('float32'))
    # b = tts.compute_style(prosody.astype('float32'))
    #
    # # L2-нормализация (как было _l2n)
    # a = l2norm(a)
    # b = l2norm(b)
    #
    # # выжимаем до (192,)
    # a = a.reshape(-1)
    # b = b.reshape(-1)
    #
    # alpha = cosine_sim(a, b)
    # mix = alpha * a + (1 - alpha) * b
    # mix /= np.linalg.norm(mix) + 1e-9
    #
    # timbre = np.asarray(mix, dtype=np.float32).reshape(1, -1)
    outs = tts.synthesize([
        SynthesisInput(text="Ну, не извольте гневаться на холодный прием. Осторожность – первое дело. Лиходеи у нас пошаливают, а хоронятся так, что не сыскать.", stress=True, timbre_wave_24k=timbre, prosody_wave_24k=prosody),
    ])
    waves_24k = outs[0]
    print("len_out:", len(waves_24k), "sec:", len(waves_24k) / 22050.0)

    soundfile.write('tmp/example.wav', waves_24k, 22050)
