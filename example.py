import resampy
import soundfile

from svr_tts import SVR_TTS
from svr_tts.core import SynthesisInput

if __name__ == '__main__':
    tts = SVR_TTS(api_key="selector:iX7bl2YQ")
    wave, sr = soundfile.read('tmp/example.ogg')
    wave_24k = resampy.resample(wave, sr, 24_000)
    outs = tts.synthesize_batch([
        SynthesisInput(text="Сбейте лестницу!", stress=True, timbre_wave_24k=wave_24k, prosody_wave_24k=wave_24k),
    ])
    waves_22050 = outs[0]
    soundfile.write('tmp/example.wav', waves_22050, 22_050)

    outs = tts.synthesize_batch([
        SynthesisInput(text="Сбейте лестницу!", stress=True, timbre_wave_24k=wave_24k, prosody_wave_24k=wave_24k),
    ])
    waves_22050 = outs[0]