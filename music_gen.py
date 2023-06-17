from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from music2vec import AlbumDownloader

import torchaudio
from typing import List

downloader = AlbumDownloader(reset=False)
downloader.download_albums(["Discovery"])

class StyleTransformer:
    def __init__(self, continuation_length):
        self.model = MusicGen.get_pretrained("melody")
        self.model.set_generation_params(duration=continuation_length)

    def generate_continuation(self, prompts: List[str]):
        melody_waveform, sr = torchaudio.load(
            "previews/Discovery/Harder, Better, Faster, Stronger.mp3"
        )

        melody_waveform = melody_waveform[:, 10*sr:20*sr]
        torchaudio.save("output/melody_waveform.wav", melody_waveform, sr)

        melody_waveform = melody_waveform.unsqueeze(0).repeat(2, 1, 1)
        output = self.model.generate_with_chroma(
            descriptions=[
                '50s rock song',
                '90s EDM song with a lot of bass and drums',
            ],
            melody_wavs=melody_waveform,
            melody_sample_rate=sr,
            progress=True
        )

        torchaudio.save("output/1.wav", output[0, : , :].squeeze(1).cpu(),
                        self.model.sample_rate)
        torchaudio.save("output/2.wav", output[1, : , :].squeeze(1).cpu(),
                        self.model.sample_rate)

style_transformer = StyleTransformer("Happy rock", 30)
style_transformer.generate_continuation()
