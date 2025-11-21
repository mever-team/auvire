import torchaudio

from speechbrain.utils.data_utils import split_path
from speechbrain.utils.fetching import LocalStrategy, fetch


def load_audio_for_speechbrain(path, audio_normalizer, frame_offset=0, num_frames=-1, savedir=None):
    source, fl = split_path(path)
    path = fetch(
        fl,
        source=source,
        savedir=savedir,
        local_strategy=LocalStrategy.SYMLINK,
    )
    signal, sr = torchaudio.load(str(path), frame_offset=frame_offset, num_frames=num_frames, channels_first=False)
    return audio_normalizer(signal, sr)
