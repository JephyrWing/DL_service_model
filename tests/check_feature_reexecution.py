"""Verify persistent WAV-to-feature alignment and cache identity across runs."""
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import soundfile as sf

from check_bee_status import CACHE, load_notebook_definitions

notebook = load_notebook_definitions()


def run_checks():
    config = notebook.AudioConfig(segment_seconds=0.1, n_mels=8, n_fft=64, hop_length=32)
    with tempfile.TemporaryDirectory(prefix='cache_', dir=CACHE) as temporary:
        root = Path(temporary)
        first_output, second_output = root / 'run1', root / 'run2'
        first_output.mkdir()
        second_output.mkdir()
        files = ['a.wav', 'silent.wav', 'b.wav']
        for index, file in enumerate(files):
            time = np.arange(config.samples) / 16000
            waveform = np.zeros_like(time) if index == 1 else 0.2 * np.sin(2 * np.pi * (300 + 100 * index) * time)
            sf.write(root / file, waveform, 16000, subtype='FLOAT')
        all_arrays = []
        try:
            with patch.object(notebook, 'ROOT', root, create=True), patch.object(
                    notebook, 'OUTPUT', first_output, create=True), patch.object(
                    notebook, 'CONFIG', config, create=True):
                arrays, rows = notebook.load_audio_features(files)
                all_arrays.extend(arrays)
                audio = notebook.IndexedAudio(arrays, rows, ['b.wav', 'a.wav'])
                expected = audio[:]
                assert 'silent.wav' not in rows
                for index, file in enumerate(['b.wav', 'a.wav']):
                    feature = notebook.log_mel(notebook.read_audio(root / file, config), config)
                    np.testing.assert_array_equal(expected[index], feature)
                with patch.object(notebook, 'OUTPUT', second_output), patch.object(
                        notebook, 'read_audio', side_effect=AssertionError('Cache should be reused')):
                    arrays, rows = notebook.load_audio_features(files[::-1])
                    all_arrays.extend(arrays)
                    np.testing.assert_array_equal(notebook.IndexedAudio(arrays, rows, ['b.wav', 'a.wav'])[:], expected)
                    assert not arrays[0].flags.writeable
                assert len(list(root.glob('artifacts/main/feature_cache/*/features.npy'))) == 1
                stamps = {file: notebook.wav_stamp(file) for file in files}
                sf.write(root / 'b.wav', np.zeros(config.samples), 16000)
                try:
                    notebook.load_audio_features(files, stamps)
                except ValueError as error:
                    assert 'WAV' in str(error)
                else:
                    raise AssertionError('Evaluation accepted a changed WAV')
                arrays, rows = notebook.load_audio_features(files)
                all_arrays.extend(arrays)
                assert 'b.wav' not in rows
                changed = notebook.AudioConfig(segment_seconds=0.1, n_mels=10, n_fft=64, hop_length=32)
                with patch.object(notebook, 'CONFIG', changed):
                    arrays, rows = notebook.load_audio_features(files)
                    all_arrays.extend(arrays)
                    assert notebook.IndexedAudio(arrays, rows, ['a.wav']).shape == (1, *changed.shape)
                assert len(list(root.glob('artifacts/main/feature_cache/*/features.npy'))) == 3
        finally:
            for array in all_arrays:
                array._mmap.close()
    print('PASS: persistent cache reuse, row order, rejected WAVs, changed WAV and changed settings')


if __name__ == '__main__':
    run_checks()
