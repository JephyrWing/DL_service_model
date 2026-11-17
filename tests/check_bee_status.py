"""Exercise the submitted B notebook and its Streamlit bundle with a one-epoch budget."""
import argparse
import ast
import contextlib
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / '.test_cache'
CACHE.mkdir(exist_ok=True)
tempfile.tempdir = str(CACHE)
sys.path.insert(0, str(ROOT))
os.environ.update(MPLBACKEND='Agg', TF_CPP_MIN_LOG_LEVEL='2',
                  TF_NUM_INTEROP_THREADS='2', TF_NUM_INTRAOP_THREADS='2',
                  OPENBLAS_NUM_THREADS='2', NUMBA_CACHE_DIR=str(CACHE),
                  MPLCONFIGDIR=str(CACHE / 'matplotlib'))

import nbformat
import numpy as np
import pandas as pd
import soundfile as sf


def load_notebook_definitions():
    """Load actual definitions without running notebook data, plots, or training cells."""
    name = 'bee_notebook_test_definitions'
    if name in sys.modules:
        return sys.modules[name]
    module = types.ModuleType(name)
    sys.modules[name] = module
    constants = {'ENV_COLUMNS', 'SPLITS', 'SILENCE_RMS'}
    imports = {'contextlib', 'dataclasses', 'datetime', 'hashlib', 'pathlib', 'uuid', 'json',
               'math', 're', 'tempfile', 'librosa', 'numpy', 'pandas', 'soundfile'}
    notebook = nbformat.read(ROOT / 'bee_status.ipynb', as_version=4)
    for cell in notebook.cells:
        if cell.cell_type != 'code':
            continue
        nodes = []
        for node in ast.parse(cell.source).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                nodes.append(node)
            elif isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id in constants for target in node.targets):
                nodes.append(node)
            elif isinstance(node, ast.ImportFrom) and node.module in imports:
                nodes.append(node)
            elif isinstance(node, ast.Import) and all(alias.name in imports for alias in node.names):
                nodes.append(node)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), cell.id, 'exec'), module.__dict__)
    return module


def synthetic_dataset(root):
    data = root / 'dataset'
    data.mkdir(parents=True)
    records = []
    for index in range(12):
        for label in range(4):
            source = f'd{index}_c{label}'
            date = pd.Timestamp('2026-01-01') + pd.Timedelta(days=index, hours=label)
            row = {'file name': source + '.raw', 'hive number': index % 2 + 1,
                   'date': str(date), 'queen status': label}
            row.update({c: 20. + label + i for i, c in enumerate(load_notebook_definitions().ENV_COLUMNS)})
            records.append(row)
            wave = 0.2 * np.sin(2 * np.pi * (200 + label * 100) * np.arange(16000) / 16000)
            for segment in (0, 1):
                sf.write(data / f'{source}__segment{segment}.wav', wave, 16000)
    pd.DataFrame(records).to_csv(data / 'all_data_updated.csv', index=False)


def check_ui(output_root, data_root):
    import faulthandler
    from unittest.mock import patch
    from streamlit.testing.v1 import AppTest
    import streamlit_bee_status as app

    directory = output_root / 'cnn_mlp'
    code = f"""
from pathlib import Path
from unittest.mock import patch
import streamlit_bee_status as app

directory = Path({str(directory)!r})
with patch.object(app, 'latest_model_bundle', return_value=(directory, app.bundle_stamp(directory))):
    app.main()
"""
    test_manifest = pd.read_csv(directory / 'test_manifest.csv')
    wav = (data_root / test_manifest.iloc[0]['file_path']).read_bytes()
    first = app.DEVICE_JSON_TEMPLATE
    second = {**first, 'source_id': 'second-observation'}

    def upload(name, content):
        file = io.BytesIO(content)
        file.name = name
        return file

    uploads = {
        '장치 입력 JSON': [
            upload('first.json', json.dumps(first).encode('utf-8')),
            upload('second.json', json.dumps(second).encode('utf-8')),
        ],
        '녹음 WAV': [upload('second.wav', wav), upload('first.wav', wav)],
    }

    def uploader(label, *, type, accept_multiple_files):
        assert accept_multiple_files
        return uploads[label]

    faulthandler.dump_traceback_later(40, repeat=True)
    try:
        with patch.object(app.st, 'file_uploader', side_effect=uploader):
            ui = AppTest.from_string(code, default_timeout=120).run()
            assert not ui.exception, list(ui.exception)
            assert not ui.get('tab') and not ui.radio
            assert not ui.sidebar.text_input and not ui.sidebar.selectbox
            assert 'analysis_batch' not in ui.session_state
            next(b for b in ui.button if b.label == '입력 완료').click().run()
            assert not ui.exception and not ui.error
            results, errors = ui.session_state['analysis_batch']
            assert not errors
            assert [result['name'] for result in results] == ['first.json', 'second.json']
            np.testing.assert_allclose(results[0]['scores'], results[1]['scores'], atol=2e-6)
            ui.selectbox[0].set_value(1).run()
            assert not ui.exception and not ui.error
            assert ui.metric[0].value in app.CLASS_NAMES
    finally:
        faulthandler.cancel_dump_traceback_later()
    print('PASS: Streamlit batch submit, paired JSON and WAV, and result detail', flush=True)


def run_cells(notebook, namespace, evaluate_only=False):
    def forbid_fit(*args, **kwargs):
        raise AssertionError('Evaluation tried to retrain')
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != 'code':
            continue
        assert notebook.cells[index - 1].cell_type == 'markdown'
        if not evaluate_only and cell.source.startswith(('cnn_dir =', 'def make_mlp_inputs')):
            name = 'cnn' if cell.source.startswith('cnn_dir =') else 'cnn_mlp'
            (namespace['OUTPUT'] / name).mkdir()
        exec(compile(cell.source, cell.id, 'exec'), namespace)
        if evaluate_only and cell.source.startswith('@contextmanager'):
            namespace['fit_model'] = forbid_fit
        if cell.source.startswith('def evaluate_split'):
            exec(compile(cell.source, cell.id, 'exec'), namespace)


def run_checks(check_streamlit=True, force_synthetic=False, autotune=False):
    notebook = nbformat.read(ROOT / 'bee_status.ipynb', as_version=4)
    nbformat.validate(notebook)
    definitions = load_notebook_definitions()
    config = definitions.AudioConfig()
    assert config.shape == (64, 1876, 1)
    late = np.zeros(config.samples, dtype='float32')
    late[-16000:] = 0.2 * np.sin(2 * np.pi * 400 * np.arange(16000) / 16000)
    feature = definitions.log_mel(late, config)
    assert feature[:, -25:, :].max() > feature[:, :25, :].max() + 20
    with tempfile.TemporaryDirectory(prefix='integration_', dir=CACHE) as temporary:
        temp = Path(temporary)
        real = (ROOT / 'dataset/all_data_updated.csv').is_file() and any((ROOT / 'dataset').rglob('*.wav'))
        real = real and not force_synthetic
        root = ROOT if real else temp / 'synthetic'
        if not real:
            synthetic_dataset(root)
        output = temp / 'artifacts'
        original = {'RUN_OPTIONS': {'root': root, 'output_root': output, 'epochs': 1,
                                    'smoke_per_class': 2, 'autotune': autotune}}
        replay = {'RUN_OPTIONS': {'root': root, 'evaluate_run_dir': output}}
        log_path = CACHE / f'bee_b_{"tuned" if autotune else "default"}.log'
        try:
            with log_path.open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log):
                run_cells(notebook, original)
                model_dir = output / 'cnn_mlp'
                paths = [output / 'cnn/model.keras', model_dir / 'mlp.keras', model_dir / 'audio_encoder.keras',
                         model_dir / 'preprocessing.json', output / 'manifest.csv']
                model_hashes = {path: sha256(path.read_bytes()).hexdigest() for path in paths}
                before = pd.read_csv(output / 'evaluation_summary.csv')
                (output / 'completed.json').unlink()
                run_cells(notebook, replay, evaluate_only=True)
            pd.testing.assert_frame_equal(before, pd.read_csv(output / 'evaluation_summary.csv'))
            assert all(sha256(path.read_bytes()).hexdigest() == expected for path, expected in model_hashes.items())
            assert original['cnn'].get_layer('audio_features').count_params() == 4160
            assert not original['encoder'].trainable_weights
            assert all(values.shape == (8, 75) for values in original['mlp_inputs'].values())
            for stage in ('cnn', 'cnn_mlp'):
                trials = pd.read_csv(output / stage / 'tuning_trials.csv')
                selected = json.loads((output / stage / 'training_metrics.json').read_text())
                assert len(trials) == (4 if autotune else 1)
                assert selected['trial'] == trials.loc[trials['val_loss'].idxmin(), 'trial']
            for split in ('train', 'val', 'test'):
                for unit in ('wav', 'source'):
                    report = pd.read_csv(model_dir / split / f'{unit}_class_report.csv', index_col='class')
                    assert report.index.tolist() == [0, 1, 2, 3]
                    assert pd.api.types.is_integer_dtype(report['support'])
            assert (output / 'test_results.png').is_file()

            import streamlit_bee_status as app
            bundle = app.load_bundle(model_dir)
            manifest = pd.read_csv(model_dir / 'test_manifest.csv', dtype={'hive_id': str},
                                   float_precision='round_trip', parse_dates=['observed_at', 'recorded_at'])
            first = manifest.iloc[0]
            waveform = app.read_audio(root / first['file_path'], bundle.config)
            scores, _ = app.predict_waveform(bundle, waveform, first)
            expected = pd.read_csv(model_dir / 'test/wav_predictions.csv').set_index('file_path').loc[first['file_path']]
            np.testing.assert_allclose(scores, expected[original['SCORE_COLUMNS']].to_numpy(dtype='float32'), atol=2e-6)
            if check_streamlit:
                check_ui(output, root)
            print(f'PASS: {"real" if real else "synthetic"} B training, Autotune={autotune}, '
                  'checkpoint replay, class reports and Streamlit bundle', flush=True)
        finally:
            for scope in (original, replay):
                for array in scope.get('arrays', []):
                    if isinstance(array, np.memmap):
                        array._mmap.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--notebook-only', action='store_true')
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--autotune', action='store_true')
    args = parser.parse_args()
    run_checks(not args.notebook_only, args.synthetic, args.autotune)
