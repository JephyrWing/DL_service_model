"""Current sensor preprocessing and paired JSON/WAV uploads."""
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_bee_status import load_notebook_definitions

import streamlit_bee_status as app

notebook = load_notebook_definitions()
ENV_COLUMNS = notebook.ENV_COLUMNS

def app_settings(current):
    return {**current, 'current_dim': 11, 'time_part_categories': list(range(6))}

def observation(source, time, split='train', hive='1', value=30.):
    row = {'source_id': source, 'hive_id': hive, 'split': split,
           'observed_at': pd.Timestamp(time), 'recorded_at': pd.Timestamp(time)}
    row.update({c: value + i for i, c in enumerate(ENV_COLUMNS)})
    return row

class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame([observation(f's{i}', f'2026-11-03 {i:02d}:00') for i in range(10)])
        self.current_settings = notebook.fit_environment(self.frame)
        self.settings = app_settings(self.current_settings)

    def test_notebook_and_app_current_inputs_match(self):
        rows = self.frame.copy()
        rows.loc[0, 'weather pressure'] = 0
        rows.loc[1, 'weather temp'] = np.nan
        np.testing.assert_array_equal(notebook.current_environment(rows, self.current_settings),
                                      app.transform_current(rows, self.settings))

    def test_train_only_unique_statistics(self):
        other = observation('other', '2026-11-04', 'test', value=100000.)
        data = pd.concat([self.frame, self.frame.iloc[:3], pd.DataFrame([other])])
        self.assertEqual(notebook.fit_environment(data), self.current_settings)
        self.assertEqual(self.current_settings['training_observations'], 10)
        np.testing.assert_array_equal(self.current_settings['std'], np.ones(5))

    def test_missing_pressure_and_degenerate_columns(self):
        data = self.frame.copy()
        data['weather pressure'] = [0., 1000.] * 5
        data['weather temp'] = np.nan
        settings = notebook.fit_environment(data)
        self.assertEqual(settings['median'][4], 1000.)
        self.assertEqual(settings['mean'][4], 1000.)
        self.assertEqual(settings['median'][3], 0.)
        self.assertEqual(settings['std'][3], 1.)
        self.assertTrue(np.isfinite(notebook.current_environment(data, settings)).all())

    def test_device_json_preserves_observations(self):
        message = deepcopy(app.DEVICE_JSON_TEMPLATE)
        message['environment']['hive_temp'] = 34.5
        current = app.parse_device_json(json.dumps(message).encode('utf-8'))
        self.assertEqual(current['hive temp'], 34.5)
        self.assertTrue(np.isnan(current['hive humidity']))
        self.assertEqual(current['recorded_at'], pd.Timestamp(message['recorded_at']))
        values = app.transform_current(pd.DataFrame([current]), self.settings)
        self.assertEqual(values.shape, (1, 11))
        np.testing.assert_array_equal(values[0, 5:], np.eye(6)[2])

    def test_device_json_rejects_invalid_inputs(self):
        message = app.DEVICE_JSON_TEMPLATE
        invalid = [
            ('recorded_at', '2026-11-05T10:00:00Z', 'recorded_at'),
            ('recorded_at', '2026-11-05T09:00:00', 'recorded_at'),
            ('environment', {}, 'environment'),
            ('environment', {**message['environment'], 'hive_temp': True}, 'hive_temp'),
        ]
        for field, value, error in invalid:
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, error):
                app.parse_device_json(json.dumps({**message, field: value}))

    def test_batch_upload_matches_names_and_keeps_valid_results(self):
        def upload(name, content):
            file = BytesIO(content)
            file.name = name
            return file

        wav = BytesIO()
        sf.write(wav, np.full(1600, 0.1), 16000, format='WAV')
        payload = json.dumps(app.DEVICE_JSON_TEMPLATE).encode('utf-8')
        json_uploads = [upload('first.json', payload), upload('second.json', payload),
                        upload('missing.json', payload), upload('duplicate.json', payload),
                        upload('invalid.json', b'bad JSON')]
        wav_uploads = [upload('second.wav', wav.getvalue()), upload('orphan.wav', wav.getvalue()),
                       upload('FIRST.wav', wav.getvalue()),
                       upload('duplicate.wav', wav.getvalue()), upload('DUPLICATE.wav', wav.getvalue()),
                       upload('invalid.wav', wav.getvalue())]
        bundle = SimpleNamespace(config=app.AudioConfig())
        prediction = (np.array([0., 1., 0., 0.]), None)
        with patch.object(app, 'predict_waveform', return_value=prediction) as predict:
            results, errors = app.analyze_uploads(bundle, json_uploads, wav_uploads)
        self.assertEqual([result['name'] for result in results], ['first.json', 'second.json'])
        self.assertEqual(predict.call_count, 2)
        self.assertEqual([error.split(' · ')[0] for error in errors],
                         ['orphan.wav', 'missing.json', 'duplicate.json', 'invalid.json'])
        self.assertIn('같은 이름의 WAV', errors[1])

if __name__ == '__main__':
    unittest.main(verbosity=2)
