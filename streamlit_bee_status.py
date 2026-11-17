"""Batch CNN–MLP analysis of device JSON and WAV recordings."""
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import math

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st


ROOT = Path(__file__).resolve().parent
LABEL_MAP = {label: f"queen status {label}" for label in range(4)}
CLASS_NAMES = (
    '기존 여왕벌 존재',
    '여왕벌 없음',
    '새 여왕벌 거부됨',
    '새 여왕벌 수용됨',
)
ENV_COLUMNS = ['hive temp', 'hive humidity', 'hive pressure',
               'weather temp', 'weather pressure']
DEVICE_ENV_FIELDS = {column.replace(' ', '_'): column for column in ENV_COLUMNS}
DEVICE_JSON_TEMPLATE = {
    'hive_id': 'hive-1',
    'source_id': 'observation-001',
    'observed_at': '2026-11-05T10:00:00',
    'recorded_at': '2026-11-05T10:00:00',
    'environment': dict.fromkeys(DEVICE_ENV_FIELDS),
}


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    segment_seconds: float = 60.0  # WAV 하나 전체를 모델 입력 하나로 사용한다.
    n_mels: int = 64
    n_fft: int = 1024
    hop_length: int = 512
    fmin: float = 0.0
    fmax: float = 8000.0
    top_db: float = 80.0

    def __post_init__(self):
        if min(self.sample_rate, self.segment_seconds, self.n_mels,
               self.n_fft, self.hop_length) <= 0:
            raise ValueError("전처리 크기와 시간은 양수여야 합니다.")
        if not 0 <= self.fmin < self.fmax <= self.sample_rate / 2:
            raise ValueError("멜 주파수 범위를 확인하세요.")

    @property
    def samples(self):
        return round(self.sample_rate * self.segment_seconds)

    @property
    def shape(self):
        return (self.n_mels, 1 + self.samples // self.hop_length, 1)


def read_audio(source, config):
    """WAV 전체를 모노 16kHz 파형으로 읽는다."""
    if hasattr(source, "seek"):
        source.seek(0)
    with sf.SoundFile(source) as audio:
        original_sr = audio.samplerate
        waveform = audio.read(dtype="float32", always_2d=True)
    if not len(waveform):
        raise ValueError("읽을 수 있는 음성 구간이 없습니다.")
    if not np.isfinite(waveform).all():
        raise ValueError("음성에 NaN 또는 무한대 값이 있습니다.")
    waveform = waveform.mean(axis=1)
    if original_sr != config.sample_rate:
        waveform = librosa.resample(
            waveform, orig_sr=original_sr, target_sr=config.sample_rate,
            res_type="soxr_hq",
        )
    return np.asarray(waveform, dtype="float32")


def log_mel(waveform, config):
    """녹음 전체를 (mel, time, channel) 로그 멜 배열로 변환한다."""
    waveform = np.asarray(waveform, dtype="float32")
    if waveform.ndim != 1 or not len(waveform) or not np.isfinite(waveform).all():
        raise ValueError("비어 있지 않은 유한한 mono 음성이 필요합니다.")
    if len(waveform) > config.samples:
        raise ValueError("녹음 길이가 모델 입력 길이와 맞지 않습니다.")
    waveform = librosa.util.fix_length(waveform, size=config.samples)
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=config.sample_rate, n_fft=config.n_fft,
        hop_length=config.hop_length, n_mels=config.n_mels,
        fmin=config.fmin, fmax=config.fmax, power=2.0,
        center=True, pad_mode="constant",
    )
    return librosa.power_to_db(mel, ref=1.0, top_db=config.top_db)[..., None].astype("float32")


def audio_quality(waveform):
    """RMS·최대 진폭·±1 이상 샘플 비율을 계산한다."""
    return {
        "rms": float(np.sqrt(np.mean(np.square(waveform, dtype="float64")))),
        "peak": float(np.max(np.abs(waveform))),
        "over_full_scale_ratio": float(np.mean(np.abs(waveform) >= 1.0)),
    }


def hive_key(value):
    if value is None or pd.isna(value) or str(value).strip().lower() in ('', 'nan', 'none'):
        raise ValueError('Missing hive identifier')
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    if not np.isfinite(number) or number <= 0 or not number.is_integer():
        raise ValueError('Invalid hive identifier')
    return str(int(number))


def timestamp(value):
    value = pd.Timestamp(value)
    if pd.isna(value) or value.tzinfo is not None:
        raise ValueError('A valid local timestamp without timezone is required')
    return value


def numeric_values(frame):
    values = frame[ENV_COLUMNS].apply(pd.to_numeric, errors='coerce').to_numpy(dtype='float64')
    values[~np.isfinite(values)] = np.nan
    values[values[:, 4] == 0, 4] = np.nan
    return values


def validate_preprocessing(settings):
    if (settings['columns'] != ENV_COLUMNS or settings['time_part_categories'] != list(range(6))
            or settings['current_dim'] != 11):
        raise ValueError('Incompatible environment preprocessing schema')
    for field in ('median', 'mean', 'std'):
        value = np.asarray(settings[field])
        if value.shape != (5,) or not np.isfinite(value).all():
            raise ValueError(f'Invalid preprocessing statistics: {field}')
    if np.any(np.asarray(settings['std']) <= 0):
        raise ValueError('Invalid standard deviation')


def transform_current(frame, settings):
    validate_preprocessing(settings)
    values = numeric_values(frame)
    values = np.where(np.isnan(values), settings['median'], values)
    numeric = (values - settings['mean']) / settings['std']
    parts = np.array([timestamp(t).hour // 4 for t in frame['recorded_at']], dtype=int)
    return np.concatenate([numeric, np.eye(6)[parts]], axis=1).astype('float32')


@dataclass
class Bundle:
    model: object
    encoder: object
    config: AudioConfig
    settings: dict


def read_model_settings(directory):
    settings = json.loads((directory / 'preprocessing.json').read_text(encoding='utf-8'))
    if not isinstance(settings, dict) or settings.get('model_kind') != 'cnn_mlp':
        raise ValueError('CNN–MLP 모델이 필요합니다.')
    return settings


def load_bundle(directory):
    from keras.models import load_model
    directory = Path(directory)
    settings = read_model_settings(directory)
    if settings.get('schema_version') != 2:
        raise ValueError('bee_status.ipynb에서 CNN–MLP 모델을 다시 저장하세요.')
    config = AudioConfig(**settings['audio'])
    if config.shape != (64, 1876, 1):
        raise ValueError('Expected a whole 60-second audio input')
    validate_preprocessing(settings['environment'])
    if {int(k): v for k, v in settings['label_map'].items()} != LABEL_MAP:
        raise ValueError('Expected numeric queen status labels 0..3')
    encoder = load_model(directory / 'audio_encoder.keras', compile=False)
    model = load_model(directory / 'mlp.keras', compile=False)
    if encoder.input_shape != (None, *config.shape) or encoder.output_shape != (None, 64):
        raise ValueError('Incompatible audio encoder')
    if model.input_shape != (None, 75):
        raise ValueError('MLP requires 75 input features')
    encoder.trainable = False
    if model.output_shape != (None, 4):
        raise ValueError('Expected four class scores')
    return Bundle(model, encoder, config, settings)


def predict_waveform(bundle, waveform, current):
    quality = audio_quality(waveform)
    if quality['rms'] < bundle.settings['quality']['silence_rms']:
        raise ValueError('거의 무음인 녹음입니다.')
    feature = log_mel(waveform, bundle.config)
    env = transform_current(pd.DataFrame([current]), bundle.settings['environment'])
    audio = np.asarray(bundle.encoder(feature[None], training=False))
    inputs = np.concatenate([audio, env], axis=1)
    scores = np.asarray(bundle.model(inputs, training=False))
    if scores.shape != (1, 4) or not np.isfinite(scores).all():
        raise ValueError('Invalid model scores')
    return scores[0], feature

def bundle_stamp(directory):
    read_model_settings(directory)
    files = []
    for name in ('preprocessing.json', 'audio_encoder.keras', 'mlp.keras'):
        stat = (directory / name).stat()
        files.append((name, stat.st_mtime_ns, stat.st_size))
    return tuple(files)


def latest_model_bundle():
    artifacts = ROOT / 'artifacts'
    for run_root in (artifacts / 'main', artifacts / 'submission/ex_b'):
        for marker in sorted(run_root.glob('*/completed.json'), reverse=True):
            directory = marker.parent / 'cnn_mlp'
            try:
                return directory, bundle_stamp(directory)
            except (OSError, ValueError):
                continue
    directory = artifacts / 'cnn_mlp'
    return directory, bundle_stamp(directory)


@st.cache_resource(show_spinner='학습 모델을 불러오는 중입니다…')
def cached_bundle(directory, stamp):
    return load_bundle(directory)


def device_timestamp(value, field):
    try:
        return timestamp(datetime.fromisoformat(value))
    except (TypeError, ValueError) as error:
        raise ValueError(f'{field}에 시간대 표기 없는 ISO 8601 현지 시각을 입력하세요.') from error


def parse_device_json(payload):
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError('각 관측은 JSON 객체여야 합니다.')
    hive, source = data.get('hive_id'), data.get('source_id')
    if type(hive) not in (int, str):
        raise ValueError('hive_id에 문자열 또는 양의 정수를 입력하세요.')
    if not isinstance(source, str) or not source.strip():
        raise ValueError('source_id에 관측 식별자를 입력하세요.')
    environment = data.get('environment')
    if not isinstance(environment, dict) or DEVICE_ENV_FIELDS.keys() - environment.keys():
        raise ValueError('environment에 예제의 센서 항목 5개를 넣으세요.')
    observed_at = device_timestamp(data.get('observed_at'), 'observed_at')
    recorded_at = device_timestamp(data.get('recorded_at'), 'recorded_at')
    if recorded_at < observed_at:
        raise ValueError('recorded_at은 observed_at과 같거나 이후여야 합니다.')
    row = {'hive_id': hive_key(hive), 'source_id': source.strip(),
           'observed_at': observed_at, 'recorded_at': recorded_at}
    for field, column in DEVICE_ENV_FIELDS.items():
        value = environment[field]
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
            raise ValueError(f'environment.{field}에는 유한한 숫자 또는 null을 입력하세요.')
        row[column] = np.nan if value is None else value
    return row


def upload_stem(upload):
    return Path(upload.name).stem.casefold()


def analyze_uploads(bundle, json_uploads, wav_uploads):
    json_counts = Counter(map(upload_stem, json_uploads))
    wav_counts = Counter(map(upload_stem, wav_uploads))
    wav_by_stem = {upload_stem(upload): upload for upload in wav_uploads}
    results = []
    errors = []
    if not json_uploads:
        errors.append('분석할 장치 입력 JSON을 한 개 이상 올리세요.')
    for stem, upload in wav_by_stem.items():
        if stem not in json_counts:
            errors.append(f'{upload.name} · 같은 이름의 JSON을 올리세요.')

    for upload in json_uploads:
        stem = upload_stem(upload)
        try:
            if json_counts[stem] > 1 or wav_counts[stem] > 1:
                raise ValueError('같은 이름의 JSON과 WAV를 각각 한 개씩 올리세요.')
            source = wav_by_stem.get(stem)
            if source is None:
                raise ValueError('같은 이름의 WAV를 올리세요.')
            current = parse_device_json(upload.getvalue())
            source.seek(0)
            info = sf.info(source)
            if info.format not in ('WAV', 'WAVEX', 'RF64') or not 0 < info.duration <= bundle.config.segment_seconds:
                raise ValueError('0초보다 길고 60초 이하인 WAV가 필요합니다.')
            waveform = read_audio(source, bundle.config)
            scores, feature = predict_waveform(bundle, waveform, current)
            results.append({'name': upload.name, 'current': current, 'waveform': waveform,
                            'scores': scores, 'feature': feature})
        except (OSError, RuntimeError, ValueError, TypeError, OverflowError) as error:
            errors.append(f'{upload.name} · {error}')
    return results, errors


def show_file_analysis(bundle):
    st.subheader('녹음 파일 분석')
    with st.expander('장치 JSON 형식과 예제'):
        template = json.dumps(DEVICE_JSON_TEMPLATE, indent=2)
        st.caption('센서값을 모르면 null을 사용하세요. 시각은 시간대 표기 없는 현지 시각으로 입력하세요.')
        st.code(template, language='json')
        st.download_button('예제 JSON 다운로드', template + '\n', 'bee_device_input.json', 'application/json')
    st.caption('sample_01.json과 sample_01.wav처럼 이름을 맞춰 두 파일을 함께 올리세요. WAV는 60초 이하여야 합니다.')
    with st.form('device_uploads'):
        json_uploads = st.file_uploader('장치 입력 JSON', type=['json'], accept_multiple_files=True)
        wav_uploads = st.file_uploader('녹음 WAV', type=['wav'], accept_multiple_files=True)
        submitted = st.form_submit_button('입력 완료', type='primary')
    if submitted:
        with st.spinner('소리와 환경을 분석하고 있습니다…'):
            st.session_state.analysis_batch = analyze_uploads(bundle, json_uploads, wav_uploads)
    results, errors = st.session_state.get('analysis_batch', ([], []))
    for error in errors:
        st.error(error)
    if not results:
        return

    st.subheader(f'분석 결과 {len(results)}건')
    summary = pd.DataFrame([
        {'입력 파일': result['name'], '벌통': result['current']['hive_id'],
         '녹음 시각': result['current']['recorded_at'],
         '예측 상태': CLASS_NAMES[int(result['scores'].argmax())],
         '최대 점수': float(result['scores'].max())}
        for result in results
    ])
    st.dataframe(summary, hide_index=True)
    scores_table = pd.DataFrame([result['scores'] for result in results], columns=list(LABEL_MAP.values()))
    csv = pd.concat([summary, scores_table], axis=1).to_csv(index=False).encode('utf-8-sig')
    st.download_button('분석 결과 CSV 저장', csv, 'bee_analysis.csv', 'text/csv')

    selected = st.selectbox('상세 결과', range(len(results)), format_func=lambda index: results[index]['name'])
    result = results[selected]
    scores = result['scores']
    st.metric('예측 상태', CLASS_NAMES[int(scores.argmax())])
    st.audio(result['waveform'], sample_rate=bundle.config.sample_rate)
    st.bar_chart(pd.DataFrame({'예측 점수': scores}, index=CLASS_NAMES))
    fig, ax = plt.subplots(figsize=(10, 3))
    image = librosa.display.specshow(result['feature'][..., 0], sr=bundle.config.sample_rate,
        hop_length=bundle.config.hop_length, x_axis='time', y_axis='mel', ax=ax)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    with st.expander('음성 품질 지표'):
        st.json(audio_quality(result['waveform']))


def main():
    st.set_page_config(page_title='벌통 상태 모니터', page_icon='🐝', layout='wide')
    st.title('🐝 벌통 상태 모니터')
    try:
        directory, stamp = latest_model_bundle()
        key = (str(directory), stamp)
        if st.session_state.get('active_bundle_key') != key:
            st.session_state.pop('analysis_batch', None)
            st.session_state.active_bundle_key = key
        bundle = cached_bundle(*key)
    except FileNotFoundError:
        st.info('저장된 CNN–MLP 모델이 없습니다. bee_status.ipynb를 마지막 셀까지 실행하세요.')
        return
    except Exception as error:
        st.error(f'모델을 불러올 수 없습니다: {error}')
        return
    st.caption(f'CNN–MLP · {directory.parent.name}')
    show_file_analysis(bundle)


if __name__ == '__main__':
    main()
