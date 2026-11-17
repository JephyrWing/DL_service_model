# 벌통 상태 분류 모델: 확정 설계와 구현

2026-11-03 · 두 제출 파일에 구현 완료. 전체 학습은 사용자 노트북 실행 대상이다.

모든 학습 구현은 `bee_status.ipynb`, 모든 앱 구현은 `streamlit_bee_status.py`에 둔다. 별도 프로젝트 Python 모듈과 노트북 소스 사본을 사용하지 않는다. CNN·MLP·LSTM 계층과 compile은 모델별 노트북 셀에서 직접 선언한다. 파일별 필요한 전처리·재로딩 함수도 각 파일 안에 포함한다.

## 공통 입력

60초 WAV 하나 전체를 한 샘플로 사용한다. `AudioConfig`의 16kHz 모노, n_fft=1024, hop=512, mel=64 로그 멜 `(64,1876,1)`을 유지한다. 정답은 정수 `queen status` 0~3이며 숫자 상태명을 표시한다. 손실은 `sparse_categorical_crossentropy`다.

환경 수치 순서는 `hive temp`, `hive humidity`, `hive pressure`, `weather temp`, `weather pressure`로 고정한다. `time_part = hour // 4`의 원핫 범주는 0~5로 고정한다. 월·계절·날짜·식별자·파일명·target·queen presence·queen acceptance는 예측 특징에서 제외한다. 날짜와 식별자는 연결·정렬·분할에 사용한다.

| 입력 | 크기 |
|---|---|
| 현재 환경 수치 + 시간대 | 5 + 6 = 11 |
| CNN 특징 + 현재 환경 | 64 + 11 = 75 |
| LSTM 시점당 환경 + 시간대 + delta_hours | 5 + 6 + 1 = 12 |
| LSTM 환경 시퀀스 | (6,12) |

## 날짜 분할과 관측 연결

`observed_at`은 CSV date, `recorded_at`은 observed_at + 60초 × segment 번호다. session_day는 원본 observed_at 날짜다. 기존 GroupShuffleSplit 128후보 방식과 SEED 42를 유지한다. 모든 분할에 네 클래스를 포함하는 후보 중 WAV 수 비율과 라벨 분포를 고려해 선택한다. 같은 source_id와 session_day는 다른 분할에 들어가지 않는다. 두 모델은 품질 검사 뒤 동일한 WAV 순서와 라벨을 공유한다.

WAV 분할로 `날짜 → split` 대응표를 만든다. 원본 CSV 관측에 이 표를 적용하며 환경 행을 별도 무작위 분할하지 않는다. WAV 없는 관측도 유효한 식별자·시각·배정 날짜를 가지면 포함한다. 이번 실데이터 확인에서는 WAV 없는 53행 모두 포함됐고 제외 행은 0개다. 데이터가 바뀌면 실제 수와 제외 사유를 다시 저장한다.

관측은 `(hive_id, source_id)`로 한 번만 등록한다. 동일 키의 동일 값은 재등록해도 한 행이며, 관측 시각·분할·환경값 중 다른 값이 들어오면 오류다. WAV별 복제 목록을 환경 이력이나 환경 통계로 사용하지 않는다.

## 환경 전처리와 윈도

노트북과 앱은 각 파일 안에 동일 규칙의 환경 전처리·시퀀스 함수를 포함한다. 외부 공통 모듈 대신 정의 및 입력 배열·예측 일치 검사로 동작을 맞춘다. weather pressure=0과 비유한 값은 결측으로 처리한다. 중복 없는 학습 분할의 원본 환경 관측만으로 중앙값을 구하고, 결측 대체 후 평균·모표준편차(ddof=0)를 계산한다. 전체 결측 열은 중앙값 0, 상수 열의 표준편차는 1을 사용한다. 검증·테스트·앱에는 저장한 기준을 적용한다. JSON에 열 순서, 통계, 범주, 차원, 학습 관측 수와 해시를 저장한다.

같은 벌통·같은 분할에서 현재 source_id를 끝점으로 잡고 현재를 포함한 최대 6개 관측을 선택한다. 현재 이후 관측은 WAV 녹음 시각보다 이르더라도 제외한다. 원본 인접 관측 사이 간격이 2시간을 초과하면 이력을 끊고, 그다음 최대 6개로 자른다. 정확히 2시간은 허용한다.

`delta_hours`는 선택된 관측의 observed_at 차이다. 선택 윈도의 첫 값은 항상 0이다. 학습 관측 각각을 끝점으로 같은 윈도를 한 번씩 만들고 그 유효 delta 값을 모아 평균·표준편차를 구한다. 윈도 간 원본 관측의 재등장은 허용하지만 WAV 복제로 중복 집계하지 않는다. delta의 표준편차가 0이면 1을 사용한다.

과거 행의 시간대는 해당 observed_at, 마지막 현재 행은 대상 WAV의 recorded_at으로 계산한다. 원본 테이블은 변경하지 않는다. 변환을 끝낸 유효 관측을 과거부터 현재 순서로 배열 앞에 배치하고 뒤에 0을 채운다. 시간대 원핫 때문에 유효 행은 전체가 0인 패딩행과 구분된다. 패딩은 모든 통계에서 제외하며 `Masking(0.0)`으로 처리한다.

## 공통 CNN 사전 학습

Sequential CNN은 정규화 → Conv2D(16) → Pool → Conv2D(32) → Pool → Conv2D(64) → Pool → GlobalAveragePooling2D → Dense(64, relu, name=audio_features) → Dropout(0.3) → Dense(4, softmax) 순서다. 합성곱은 3×3/same/ReLU이고 풀링은 2×2다.

소리 정규화 평균·분산은 품질 검사 뒤 학습 WAV에서 배치로 구해 CNN 내부 Normalization에 넣는다. 최적 가중치 복원 후 audio_features까지 복제한 인코더는 정규화를 포함한다.

평균 풀링은 마지막 맵 `(8,234,64)`의 두 공간 축을 평균해 64채널 벡터를 만든다. 뒤따르는 Dense(64)의 파라미터는 기존 Flatten 구조의 7,667,776개에서 `(64+1)×64 = 4,160`개로 줄어든다. 두 모델은 이 Dense층의 64차원 출력을 사용한다.

CNN 재사용은 구조·음성 전처리·정규화·학습 WAV 순서/라벨·전체 날짜 분할·원본 WAV stamp·품질 설정의 provenance가 모두 일치할 때만 허용한다. 이전 소리 단독 모델은 재사용 후보가 아니다. `cnn_pretraining/provenance.json`과 `audio_cnn.weights.h5`를 사용한다.

공통 학습은 batch 8, 최대 30 epoch, Adam, val_loss EarlyStopping(patience=5, restore_best_weights=True)다. 클래스 가중치는 학습 WAV 라벨에서만 계산한다. 큰 memmap은 `tf.data` generator로 배치 공급하며 시퀀스 내부 순서는 유지한다.

# 1. Sequential 기반 CNN + MLP

고정 인코더를 `training=False`로 배치 실행해 `(N,64)` 특징을 메모리에 추출한다. 현재 환경 `(N,11)`과 모델 밖에서 결합한다. Sequential MLP는 Input(75) → Dense(32, relu) → Dropout(0.3) → Dense(16, relu) → Dense(4, softmax)다. MLP 학습은 CNN 가중치를 바꾸지 않는다.

예측: WAV → 로그 멜 → 저장된 인코더 → 64차원, 현재 환경 → 저장된 전처리 → 11차원, 결합 75차원 → MLP → 네 점수다.

# 2. Functional API 기반 CNN + LSTM

모델 1과 동일한 최적 CNN 가중치를 별도 인코더 객체에 복사한다. 인코더를 고정하고 `training=False`로 호출한다. 소리 입력 `(64,1876,1)`과 환경 입력 `(6,12)`를 받는다.

환경 분기는 Masking(0) → LSTM(32, return_sequences=False, stateful=False) → Dense(16, relu)다. 음향 64차원과 환경 16차원을 결합한 후 Dense(32, relu) → Dropout(0.3) → Dense(16, relu) → Dense(4, softmax)를 적용한다. 한 fit에서 LSTM과 결합층을 학습한다. 이번 구현은 공동 CNN 미세조정을 포함하지 않는다.

## 저장 계약

| 경로 | 내용 |
|---|---|
| artifacts/recording_60s/features/{train,val,test}.npy | 기존 음성 특징 경로 |
| artifacts/recording_60s/features/{train,val,test}_config.json | 해당 분할의 음성 전처리·품질 설정 |
| artifacts/recording_60s/quality_{train,val,test}.csv | 원래 WAV 목록·품질·제외 사유·파일 stamp |
| artifacts/recording_60s/date_splits.csv | session_day → split |
| artifacts/recording_60s/environment_observations.csv | 전체 유효 원본 환경 관측 |
| artifacts/recording_60s/environment_{train,val,test}.csv | 분할별 관측 |
| artifacts/recording_60s/environment_exclusions.csv, data_audit.json | 제외 사유·실측 수 |
| artifacts/recording_60s/cnn_pretraining/ | 공통 CNN 가중치·설정·학습 기록 |
| artifacts/cnn_mlp/audio_encoder.keras | 정규화 포함 고정 인코더 |
| artifacts/cnn_mlp/mlp.keras | 75차원 Sequential MLP |
| artifacts/cnn_mlp/preprocessing.json | 음성·환경 전처리, 모델 종류, 파일 참조 |
| artifacts/cnn_lstm/model.keras | 고정 CNN·LSTM·결합층 전체 |
| artifacts/cnn_lstm/preprocessing.json | 동일 전처리 계약 |
| artifacts/comparison/model_comparison.csv, paired_predictions.csv | 동일 테스트 WAV 비교 |

음성 설정 JSON의 정확한 경로는 `artifacts/recording_60s/features/{train,val,test}_config.json`이다. 환경·모델 구조 변경은 오디오 캐시를 무효화하지 않는다. 오디오 설정이나 WAV 목록/순서가 달라지면 같은 이름을 덮어쓴다. 기존 참조와 두 입력 dataset generator를 무효화한 후 매핑을 닫는다. 새 오디오 설정에 맞춰 이후 셀을 다시 실행해야 한다. 완료되지 않은 추출에는 유효한 설정 JSON을 남기지 않는다.

캐시는 품질 검사 후 통과한 WAV만 앞부분에 압축 저장한다. 재사용·재추출 모두 살아남은 `file_path`, `source_id`, `feature_index`로 메타데이터를 새로 정렬한다. 환경값을 바꿔도 오래된 환경 배열을 캐시에서 읽지 않는다.

각 모델 폴더에는 test_manifest.csv(품질 검사 뒤 테스트 WAV), date_splits.csv, environment_observations.csv, environment_{train,val,test}.csv가 함께 저장된다. preprocessing.json의 files.test_manifest와 files.test_environment는 앱이 읽는 test_manifest.csv와 environment_test.csv를 가리킨다. 테스트 WAV 원본은 프로젝트 dataset/ 아래 상대 경로로 연결한다.

각 내보내기는 새 `모델폴더/experiments/한국시간_마이크로초/`에 모델·전처리·파라미터·학습 이력·평가·예측·분할·그림을 먼저 저장한다. 이 실행의 파일만 최신 폴더에 복사한다. 학습 이력과 모델 객체가 다르면 저장을 거부한다. 비교표도 별도 experiments 폴더에 보관한다.

정확도·macro F1·각 클래스 recall·혼동행렬, 검증/테스트 예측 CSV, 학습 곡선과 설정을 저장한다. `epoch_seconds`는 모델별 fit 전체 시간/완료 epoch 수인 스칼라 하나다. 모델 1은 MLP fit, 모델 2는 결합 모델 fit이다. 공통 CNN 사전 학습 시간은 별도로 기록하며 비교 시간에서 제외한다.

전체 평가 요약의 컬럼은 accuracy, macro_precision, macro_recall, macro_f1, weighted_f1, recordings이며 `{validation,test}_summary.csv`로 저장한다. 클래스별 precision·recall·f1-score·정수 support는 별도 `{validation,test}_report.csv`로 저장한다. 정확도는 전체 표의 컬럼 하나이며 클래스별 표의 행으로 넣지 않는다. 결과표 출력 셀은 학습과 분리해 재학습 없이 실행할 수 있다.

## Streamlit

모델을 선택하고 테스트 WAV 분석 또는 WAV 업로드를 사용한다. 테스트 선택은 환경·시각·식별자를 자동 연결한다. 업로드는 현재 5개 환경값, 관측/녹음 시각, 벌통·원본 식별자를 입력한다. 선택적 이력 CSV에 hive_id, source_id, observed_at 및 환경 5개 열을 넣는다. split은 upload로 통일한다. 없는 과거 관측을 생성하지 않으며 현재 한 관측과 뒤쪽 패딩을 사용한다. 유효 관측 수를 표시한다.

재생은 관측 시각이 재생 시계 이하인 테스트 환경 관측을 등록하고 WAV 없는 관측도 포함한다. 현재 source_id 끝점 규칙으로 이후 관측을 차단한다. 환경 이력의 2시간 단절과 최근 3개 점수 평균의 300초/날짜 초기화는 별개다. 모델 변경과 재생 초기화는 관련 상태를 정리한다. 예측 캐시는 모델·인코더·전처리 파일 stamp, WAV stamp, 현재 환경·관측/녹음 시각·식별자와 환경 이력 전체를 포함한다. 화면 결과도 같은 입력 변경을 감지한다.

## 검증

`tests/test_environment.py`는 노트북 안의 함수 정의를 직접 읽어 합성 관측으로 중복·충돌, 학습 통계, 미래/다른 split/벌통 배제, 2시간 경계, window 첫 delta, 뒤쪽 패딩, 시간대 변경·원본 불변성 및 입력 키 변경을 확인한다.

`tests/check_feature_reexecution.py`는 노트북 안의 캐시 함수를 사용하여 중간 무음 파일 제외 뒤 정렬, 읽기 전용 캐시 재사용, 환경 변경 시 재사용, FFT/크기 변경 시 같은 경로 덮어쓰기, 이전 뷰/두 입력 generator 무효화, 실패 캐시 설정 제거를 확인한다.

`tests/check_bee_status.py`는 전체 메타데이터로 분할/환경 이력을 만든 후 세트·클래스별 한 WAV(12개)로 노트북 모든 셀과 공통 CNN·MLP·CNN+LSTM을 각각 1 epoch 실행한다. 저장/재로딩 예측, 앱/노트북 입력 일치, 인코더 복사·고정, 실제 Masking, 재사용·실험 분리·시간 계측 및 Streamlit 화면/재생을 검증한다. 임시 경로만 사용하고 전체 30 epoch는 실행하지 않는다. 데이터가 없거나 --synthetic을 주면 임시 합성 CSV/WAV로 실행한다. 임의의 성능 통과 기준은 없다.

구현 참고: [Keras Sequential](https://keras.io/guides/sequential_model/), [Functional API](https://keras.io/guides/functional_api/), [LSTM](https://keras.io/api/layers/recurrent_layers/lstm/), [저장·재로딩](https://keras.io/guides/serialization_and_saving/).

## 제출 구성

`dist/bee_submission.zip`에는 `bee_status.ipynb`와 `streamlit_bee_status.py` 두 파일만 포함한다. 코드 파일을 추가로 제출할 필요가 없다. 필요한 데이터는 dataset/에 배치하고 노트북 실행으로 모델·전처리·환경 목록을 만든 뒤 앱을 실행한다. 라이브러리는 requirements.txt에 안내된 일반 Python 패키지다. 개발 검증 스크립트와 문서는 제출 ZIP에서 제외한다.
