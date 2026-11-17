# 데이터셋 설명

확인일: 2026-11-03. 아래 통계는 현재 프로젝트의 로컬 CSV와 WAV 파일을 직접 집계한 값이다.

## 원본 출처

- [Smart Bee Colony Monitor: Clips of Beehive Sounds — Kaggle](https://www.kaggle.com/datasets/annajyang/beehive-sounds)
- Kaggle 계정: `annajyang`
- 프로젝트 코드가 사용하는 로컬 데이터 구조와 전처리는 아래에 정리한다.

## 로컬 파일 구조

```text
dataset/
  all_data_updated.csv
  sound_files/
    sound_files/
      <source_id>__segment0.wav
      <source_id>__segment1.wav
      ...
```

`all_data_updated.csv`는 환경·시각·벌통·라벨을 담은 1,275행, 23열의 표다. WAV는 7,100개이며 원본 녹음 1,222개와 연결된다. WAV와 연결되지 않은 CSV 원본 항목은 53개다. WAV별 CSV 연결 실패는 0개다.

## WAV 구성

| 항목 | 현재 로컬 데이터 |
|---|---|
| 파일 수 | 7,100개 |
| 파일 길이 | 전부 60.0초 |
| 샘플링 주파수 | 전부 22,050Hz |
| 채널 | 전부 모노 1채널 |
| 저장 형식 | PCM_16 5,389개, FLOAT 1,711개 |
| 모델 입력 변환 | 16,000Hz 모노 → 60초 전체 로그 멜 `(64, 1876, 1)` |

## CSV 컬럼 사전

결측 수는 CSV 원본 1,275행 기준이다.

| 컬럼 | dtype | 결측 수 | 의미·사용 |
|---|---|---:|---|
| `device` | `int64` | 0 | 기기 식별값 |
| `hive number` | `int64` | 0 | 벌통 식별값. 현재 1, 3, 4, 5 |
| `date` | `object` | 0 | 원본 환경 관측·녹음 시각 |
| `hive temp` | `float64` | 0 | 벌통 온도. 두 모델의 수치형 입력 |
| `hive humidity` | `float64` | 0 | 벌통 습도. 두 모델의 수치형 입력 |
| `hive pressure` | `float64` | 0 | 벌통 기압. 두 모델의 수치형 입력 |
| `weather temp` | `float64` | 4 | 외기 온도. 두 모델의 수치형 입력 |
| `weather humidity` | `int64` | 0 | 외기 습도 |
| `weather pressure` | `int64` | 0 | 외기 기압. 두 모델의 수치형 입력 |
| `wind speed` | `float64` | 4 | 풍속 |
| `gust speed` | `float64` | 994 | 돌풍 속도 |
| `weatherID` | `int64` | 0 | 날씨 상태 코드 |
| `cloud coverage` | `int64` | 0 | 구름량 수치 |
| `rain` | `int64` | 0 | 강수 관련 수치 |
| `lat` | `float64` | 4 | 위도 |
| `long` | `float64` | 4 | 경도 |
| `file name` | `object` | 0 | 원본 녹음 파일명. 확장자를 제거해 source_id 생성 |
| `queen presence` | `int64` | 0 | 원본의 여왕벌 존재 관련 라벨 |
| `queen acceptance` | `int64` | 0 | 원본의 여왕벌 수용 관련 라벨 |
| `frames` | `int64` | 0 | 원본의 프레임 관련 정보 |
| `target` | `int64` | 0 | 별도의 분류용 정수 라벨 |
| `time` | `float64` | 0 | 원본의 시각 관련 수치 |
| `queen status` | `int64` | 0 | 두 모델의 정답. 정수 0~3 |

정답은 `queen status`를 사용하며 `target`, `queen presence`, `queen acceptance`는 예측 입력에 포함하지 않는다. 표시명은 `queen status 0`~`queen status 3`이다.

## CSV와 WAV의 연결

```text
CSV file name: <source_id>.raw
WAV file:     <source_id>__segmentN.wav
연결 키:      source_id
```

- 하나의 원본 CSV 행에 여러 WAV가 연결된다. 해당 WAV들은 같은 원본 환경 측정값과 정답을 공유한다.
- `observed_at = CSV date`
- `recorded_at = observed_at + 60초 × N`
- 날짜 분할 키 `session_day`는 원본 관측의 시작 날짜를 사용한다.
- 음성 배열의 각 행은 해당 `file_path`, `source_id`, 정답과 함께 정렬을 유지한다.
- 환경 시계열은 CSV의 원본 관측행을 벌통별·시간순으로 정리해 만든다.
- 동일 원본의 WAV가 여러 개여도 환경 관측은 한 번만 등록한다. 관측 간격은 `observed_at` 차이로 계산하고 WAV의 시간대는 입력 생성 시 `recorded_at`을 사용한다.
- WAV 없는 53행은 날짜별 분할 대응표에 포함된 날짜이고 시각·벌통 식별자가 유효하면 환경 이력에 포함한다. 소리 학습·평가 샘플로 추가하지는 않는다. 분할 미배정 날짜의 관측은 오프라인 실험에서 제외하고 수와 사유를 기록한다.
- 날짜 분할 대응표와 분할별 환경 관측 목록을 저장해 노트북과 테스트 재생 앱이 함께 사용한다. 자세한 시퀀스·초기화 규칙은 `MODEL_DESIGN.md`를 따른다.

## 정답 분포

| queen status | CSV 행 수 | 연결된 WAV 수 |
|---|---:|---:|
| 0 | 179 | 1038 |
| 1 | 158 | 946 |
| 2 | 259 | 1553 |
| 3 | 679 | 3563 |

## 벌통과 시간 정보

CSV 시각 범위는 `2022-06-05 17:41:01` ~ `2022-07-15 15:28:21`이고, 원본 관측 날짜는 34개다.

| hive number | CSV 관측 수 | WAV 수 | 관측 간격 중앙값(분) | 최대 관측 간격(시간) |
|---|---:|---:|---:|---:|
| 1 | 518 | 2843 | 59.20 | 178.08 |
| 3 | 189 | 1130 | 59.19 | 5.03 |
| 4 | 471 | 2546 | 59.20 | 26.21 |
| 5 | 97 | 581 | 59.20 | 50.65 |

월별 WAV 라벨 분포는 다음과 같다. 이는 데이터 분포 설명이며, 월 컬럼은 모델 입력에서 제외한다.

| month | status 0 | status 1 | status 2 | status 3 |
|---|---:|---:|---:|---:|
| 6 | 1038 | 946 | 1553 | 186 |
| 7 | 0 | 0 | 0 | 3377 |

## 구현된 모델 입력

환경 수치 5개는 `hive temp`, `hive humidity`, `hive pressure`, `weather temp`, `weather pressure`다.

- 현재 WAV와 연결된 환경값에서 `weather temp` 결측은 12개 WAV에 해당한다.
- `weather pressure == 0`도 12개 WAV에 해당하며, 해당 행의 외기 온도 결측과 함께 나타난다.
- 공통 전처리는 외기 기압 0을 결측으로 처리하고, 학습 세트에서 계산한 중앙값으로 대체한다.
- `time_part = recorded_at.hour // 4`: 0~5의 6개 범주다.
- `time_part`는 입력 배열 생성 시 계산하며 원본 관측 테이블을 변경하지 않는다.
- 월·계절 범주는 모델 입력에서 제외한다. 원본 시각은 정렬·분할과 시간대 계산을 위해 보관한다.
- CNN+MLP는 현재 환경 수치 5개 + 시간대 원핫 6개 = 11차원을 음향 특징 64개와 결합해 75차원 입력을 사용한다.
- CNN+LSTM은 여기에 관측 간격 `delta_hours`를 더해 시점당 12차원, 초기 시퀀스 길이 6을 사용한다.

## 기존 음성 특징 캐시

```text
artifacts/recording_60s/features/
  train.npy
  train_config.json
  val.npy
  val_config.json
  test.npy
  test_config.json
```

`.npy`는 로그 멜 수치 배열과 shape·dtype을 저장한다. 별도 JSON은 음성 전처리 설정과 무음·수동 제외 설정을 저장한다. 설정·목록이 같으면 기존 배열을 불러오고, 달라지면 같은 파일명으로 재추출·갱신한다. 전체 로그 멜 배열은 약 3.4GB다.

배포 패키지는 원본 CSV·WAV, 추출된 `.npy`, 학습 가중치를 포함하지 않고, 위 설명과 코드만 제공한다.

## 구현 검증 집계 (2026-11-03)

전체 WAV 메타데이터로 기존 SEED 42 날짜 그룹 분할을 실행했다. WAV 없는 53행은 모두 유효한 식별자·관측 시각과 배정 날짜를 가져 환경 이력에 포함됐다. 제외된 환경 행은 0개다. 원본 환경 관측은 train 867개, val 193개, test 215개이며 총 1,275개다. 해당 행들은 음성 샘플로 추가되지 않는다.

`prepare_project()`는 실행할 때마다 `recording_60s/data_audit.json`과 `environment_exclusions.csv`에 실제 집계와 제외 사유를 기록한다. 검증에서는 전체 메타데이터로 환경 이력을 만든 후 세트별·클래스별 WAV 1개(총 12개)만 사용했다. 전체 품질 검사와 전체 모델 학습은 별도 노트북 실행 대상이다.

환경 저장 파일, 전처리, 앱 연결은 [MODEL_DESIGN.md](MODEL_DESIGN.md)를 참고한다.
