# 벌통 상태 분류: CNN+MLP / CNN+LSTM

> 아래 내용은 실험 B를 적용하기 전 두 모델 구성의 기록입니다. 현재 제출 파일과 실행 방법은 [제출용 실행 안내](START_HERE.md)에 있습니다.

60초 WAV 전체의 로그 멜과 환경 관측을 결합해 `queen status 0`~`3`을 분류합니다. 공통 CNN을 먼저 학습하고, 현재 환경을 사용하는 Sequential MLP와 환경 이력을 사용하는 Functional CNN+LSTM을 같은 테스트 WAV로 비교합니다. 전체 30에포크 학습은 노트북에서 실행합니다.

**제출 코드 파일은 `bee_status.ipynb`와 `streamlit_bee_status.py` 두 개뿐입니다.** 데이터 준비·전처리·모델 구조·학습·평가·저장 코드는 노트북에, 앱의 전처리·모델 로딩·추론·화면·재생 코드는 Streamlit 파일에 전부 들어 있습니다. 두 파일은 별도 프로젝트 Python 모듈을 import하거나 생성하지 않습니다. 데이터와 노트북이 생성한 모델·JSON·CSV는 실행 입출력입니다.

## 실행 순서

PowerShell, Python 3.12, 프로젝트 루트 폴더에서 실행합니다. 아래 실행 명령과 데이터·코드 경로는 모두 프로젝트 루트 기준입니다. 안내 문서는 `docs/`에 모아 두었습니다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

1. `dataset/all_data_updated.csv`와 `dataset/sound_files/`의 WAV를 확인합니다.
2. `bee_status.ipynb`를 `.venv` 커널로 열고 커널 재시작 후 위에서 아래로 실행합니다.
3. 공통 준비 → CNN 사전 학습 → `# 1. Sequential 기반 CNN + MLP` → `# 2. Functional API 기반 CNN + LSTM` → 비교/저장·재로딩 검증까지 완료합니다.
4. 앱을 실행하고 모델을 선택합니다.

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_bee_status.py
```

앱의 모델 저장 루트 기본값은 `artifacts`입니다. 하위 `cnn_mlp`와 `cnn_lstm`을 선택해 불러옵니다. 이전 `recording_60s/bee_status.keras`는 새 모델로 재사용하지 않습니다.

## 입력과 학습

- 소리: 기존 16kHz 모노·로그 멜 `(64,1876,1)`, WAV 60초 전체 하나가 한 샘플.
- CNN: 학습 음성 정규화 → Conv 16/32/64와 Pool → GlobalAveragePooling2D → Dense(64, audio_features). 사전 학습에 Dropout(0.3)·Dense(4)를 사용합니다.
- 현재 환경: hive temp, hive humidity, hive pressure, weather temp, weather pressure + time_part 0~5 원핫 = 11차원.
- CNN+MLP: 고정 CNN 64차원 + 현재 11차원 → 75차원 Sequential MLP.
- CNN+LSTM: 별도 고정 CNN + 환경 `(6,12)` → Masking/LSTM32/Dense16 → 결합 분류층.
- 월·계절·다른 라벨·식별자는 예측 특징에서 제외합니다. 정수 정답과 sparse_categorical_crossentropy를 사용합니다.
- batch 8, 최대 30 epoch, Adam, val_loss EarlyStopping patience 5/최적 가중치 복원, 학습 라벨에서만 클래스 가중치를 계산합니다.

환경 통계는 중복 없는 학습 원본 관측에서 계산합니다. weather pressure=0은 결측으로 처리합니다. 원본 시작 날짜로 SEED 42 그룹 분할하고 두 모델이 같은 WAV 분할을 사용합니다. 환경 이력은 같은 벌통·분할의 현재 source_id와 이전 최대 5개 관측이며 2시간 초과 공백에서 끊습니다. 부족한 이력은 뒤쪽 0 패딩입니다. 자세한 delta 통계·시간대·분할 규칙은 [MODEL_DESIGN.md](MODEL_DESIGN.md)에 있습니다.

## 캐시와 실험 저장

`artifacts/recording_60s/features/train.npy`, `val.npy`, `test.npy`를 유지합니다. 소리 설정·품질 기준·WAV 목록과 순서가 같으면 읽기 전용 memmap을 재사용합니다. 환경이나 모델 변경은 재추출하지 않습니다. 음성 설정이 바뀌면 이전 뷰·배치 generator를 무효화하고 같은 파일을 갱신합니다. 특징 셀 재실행 후에는 환경 배열·정규화·데이터셋·모델 셀부터 다시 실행합니다. 품질 보고서/목록이 없어 재사용을 검증할 수 없는 기존 캐시는 재구축합니다. 큰 음성 배열은 통째로 복사하지 않고 배치로 읽습니다.

최종 파일은 다음과 같습니다.

```text
artifacts/
  recording_60s/       # 오디오 캐시, 전체 분할/환경 목록, 데이터 감사, 공통 CNN
  cnn_mlp/
    audio_encoder.keras
    mlp.keras
    preprocessing.json
  cnn_lstm/
    model.keras
    preprocessing.json
  comparison/
    model_comparison.csv
    paired_predictions.csv
```

각 모델 폴더는 `test_manifest.csv`, `environment_test.csv`, 전체/분할별 환경 목록과 날짜 대응표를 함께 저장합니다. preprocessing.json의 files 항목이 앱 참조를 명시합니다. test_manifest에는 품질 검사 후 살아남은 테스트 WAV만 들어갑니다.

평가는 전체 요약 표와 클래스별 표로 출력합니다. 전체 표는 `accuracy`, `macro_precision`, `macro_recall`, `macro_f1`, `weighted_f1`, `recordings`를 컬럼으로 표시하고 `validation_summary.csv`와 `test_summary.csv`에 저장합니다. 클래스별 `precision`, `recall`, `f1-score`, 정수형 `support`는 `validation_report.csv`와 `test_report.csv`에 저장합니다. 노트북의 결과표 셀은 학습 셀과 분리되어 있어 재학습 없이 표시를 다시 확인할 수 있습니다.

공통 CNN, CNN+MLP, CNN+LSTM의 학습 셀 뒤에 train과 validation의 loss·accuracy 곡선을 표시합니다. 점선은 validation loss가 가장 낮았던 에포크입니다. 마지막 셀에서는 같은 테스트 WAV의 accuracy·macro F1과 클래스별 recall을 비교합니다. 비교 그래프는 `artifacts/comparison/model_comparison.png`에 저장합니다.

모델별 `experiments/실행시각/`에 모델, 실제 파라미터·구조, 전처리, 학습 이력, 정확도·macro F1·클래스 recall·혼동행렬, 검증/테스트 예측과 그림을 보관합니다. `EXPERIMENT_NOTE`에 설명을 추가할 수 있습니다. `epoch_seconds`는 최종 모델의 fit 전체 시간/실제 완료 epoch 수인 평균 하나입니다. MLP fit과 결합 CNN+LSTM fit을 각각 재며 공통 CNN 사전 학습은 제외합니다. 이전 실험은 보존합니다.

## 앱 사용

테스트 WAV를 선택하면 연결된 환경·시각·식별자를 자동 사용합니다. 업로드 WAV에는 현재 환경값 5개와 환경 관측/녹음 시각, 벌통·원본 식별자를 입력합니다. 모르는 환경값은 빈칸으로 두면 저장된 학습 중앙값으로 대체합니다. LSTM 과거 이력은 별도 CSV로 업로드할 수 있습니다.

```text
hive_id,source_id,observed_at,hive temp,hive humidity,hive pressure,weather temp,weather pressure
1,previous-source,2026-11-03 10:00:00,32,50,1008,24,1010
```

과거 이력이 없으면 현재 관측 하나만 사용하며 유효 관측 수를 표시합니다. 동일 관측을 중복 등록하지 않으며 다른 값으로 재등록하면 오류를 표시합니다. 테스트 재생은 WAV 없는 환경 관측도 재생 시각에 맞춰 반영합니다. 환경 이력 2시간 단절과 점수 평균의 300초/날짜 초기화를 구분합니다. 모델 변경과 처음부터 버튼은 관련 상태를 정리합니다. 환경·시각·이력을 바꾸면 이전 화면 결과가 사라지고 새 분석을 할 수 있습니다.

## 검증

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_*.py' -v
.\.venv\Scripts\python.exe tests/check_feature_reexecution.py
.\.venv\Scripts\python.exe tests/check_bee_status.py
```

통합 검사는 전체 메타데이터로 날짜 분할과 원본 환경 이력을 만든 뒤 세트별·클래스별 한 WAV씩 총 12개로 세 학습을 각각 1 epoch 실행합니다. 노트북 모든 셀과 두 파일의 독립 실행, 입력 정렬, 저장·재로딩 일치, 인코더 복사·고정, 실제 Masking, 실험 보관, 앱 분석·재생·모델 전환을 확인합니다. 결과는 `.test_cache/`의 임시 폴더에만 저장하고 기존 학습 결과를 덮어쓰지 않습니다. `--notebook-only`는 UI 검사를 생략하고 `--synthetic`은 합성 데이터로 강제 실행합니다. 데이터가 없으면 자동으로 합성 CSV/임시 WAV를 생성합니다.

실측 데이터: CSV 1,275행, WAV 7,100개. WAV 없는 53행 모두 환경 이력에 포함됐고 제외 환경 행은 0개입니다. 환경 분할은 train 867/val 193/test 215 관측입니다. 전체 WAV 품질 검사와 전체 30 epoch 성능 비교는 아직 실행하지 않았습니다. 1 epoch 검사는 실행 경로 점검이며 성능 결론을 제공하지 않습니다.

## 파일 안내와 제출

| 제출 코드 | 포함된 구현 |
|---|---|
| `bee_status.ipynb` | 원본 CSV/WAV 연결, 분할, 음성·환경 전처리, 캐시·배치, CNN·MLP·LSTM 구조 직접 선언, 학습, 평가, 실험 기록, 모델 내보내기·재로딩 검증 |
| `streamlit_bee_status.py` | 독립 음성·환경 전처리, 저장 모델 로딩·예측, 업로드·테스트 파일 분석, 환경 이력, 모니터링 재생·상태 관리 |

노트북이 학습 코드의 원본입니다. 추가 Python 소스나 동기화 스크립트는 필요하지 않습니다. 두 파일에 필요한 동일 변환 함수를 각각 포함하며, 개발 검사는 함수 정의 및 같은 WAV의 입력·예측 일치를 확인합니다. 마지막 앱 비교 셀은 같은 폴더에 앱 파일이 있을 때 추가 검증하며 노트북의 독립 학습을 방해하지 않습니다.

`dist/bee_submission.zip`에는 위 두 파일만 루트 경로에 포함합니다. [패키지 명세](PACKAGE_MANIFEST.json)는 `docs/PACKAGE_MANIFEST.json`에서 두 파일의 크기와 SHA256을 기록합니다. `tests/`와 `docs/`는 개발·검증 참고자료이며 제출 ZIP에 포함하지 않습니다. 기존 보조 모듈과 Python 사본은 `.test_cache/before_two_file_refactor/`에 백업한 뒤 실행 경로에서 제거했습니다.
