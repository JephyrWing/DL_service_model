# 제출용 벌통 상태 분류 실행 안내

현재 프로젝트 설명과 결과 해석은 [프로젝트 안내](PROJECT_CONTEXT.txt)에 정리했습니다. 패키지 ZIP 루트의 `PACKAGE_MANIFEST.json`에 포함 파일, 크기와 SHA-256을 기록합니다.

1. 프로젝트 루트의 `dataset/`에 원본 CSV와 WAV를 둡니다.
2. `.venv` 커널에서 `bee_status.ipynb`를 열고 위에서부터 실행합니다. 이 노트북은 실험 B의 원본 단위 층화 분할, GAP CNN 학습, 고정된 CNN 특징과 현재 환경·시간대 특징을 사용하는 MLP 학습을 수행합니다. 고정 배정 CSV가 있으면 우선 사용하고, 없으면 같은 원본 목록에서 seed 42로 배정을 재현합니다.
3. 실행 결과는 `artifacts/main/<실행 ID>/`에 저장됩니다. 모델을 저장한 실행은 첫 셀의 `EVALUATE_RUN_DIR`에 해당 실행 폴더를 지정해 다시 평가할 수 있습니다.
4. `python -m streamlit run streamlit_bee_status.py`로 앱을 실행합니다. 앱은 `artifacts/main/`의 최근 완료된 CNN–MLP 실행을 자동으로 불러옵니다. 새 실행이 없으면 `artifacts/submission/ex_b/`의 최근 완료된 실행을 사용합니다.

## 장치 입력 JSON

앱의 **장치 JSON 형식과 예제**에서 틀을 내려받을 수 있습니다. 관측 정보는 **장치 입력 JSON**에, 60초 이하 녹음은 **녹음 WAV**에 올립니다. `sample_01.json`과 `sample_01.wav`처럼 확장자 앞 이름을 맞춰 두 파일을 함께 제출합니다. 두 칸 모두 여러 파일을 추가할 수 있습니다. **입력 완료**를 누르면 전체 결과와 파일별 상세 분석을 확인할 수 있습니다.

- `hive_id`와 `source_id`는 각각 벌통과 환경 관측의 식별자입니다. 같은 벌통의 새 환경 관측마다 새 `source_id`를 사용합니다.
- `observed_at`과 `recorded_at`은 환경 관측과 WAV 녹음 시각입니다. 시간대 표기 없는 현지 시각을 ISO 8601 형식으로 입력합니다. 예를 들어 `2026-11-05T10:00:00`입니다. 녹음 시각은 관측 시각과 같거나 이후여야 합니다.
- `environment`에 `hive_temp`, `hive_humidity`, `hive_pressure`, `weather_temp`, `weather_pressure`를 숫자로 넣습니다. 측정값이 없으면 `null`을 넣습니다.

Autotune은 현재 기본값이 켜져 있습니다. CNN과 MLP에서 각각 후보 네 개를 검증 손실로 비교합니다. 새 로그멜 특징은 `artifacts/main/feature_cache/`에 저장해 후속 실행에서도 사용합니다.

개발 검사는 `python -m unittest discover -s tests -v`, `python tests/check_feature_reexecution.py`, `python tests/check_bee_status.py`로 실행합니다. 기존 [설계서](MODEL_DESIGN.md)와 [검증 기록](VALIDATION.md)은 실험 B를 적용하기 전 두 모델 구성의 기록입니다.
