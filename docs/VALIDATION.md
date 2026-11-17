# 검증 기록

검증일: 2026-11-03 (Asia/Seoul). 두 제출 파일로 통합한 후 실데이터 기반 검사를 다시 완료했다.

검사 대상은 `bee_status.ipynb`와 `streamlit_bee_status.py`다. 별도 학습·전처리 Python 모듈과 노트북 Python 사본을 실행 경로에서 제거한 상태에서 노트북 모든 셀과 앱을 실행했다. CNN·MLP·LSTM은 노트북 셀에서 직접 구성한다. 앱 전처리·추론·화면·재생은 Streamlit 파일 안에 모두 포함된다.

## 수행 결과

| 검사 | 결과 |
|---|---|
| 합성 환경·모니터링 unittest | 11개 통과 (두 파일에 포함한 변환 정의 일치 검사 추가) |
| 캐시 회귀 검사 | 통과: 중간 제외 파일 정렬, 환경 변경 시 재사용, 동일 경로 FFT/shape 갱신, 이전 뷰/두 입력 generator 무효화, 실패 캐시 설정 제거 |
| 노트북 | 모든 코드 셀 실행, nbformat/구문 검사 및 외부 프로젝트 모듈 의존성 없음 확인 |
| 공통 CNN 사전 학습 | 실데이터 소수 WAV, 1 epoch 완료 |
| CNN+MLP | 동일 분할의 소수 WAV, MLP 1 epoch 완료 |
| CNN+LSTM | 동일 분할의 소수 WAV, 결합 모델 1 epoch 완료 |
| 입력·저장 | 저장/재로딩 예측 allclose(rtol=1e-5, atol=1e-6); WAV 로그 멜과 앱/노트북 환경 배열 정확히 일치 |
| 인코더·Masking | 별도 객체·동일 가중치·고정 확인, 실제 LSTM 패딩 무시 확인 |
| 실험 기록 | 재저장 시 새 폴더·이전 파일 불변, 그림 PNG 검증, fit 측정 범위·epoch 평균 계산 확인 |
| Streamlit AppTest | 두 모델, 테스트/업로드 분석, 환경 수정 시 기존 결과 제거·재분석, 변경 없는 재실행 결과 유지, 재생 시작/일시정지/초기화/모델 전환 통과 |
| 재생 | test 전용/전체 WAV 검증, 누락 WAV 처리, WAV 없는 관측 등록·중복 방지, 300초/녹음 날짜 점수 초기화 확인 |

## 사용한 실데이터 범위

CSV 1,275행과 WAV 7,100개의 전체 메타데이터로 날짜 분할 및 환경 이력을 구성했다. 같은 source_id 및 원본 시작 날짜는 한 분할에만 속한다. WAV 없는 CSV 53행 모두 환경 이력에 포함됐으며 환경 제외는 0행이다. 중복 없는 환경 관측은 train 867 / val 193 / test 215다. 소리·라벨 샘플로는 세트별·클래스별 1 WAV씩, train/val/test 각 4개, 총 12개를 사용했다.

1 epoch 통합 점검에서 두 모델의 테스트 정확도는 각각 0.25/0.25, macro F1은 각각 0.10/0.10였다. 테스트 4 WAV의 실행 점검 결과이며 전체 데이터의 성능이나 모델 우열을 뜻하지 않는다. MLP fit 평균 epoch 시간은 0.526초, 결합 CNN+LSTM fit은 1.632초였다. 공통 CNN 사전 학습 시간은 포함하지 않았다. 임의의 정확도 통과 기준은 사용하지 않았다.

검증용 모델·특징·결과는 `.test_cache/integration_*/artifacts` 임시 경로에서 생성하고 성공 후 제거했다. 기존 `artifacts` 아래 음성 캐시와 학습 모델은 덮어쓰지 않았다. 실행 로그와 요약은 로컬 `.test_cache/integration.log`, `.test_cache/two_file_integration_tests.log`, `.test_cache/verification_report.json`에 남긴다. 구현 변경 전 노트북·앱·설계 사본도 `.test_cache/original_*`로 보관한다.

검증 환경에서 Keras 임시 파일 경로와 Streamlit AppTest의 로컬 소켓 제한을 확인했다. 검사 스크립트는 임시 경로를 `.test_cache`로 고정하고, 화면 검사는 로컬 소켓을 허용하는 실행 환경에서 수행했다. 전체 학습 결과 경로는 변경하지 않았다.

## 미실행

전체 7,100 WAV의 새 품질 검사/특징 재구축과 전체 데이터 최대 30 epoch 학습·최종 성능 비교는 실행하지 않았다. 합성 데이터만으로 전체 통합 검사를 강제하는 --synthetic 경로는 이번 실데이터 검사에서는 실행하지 않았다. 합성 관측·임시 WAV 경계조건 검사는 별도로 통과했다. 앱에서 오디오 재생 브라우저의 실제 소리 출력은 청취 검증하지 않았다.

## 재현

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_*.py' -v
.\.venv\Scripts\python.exe tests/check_feature_reexecution.py
.\.venv\Scripts\python.exe tests/check_bee_status.py
```

전체 학습은 `.venv` 커널로 bee_status.ipynb를 위에서 아래로 실행한다. 새 두 모델 내보내기와 재로딩 검증 후 `.venv\Scripts\python.exe -m streamlit run streamlit_bee_status.py`를 실행한다.

## 두 파일 제출 구성 검증

- 루트 Python 소스는 streamlit_bee_status.py 하나다. 모델 구현은 bee_status.ipynb에 들어 있다.
- 코드가 별도 프로젝트 모듈을 import하거나 Python 보조 파일을 생성하지 않는다.
- notebook의 정의 셀을 직접 불러 캐시·환경 검사를 수행했다. 개발용 tests/는 제출 ZIP에서 제외한다.
- 노트북과 앱에 각각 포함한 변환 함수 19개의 AST 정의가 일치하고, 저장된 CSV/JSON을 통해 만든 입력 배열 및 두 모델 예측이 일치한다.
- dist/bee_submission.zip은 bee_status.ipynb, streamlit_bee_status.py 두 항목만 포함하며 파일 크기와 SHA256을 검증했다. docs/PACKAGE_MANIFEST.json은 ZIP 밖의 로컬 검증 자료다.
- 이전 보조 모듈·소스 사본·패키지는 .test_cache/before_two_file_refactor/에 보관했다. 원본 데이터, 기존 음성 캐시와 전체 학습 결과는 수정하지 않았다.

## 2026-11-04 평가표 표시 수정 검증

`classification_report`의 전체 accuracy 스칼라를 클래스별 딕셔너리와 함께 DataFrame으로 변환하면 accuracy가 행으로 들어가고 precision·recall·f1-score·support에 같은 값이 반복됐다. 전체 요약 표와 클래스별 표를 분리했다. 전체 요약은 accuracy 등 지표가 컬럼이며, 클래스별 support는 정수다. 두 모델의 결과표 셀은 학습과 분리했다.

현재 저장된 CNN+MLP·CNN+LSTM의 검증/테스트 예측으로 네 표를 확인했다. 검증은 각각 1,144 WAV, 테스트는 각각 1,090 WAV이며, 요약 accuracy와 macro F1이 기존 metrics.json 및 예측에서 계산한 값과 일치했다. CSV 저장·재로딩 후 컬럼과 support 정수형도 확인했다. 기존 노트북 출력과 최신 `{validation,test}_report.csv`를 수정하고 `{validation,test}_summary.csv`를 추가했다. 학습은 실행하지 않았으며 모델 구조·학습 설정·저장된 가중치·이전 실험 보관물은 변경하지 않았다.

수정 전 파일은 `.test_cache/before_evaluation_table_fix/`에 보관했다.
