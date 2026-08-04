# KBL LSTM/GRU 승패 예측 데이터 파이프라인

KBL 공식 경기 기록을 이용해 2023-24, 2024-25 정규리그 540경기를 수집하고,
팀 관점 데이터와 LSTM/GRU용 시퀀스를 생성합니다.

## 산출물

파이프라인 실행 후 `data/processed/`에 다음 파일이 생성됩니다.

- `games_raw.csv`: 경기당 1행(총 540행)
- `team_games.csv`: 팀 관점 경기당 2행(총 1,080행)
- `team_games_chronological.csv`: 팀·경기일 순으로 정렬된 모델 준비 데이터
- `pregame_model_features.csv`: Elo·연승/연패·이동평균을 포함한 부스팅 모델용 경기당 1행 데이터
- `elo_baseline_predictions.csv`, `elo_baseline_metrics.json`: 경기 전 Elo 기준선
- `sequence_metadata_n{N}.csv`: 시퀀스 목표 경기와 학습/검증/테스트 구간
- `sequences_n{N}.npz`: 홈·원정 최근 N경기 시퀀스, 라벨, 스케일링 정보
- `sequences_carryover_n{N}.npz`: 이전 시즌 기록을 이어 쓰는 시퀀스
- `sequences_season_reset_n{N}.npz`: 시즌마다 과거 기록을 초기화하는 시퀀스
- `quality_report.json`, `quality_report.md`: 결측·중복·논리 오류 검사 결과
- `kbl_data_quality.xlsx`: 팀원이 확인하기 쉬운 품질 검증용 엑셀 파일

원본 API 응답은 `data/raw/`에 저장되므로 중간에 실행이 끊겨도 이어서 받을 수 있습니다.

## 팀 공유 및 시작 방법

저장소를 내려받은 팀원은 다음 순서로 환경을 준비할 수 있습니다.

```powershell
git clone https://github.com/ToySoldier7/kbl-win-prediction.git
cd kbl-win-prediction
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

모델 학습만 진행한다면 `data/processed/`의 CSV와 NPZ 파일을 바로 사용하면 됩니다.
원본 KBL API 응답은 GitHub Release의 `kbl_raw_data_2023-24_2024-25.zip`에서 내려받아
프로젝트의 `data/` 아래에 압축을 풀면 됩니다. 원본 캐시는 파일 수가 많아 Git 저장소에는
직접 포함하지 않습니다.

팀원이 새 실험을 추가할 때는 개인 브랜치를 만든 뒤 코드와 결과 설명을 함께 커밋하는 방식을 권장합니다.

```powershell
git switch -c feature/모델이름
```

## 통계 분석 및 시각화

540경기의 승리팀과 패배팀을 대응쌍으로 구성해 분포, 승패 차이 검정, 효과크기, 상관관계,
VIF, PCA, 팀별 이동평균을 분석합니다.

```powershell
python scripts/run_statistical_analysis.py --rolling-window 5
```

- 핵심 해석: [`analysis/statistical_analysis_report.md`](analysis/statistical_analysis_report.md)
- 통계 결과 CSV: [`analysis/results/`](analysis/results/)
- 발표용 PNG 그래프: [`analysis/figures/`](analysis/figures/)
- 분석 설계와 주의사항: [`analysis/README.md`](analysis/README.md)

승리팀과 패배팀은 같은 경기에서 나온 두 관측치이므로 독립표본 검정이 아니라 대응표본 t-검정 또는
Wilcoxon 부호순위 검정을 사용합니다. 여러 스탯을 동시에 검정할 때는 Benjamini-Hochberg FDR 보정
p-value와 효과크기를 함께 확인합니다.

## 실행

Python 3.10 이상에서:

```powershell
pip install -r requirements.txt
python src/kbl_pipeline.py --windows 3 5 10
```

캐시를 무시하고 다시 수집하려면 `--refresh`를 추가합니다.

## 데이터 누수 방지

시퀀스의 목표는 `y_home_win`이며 목표 경기의 스탯은 입력에 포함되지 않습니다.
각 샘플은 다음 구조입니다.

```text
X_home: 목표 경기 이전 홈팀 최근 N경기 (N, F)
X_away: 목표 경기 이전 원정팀 최근 N경기 (N, F)
context: 목표 경기 전 휴식일·백투백 등
y_home_win: 목표 경기 홈팀 승리=1, 패배=0
```

`sequences_n{N}.npz`에는 전체 기간으로 계산한 값이 아니라 학습 구간에서만 구한
평균·표준편차로 결측치 대체 및 표준화한 배열도 함께 저장됩니다.
기존 파일명은 호환성을 위해 `carryover` 버전과 동일하게 유지합니다.

## 주요 파생 변수

- `fg_pct = (2PM + 3PM) / (2PA + 3PA)`
- `three_pt_pct = 3PM / 3PA`
- `free_throw_pct = FTM / FTA`
- `efg_pct = (FGM + 0.5 × 3PM) / FGA`
- `ts_pct = 득점 / (2 × (FGA + 0.44 × FTA))`
- `tov_pct = TO / (FGA + 0.44 × FTA + TO)`
- `oreb_pct = ORB / (ORB + 상대 DRB)`
- `fta_rate = FTA / FGA`
- `ast_to_ratio`: 턴오버가 0이면 결측
- `ast_to_smoothed = AST / (TO + 1)`: 모델 입력용 안정화 지표
- `rebound_margin = 팀 리바운드 - 상대 리바운드`
- `*_per40`: 연장전 영향을 줄이기 위해 40분 기준으로 환산
- 경기 전 연승·연패 길이와 최근 3·5·10경기 이동평균
- 경기 전 Elo, 상대 Elo, Elo 차이, Elo 승리확률

## 시즌 경계 처리

두 시즌 사이의 로스터 변화 영향을 비교할 수 있도록 두 종류의 시퀀스를 제공합니다.

- `carryover`: 이전 시즌 후반 기록을 다음 시즌 초반 입력으로 사용하고 경계 여부를 컨텍스트에 포함
- `season_reset`: 시즌이 바뀌면 팀별 시퀀스를 초기화

Elo는 새 시즌 시작 시 리그 평균 1500 방향으로 35% 회귀합니다. 정확한 설정값과
기준선 성능은 `elo_baseline_metrics.json`에 기록됩니다.

쿼터 단위 표본 증가는 같은 경기 내 상관과 진행 중 점수 누수 때문에 적용하지 않았습니다.

## 주의

KBL 웹사이트에서 사용하는 공개 호출 구조를 기반으로 하므로 주소나 필드명이 바뀔 수 있습니다.
요청 간격, 재시도, 로컬 캐시를 적용했으며 연구 목적에 맞게 사용하세요.
