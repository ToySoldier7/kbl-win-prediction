# KBL 통계 분석 및 시각화

`scripts/run_statistical_analysis.py`를 실행하면 KBL 2023-24, 2024-25 정규리그 팀 경기 데이터를 이용해
분포, 승패 차이, 효과크기, 상관관계, VIF, PCA, 팀별 이동평균 분석을 생성합니다.

## 실행

```powershell
python scripts/run_statistical_analysis.py --rolling-window 5
```

## 분석 원칙

- 승리팀과 패배팀은 같은 경기에서 나온 쌍이므로 **대응표본 분석**을 사용합니다.
- 경기별 `승리팀 값 - 패배팀 값` 차이의 정규성을 Shapiro-Wilk와 Q-Q plot으로 확인합니다.
- 차이가 정규적이면 대응표본 t-검정과 Cohen's dz, 그렇지 않으면 Wilcoxon 부호순위 검정과
  matched-pairs rank-biserial correlation을 사용합니다.
- 여러 스탯을 동시에 검정하므로 원 p-value와 Benjamini-Hochberg FDR 보정 p-value를 모두 제공합니다.
- VIF는 진단 도구로 사용하며, 값이 높다는 이유만으로 변수를 자동 삭제하지 않습니다.
- PCA는 승패 예측용 변수가 아니라 경기 스타일과 스탯 구조를 설명하기 위한 탐색적 분석입니다.

생성된 표는 `analysis/results/`, 그림은 `analysis/figures/`, 핵심 해석은
`analysis/statistical_analysis_report.md`에 저장됩니다.
