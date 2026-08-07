# KBL 데이터 품질 보고서

- 생성 시각: 2026-08-07T22:05:46.035023+09:00
- 경기 수: 3183
- 팀 관점 행 수: 6366
- 연장전 경기 수: 148
- 전체 검사 통과: PASS

## 검사 결과

| 검사 | 결과 |
|---|---|
| requested_12_seasons_present | PASS |
| total_games_match_expected_3183 | PASS |
| season_game_counts_match_expected | PASS |
| team_game_counts_match_schedule | PASS |
| team_rows_twice_total_games | PASS |
| no_game_duplicates | PASS |
| no_team_game_duplicates | PASS |
| two_team_rows_per_game | PASS |
| one_winner_per_game | PASS |
| valid_shot_counts | PASS |
| scores_reconcile | PASS |
| required_values_complete | PASS |
| pregame_rows_match_total_games | PASS |
| four_factors_complete | PASS |
| no_sequence_date_leakage | PASS |
| season_reset_has_no_boundary_crossing | PASS |
| scaled_sequences_have_no_missing_values | PASS |
| elo_probabilities_are_valid | PASS |

## 시즌별 경기 수

| 시즌 | 경기 수 |
|---|---:|
| 2013-2014 | 270 |
| 2014-2015 | 270 |
| 2015-2016 | 270 |
| 2016-2017 | 270 |
| 2017-2018 | 270 |
| 2018-2019 | 270 |
| 2019-2020 | 213 |
| 2020-2021 | 270 |
| 2021-2022 | 270 |
| 2022-2023 | 270 |
| 2023-2024 | 270 |
| 2024-2025 | 270 |

## 시퀀스 샘플

| 모드 | N | 전체 | 학습 | 검증 | 테스트 |
|---|---:|---:|---:|---:|---:|
| carryover | 3 | 3168 | 2088 | 540 | 540 |
| carryover | 5 | 3156 | 2076 | 540 | 540 |
| carryover | 10 | 3132 | 2052 | 540 | 540 |
| season_reset | 3 | 2984 | 1971 | 508 | 505 |
| season_reset | 5 | 2865 | 1892 | 488 | 485 |
| season_reset | 10 | 2562 | 1691 | 435 | 436 |

## Elo 테스트 기준선

- 샘플 수: 540
- 정확도: 0.6296
- Brier score: 0.2283
- Log loss: 0.6489
