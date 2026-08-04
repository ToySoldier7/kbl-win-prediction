# KBL 데이터 품질 보고서

- 생성 시각: 2026-08-04T18:11:45.130735+09:00
- 경기 수: 540
- 팀 관점 행 수: 1080
- 연장전 경기 수: 19
- 전체 검사 통과: PASS

## 검사 결과

| 검사 | 결과 |
|---|---|
| total_games_is_540 | PASS |
| each_season_is_270 | PASS |
| each_team_has_54_games | PASS |
| team_rows_is_1080 | PASS |
| no_game_duplicates | PASS |
| no_team_game_duplicates | PASS |
| two_team_rows_per_game | PASS |
| one_winner_per_game | PASS |
| valid_shot_counts | PASS |
| scores_reconcile | PASS |
| required_values_complete | PASS |
| pregame_rows_is_540 | PASS |
| four_factors_complete | PASS |
| no_sequence_date_leakage | PASS |
| season_reset_has_no_boundary_crossing | PASS |
| scaled_sequences_have_no_missing_values | PASS |
| elo_probabilities_are_valid | PASS |

## 시즌별 경기 수

| 시즌 | 경기 수 |
|---|---:|
| 2023-2024 | 270 |
| 2024-2025 | 270 |

## 시퀀스 샘플

| 모드 | N | 전체 | 학습 | 검증 | 테스트 |
|---|---:|---:|---:|---:|---:|
| carryover | 3 | 522 | 366 | 78 | 78 |
| carryover | 5 | 512 | 356 | 78 | 78 |
| carryover | 10 | 488 | 332 | 78 | 78 |
| season_reset | 3 | 505 | 349 | 78 | 78 |
| season_reset | 5 | 485 | 329 | 78 | 78 |
| season_reset | 10 | 436 | 280 | 78 | 78 |

## Elo 테스트 기준선

- 샘플 수: 78
- 정확도: 0.6410
- Brier score: 0.2302
- Log loss: 0.6581
