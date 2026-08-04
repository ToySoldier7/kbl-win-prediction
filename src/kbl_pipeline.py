from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


API_BASE = "https://api.kbl.or.kr"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ELO_INITIAL = 1500.0
ELO_K_FACTOR = 20.0
ELO_HOME_ADVANTAGE = 65.0
ELO_OFFSEASON_RETENTION = 0.65

SEASONS = (
    {"name": "2023-2024", "slug": "2023_2024", "from": "20231001", "to": "20240630"},
    {"name": "2024-2025", "slug": "2024_2025", "from": "20241001", "to": "20250630"},
)

HEADERS = {
    "Channel": "WEB",
    "TeamCode": "XX",
    "X-Requested-With": "XMLHttpRequest",
    "lang": "ko",
    "User-Agent": "KBL-University-Research/1.0",
    "Accept": "application/json",
}

STAT_MAP = {
    "points": "score",
    "two_pt_made": "fg",
    "two_pt_attempted": "fgA",
    "three_pt_made": "threep",
    "three_pt_attempted": "threepA",
    "free_throw_made": "ft",
    "free_throw_attempted": "ftA",
    "off_rebounds": "offr",
    "def_rebounds": "defr",
    "total_rebounds": "rb",
    "team_rebounds": "teamR",
    "assists": "ast",
    "steals": "stl",
    "blocks": "bs",
    "turnovers": "to",
    "fouls": "foul",
    "possessions": "poss",
    "pace": "pace",
    "offensive_rating": "offrtg",
    "defensive_rating": "defrtg",
    "net_rating": "netrtg",
}

COUNT_STATS = (
    "points",
    "two_pt_made",
    "two_pt_attempted",
    "three_pt_made",
    "three_pt_attempted",
    "free_throw_made",
    "free_throw_attempted",
    "off_rebounds",
    "def_rebounds",
    "total_rebounds",
    "assists",
    "steals",
    "blocks",
    "turnovers",
    "fouls",
)

SEQUENCE_FEATURES = (
    "is_home",
    "win",
    "points_per40",
    "opp_points_per40",
    "score_margin_per40",
    "fg_pct",
    "two_pt_pct",
    "three_pt_pct",
    "free_throw_pct",
    "efg_pct",
    "ts_pct",
    "tov_pct",
    "fta_rate",
    "off_rebounds_per40",
    "def_rebounds_per40",
    "total_rebounds_per40",
    "rebound_margin_per40",
    "assists_per40",
    "steals_per40",
    "blocks_per40",
    "turnovers_per40",
    "fouls_per40",
    "ast_to_smoothed",
    "oreb_pct",
    "dreb_pct",
    "possessions",
    "pace",
    "offensive_rating",
    "defensive_rating",
    "net_rating",
    "rest_days_capped14",
    "back_to_back",
    "overtime_periods",
)

CONTEXT_FEATURES = (
    "home_rest_days_capped14",
    "away_rest_days_capped14",
    "home_back_to_back",
    "away_back_to_back",
    "home_season_game_number",
    "away_season_game_number",
    "home_win_streak_before",
    "away_win_streak_before",
    "home_loss_streak_before",
    "away_loss_streak_before",
    "home_elo_before",
    "away_elo_before",
    "elo_difference",
    "elo_home_win_probability",
    "home_season_opener",
    "away_season_opener",
    "home_long_break",
    "away_long_break",
    "sequence_crosses_season_boundary",
)

ROLLING_WINDOWS = (3, 5, 10)
ROLLING_BASE_FEATURES = (
    "win",
    "score_margin_per40",
    "efg_pct",
    "tov_pct",
    "oreb_pct",
    "fta_rate",
    "offensive_rating",
    "defensive_rating",
)


@dataclass
class RateLimiter:
    interval: float = 0.06

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self._next_allowed - now
            if delay > 0:
                time.sleep(delay)
            self._next_allowed = time.monotonic() + self.interval


RATE_LIMITER = RateLimiter()


def fetch_json(path: str, params: dict[str, Any] | None = None, retries: int = 5) -> Any:
    url = f"{API_BASE}{path}"
    if params:
        url = f"{url}?{urlencode(params, doseq=True)}"
    last_error: Exception | None = None
    for attempt in range(retries):
        RATE_LIMITER.wait()
        request = Request(url, headers=HEADERS, method="GET")
        try:
            with urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            last_error = exc
            if exc.code not in (429, 500, 502, 503, 504):
                raise
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
        time.sleep(min(2**attempt, 12))
    raise RuntimeError(f"KBL API 요청 실패: {url}") from last_error


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp_path, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def as_list(payload: Any) -> list[dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "value" in payload and isinstance(payload["value"], list):
        return payload["value"]
    raise TypeError(f"예상하지 못한 API 응답 형태: {type(payload)!r}")


def get_schedule(season: dict[str, str], refresh: bool) -> list[dict[str, Any]]:
    cache_path = RAW_DIR / "schedules" / f"{season['slug']}.json"
    if cache_path.exists() and not refresh:
        games = as_list(read_json(cache_path))
    else:
        games = as_list(
            fetch_json(
                "/match/list",
                {
                    "fromDate": season["from"],
                    "toDate": season["to"],
                    "tcodeList": "all",
                    "seasonCategory": "R",
                    "seasonGrade": 1,
                },
            )
        )
        write_json(cache_path, games)

    filtered = [
        game
        for game in games
        if game.get("seasonName1", game.get("seasonName")) == season["name"]
        and game.get("seasonCategory") == "R"
        and int(game.get("seasonGrade", 1)) == 1
        and int(game.get("isEnded", 0)) == 1
    ]
    filtered.sort(key=lambda game: (game["gameDate"], game.get("gameStart", ""), game["gmkey"]))
    return filtered


def get_game_payload(game: dict[str, Any], season: dict[str, str], refresh: bool) -> dict[str, Any]:
    cache_path = RAW_DIR / "matches" / season["slug"] / f"{game['gmkey']}.json"
    if cache_path.exists() and not refresh:
        return read_json(cache_path)

    payload = {
        "schedule": game,
        "match_info": fetch_json(f"/match/{game['gmkey']}"),
        "team_records": as_list(fetch_json(f"/match/{game['gmkey']}/team-record")),
        "sources": {
            "match_info": f"{API_BASE}/match/{game['gmkey']}",
            "team_records": f"{API_BASE}/match/{game['gmkey']}/team-record",
        },
    }
    write_json(cache_path, payload)
    return payload


def crawl_all(refresh: bool, max_workers: int) -> list[dict[str, Any]]:
    all_payloads: list[dict[str, Any]] = []
    for season in SEASONS:
        games = get_schedule(season, refresh)
        if len(games) != 270:
            raise ValueError(f"{season['name']} 정규리그 경기 수가 270이 아닙니다: {len(games)}")

        print(f"[{season['name']}] {len(games)}경기 상세 기록 수집")
        payload_by_key: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(get_game_payload, game, season, refresh): game["gmkey"]
                for game in games
            }
            completed = 0
            for future in as_completed(futures):
                gmkey = futures[future]
                payload_by_key[gmkey] = future.result()
                completed += 1
                if completed % 25 == 0 or completed == len(games):
                    print(f"  {completed}/{len(games)} 완료")
        all_payloads.extend(payload_by_key[game["gmkey"]] for game in games)
    return all_payloads


def safe_number(value: Any) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def safe_ratio(numerator: Any, denominator: Any) -> float:
    num = safe_number(numerator)
    den = safe_number(denominator)
    if math.isnan(num) or math.isnan(den) or den == 0:
        return math.nan
    return num / den


def overtime_periods(match_info: dict[str, Any], records: dict[str, dict[str, Any]]) -> int:
    teamrecords = match_info.get("teamrecords", {})
    quarter_counts = []
    for side in ("home", "away"):
        eq = teamrecords.get(side, {}).get("scoreeq", []) or []
        quarter_counts.append(len(eq))
    if max(quarter_counts, default=0) > 0:
        return max(quarter_counts)

    play_minutes = [safe_number(record.get("playMin")) for record in records.values()]
    estimates = [max(0, round((minutes - 200) / 25)) for minutes in play_minutes if not math.isnan(minutes)]
    return max(estimates, default=0)


def add_pregame_form_features(team_df: pd.DataFrame) -> pd.DataFrame:
    """과거 경기만 사용해 연승/연패와 이동평균을 만든다."""
    result = team_df.copy()
    result["win_streak_before"] = 0
    result["loss_streak_before"] = 0

    for _, group in result.groupby("team_code", sort=False):
        win_streak = 0
        loss_streak = 0
        for index in group.index:
            result.at[index, "win_streak_before"] = win_streak
            result.at[index, "loss_streak_before"] = loss_streak
            if int(result.at[index, "win"]) == 1:
                win_streak += 1
                loss_streak = 0
            else:
                loss_streak += 1
                win_streak = 0

    grouped = result.groupby("team_code", sort=False)
    for window in ROLLING_WINDOWS:
        for feature in ROLLING_BASE_FEATURES:
            result[f"rolling_{feature}_{window}"] = grouped[feature].transform(
                lambda values, n=window: values.shift(1).rolling(n, min_periods=1).mean()
            )
    return result


def add_elo_features(games_df: pd.DataFrame, team_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """경기 전 Elo를 저장하고 경기 종료 후에만 평점을 갱신한다."""
    games = games_df.sort_values(["game_date", "game_start", "game_id"]).copy()
    ratings: dict[str, float] = {}
    previous_season: str | None = None
    elo_rows: list[dict[str, Any]] = []

    for game in games.itertuples(index=False):
        if previous_season is not None and game.season != previous_season:
            ratings = {
                code: ELO_INITIAL + ELO_OFFSEASON_RETENTION * (rating - ELO_INITIAL)
                for code, rating in ratings.items()
            }
        previous_season = game.season

        home_code = str(game.home_team_code)
        away_code = str(game.away_team_code)
        home_before = ratings.get(home_code, ELO_INITIAL)
        away_before = ratings.get(away_code, ELO_INITIAL)
        expected_home = 1.0 / (
            1.0 + 10.0 ** ((away_before - (home_before + ELO_HOME_ADVANTAGE)) / 400.0)
        )
        actual_home = float(game.home_score > game.away_score)
        change = ELO_K_FACTOR * (actual_home - expected_home)
        home_after = home_before + change
        away_after = away_before - change
        ratings[home_code] = home_after
        ratings[away_code] = away_after
        elo_rows.append(
            {
                "game_id": game.game_id,
                "home_elo_before": home_before,
                "away_elo_before": away_before,
                "elo_difference": home_before - away_before,
                "elo_home_win_probability": expected_home,
                "home_elo_after": home_after,
                "away_elo_after": away_after,
            }
        )

    elo_df = pd.DataFrame(elo_rows)
    games = games.merge(elo_df, on="game_id", how="left")
    team = team_df.copy()
    elo_by_game = elo_df.set_index("game_id")

    def team_elo(row: pd.Series) -> pd.Series:
        elo = elo_by_game.loc[row["game_id"]]
        if int(row["is_home"]) == 1:
            return pd.Series(
                [
                    elo["home_elo_before"],
                    elo["away_elo_before"],
                    elo["elo_difference"],
                    elo["elo_home_win_probability"],
                ]
            )
        return pd.Series(
            [
                elo["away_elo_before"],
                elo["home_elo_before"],
                -elo["elo_difference"],
                1.0 - elo["elo_home_win_probability"],
            ]
        )

    team[["elo_before", "opp_elo_before", "elo_difference", "elo_win_probability"]] = team.apply(
        team_elo, axis=1
    )
    return games.reset_index(drop=True), team


def extract_datasets(payloads: Iterable[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    game_rows: list[dict[str, Any]] = []
    team_rows: list[dict[str, Any]] = []

    for payload in payloads:
        schedule = payload["schedule"]
        info = payload["match_info"]
        game = info.get("game", schedule)
        records_by_team = {row["tcode"]: row.get("records", {}) for row in payload["team_records"]}
        home_code = str(game["tcodeH"])
        away_code = str(game["tcodeA"])
        if home_code not in records_by_team or away_code not in records_by_team:
            raise ValueError(f"{game['gmkey']}: 홈/원정 팀 기록이 모두 존재하지 않습니다")

        ot_periods = overtime_periods(info, records_by_team)
        game_minutes = 40 + 5 * ot_periods
        q_home = info.get("teamrecords", {}).get("home", {})
        q_away = info.get("teamrecords", {}).get("away", {})
        season_name = schedule.get("seasonName1", game.get("seasonName"))

        game_row = {
            "game_id": game["gmkey"],
            "season": season_name,
            "season_code": game.get("seasonCode"),
            "game_number": game.get("gameNo"),
            "game_date": datetime.strptime(game["gameDate"], "%Y%m%d").date().isoformat(),
            "game_start": game.get("gameStart"),
            "home_team_code": home_code,
            "home_team": game.get("tnameH"),
            "away_team_code": away_code,
            "away_team": game.get("tnameA"),
            "home_score": int(schedule.get("scoreH", records_by_team[home_code].get("score"))),
            "away_score": int(schedule.get("scoreA", records_by_team[away_code].get("score"))),
            "home_q1": q_home.get("scoreq1"),
            "home_q2": q_home.get("scoreq2"),
            "home_q3": q_home.get("scoreq3"),
            "home_q4": q_home.get("scoreq4"),
            "away_q1": q_away.get("scoreq1"),
            "away_q2": q_away.get("scoreq2"),
            "away_q3": q_away.get("scoreq3"),
            "away_q4": q_away.get("scoreq4"),
            "home_ot_scores": "|".join(map(str, q_home.get("scoreeq", []) or [])),
            "away_ot_scores": "|".join(map(str, q_away.get("scoreeq", []) or [])),
            "overtime": int(ot_periods > 0),
            "overtime_periods": ot_periods,
            "game_minutes": game_minutes,
            "stadium": game.get("stadiumnameF") or game.get("stadiumname"),
            "source_match_url": payload["sources"]["match_info"],
            "source_team_record_url": payload["sources"]["team_records"],
        }
        game_rows.append(game_row)

        sides = (
            (home_code, away_code, game.get("tnameH"), game.get("tnameA"), 1),
            (away_code, home_code, game.get("tnameA"), game.get("tnameH"), 0),
        )
        for team_code, opponent_code, team_name, opponent_name, is_home in sides:
            own = records_by_team[team_code]
            opp = records_by_team[opponent_code]
            row: dict[str, Any] = {
                "game_id": game["gmkey"],
                "season": season_name,
                "game_date": game_row["game_date"],
                "game_start": game.get("gameStart"),
                "game_number": game.get("gameNo"),
                "team_code": team_code,
                "team": team_name,
                "opponent_code": opponent_code,
                "opponent": opponent_name,
                "is_home": is_home,
                "overtime": int(ot_periods > 0),
                "overtime_periods": ot_periods,
                "game_minutes": game_minutes,
                "source_team_record_url": payload["sources"]["team_records"],
            }
            for output_name, api_name in STAT_MAP.items():
                row[output_name] = safe_number(own.get(api_name))
                row[f"opp_{output_name}"] = safe_number(opp.get(api_name))

            row["field_goals_made"] = row["two_pt_made"] + row["three_pt_made"]
            row["field_goals_attempted"] = row["two_pt_attempted"] + row["three_pt_attempted"]
            row["opp_field_goals_made"] = row["opp_two_pt_made"] + row["opp_three_pt_made"]
            row["opp_field_goals_attempted"] = row["opp_two_pt_attempted"] + row["opp_three_pt_attempted"]
            row["win"] = int(row["points"] > row["opp_points"])
            row["score_margin"] = row["points"] - row["opp_points"]
            row["rebound_margin"] = row["total_rebounds"] - row["opp_total_rebounds"]
            row["fg_pct"] = safe_ratio(row["field_goals_made"], row["field_goals_attempted"])
            row["two_pt_pct"] = safe_ratio(row["two_pt_made"], row["two_pt_attempted"])
            row["three_pt_pct"] = safe_ratio(row["three_pt_made"], row["three_pt_attempted"])
            row["free_throw_pct"] = safe_ratio(row["free_throw_made"], row["free_throw_attempted"])
            row["ast_to_ratio"] = safe_ratio(row["assists"], row["turnovers"])
            row["ast_to_smoothed"] = row["assists"] / (row["turnovers"] + 1.0)
            row["efg_pct"] = safe_ratio(
                row["field_goals_made"] + 0.5 * row["three_pt_made"], row["field_goals_attempted"]
            )
            row["ts_pct"] = safe_ratio(
                row["points"], 2.0 * (row["field_goals_attempted"] + 0.44 * row["free_throw_attempted"])
            )
            row["oreb_pct"] = safe_ratio(
                row["off_rebounds"], row["off_rebounds"] + row["opp_def_rebounds"]
            )
            row["dreb_pct"] = safe_ratio(
                row["def_rebounds"], row["def_rebounds"] + row["opp_off_rebounds"]
            )
            row["tov_pct"] = safe_ratio(
                row["turnovers"],
                row["field_goals_attempted"] + 0.44 * row["free_throw_attempted"] + row["turnovers"],
            )
            row["fta_rate"] = safe_ratio(row["free_throw_attempted"], row["field_goals_attempted"])
            row["ftm_rate"] = safe_ratio(row["free_throw_made"], row["field_goals_attempted"])

            per40_factor = 40.0 / game_minutes
            for stat in COUNT_STATS:
                row[f"{stat}_per40"] = row[stat] * per40_factor
                row[f"opp_{stat}_per40"] = row[f"opp_{stat}"] * per40_factor
            row["score_margin_per40"] = row["score_margin"] * per40_factor
            row["rebound_margin_per40"] = row["rebound_margin"] * per40_factor
            team_rows.append(row)

    games_df = pd.DataFrame(game_rows).sort_values(["game_date", "game_start", "game_id"]).reset_index(drop=True)
    team_df = pd.DataFrame(team_rows)
    team_df["game_date"] = pd.to_datetime(team_df["game_date"])
    team_df = team_df.sort_values(["team_code", "game_date", "game_start", "game_id"]).reset_index(drop=True)
    team_df["team_game_number"] = team_df.groupby("team_code").cumcount() + 1
    team_df["season_game_number"] = team_df.groupby(["season", "team_code"]).cumcount() + 1
    team_df["previous_game_date"] = team_df.groupby("team_code")["game_date"].shift(1)
    team_df["rest_days"] = (team_df["game_date"] - team_df["previous_game_date"]).dt.days
    team_df["rest_days_capped14"] = team_df["rest_days"].clip(upper=14).fillna(7)
    team_df["back_to_back"] = (team_df["rest_days"] == 1).astype(int)
    team_df["long_break"] = (team_df["rest_days"] > 30).astype(int)
    team_df["season_opener"] = (team_df["season_game_number"] == 1).astype(int)
    team_df["previous_season"] = team_df.groupby("team_code")["season"].shift(1)
    team_df["season_transition"] = (
        team_df["previous_season"].notna() & (team_df["previous_season"] != team_df["season"])
    ).astype(int)
    team_df = add_pregame_form_features(team_df)
    games_df, team_df = add_elo_features(games_df, team_df)
    team_df["game_date"] = team_df["game_date"].dt.strftime("%Y-%m-%d")
    return games_df, team_df


def chronological_split(
    target_dates: list[str], reference_dates: list[str] | None = None
) -> tuple[np.ndarray, str, str]:
    unique_dates = sorted(set(reference_dates or target_dates))
    train_idx = max(1, int(len(unique_dates) * 0.70))
    val_idx = max(train_idx + 1, int(len(unique_dates) * 0.85))
    val_idx = min(val_idx, len(unique_dates) - 1)
    train_end = unique_dates[train_idx - 1]
    val_end = unique_dates[val_idx - 1]
    split = np.array(
        [0 if date <= train_end else 1 if date <= val_end else 2 for date in target_dates], dtype=np.uint8
    )
    return split, train_end, val_end


def fit_impute_scale(array: np.ndarray, training_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    training = array[training_mask]
    axes = tuple(range(training.ndim - 1))
    mean = np.nanmean(training, axis=axes)
    mean = np.where(np.isnan(mean), 0.0, mean)
    std = np.nanstd(training, axis=axes)
    std = np.where((std == 0) | np.isnan(std), 1.0, std)
    filled = np.where(np.isnan(array), mean, array)
    scaled = (filled - mean) / std
    return scaled.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def build_sequences(
    team_df: pd.DataFrame, games_df: pd.DataFrame, window: int, mode: str
) -> pd.DataFrame:
    if mode not in {"carryover", "season_reset"}:
        raise ValueError(f"지원하지 않는 시퀀스 모드: {mode}")
    ordered = team_df.sort_values(["team_code", "game_date", "game_start", "game_id"]).reset_index(drop=True)
    histories: dict[Any, pd.DataFrame] = {}
    grouped_histories = (
        ordered.groupby("team_code", sort=False)
        if mode == "carryover"
        else ordered.groupby(["season", "team_code"], sort=False)
    )
    for key, group in grouped_histories:
        normalized_key = str(key) if mode == "carryover" else (str(key[0]), str(key[1]))
        histories[normalized_key] = group.reset_index(drop=True)
    positions: dict[tuple[str, str], int] = {}
    for _, history in histories.items():
        for position, game_id in enumerate(history["game_id"].tolist()):
            code = str(history.iloc[position]["team_code"])
            positions[(str(game_id), code)] = position

    home_sequences: list[np.ndarray] = []
    away_sequences: list[np.ndarray] = []
    contexts: list[list[float]] = []
    labels: list[int] = []
    metadata_rows: list[dict[str, Any]] = []

    team_lookup = team_df.set_index(["game_id", "team_code"], drop=False)
    for game in games_df.sort_values(["game_date", "game_start", "game_id"]).itertuples(index=False):
        home_code = str(game.home_team_code)
        away_code = str(game.away_team_code)
        home_key: Any = home_code if mode == "carryover" else (str(game.season), home_code)
        away_key: Any = away_code if mode == "carryover" else (str(game.season), away_code)
        home_position = positions[(game.game_id, home_code)]
        away_position = positions[(game.game_id, away_code)]
        if home_position < window or away_position < window:
            continue

        home_history = histories[home_key].iloc[home_position - window : home_position]
        away_history = histories[away_key].iloc[away_position - window : away_position]
        home_target = team_lookup.loc[(game.game_id, home_code)]
        away_target = team_lookup.loc[(game.game_id, away_code)]
        history_seasons = set(home_history["season"]).union(set(away_history["season"]))
        crosses_boundary = int(any(season != game.season for season in history_seasons))

        home_sequences.append(home_history.loc[:, SEQUENCE_FEATURES].to_numpy(dtype=np.float32))
        away_sequences.append(away_history.loc[:, SEQUENCE_FEATURES].to_numpy(dtype=np.float32))
        context = [
            float(home_target["rest_days_capped14"]),
            float(away_target["rest_days_capped14"]),
            float(home_target["back_to_back"]),
            float(away_target["back_to_back"]),
            float(home_target["season_game_number"]),
            float(away_target["season_game_number"]),
            float(home_target["win_streak_before"]),
            float(away_target["win_streak_before"]),
            float(home_target["loss_streak_before"]),
            float(away_target["loss_streak_before"]),
            float(home_target["elo_before"]),
            float(away_target["elo_before"]),
            float(home_target["elo_difference"]),
            float(home_target["elo_win_probability"]),
            float(home_target["season_opener"]),
            float(away_target["season_opener"]),
            float(home_target["long_break"]),
            float(away_target["long_break"]),
            float(crosses_boundary),
        ]
        contexts.append(context)
        labels.append(int(home_target["win"]))
        metadata_rows.append(
            {
                "sample_index": len(metadata_rows),
                "target_game_id": game.game_id,
                "target_date": game.game_date,
                "season": game.season,
                "home_team_code": home_code,
                "home_team": game.home_team,
                "away_team_code": away_code,
                "away_team": game.away_team,
                "y_home_win": int(home_target["win"]),
                "window": window,
                "sequence_mode": mode,
                "crosses_season_boundary": crosses_boundary,
                "home_history_start": home_history.iloc[0]["game_date"],
                "home_history_end": home_history.iloc[-1]["game_date"],
                "away_history_start": away_history.iloc[0]["game_date"],
                "away_history_end": away_history.iloc[-1]["game_date"],
            }
        )

    if not metadata_rows:
        raise ValueError(f"N={window} 시퀀스가 생성되지 않았습니다")

    metadata = pd.DataFrame(metadata_rows)
    split, train_end, val_end = chronological_split(
        metadata["target_date"].tolist(), games_df["game_date"].astype(str).tolist()
    )
    metadata["split"] = np.choose(split, ["train", "validation", "test"])
    metadata["train_end_date"] = train_end
    metadata["validation_end_date"] = val_end

    x_home = np.stack(home_sequences).astype(np.float32)
    x_away = np.stack(away_sequences).astype(np.float32)
    context_array = np.asarray(contexts, dtype=np.float32)
    y = np.asarray(labels, dtype=np.uint8)
    training_mask = split == 0

    combined = np.concatenate([x_home, x_away], axis=0)
    combined_training_mask = np.concatenate([training_mask, training_mask])
    combined_scaled, feature_mean, feature_std = fit_impute_scale(combined, combined_training_mask)
    x_home_scaled = combined_scaled[: len(x_home)]
    x_away_scaled = combined_scaled[len(x_home) :]
    context_scaled, context_mean, context_std = fit_impute_scale(context_array, training_mask)

    output_path = PROCESSED_DIR / f"sequences_{mode}_n{window}.npz"
    np.savez_compressed(
        output_path,
        X_home=x_home,
        X_away=x_away,
        context=context_array,
        X_home_scaled=x_home_scaled,
        X_away_scaled=x_away_scaled,
        context_scaled=context_scaled,
        y_home_win=y,
        split=split,
        target_game_id=metadata["target_game_id"].to_numpy(dtype=str),
        feature_names=np.asarray(SEQUENCE_FEATURES, dtype=str),
        context_names=np.asarray(CONTEXT_FEATURES, dtype=str),
        feature_mean=feature_mean,
        feature_std=feature_std,
        context_mean=context_mean,
        context_std=context_std,
        train_end_date=np.asarray(train_end),
        validation_end_date=np.asarray(val_end),
    )
    metadata_path = PROCESSED_DIR / f"sequence_metadata_{mode}_n{window}.csv"
    metadata.to_csv(metadata_path, index=False, encoding="utf-8-sig")
    if mode == "carryover":
        shutil.copyfile(output_path, PROCESSED_DIR / f"sequences_n{window}.npz")
        shutil.copyfile(metadata_path, PROCESSED_DIR / f"sequence_metadata_n{window}.csv")
    return metadata


def build_pregame_model_features(team_df: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
    """부스팅·로지스틱 회귀용 경기 전 한 행 데이터셋을 만든다."""
    lookup = team_df.set_index(["game_id", "team_code"], drop=False)
    rows: list[dict[str, Any]] = []
    for game in games_df.sort_values(["game_date", "game_start", "game_id"]).itertuples(index=False):
        home = lookup.loc[(game.game_id, str(game.home_team_code))]
        away = lookup.loc[(game.game_id, str(game.away_team_code))]
        row: dict[str, Any] = {
            "game_id": game.game_id,
            "season": game.season,
            "game_date": game.game_date,
            "home_team_code": str(game.home_team_code),
            "home_team": game.home_team,
            "away_team_code": str(game.away_team_code),
            "away_team": game.away_team,
            "y_home_win": int(home["win"]),
            "home_prior_games": int(home["team_game_number"] - 1),
            "away_prior_games": int(away["team_game_number"] - 1),
            "home_season_prior_games": int(home["season_game_number"] - 1),
            "away_season_prior_games": int(away["season_game_number"] - 1),
        }
        paired_features = (
            "rest_days_capped14",
            "back_to_back",
            "long_break",
            "season_opener",
            "season_transition",
            "win_streak_before",
            "loss_streak_before",
            "elo_before",
            "elo_win_probability",
        )
        for feature in paired_features:
            row[f"home_{feature}"] = safe_number(home[feature])
            row[f"away_{feature}"] = safe_number(away[feature])
            row[f"diff_{feature}"] = safe_number(home[feature]) - safe_number(away[feature])

        row["elo_difference"] = safe_number(home["elo_difference"])
        row["elo_home_win_probability"] = safe_number(home["elo_win_probability"])
        for window in ROLLING_WINDOWS:
            for feature in ROLLING_BASE_FEATURES:
                column = f"rolling_{feature}_{window}"
                row[f"home_{column}"] = safe_number(home[column])
                row[f"away_{column}"] = safe_number(away[column])
                row[f"diff_{column}"] = safe_number(home[column]) - safe_number(away[column])
        rows.append(row)

    frame = pd.DataFrame(rows)
    split, train_end, val_end = chronological_split(
        frame["game_date"].astype(str).tolist(), games_df["game_date"].astype(str).tolist()
    )
    frame["split"] = np.choose(split, ["train", "validation", "test"])
    frame["train_end_date"] = train_end
    frame["validation_end_date"] = val_end
    frame.to_csv(PROCESSED_DIR / "pregame_model_features.csv", index=False, encoding="utf-8-sig")
    return frame


def evaluate_elo_baseline(pregame_df: pd.DataFrame) -> dict[str, Any]:
    predictions = pregame_df[
        [
            "game_id",
            "season",
            "game_date",
            "home_team",
            "away_team",
            "y_home_win",
            "elo_home_win_probability",
            "split",
        ]
    ].copy()
    predictions["elo_prediction"] = (predictions["elo_home_win_probability"] >= 0.5).astype(int)
    predictions.to_csv(PROCESSED_DIR / "elo_baseline_predictions.csv", index=False, encoding="utf-8-sig")

    def metrics(frame: pd.DataFrame) -> dict[str, float | int]:
        y = frame["y_home_win"].to_numpy(dtype=float)
        probability = frame["elo_home_win_probability"].to_numpy(dtype=float)
        clipped = np.clip(probability, 1e-7, 1 - 1e-7)
        prediction = (probability >= 0.5).astype(float)
        return {
            "samples": int(len(frame)),
            "accuracy": float(np.mean(prediction == y)),
            "brier_score": float(np.mean((probability - y) ** 2)),
            "log_loss": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))),
            "home_win_rate": float(np.mean(y)),
        }

    report = {
        "configuration": {
            "initial_rating": ELO_INITIAL,
            "k_factor": ELO_K_FACTOR,
            "home_advantage": ELO_HOME_ADVANTAGE,
            "offseason_retention": ELO_OFFSEASON_RETENTION,
        },
        "all": metrics(predictions),
        "train": metrics(predictions[predictions["split"] == "train"]),
        "validation": metrics(predictions[predictions["split"] == "validation"]),
        "test": metrics(predictions[predictions["split"] == "test"]),
    }
    write_json(PROCESSED_DIR / "elo_baseline_metrics.json", report)
    return report


def write_feature_dictionary() -> None:
    rows = [
        ("identifier", "game_id", "KBL 경기 고유키", "known", "all"),
        ("label", "y_home_win", "목표 경기 홈팀 승리=1", "target", "all"),
        ("context", "rest_days_capped14", "직전 경기 후 휴식일, 최대 14일", "pregame", "all"),
        ("context", "back_to_back", "직전 경기 다음 날 경기 여부", "pregame", "all"),
        ("context", "long_break", "직전 경기 후 30일 초과 여부", "pregame", "all"),
        ("form", "win_streak_before", "목표 경기 전 연승 길이", "pregame", "all"),
        ("form", "loss_streak_before", "목표 경기 전 연패 길이", "pregame", "all"),
        ("four_factors", "efg_pct", "(FGM + 0.5×3PM) / FGA", "postgame_history", "all"),
        ("four_factors", "tov_pct", "TO / (FGA + 0.44×FTA + TO)", "postgame_history", "all"),
        ("four_factors", "oreb_pct", "ORB / (ORB + 상대 DRB)", "postgame_history", "all"),
        ("four_factors", "fta_rate", "FTA / FGA", "postgame_history", "all"),
        ("rolling", "rolling_*_3/5/10", "목표 경기 이전 3·5·10경기 평균", "pregame", "boosting"),
        ("elo", "elo_before", "목표 경기 시작 전 팀 Elo", "pregame", "all"),
        ("elo", "elo_difference", "홈팀 경기 전 Elo - 원정팀 경기 전 Elo", "pregame", "all"),
        ("elo", "elo_home_win_probability", "홈 어드밴티지를 반영한 Elo 홈 승리확률", "pregame", "all"),
        ("season", "season_transition", "직전 경기와 시즌이 다른 첫 경기 여부", "pregame", "all"),
        ("season", "crosses_season_boundary", "시퀀스가 이전 시즌 기록을 포함하는지", "pregame", "sequence"),
        ("normalization", "*_per40", "연장전 보정 40분 환산 스탯", "postgame_history", "sequence"),
    ]
    dictionary = pd.DataFrame(
        rows, columns=["category", "feature", "definition", "availability", "recommended_for"]
    )
    dictionary.to_csv(PROCESSED_DIR / "feature_dictionary.csv", index=False, encoding="utf-8-sig")


def validate(
    games_df: pd.DataFrame,
    team_df: pd.DataFrame,
    sequence_meta: dict[str, dict[int, pd.DataFrame]],
    pregame_df: pd.DataFrame,
    elo_report: dict[str, Any],
) -> dict[str, Any]:
    required_game = ["game_id", "season", "game_date", "home_team_code", "away_team_code", "home_score", "away_score"]
    required_team = [
        "game_id",
        "season",
        "game_date",
        "team_code",
        "opponent_code",
        "is_home",
        "points",
        "opp_points",
        "win",
    ]
    season_games = games_df.groupby("season")["game_id"].nunique().astype(int).to_dict()
    team_season_counts = (
        team_df.groupby(["season", "team_code"]).size().rename("games").reset_index().to_dict("records")
    )
    per_game_rows = team_df.groupby("game_id").size()
    label_sums = team_df.groupby("game_id")["win"].sum()
    shot_violations = int(
        (
            (team_df["two_pt_made"] > team_df["two_pt_attempted"])
            | (team_df["three_pt_made"] > team_df["three_pt_attempted"])
            | (team_df["free_throw_made"] > team_df["free_throw_attempted"])
        ).sum()
    )
    score_match = team_df["points"] == team_df.apply(
        lambda row: games_df.set_index("game_id").loc[row["game_id"], "home_score"]
        if row["is_home"] == 1
        else games_df.set_index("game_id").loc[row["game_id"], "away_score"],
        axis=1,
    )
    sequence_date_leaks = 0
    reset_boundary_crossings = 0
    scaled_sequence_missing_values = 0
    for mode, by_window in sequence_meta.items():
        for window, metadata in by_window.items():
            sequence_date_leaks += int(
                (
                    (pd.to_datetime(metadata["home_history_end"]) >= pd.to_datetime(metadata["target_date"]))
                    | (pd.to_datetime(metadata["away_history_end"]) >= pd.to_datetime(metadata["target_date"]))
                ).sum()
            )
            if mode == "season_reset":
                reset_boundary_crossings += int(metadata["crosses_season_boundary"].sum())
            arrays = np.load(PROCESSED_DIR / f"sequences_{mode}_n{window}.npz")
            scaled_sequence_missing_values += int(
                np.isnan(arrays["X_home_scaled"]).sum()
                + np.isnan(arrays["X_away_scaled"]).sum()
                + np.isnan(arrays["context_scaled"]).sum()
            )
    four_factor_missing_values = int(
        team_df[["efg_pct", "tov_pct", "oreb_pct", "fta_rate"]].isna().sum().sum()
    )
    report: dict[str, Any] = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "source": "KBL official website API used by https://www.kbl.or.kr/match/schedule",
        "total_games": int(len(games_df)),
        "unique_games": int(games_df["game_id"].nunique()),
        "team_game_rows": int(len(team_df)),
        "seasons": season_games,
        "team_season_counts": team_season_counts,
        "min_team_games_per_season": int(min(row["games"] for row in team_season_counts)),
        "max_team_games_per_season": int(max(row["games"] for row in team_season_counts)),
        "duplicate_game_rows": int(games_df.duplicated("game_id").sum()),
        "duplicate_team_game_rows": int(team_df.duplicated(["game_id", "team_code"]).sum()),
        "games_without_exactly_two_team_rows": int((per_game_rows != 2).sum()),
        "games_with_invalid_label_sum": int((label_sums != 1).sum()),
        "shot_make_attempt_violations": shot_violations,
        "score_mismatches": int((~score_match).sum()),
        "missing_required_game_values": int(games_df[required_game].isna().sum().sum()),
        "missing_required_team_values": int(team_df[required_team].isna().sum().sum()),
        "overtime_games": int(games_df["overtime"].sum()),
        "pregame_model_rows": int(len(pregame_df)),
        "four_factor_missing_values": four_factor_missing_values,
        "sequence_date_leaks": sequence_date_leaks,
        "reset_boundary_crossings": reset_boundary_crossings,
        "scaled_sequence_missing_values": scaled_sequence_missing_values,
        "sequence_samples": {
            mode: {str(window): int(len(metadata)) for window, metadata in by_window.items()}
            for mode, by_window in sequence_meta.items()
        },
        "sequence_splits": {
            mode: {
                str(window): {
                    str(name): int(count) for name, count in metadata["split"].value_counts().items()
                }
                for window, metadata in by_window.items()
            }
            for mode, by_window in sequence_meta.items()
        },
        "elo_baseline": elo_report,
    }
    checks = {
        "total_games_is_540": report["total_games"] == 540,
        "each_season_is_270": season_games == {"2023-2024": 270, "2024-2025": 270},
        "each_team_has_54_games": report["min_team_games_per_season"] == 54
        and report["max_team_games_per_season"] == 54,
        "team_rows_is_1080": report["team_game_rows"] == 1080,
        "no_game_duplicates": report["duplicate_game_rows"] == 0,
        "no_team_game_duplicates": report["duplicate_team_game_rows"] == 0,
        "two_team_rows_per_game": report["games_without_exactly_two_team_rows"] == 0,
        "one_winner_per_game": report["games_with_invalid_label_sum"] == 0,
        "valid_shot_counts": report["shot_make_attempt_violations"] == 0,
        "scores_reconcile": report["score_mismatches"] == 0,
        "required_values_complete": report["missing_required_game_values"] == 0
        and report["missing_required_team_values"] == 0,
        "pregame_rows_is_540": report["pregame_model_rows"] == 540,
        "four_factors_complete": report["four_factor_missing_values"] == 0,
        "no_sequence_date_leakage": report["sequence_date_leaks"] == 0,
        "season_reset_has_no_boundary_crossing": report["reset_boundary_crossings"] == 0,
        "scaled_sequences_have_no_missing_values": report["scaled_sequence_missing_values"] == 0,
        "elo_probabilities_are_valid": bool(
            pregame_df["elo_home_win_probability"].between(0, 1, inclusive="both").all()
        ),
    }
    report["checks"] = checks
    report["all_checks_passed"] = all(checks.values())
    return report


def write_quality_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# KBL 데이터 품질 보고서",
        "",
        f"- 생성 시각: {report['generated_at']}",
        f"- 경기 수: {report['total_games']}",
        f"- 팀 관점 행 수: {report['team_game_rows']}",
        f"- 연장전 경기 수: {report['overtime_games']}",
        f"- 전체 검사 통과: {'PASS' if report['all_checks_passed'] else 'CHECK'}",
        "",
        "## 검사 결과",
        "",
        "| 검사 | 결과 |",
        "|---|---|",
    ]
    lines.extend(f"| {name} | {'PASS' if passed else 'FAIL'} |" for name, passed in report["checks"].items())
    lines.extend(["", "## 시즌별 경기 수", "", "| 시즌 | 경기 수 |", "|---|---:|"])
    lines.extend(f"| {season} | {count} |" for season, count in report["seasons"].items())
    lines.extend(
        [
            "",
            "## 시퀀스 샘플",
            "",
            "| 모드 | N | 전체 | 학습 | 검증 | 테스트 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for mode, by_window in report["sequence_samples"].items():
        for window, total in by_window.items():
            splits = report["sequence_splits"][mode][window]
            lines.append(
                f"| {mode} | {window} | {total} | {splits.get('train', 0)} | "
                f"{splits.get('validation', 0)} | {splits.get('test', 0)} |"
            )
    elo_test = report["elo_baseline"]["test"]
    lines.extend(
        [
            "",
            "## Elo 테스트 기준선",
            "",
            f"- 샘플 수: {elo_test['samples']}",
            f"- 정확도: {elo_test['accuracy']:.4f}",
            f"- Brier score: {elo_test['brier_score']:.4f}",
            f"- Log loss: {elo_test['log_loss']:.4f}",
        ]
    )
    (PROCESSED_DIR / "quality_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_workbook_payload(
    games_df: pd.DataFrame,
    team_df: pd.DataFrame,
    pregame_df: pd.DataFrame,
    sequence_carryover: pd.DataFrame,
    sequence_reset: pd.DataFrame,
    report: dict[str, Any],
) -> None:
    game_columns = [
        "game_id",
        "season",
        "game_date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "overtime_periods",
        "game_minutes",
        "home_elo_before",
        "away_elo_before",
        "elo_home_win_probability",
        "stadium",
        "source_match_url",
    ]
    team_columns = [
        "game_id",
        "season",
        "game_date",
        "team_code",
        "team",
        "opponent",
        "is_home",
        "win",
        "points",
        "opp_points",
        "score_margin",
        "fg_pct",
        "three_pt_pct",
        "free_throw_pct",
        "efg_pct",
        "tov_pct",
        "oreb_pct",
        "fta_rate",
        "off_rebounds",
        "def_rebounds",
        "total_rebounds",
        "rebound_margin",
        "assists",
        "steals",
        "blocks",
        "turnovers",
        "fouls",
        "overtime_periods",
        "rest_days",
        "back_to_back",
        "win_streak_before",
        "loss_streak_before",
        "rolling_win_5",
        "elo_before",
        "opp_elo_before",
        "elo_win_probability",
    ]
    sequence_columns = [
        "sample_index",
        "target_game_id",
        "target_date",
        "season",
        "home_team",
        "away_team",
        "y_home_win",
        "window",
        "sequence_mode",
        "split",
        "crosses_season_boundary",
    ]
    pregame_columns = [
        "game_id",
        "season",
        "game_date",
        "home_team",
        "away_team",
        "y_home_win",
        "split",
        "home_rest_days_capped14",
        "away_rest_days_capped14",
        "home_win_streak_before",
        "away_win_streak_before",
        "home_rolling_win_5",
        "away_rolling_win_5",
        "diff_rolling_efg_pct_5",
        "diff_rolling_tov_pct_5",
        "diff_rolling_oreb_pct_5",
        "diff_rolling_fta_rate_5",
        "elo_difference",
        "elo_home_win_probability",
    ]

    def clean_records(frame: pd.DataFrame, columns: list[str]) -> list[list[Any]]:
        selected = frame.loc[:, columns].copy()
        selected = selected.astype(object).where(pd.notna(selected), None)
        return [columns] + selected.values.tolist()

    payload = {
        "report": report,
        "games": clean_records(games_df, game_columns),
        "team_games": clean_records(team_df, team_columns),
        "pregame": clean_records(pregame_df, pregame_columns),
        "sequences_carryover": clean_records(sequence_carryover, sequence_columns),
        "sequences_reset": clean_records(sequence_reset, sequence_columns),
        "elo_predictions": clean_records(
            pd.read_csv(PROCESSED_DIR / "elo_baseline_predictions.csv"),
            [
                "game_id",
                "season",
                "game_date",
                "home_team",
                "away_team",
                "y_home_win",
                "elo_home_win_probability",
                "elo_prediction",
                "split",
            ],
        ),
    }
    write_json(PROCESSED_DIR / "workbook_payload.json", payload)


def run(args: argparse.Namespace) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    payloads = crawl_all(refresh=args.refresh, max_workers=args.workers)
    games_df, team_df = extract_datasets(payloads)

    games_df.to_csv(PROCESSED_DIR / "games_raw.csv", index=False, encoding="utf-8-sig")
    team_df.to_csv(PROCESSED_DIR / "team_games.csv", index=False, encoding="utf-8-sig")
    chronological = team_df.sort_values(["team_code", "game_date", "game_start", "game_id"])
    chronological.to_csv(PROCESSED_DIR / "team_games_chronological.csv", index=False, encoding="utf-8-sig")

    pregame_df = build_pregame_model_features(team_df, games_df)
    elo_report = evaluate_elo_baseline(pregame_df)
    write_feature_dictionary()
    sequence_meta = {
        mode: {window: build_sequences(team_df, games_df, window, mode) for window in args.windows}
        for mode in ("carryover", "season_reset")
    }
    report = validate(games_df, team_df, sequence_meta, pregame_df, elo_report)
    write_json(PROCESSED_DIR / "quality_report.json", report)
    write_quality_markdown(report)
    workbook_window = 5 if 5 in args.windows else args.windows[0]
    make_workbook_payload(
        games_df,
        team_df,
        pregame_df,
        sequence_meta["carryover"][workbook_window],
        sequence_meta["season_reset"][workbook_window],
        report,
    )

    if not report["all_checks_passed"]:
        failed = [name for name, passed in report["checks"].items() if not passed]
        raise RuntimeError(f"데이터 품질 검사 실패: {', '.join(failed)}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KBL 두 시즌 경기 데이터 수집·전처리·시퀀스 생성")
    parser.add_argument("--windows", nargs="+", type=int, default=[3, 5, 10], help="시퀀스 길이 N")
    parser.add_argument("--workers", type=int, default=6, help="동시 요청 수")
    parser.add_argument("--refresh", action="store_true", help="기존 원본 캐시를 무시하고 다시 수집")
    args = parser.parse_args()
    if any(window < 1 for window in args.windows):
        parser.error("시퀀스 길이는 1 이상이어야 합니다")
    if not 1 <= args.workers <= 12:
        parser.error("workers는 1~12 사이여야 합니다")
    args.windows = sorted(set(args.windows))
    return args


if __name__ == "__main__":
    run(parse_args())
