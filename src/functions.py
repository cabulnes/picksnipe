"""
Functions for NBA player data retrieval and probability calculations.

This module provides utilities for fetching NBA player statistics
and calculating prop bet probabilities based on historical performance.
"""

import time
from typing import Literal

import pandas as pd
import streamlit as st
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
from pydantic import BaseModel, ConfigDict

# Type definitions
StatType = Literal[
    "Points",
    "Rebounds",
    "Assists",
    "3-PT-Made",
    "Pts+Rebs+Asts",
    "Pts+Asts",
    "FG Made",
    "Fantasy Score",
]

PredictionType = Literal["More", "Less"]


class PlayerInfo(BaseModel):
    """Pydantic model for NBA player information."""

    model_config = ConfigDict(frozen=True)

    id: int
    full_name: str
    first_name: str
    last_name: str
    is_active: bool


class ProbabilityResult(BaseModel):
    """Pydantic model for probability calculation results."""

    model_config = ConfigDict(frozen=True)

    probability: float
    games_met: int
    total_games: int
    mean: float
    median: float
    std: float
    min: float
    max: float
    stat_values: list[float]  # Converting pd.Series to list for JSON serialization


# Get all NBA players - converting dict data to Pydantic models
_raw_players = players.get_players()
all_players: list[PlayerInfo] = [PlayerInfo(**player) for player in _raw_players]
player_names: list[str] = [player.full_name for player in all_players]

if "player_results" not in st.session_state:
    st.session_state.player_results = {}


def search_players(searchterm: str) -> list[str]:
    """Search function for player autocomplete."""
    if not searchterm:
        return []
    return [name for name in player_names if searchterm.lower() in name.lower()][:10]


def get_player_id(player_name: str) -> int | None:
    """Get player ID from name."""
    player = [p for p in all_players if p.full_name.lower() == player_name.lower()]
    if player:
        return player[0].id
    return None


def get_last_n_games(
    player_id: int, n_games: int = 10, season: str = "2024-25"
) -> pd.DataFrame | None:
    """Get last N games for a player."""
    try:
        time.sleep(0.6)  # Rate limit

        gamelog = playergamelog.PlayerGameLog(
            player_id=player_id, season=season, season_type_all_star="Regular Season"
        )

        df = gamelog.get_data_frames()[0]
        return df.head(n_games)
    except (ValueError, KeyError, IndexError, ConnectionError) as e:
        st.error(f"Error fetching data: {e}")
        return None


def calculate_stat_value(games_df: pd.DataFrame, stat_type: StatType) -> pd.Series:
    """Calculate the stat value based on stat type."""
    stat_mapping: dict[str, str] = {
        "Points": "PTS",
        "Rebounds": "REB",
        "Assists": "AST",
        "3-PT-Made": "FG3M",
        "FG Made": "FGM",
        "Fantasy Score": "FANTASY_PTS",  # If available
    }

    # Handle combined stats
    if stat_type == "Pts+Rebs+Asts":
        return games_df["PTS"] + games_df["REB"] + games_df["AST"]
    elif stat_type == "Pts+Asts":
        return games_df["PTS"] + games_df["AST"]
    else:
        return games_df[stat_mapping.get(stat_type, "PTS")]


def calculate_probability(
    games_df: pd.DataFrame | None,
    stat_type: StatType,
    threshold: int | float,
    prediction_type: PredictionType,
) -> ProbabilityResult | None:
    """Calculate probability based on stat type and prediction (More/Less)."""
    if games_df is None or games_df.empty:
        return None

    stat_values = calculate_stat_value(games_df, stat_type)

    # Determine if threshold is met based on More/Less
    if prediction_type == "More":
        games_met_threshold = (stat_values > threshold).sum()
    else:  # "Less"
        games_met_threshold = (stat_values < threshold).sum()

    total_games = len(games_df)
    probability = games_met_threshold / total_games

    return ProbabilityResult(
        probability=probability,
        games_met=games_met_threshold,
        total_games=total_games,
        mean=stat_values.mean(),
        median=stat_values.median(),
        std=stat_values.std(),
        min=stat_values.min(),
        max=stat_values.max(),
        stat_values=stat_values.tolist(),
    )
