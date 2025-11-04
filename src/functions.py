"""
Functions for NBA player data retrieval and probability calculations.

This module provides utilities for fetching NBA player statistics
and calculating prop bet probabilities based on historical performance.
"""

import time
from typing import Literal
from datetime import datetime

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


def get_current_season() -> str:
    """Get the current NBA season string (e.g., '2025-26')."""
    now = datetime.now()
    year = now.year
    month = now.month
    
    # NBA season starts in October, so if we're in Oct-Dec, 
    # the season is current_year to next_year
    # If we're in Jan-Sep, the season started last year
    if month >= 10:
        return f"{year}-{str(year + 1)[-2:]}"
    else:
        return f"{year - 1}-{str(year)[-2:]}"


def get_last_n_games(
    player_id: int, n_games: int = 10, season: str | None = None
) -> pd.DataFrame | None:
    """Get last N games for a player across multiple seasons if needed.
    
    Args:
        player_id: NBA player ID
        n_games: Number of games to retrieve
        season: Season to start from (defaults to current season)
    
    Returns:
        DataFrame with the most recent N games (or all available games)
    """
    try:
        # Use current season if not specified
        if season is None:
            season = get_current_season()
        
        all_games = []
        year = int(season.split("-")[0])
        seasons_to_check = 3  # Check up to 3 seasons back
        
        for i in range(seasons_to_check):
            season_year = year - i
            season_str = f"{season_year}-{str(season_year + 1)[-2:]}"
            
            try:
                time.sleep(0.6)  # Rate limit
                
                gamelog = playergamelog.PlayerGameLog(
                    player_id=player_id, 
                    season=season_str, 
                    season_type_all_star="Regular Season"
                )
                df = gamelog.get_data_frames()[0]
                
                if not df.empty:
                    all_games.append(df)
                    
                # Stop if we have enough games
                total_games = sum(len(games) for games in all_games)
                if total_games >= n_games:
                    break
                    
            except Exception:
                # Continue to next season if this one fails
                continue
        
        if all_games:
            # Combine and sort all games
            combined_df = pd.concat(all_games, ignore_index=True)
            combined_df['GAME_DATE'] = pd.to_datetime(combined_df['GAME_DATE'])
            combined_df = combined_df.sort_values('GAME_DATE', ascending=False)
            
            # Take up to n_games
            result_df = combined_df.head(n_games)
            
            # Show info if fewer games available
            if len(result_df) < n_games:
                st.info(
                    f"Only {len(result_df)} games available for this player "
                    f"(requested {n_games}). Using all available games."
                )
            
            return result_df
        else:
            st.warning(f"No game data found for player ID {player_id}")
            return None
            
    except Exception as e:
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
    elif stat_type == "Pts+Rebs":
        return games_df["PTS"] + games_df["REB"]
    elif stat_type == "Rebs+Asts":
        return games_df["REB"] + games_df["AST"]
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