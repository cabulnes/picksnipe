from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog

import pandas as pd
import time

import streamlit as st

# Get all NBA players
all_players = players.get_players()
player_names = [player['full_name'] for player in all_players]

if 'player_results' not in st.session_state:
    st.session_state.player_results = {}

def search_players(searchterm: str):
    """Search function for player autocomplete"""
    if not searchterm:
        return []
    all_players = players.get_players()
    player_names = [player['full_name'] for player in all_players]
    return [name for name in player_names if searchterm.lower() in name.lower()][:10]


def get_player_id(player_name):
    """Get player ID from name"""
    all_players = players.get_players()
    player = [p for p in all_players if p['full_name'].lower() == player_name.lower()]
    if player:
        return player[0]['id']
    return None


def get_last_n_games(player_id, n_games=10, season='2024-25'):
    """Get last N games for a player"""
    try:
        time.sleep(0.6)  # Rate limit
        
        gamelog = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star='Regular Season'
        )
        
        df = gamelog.get_data_frames()[0]
        return df.head(n_games)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None


def calculate_stat_value(games_df, stat_type):
    """Calculate the stat value based on stat type"""
    stat_mapping = {
        "Points": "PTS",
        "Rebounds": "REB",
        "Assists": "AST",
        "3-PT-Made": "FG3M",
        "FG Made": "FGM",
        "Fantasy Score": "FANTASY_PTS"  # If available
    }
    
    # Handle combined stats
    if stat_type == "Pts+Rebs+Asts":
        return games_df["PTS"] + games_df["REB"] + games_df["AST"]
    elif stat_type == "Pts+Asts":
        return games_df["PTS"] + games_df["AST"]
    else:
        return games_df[stat_mapping.get(stat_type, "PTS")]


def calculate_probability(games_df, stat_type, threshold, prediction_type):
    """Calculate probability based on stat type and prediction (More/Less)"""
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
    
    return {
        'probability': probability,
        'games_met': games_met_threshold,
        'total_games': total_games,
        'mean': stat_values.mean(),
        'median': stat_values.median(),
        'std': stat_values.std(),
        'min': stat_values.min(),
        'max': stat_values.max(),
        'stat_values': stat_values
    }