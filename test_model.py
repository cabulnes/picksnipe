from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog

import pandas as pd
import time

import streamlit as st
from streamlit_searchbox import st_searchbox

from functions import search_players, get_player_id, get_last_n_games, calculate_probability, calculate_stat_value


# Streamlit App
st.title("NBA Multi-Player Prop Probability Calculator")

st.write("Enter up to 6 players and their prop bets to calculate the probability that ALL will hit.")

# Configuration
n_games = st.slider("Number of recent games to analyze:", min_value=5, max_value=20, value=10)

st.divider()

# Player input arrays
options = ["Less", "More"]
player_input_keys = [
    "player_searchbox_1",
    "player_searchbox_2",
    "player_searchbox_3",
    "player_searchbox_4",
    "player_searchbox_5",
    "player_searchbox_6"
]
number_input_strings = [
    "Input first pick",
    "Input second pick",
    "Input third pick",
    "Input fourth pick",
    "Input fifth pick",
    "Input sixth pick"
]
type_input_keys = [
    "type_1",
    "type_2",
    "type_3",
    "type_4",
    "type_5",
    "type_6"
]
prediction_input_keys = [
    "prediction_pill_1",
    "prediction_pill_2",
    "prediction_pill_3",
    "prediction_pill_4",
    "prediction_pill_5",
    "prediction_pill_6"
]

# Store picks
picks = []

for idx, (player_input_key, type_input_key, number_input_string, prediction_input_key) in enumerate(
    zip(player_input_keys, type_input_keys, number_input_strings, prediction_input_keys), 1
):
    with st.container():
        col1, col2, col3, col4 = st.columns([2,1,1,1])
        with col1:
            col1.space("small")
            player = st_searchbox(
                search_players, 
                key=player_input_key, 
                placeholder="Input player name..."
            )
        with col2:
            stat_type = st.selectbox(
                "Points, Rebounds...", 
                ("Points","Rebounds","Pts+Rebs+Asts","Assists","3-PT-Made","Pts+Asts","FG Made","Fantasy Score"), 
                key=type_input_key
            )
        with col3:
            threshold = st.number_input(
                number_input_string,
                step=0.5, 
                format="%.1f",
                value=0.0
            )
        with col4:
            prediction = st.pills(
                "Type", 
                options, 
                key=prediction_input_key, 
                selection_mode="single"
            )
        
        # Store pick if player is selected
        if player and prediction and threshold > 0:
            picks.append({
                'player': player,
                'stat_type': stat_type,
                'threshold': threshold,
                'prediction': prediction,
                'pick_number': idx
            })
    
    st.divider()

# Calculate button
if st.button("Calculate Probabilities", type="primary"):
    if not picks:
        st.warning("Please enter at least one player pick.")
    else:
        st.subheader(f"Analyzing {len(picks)} Pick(s)")
        
        individual_probabilities = []
        results_data = []
        
        with st.spinner("Fetching player data..."):
            for pick in picks:
                player_id = get_player_id(pick['player'])
                
                if player_id:
                    games_df = get_last_n_games(player_id, n_games)
                    
                    if games_df is not None and not games_df.empty:
                        result = calculate_probability(
                            games_df, 
                            pick['stat_type'], 
                            pick['threshold'], 
                            pick['prediction']
                        )
                        
                        if result:
                            individual_probabilities.append(result['probability'])
                            results_data.append({
                                'Pick': f"Pick {pick['pick_number']}",
                                'Player': pick['player'],
                                'Stat': pick['stat_type'],
                                'Threshold': f"{pick['prediction']} {pick['threshold']}",
                                'Probability': f"{result['probability']:.1%}",
                                'Hit Rate': f"{result['games_met']}/{result['total_games']}",
                                'Avg': f"{result['mean']:.1f}"
                            })
        
        # Display individual results
        if results_data:
            st.subheader("Individual Pick Probabilities")
            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df, use_container_width=True, hide_index=True)
            
            # Calculate combined probability (all must hit)
            if len(individual_probabilities) > 0:
                combined_probability = 1.0
                for prob in individual_probabilities:
                    combined_probability *= prob
                
                st.divider()
                st.subheader("Combined Parlay Probability")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Probability ALL Hit", 
                        f"{combined_probability:.2%}",
                        help="Probability that all picks hit"
                    )
                with col2:
                    st.metric(
                        "Number of Picks", 
                        len(picks)
                    )
                with col3:
                    if combined_probability > 0:
                        implied_odds = 1 / combined_probability
                        st.metric(
                            "Implied Odds", 
                            f"+{int((implied_odds - 1) * 100)}",
                            help="American odds format"
                        )
                
                # Show probability breakdown
                st.subheader("Probability Breakdown")
                st.write("For all picks to hit, each individual pick must succeed:")
                
                prob_breakdown = " × ".join([f"{p:.1%}" for p in individual_probabilities])
                st.write(f"**{prob_breakdown} = {combined_probability:.2%}**")
                
                # Visual representation
                import numpy as np
                fig_data = pd.DataFrame({
                    'Pick': [f"Pick {i+1}" for i in range(len(individual_probabilities))],
                    'Probability': individual_probabilities
                })
                st.bar_chart(fig_data.set_index('Pick'))
        else:
            st.error("Could not calculate probabilities for any picks.")