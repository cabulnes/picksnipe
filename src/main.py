"""
NBA Multi-Player Prop Probability Calculator Streamlit App.

This module provides a web interface for calculating the probability
that multiple NBA player prop bets will all hit based on historical data.
"""

import pandas as pd
import streamlit as st
from pydantic import BaseModel, ConfigDict
from streamlit_searchbox import st_searchbox
import plotly.graph_objects as go

from functions import (
    PredictionType,
    StatType,
    ProbabilityResult,
    calculate_probability,
    get_last_n_games,
    get_player_id,
    search_players,
)


# Pydantic models for picks
class PickData(BaseModel):
    """Pydantic model for individual player pick data."""

    model_config = ConfigDict(frozen=True)

    player: str
    stat_type: StatType
    threshold: float
    prediction: PredictionType
    pick_number: int


class DetailedResult(BaseModel):
    """Pydantic model for detailed results including pick and probability data."""

    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)

    pick: PickData
    result: ProbabilityResult
    games_df: pd.DataFrame

    #class Config:
    #    arbitrary_types_allowed = True


# Streamlit App
st.title("NBA Multi-Player Prop Probability Calculator")

st.write(
    "Enter up to 6 players and their prop bets to calculate "
    "the probability that ALL will hit."
)

# Configuration
n_games: int = st.slider(
    "Number of recent games to analyze:",
    min_value=5,
    max_value=20,
    value=10,
)

st.divider()

# Player input arrays
options: list[PredictionType] = ["Less", "More"]
player_input_keys: list[str] = [
    "player_searchbox_1",
    "player_searchbox_2",
    "player_searchbox_3",
    "player_searchbox_4",
    "player_searchbox_5",
    "player_searchbox_6",
]
number_input_strings: list[str] = [
    "Input first pick",
    "Input second pick",
    "Input third pick",
    "Input fourth pick",
    "Input fifth pick",
    "Input sixth pick",
]
type_input_keys: list[str] = [
    "type_1",
    "type_2",
    "type_3",
    "type_4",
    "type_5",
    "type_6",
]
prediction_input_keys: list[str] = [
    "prediction_pill_1",
    "prediction_pill_2",
    "prediction_pill_3",
    "prediction_pill_4",
    "prediction_pill_5",
    "prediction_pill_6",
]

# Store picks
picks: list[PickData] = []

for idx, (
    player_input_key,
    type_input_key,
    number_input_string,
    prediction_input_key,
) in enumerate(
    zip(
        player_input_keys, type_input_keys, number_input_strings, prediction_input_keys
    ),
    1,
):
    with st.container():
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        with col1:
            col1.space("small")
            player = st_searchbox(
                search_players,
                key=player_input_key,
                placeholder="Input player name...",
            )
        with col2:
            stat_type = st.selectbox(
                "Points, Rebounds...",
                (
                    "Points",
                    "Rebounds",
                    "Assists",
                    "Pts+Rebs",
                    "Pts+Asts",
                    "Rebs+Asts",
                    "Pts+Rebs+Asts",
                    "3-PT-Made",
                    "FG Made",
                    "Fantasy Score",
                ),
                key=type_input_key,
            )
        with col3:
            threshold = st.number_input(
                number_input_string,
                step=0.5,
                format="%.1f",
                value=0.0,
            )
        with col4:
            prediction = st.pills(
                "Type",
                options,
                key=prediction_input_key,
                selection_mode="single",
            )

        # Store pick if player is selected
        if player and prediction and threshold > 0:
            picks.append(
                PickData(
                    player=player,
                    stat_type=stat_type,
                    threshold=threshold,
                    prediction=prediction,
                    pick_number=idx,
                )
            )

    st.divider()

# Calculate button
if st.button("Calculate Probabilities", type="primary"):
    if not picks:
        st.warning("Please enter at least one player pick.")
    else:
        st.subheader(f"Analyzing {len(picks)} Pick(s)")

        individual_probabilities: list[float] = []
        results_data: list[dict[str, str]] = []
        detailed_results: list[DetailedResult] = []

        with st.spinner("Fetching player data..."):
            for pick in picks:
                player_id: int | None = get_player_id(pick.player)

                if player_id:
                    games_df: pd.DataFrame | None = get_last_n_games(player_id, n_games)

                    if games_df is not None and not games_df.empty:
                        result = calculate_probability(
                            games_df,
                            pick.stat_type,
                            pick.threshold,
                            pick.prediction,
                        )

                        if result:
                            # Access Pydantic attributes with dot notation
                            individual_probabilities.append(result.probability)
                            results_data.append(
                                {
                                    "Pick": f"Pick {pick.pick_number}",
                                    "Player": pick.player,
                                    "Stat": pick.stat_type,
                                    "Threshold": (
                                        f"{pick.prediction} {pick.threshold}"
                                    ),
                                    "Probability": f"{result.probability:.1%}",
                                    "Hit Rate": (
                                        f"{result.games_met}/{result.total_games}"
                                    ),
                                    "Avg": f"{result.mean:.1f}",
                                }
                            )

                            # Store detailed results for game logs and visualization
                            detailed_results.append(
                                DetailedResult(
                                    pick=pick,
                                    result=result,
                                    games_df=games_df
                                )
                            )

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
                        help="Probability that all picks hit",
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
                            help="American odds format",
                        )

                # Show probability breakdown
                st.subheader("Probability Breakdown")
                st.write("For all picks to hit, each individual pick must succeed:")

                prob_breakdown = " × ".join(
                    [f"{p:.1%}" for p in individual_probabilities]
                )
                st.write(f"**{prob_breakdown} = {combined_probability:.2%}**")

                # Visual representation - Individual probabilities
                prob_fig_data = pd.DataFrame({
                    'Pick': [f"Pick {i+1}" for i in range(len(individual_probabilities))],
                    'Probability': individual_probabilities
                })
                st.bar_chart(prob_fig_data.set_index('Pick'))
                
                st.divider()
                
                # Game-by-game grouped bar chart
                st.subheader("Game-by-Game Performance (All Players)")
                
                # Create grouped bar chart using plotly
                fig = go.Figure()
                
                if detailed_results:
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
                    
                    for idx, detail in enumerate(detailed_results):
                        pick = detail.pick
                        result = detail.result
                        games_df = detail.games_df
                        
                        # Get dates and stats - result.stat_values is already a list
                        game_dates = games_df['GAME_DATE'].tolist()
                        game_dates.reverse()  # Chronological order
                        stat_values = list(reversed(result.stat_values))
                        
                        # Create label for legend
                        label = f"{pick.player} - {pick.stat_type}"
                        
                        fig.add_trace(go.Bar(
                            name=label,
                            x=game_dates,
                            y=stat_values,
                            marker_color=colors[idx % len(colors)],
                            hovertemplate=f"{label}<br>Date: %{{x}}<br>Value: %{{y}}<extra></extra>"
                        ))
                    
                    fig.update_layout(
                        barmode='group',
                        xaxis_title="Game Date",
                        yaxis_title="Stat Value",
                        legend_title="Players & Stats",
                        height=500,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                st.divider()
                
                # Individual player game logs
                st.subheader("Detailed Game Logs")
                
                for detail in detailed_results:
                    pick = detail.pick
                    result = detail.result
                    games_df = detail.games_df
                    
                    with st.expander(f"📊 {pick.player} - {pick.stat_type} ({pick.prediction} {pick.threshold})"):
                        # Create game log dataframe
                        stat_values = pd.Series(result.stat_values)
                        
                        # Determine if threshold was met
                        if pick.prediction == "More":
                            met_threshold = stat_values > pick.threshold
                        else:
                            met_threshold = stat_values < pick.threshold
                        
                        # Create display dataframe
                        display_df = pd.DataFrame({
                            'Game Date': games_df['GAME_DATE'].values,
                            'Matchup': games_df['MATCHUP'].values,
                            'Pick Type': pick.stat_type,
                            'Value': stat_values.round(1),
                            'Threshold': f"{pick.prediction} {pick.threshold}",
                            'Hit': met_threshold.map({True: '✅', False: '❌'})
                        })
                        
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                        
                        # Individual player line chart
                        st.write("**Game-by-Game Trend**")
                        
                        # Create line chart with threshold line
                        chart_df = display_df[['Game Date', 'Value']].copy()
                        chart_df = chart_df.iloc[::-1]  # Reverse for chronological order
                        
                        individual_fig = go.Figure()
                        
                        # Add stat values line
                        individual_fig.add_trace(go.Scatter(
                            x=chart_df['Game Date'],
                            y=chart_df['Value'],
                            mode='lines+markers',
                            name=pick.stat_type,
                            line=dict(color='blue', width=2),
                            marker=dict(size=8)
                        ))
                        
                        # Add threshold line
                        individual_fig.add_trace(go.Scatter(
                            x=chart_df['Game Date'],
                            y=[pick.threshold] * len(chart_df),
                            mode='lines',
                            name=f"Threshold ({pick.threshold})",
                            line=dict(color='red', width=2, dash='dash')
                        ))
                        
                        individual_fig.update_layout(
                            xaxis_title="Game Date",
                            yaxis_title=pick.stat_type,
                            height=300,
                            showlegend=True
                        )
                        
                        st.plotly_chart(individual_fig, use_container_width=True)
                        
                        # Stats summary - access Pydantic attributes with dot notation
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Mean", f"{result.mean:.1f}")
                        with col2:
                            st.metric("Median", f"{result.median:.1f}")
                        with col3:
                            st.metric("Min", f"{result.min:.0f}")
                        with col4:
                            st.metric("Max", f"{result.max:.0f}")
        else:
            st.error("Could not calculate probabilities for any picks.")