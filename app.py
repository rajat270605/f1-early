import streamlit as st
import plotly.graph_objects as go
import numpy as np

from data.fetcher import get_session_laps
from data.cleaner import clean_laps
from data.selector import get_available_races, get_pit_loss_delta
from data.features import engineer_features
from model.trainer import train_degradation_model
from simulation.undercut import run_undercut_simulation
from simulation.verdict import get_undercut_verdict

# Constants for themes
COLOR_SOFT = "#E8002D"
COLOR_MEDIUM = "#FFF200"
COLOR_HARD = "#FFFFFF"
COLOR_UNKNOWN = "#888888"
COLOR_GAP_LINE = "#00D2FF"
COLOR_PIT_LINE = "#FFF200"
COLOR_STAY_OUT = "#FF6B6B"
COLOR_PIT_NOW = "#51CF66"
BG_COLOR = "#1a1a1a"
FONT_COLOR = "#ffffff"

st.set_page_config(page_title="F1 Strategy Engineer", page_icon="🏎️", layout="wide")

st.title("🏎️ F1 Strategy Engineer")

st.sidebar.header("Race Selection")
# Season selectbox: years 2018 to 2025 descending. Do NOT include 2026.
selected_year = st.sidebar.selectbox("Season", list(range(2025, 2017, -1)))

with st.spinner("Fetching available races..."):
    try:
        races = get_available_races(selected_year)
    except Exception as e:
        st.sidebar.error(f"Error fetching schedule: {e}")
        races = []

if races:
    selected_race = st.sidebar.selectbox("Race", races)

    # Main data fetching
    with st.spinner("Loading race data..."):
        try:
            laps, session = get_session_laps(selected_year, selected_race)
            cleaned_laps = clean_laps(laps)
            featured_laps = engineer_features(cleaned_laps)
            pit_loss_delta = get_pit_loss_delta(session)
        except Exception as e:
            st.error(f"Failed to load or process race data: {e}")
            st.stop()
            
        try:
            model, feat_imp_dict, mae = train_degradation_model(featured_laps)
        except Exception as e:
            st.error(f"Model training failed: {e}")
            st.stop()

    # Get unique drivers
    drivers = sorted(cleaned_laps['Driver'].dropna().unique().tolist())
    
    if len(drivers) < 2:
        st.warning("Not enough drivers found in the data.")
        st.stop()

    st.sidebar.header("Driver Comparison")
    driver_a = st.sidebar.selectbox("Driver A", drivers, index=0)
    
    drivers_b = [d for d in drivers if d != driver_a]
    driver_b = st.sidebar.selectbox("Driver B", drivers_b, index=0)
    
    if driver_a:
        max_lap = int(cleaned_laps[cleaned_laps['Driver'] == driver_a]['LapNumber'].max())
        sim_start_lap = st.sidebar.slider(
            "Simulate undercut from lap",
            min_value=1,
            max_value=max_lap,
            value=1,
            step=1
        )
    else:
        sim_start_lap = 1
        
    if driver_a and driver_b:
        pit_compound = st.sidebar.selectbox(
            "Pit onto compound",
            options=['SOFT', 'MEDIUM', 'HARD'],
            index=2
        )
    else:
        pit_compound = 'HARD'
        
    st.sidebar.divider()
    st.sidebar.metric("Pit loss delta", f"{pit_loss_delta:.1f}s")
    st.sidebar.metric("Model MAE", f"{mae:.3f}s")

    # Main panel
    if driver_a and driver_b:
        a_clean_laps = featured_laps[featured_laps['Driver'] == driver_a]
        b_clean_laps = featured_laps[featured_laps['Driver'] == driver_b]
        
        if len(a_clean_laps) < 10 or len(b_clean_laps) < 10:
            st.warning(f"Insufficient data for {driver_a} or {driver_b} — try a different race or driver pairing")
            st.stop()

        with st.spinner("Running simulation..."):
            sim_results = run_undercut_simulation(featured_laps, model, driver_a, driver_b, pit_loss_delta, start_lap=sim_start_lap, pit_compound=pit_compound)
            
            if not sim_results:
                st.warning("Simulation returned no results")
                st.stop()
                
            verdict_str = get_undercut_verdict(sim_results, driver_a, driver_b)

        # Section 1: Tyre Degradation Chart
        st.subheader("Tyre Degradation")
        
        fig1 = go.Figure()
        
        def get_compound_color(compound):
            c = str(compound).upper()
            if c == 'SOFT': return COLOR_SOFT
            if c == 'MEDIUM': return COLOR_MEDIUM
            if c == 'HARD': return COLOR_HARD
            return COLOR_UNKNOWN
            
        for drv in [driver_a, driver_b]:
            drv_df = featured_laps[featured_laps['Driver'] == drv]
            stints = drv_df['StintNumber'].unique()
            
            for stint in stints:
                stint_df = drv_df[drv_df['StintNumber'] == stint].sort_values('LapNumber')
                if stint_df.empty: continue
                
                compound_val = stint_df['Compound'].iloc[0] if 'Compound' in stint_df.columns else 'UNKNOWN'
                color = get_compound_color(compound_val)
                
                symbol = "circle" if drv == driver_a else "diamond"
                line_dict = dict(width=1, color="#888888") if str(compound_val).upper() == 'HARD' else None
                
                fig1.add_trace(go.Scatter(
                    x=stint_df['LapNumber'],
                    y=stint_df['fuel_corrected_time'],
                    mode='markers',
                    marker=dict(size=6, color=color, symbol=symbol, line=line_dict),
                    name=f"{drv} — {compound_val} Stint {stint}"
                ))

        fig1.update_layout(
            xaxis_title="Lap",
            yaxis_title="Fuel-Corrected Lap Time (s)",
            plot_bgcolor=BG_COLOR,
            paper_bgcolor=BG_COLOR,
            font=dict(color=FONT_COLOR),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Section 2: Gap Evolution Chart
        st.subheader("Gap Evolution — Driver A vs Driver B")
        fig2 = go.Figure()
        
        # We also need gap logic here to be based on the simulation results. 
        # If sim_results only has laps >= sim_start_lap, the gap chart will only plot from sim_start_lap onwards.
        laps_x = [res['lap'] for res in sim_results if res['gap_to_driver_b'] is not None]
        gaps_y = [res['gap_to_driver_b'] for res in sim_results if res['gap_to_driver_b'] is not None]
        
        fig2.add_trace(go.Scatter(
            x=laps_x,
            y=gaps_y,
            mode='lines',
            line=dict(color=COLOR_GAP_LINE, width=2),
            name="Gap"
        ))
        
        fig2.add_hline(y=0, line_dash="dash", line_color="#888888")
        
        pit_laps_a = cleaned_laps[(cleaned_laps['Driver'] == driver_a) & (cleaned_laps['PitInTime'].notna())]['LapNumber'].tolist() if 'PitInTime' in cleaned_laps.columns else []
        pit_laps_b = cleaned_laps[(cleaned_laps['Driver'] == driver_b) & (cleaned_laps['PitInTime'].notna())]['LapNumber'].tolist() if 'PitInTime' in cleaned_laps.columns else []
        
        for p_lap in pit_laps_a:
            fig2.add_vline(x=p_lap, line_dash="dash", line_color=COLOR_PIT_LINE)
            fig2.add_annotation(x=p_lap, y=max(gaps_y) if gaps_y else 0, text=f"PIT {driver_a}", showarrow=False, yshift=10, font=dict(size=10, color=COLOR_PIT_LINE))
            
        for p_lap in pit_laps_b:
            fig2.add_vline(x=p_lap, line_dash="dash", line_color=COLOR_PIT_LINE)
            fig2.add_annotation(x=p_lap, y=max(gaps_y) if gaps_y else 0, text=f"PIT {driver_b}", showarrow=False, yshift=25, font=dict(size=10, color=COLOR_PIT_LINE))
            
        fig2.update_layout(
            xaxis_title="Lap",
            yaxis_title="Cumulative Gap (s)",
            plot_bgcolor=BG_COLOR,
            paper_bgcolor=BG_COLOR,
            font=dict(color=FONT_COLOR)
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Section 3: Undercut Simulation
        st.subheader(f"Undercut Window Simulation — Lap {sim_start_lap}")
        
        try:
            focus = next(r for r in sim_results if r['lap'] == sim_start_lap)
        except StopIteration:
            st.warning("No simulation data available for this lap — adjust the lap slider")
            focus = None
                
        if focus and len(focus.get('stay_out_lap_times', [])) > 0:
            drv_a_lap_df = a_clean_laps[a_clean_laps['LapNumber'] == sim_start_lap]
            current_compound = drv_a_lap_df['Compound'].iloc[0] if not drv_a_lap_df.empty and 'Compound' in drv_a_lap_df.columns else "Unknown"
            
            fig3 = go.Figure()
            
            num_points = len(focus['stay_out_lap_times'])
            laps_axis = list(range(sim_start_lap, sim_start_lap + num_points))
            
            fig3.add_trace(go.Scatter(
                x=laps_axis,
                y=focus['stay_out_lap_times'],
                mode='lines+markers',
                marker=dict(size=5),
                line=dict(color=COLOR_STAY_OUT),
                name=f"Stay Out ({driver_a} current compound)"
            ))
            
            fig3.add_trace(go.Scatter(
                x=laps_axis,
                y=focus['pit_now_lap_times'],
                mode='lines+markers',
                marker=dict(size=5),
                line=dict(color=COLOR_PIT_NOW),
                name=f"Pit Now ({pit_compound})"
            ))
            
            fig3.add_annotation(
                x=laps_axis[0],
                y=focus['pit_now_lap_times'][0],
                text=f"+{pit_loss_delta:.1f}s pit stop",
                showarrow=False,
                yshift=15,
                font=dict(size=10, color="#FFF200")
            )
            
            crossover = focus['crossover_lap']
            if crossover is not None:
                cross_x = sim_start_lap + crossover
                fig3.add_vline(x=cross_x, line_dash="dash", line_color="#FFF200", line_width=1.5)
                
                # Approximate max y to place annotation at top
                max_y = max(max(focus['stay_out_lap_times']), max(focus['pit_now_lap_times']))
                fig3.add_annotation(
                    x=cross_x,
                    y=max_y,
                    text=f"Crossover — Lap {cross_x}",
                    showarrow=False,
                    yshift=10,
                    font=dict(size=11, color="#FFF200")
                )
                
            fig3.update_layout(
                xaxis_title="Race Lap",
                yaxis_title="Predicted Lap Time (s)",
                plot_bgcolor=BG_COLOR,
                paper_bgcolor=BG_COLOR,
                font=dict(color=FONT_COLOR),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig3, use_container_width=True)
            
            window_open = focus['undercut_window_open']

            if window_open:
                st.warning(verdict_str)
            elif crossover is not None:
                st.info(f"Crossover detected at lap {sim_start_lap + crossover} — undercut becomes viable but not urgent. {verdict_str}")
            else:
                st.info(verdict_str)
