import streamlit as st
import pydeck as pdk
import pandas as pd

def create_hub_spoke_map(origin=None, destinations=None):
    """Create a hub and spoke map visualization"""
    
    # Sample data - replace with your own
    # origin = {
    #     'lat': 40.7128,  # New York City
    #     'lon': -74.0060,
    #     'name': 'New York (Origin)'
    # }
    
    # destinations = [
    #     {'lat': 34.0522, 'lon': -118.2437, 'name': 'Los Angeles'},
    #     {'lat': 41.8781, 'lon': -87.6298, 'name': 'Chicago'},
    #     {'lat': 25.7617, 'lon': -80.1918, 'name': 'Miami'},
    #     {'lat': 47.6062, 'lon': -122.3321, 'name': 'Seattle'},
    #     {'lat': 39.2904, 'lon': -76.6122, 'name': 'Baltimore'},
    #     {'lat': 32.7767, 'lon': -96.7970, 'name': 'Dallas'},
    #     {'lat': 42.3601, 'lon': -71.0589, 'name': 'Boston'}
    # ]
    
    # Create DataFrame for lines (arcs between origin and destinations)
    arc_data = []
    for dest in destinations:
        arc_data.append({
            'source_lat': origin['lat'],
            'source_lon': origin['lon'],
            'target_lat': dest['lat'],
            'target_lon': dest['lon'],
            'source_name': origin['name'],
            'target_name': dest['target_name']
        })
    
    arc_df = pd.DataFrame(arc_data)
    #print("Arc DataFrame columns:", arc_df.columns.tolist())
    
    # Create DataFrame for points (origin + destinations)
    points_data = [origin] + destinations
    points_df = pd.DataFrame(points_data)
    #print("Points DataFrame columns:", points_df.columns.tolist())
    
    
    # Create the pydeck chart
    st.pydeck_chart(pdk.Deck(
        map_style='road',
        initial_view_state=pdk.ViewState(
            latitude=origin['lat'],
            longitude=origin['lon'],
            zoom=1,
            pitch=50,
        ),
        layers=[
            # Arc layer for the lines
            pdk.Layer(
                'ArcLayer',
                data=arc_df,
                get_width=2,
                get_source_position=['source_lon', 'source_lat'],
                get_target_position=['target_lon', 'target_lat'],
                get_tilt=0,
                get_source_color=[255, 0, 0, 160],  # Red color
                get_target_color=[0, 128, 200, 160],  # Blue color
                pickable=True,
                auto_highlight=True,
            ),
            # Scatter plot layer for the points
            pdk.Layer(
                'ScatterplotLayer',
                data=points_df,
                get_position=['lon', 'lat'],
                get_color='[200, 30, 0, 160]',
                get_radius=50000,
                pickable=True,
            ),
        ],
        tooltip={
            'html': '<b>{source_name}</b> → <b>{target_name}</b>',
            'style': {
                'backgroundColor': 'steelblue',
                'color': 'white'
            }
        }
    ))
    
    

    



