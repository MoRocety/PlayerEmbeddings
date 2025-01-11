import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import umap.umap_ as umap
import plotly.express as px

# Load the DataFrame from the saved Parquet file
def load_embeddings():
    df = pd.read_parquet('player_embeddings_with_metadata.parquet')
    return df

# Function to create UMAP projection
def create_umap_projection(embeddings_df):
    """Create UMAP projection from embeddings"""
    # Stack all embeddings into a matrix
    embeddings_matrix = np.vstack(embeddings_df['embedding'].values)
    
    # Create and fit UMAP
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric='cosine',
        random_state=42
    )
    
    # Perform dimensionality reduction
    umap_embeddings = reducer.fit_transform(embeddings_matrix)
    
    # Create DataFrame with UMAP coordinates and set column names
    umap_df = pd.DataFrame(
        umap_embeddings, 
        columns=['UMAP1', 'UMAP2']  # Correct column names
    )
    
    # Add metadata
    umap_df['player_name'] = embeddings_df['player_name'].values
    umap_df['position'] = embeddings_df['position'].values
    umap_df['team'] = embeddings_df['team'].values
    
    return umap_df

# Function to find similar players using multiple distance metrics
def find_similar_players(player_name, embeddings_df, metric='cosine', top_n=10):
    # Find the player by name
    player_data = embeddings_df[embeddings_df['player_name'] == player_name]
    
    if player_data.empty:
        st.error(f"Player '{player_name}' not found.")
        return pd.DataFrame()
    
    # Get the embedding of the searched player
    player_embedding = player_data['embedding'].iloc[0]
    
    # Compute similarity/distance between the input player and all other players
    embeddings_matrix = np.vstack(embeddings_df['embedding'].values)
    
    if metric == 'cosine':
        similarity_scores = cosine_similarity([player_embedding], embeddings_matrix).flatten()
        similar_indices = similarity_scores.argsort()[-top_n-1:-1][::-1]
        similarity_values = similarity_scores[similar_indices]
        metric_name = 'cosine_similarity'
    else:  # euclidean
        distances = euclidean_distances([player_embedding], embeddings_matrix).flatten()
        similar_indices = distances.argsort()[1:top_n+1]
        similarity_values = distances[similar_indices]
        metric_name = 'euclidean_distance'
    
    similar_players = embeddings_df.iloc[similar_indices]
    similar_players[metric_name] = similarity_values
    
    return similar_players[['player_name', 'position', 'team', metric_name]]

# Streamlit app layout
st.title("Player Similarity Analysis")

# Load the embeddings dataset
embeddings_df = load_embeddings()

# Create UMAP projection
@st.cache_data
def get_umap_projection():
    return create_umap_projection(embeddings_df)

umap_df = get_umap_projection()

# Create tabs for different views
tab1, tab2 = st.tabs(["Player Similarity Search", "UMAP Visualization"])

with tab1:
    # Input for player name search with suggestions
    player_names = embeddings_df['player_name'].unique()
    
    col1, col2 = st.columns(2)
    
    with col1:
        player_name_input = st.selectbox("Select Player Name:", player_names)
    
    with col2:
        similarity_metric = st.radio(
            "Select Similarity Metric:",
            ('cosine', 'euclidean'),
            help="Cosine similarity measures angle between vectors (higher is more similar). Euclidean distance measures straight-line distance (lower is more similar)."
        )
    
    if player_name_input:
        similar_players = find_similar_players(player_name_input, embeddings_df, metric=similarity_metric)
        
        if not similar_players.empty:
            st.write(f"Most similar players to {player_name_input} using {similarity_metric} {'similarity' if similarity_metric == 'cosine' else 'distance'}:")
            st.dataframe(similar_players)

with tab2:
    # UMAP Visualization controls
    col1, col2 = st.columns(2)
    
    with col1:
        color_by = st.selectbox(
            "Color by:",
            ['position', 'team']
        )
    
    with col2:
        # Optional player highlight
        highlight_player = st.selectbox(
            "Highlight player (optional):",
            ['None'] + list(player_names)
        )
    
    fig = px.scatter(
        umap_df,
        x='UMAP1',
        y='UMAP2',
        color=color_by,
        hover_data=['player_name', 'position', 'team'],
        title=f'UMAP Projection of Player Embeddings (colored by {color_by})'
    )

    # If a player is selected to highlight
    if highlight_player != 'None':
        highlight_data = umap_df[umap_df['player_name'] == highlight_player]
        
        # Add a larger point for the highlighted player
        fig.add_trace(
            px.scatter(
                highlight_data,
                x='UMAP1',
                y='UMAP2',
                text='player_name'
            ).data[0]
        )
        
        # Update the new trace
        fig.data[-1].update(
            mode='markers+text',
            marker=dict(size=15, symbol='star', color='red'),
            textposition='top center',
            name=highlight_player
        )

    # Get the min and max for UMAP1 and UMAP2
    x_min, x_max = umap_df['UMAP1'].min(), umap_df['UMAP1'].max()
    y_min, y_max = umap_df['UMAP2'].min(), umap_df['UMAP2'].max()

    

    # Update layout
    fig.update_layout(
        height=700,
        template='plotly_white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.99
        )
    )
        
    st.plotly_chart(fig, use_container_width=True)

    
    # Add some explanatory text
    st.markdown("""
    ### About the UMAP Visualization
    
    This visualization shows a 2D projection of the high-dimensional player embeddings using UMAP (Uniform Manifold Approximation and Projection). 
    Players that are closer together in this visualization have more similar playing styles based on their match descriptions.
    
    - Use the 'Color by' dropdown to switch between coloring by position or team
    - Select a player in the 'Highlight player' dropdown to highlight them in the visualization
    - Hover over points to see player details
    """)