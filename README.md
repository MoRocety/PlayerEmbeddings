# PlayerEmbeddings

## Overview
This project provides an interactive Streamlit web application for exploring and comparing player embeddings derived from match event data. The embeddings are generated using a transformer-based text embedding model, treating each event as a sentence and applying mean pooling to create 768-dimensional player representations.

### Key Features:
- **Player Similarity Search**: Find players with similar playing styles using cosine similarity and Euclidean distance.
- **Interactive UMAP Visualization**: Reduce embeddings to 2D using UMAP and visualize them with Plotly.
- **Dynamic Filtering & Highlighting**: Explore players based on position, team, and individual selection.

## Data Processing Pipeline
1. **Data Collection**:
   - Player match data is obtained via the StatsBomb API.
2. **Embedding Generation**:
   - A transformer-based model processes match events into text embeddings.
   - Mean pooling is applied to obtain 768-dimensional player embeddings.
3. **Dimensionality Reduction**:
   - UMAP (Uniform Manifold Approximation and Projection) reduces embeddings to 2D for visualization.
4. **Similarity Computation**:
   - Cosine similarity and Euclidean distance are used to compare players.

## Installation
Ensure you have Python 3.8+ installed. Then, clone this repository and install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application
To start the Streamlit app, run:

```bash
streamlit run smlit.py
```

## Usage
### 1. Player Similarity Search
- Select a player to find similar ones based on cosine similarity or Euclidean distance.
- View a ranked list of the closest players.

### 2. UMAP Visualization
- Explore player embeddings in a 2D space.
- Color the points based on position or team.
- Highlight specific players to examine their relationships.

## Dependencies
- `streamlit`
- `pandas`
- `numpy`
- `scikit-learn`
- `umap-learn`
- `plotly`

## Future Enhancements
- Incorporate more advanced embedding techniques.
- Support real-time updates with new match data.
- Extend filtering and analysis capabilities.

## License
This project is open-source under the MIT License.

