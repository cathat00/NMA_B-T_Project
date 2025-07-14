from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os


# ====================
# == SAVE FIGS UTIL ==
# ====================

def save_figs(figs, filepath, overwrite=False):

  # -- Check if file already exists
  if os.path.exists(filepath):
    if overwrite:
      # File exists, overwrite it
      os.remove(filepath)
    else:
      # File exists, don't overwrite
      msg = f"{filepath} already exists. Overwrite set to False in call to save_figs."
      raise Exception(msg)

  # -- Write to HTML file
  with open(filepath, 'a') as f:
    for fig in all_figs:
      f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))


# =================================
# == MANIFOLD / PCA SUMMARY PLOT ==
# =================================

def plot_pca_summary(manifold, eigenvalues, eigenval_thresh=0.9):

    # -- Get time component from manifold 
    t = np.arange(manifold.shape[0])

    # -- Calculate variance explained
    explained_variance = eigenvalues / np.sum(eigenvalues)
    cumulative_variance = np.cumsum(explained_variance)
    
    # -- Initialize subplots
    suplot_names = ['Activity Projected Onto Manifold', 'Variance Explained Plot']
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'xy'}]],
        subplot_titles=suplot_names,
    )

    # -- 3D manifold trace
    fig.add_trace(go.Scatter3d(
        x=manifold[:, 0],
        y=manifold[:, 1],
        z=manifold[:, 2],
        mode='markers',
        marker=dict(
            size=1,
            color=t,
            colorscale='Viridis',
            colorbar=dict(title='Time')
        ),
        showlegend=False,
    ), row=1, col=1)

    # -- Cumulative variance line trace
    fig.add_trace(go.Scatter(
        x=np.arange(1, len(eigenvalues) + 1),
        y=cumulative_variance,
        mode='lines+markers',
        name='Cumulative Variance',
        line=dict(color='blue'),
        showlegend=False,
    ), row=1, col=2)

    # -- Eigenvaue threshold line
    fig.add_trace(go.Scatter(
        x=[1, len(eigenvalues)],
        y=[eigenval_thresh, eigenval_thresh],
        mode='lines',
        name=f'{int(eigenval_thresh * 100)}% Threshold',
        line=dict(dash='dash', color='gray'),
        showlegend=False,
    ), row=1, col=2)

    # -- Update master plot layout with proper axis titles
    fig.update_layout(
        height=500,
        title_text="Intrinsic Dimensionality Analysis",
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3'
        ),
        xaxis=dict(title='Principal Component Number'),
        yaxis=dict(title='Cumulative Variance Explained', range=[0, 1.05]) 
    )

    return fig
