from plotly.subplots import make_subplots
import numpy as np
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
    for fig in figs:
      f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))


# ================
# == TASK PLOTS ==
# ================

def plot_reaching_targets(all_targets, dt=0.01):
    """
    Parameters:
    - target: np.array of shape (n_targets, tsteps, 2)
    - dt: time per timestep (e.g., 0.01s for 10ms)
    """
    n_targets, tsteps, _ = all_targets.shape
    time = np.arange(tsteps) * dt

    # Build long-form dataframe for Plotly
    data = {
        "time": [],
        "coord": [],
        "axis": [],
        "target": []
    }

    for target_idx in range(n_targets):
        for axis, name in enumerate(["x", "y"]):
            data["time"].extend(time)
            data["coord"].extend(all_targets[target_idx, :, axis])
            data["axis"].extend([name] * tsteps)
            data["target"].extend([target_idx] * tsteps)

    df = pd.DataFrame(data)

    fig = px.line(
        df,
        x="time",
        y="coord",
        color="axis",
        animation_frame="target",
        labels={"coord": "Target Coordinate (a.u.)", "time": "Time (s)"},
        title="Target X/Y Coordinates Over Time"
    )

    fig.update_layout(
        legend_title_text="Axis",
        yaxis_range=[all_targets.min() - 0.01, all_targets.max() + 0.01],
        transition={"duration": 0},
    )

    fig.show()


# ==================================
# == MANIFOLD / PCA SUMMARY PLOTS ==
# ==================================

def plot_pca_summary(proj, eigenvalues, eigenval_thresh=0.9):

    # -- Projected data shape should be (trials, time, n_pcs)
    n_trials, n_tsteps, n_pcs = proj.shape
    t = np.arange(n_tsteps)

    # -- Calculate variance explained
    explained_variance = eigenvalues / np.sum(eigenvalues)
    cumulative_variance = np.cumsum(explained_variance)
    total_under_thresh = np.count_nonzero(cumulative_variance < eigenval_thresh)
    
    # -- Initialize subplots
    suplot_names = [
        f'Activity Projected Onto Manifold', 
        f'Variance Explained Plot ({total_under_thresh} PCs below thresh)'
    ]
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'xy'}]],
        subplot_titles=suplot_names,
    )

    # -- Add trace from each manifold trial
    for trial_idx in range(n_trials):
      fig.add_trace(go.Scatter3d(
          x=proj[trial_idx,:, 0],
          y=proj[trial_idx,:, 1],
          z=proj[trial_idx,:, 2],
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
