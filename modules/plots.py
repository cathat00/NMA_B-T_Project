from plotly.subplots import make_subplots
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.colors as pc
import pandas as pd
import os


# ====================
# == SAVE FIGS UTIL ==
# ====================

def save_figs(figs, filepath, overwrite=False):

  # -- Create directory path if it doesn't already exist.
  dir_path = os.path.dirname(filepath)
  os.makedirs(dir_path, exist_ok=True)

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


# ======================
# == EXPERIMENT PLOTS ==
# ======================

def plot_cumulative_variance(eigenvalues, eigenval_thresh=0.9):
    explained_variance = eigenvalues / np.sum(eigenvalues)
    cumulative_variance = np.cumsum(explained_variance)

    df = pd.DataFrame({
        'Principal Component': np.arange(1, len(eigenvalues)+1),
        'Cumulative Variance Explained': cumulative_variance
    })

    fig = px.line(
        df,
        x='Principal Component',
        y='Cumulative Variance Explained',
        markers=True
    )

    fig.add_hline(
        y=eigenval_thresh,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"{int(eigenval_thresh*100)}% Threshold",
        annotation_position="top left"
    )

    fig.update_layout(
        yaxis_range=[0, 1.05],
        xaxis_title='Principal Component Number',
        yaxis_title='Cumulative Variance Explained'
    )

    return fig


def plot_traj(proj, targets_per_trial):
  """
  Plot the trajectories of the RNN's activity in each trial onto the neural manifold. 
  Two different views of the trajectories are plotted  together in a subplot: 
  1) Trajectories, colored by time
  2) Trajectories, colored by target

  Parameters:
  - proj: ndarray of shape (n_trials, n_tsteps, n_pcs)
      A list containing the projected activity for each trial across all timesteps.
  - targets_per_trial: list
      List of the target index for each trial. E.g., for 3 trials [0,1,5] would mean
      that the target index of trial 1 was 0, for trial 2 was 1, etc.
      
  """
  # -- Projected data shape should be (trials, time, n_pcs)
  n_trials, n_tsteps, n_pcs = proj.shape
  t = np.arange(n_tsteps)

  # -- Initialize colors corresponding to each target
  color_list = pc.qualitative.Plotly
    
  # -- Keep track of which targets we've already added to the legend
  targets_in_legend = set()

  # -- Initialize subplots
  fig = make_subplots(
      rows=1, cols=2,
      specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
      subplot_titles=[
          f'Manifold Trajectories (Colored by Time)', 
          f'Manifold Trajectories (Colored by Target)',
      ],
  )

  for trial_idx, target_idx in enumerate(targets_per_trial):

    color = color_list[target_idx % len(color_list)]
    label = f'Target {target_idx}'

    show_legend = target_idx not in targets_in_legend
    if show_legend:
        targets_in_legend.add(target_idx)

    # -- Add trace from each manifold trial, colored by time
    fig.add_trace(go.Scatter3d(
        x=proj[trial_idx,:, 0],
        y=proj[trial_idx,:, 1],
        z=proj[trial_idx,:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=t,
            colorscale='Viridis',
            #colorbar=dict(title='Time'),
            colorbar=dict(
                x=0.5,
                thickness=15,
                title='Time'
            )        
        ),
        legendgroup=label,
        name=label,
        showlegend=False,
    ), row=1, col=1)

    # -- Add trace from each manifold trial, colored by target
    fig.add_trace(go.Scatter3d(
        x=proj[trial_idx,:,0],
        y=proj[trial_idx,:,1],
        z=proj[trial_idx,:,2],
        mode='markers',
        marker=dict(size=2, color=color),
        legendgroup=label,
        name=label,
        showlegend=show_legend
    ), row=1, col=2)

  # -- Update master plot layout with proper axis titles
  fig.update_layout(
      scene=dict(
          xaxis_title='PC1',
          yaxis_title='PC2',
          zaxis_title='PC3'
      ),
      height=500 
  )

  return fig


# ==============================
# == EXPERIMENT SUMMARY PLOTS ==
# ==============================

def plot_num_pcs_vs_targets(num_targets_list, eigenval_list, eigenval_thresh=0.9):
    """
    Plot the number of principal components (PCs) needed 
    to reach a cumulative variance threshold as a function of the number of targets.

    Parameters:
    - num_targets_list: list of int
        A list containing the number of targets used in each task (e.g., [2, 3, 4, ...]).
    - eigenval_list: list of np.ndarray
        A list of 1D numpy arrays, where each array contains the PCA eigenvalues 
        (variance explained) for one task condition.
    - eigenval_thresh: float, optional (default=0.9)
        The cumulative variance threshold used to determine how many PCs are required 
        (e.g., 0.9 for 90% variance explained).
    """

    num_pcs_list = []

    for eigenvalues in eigenval_list:
        explained_variance = eigenvalues / np.sum(eigenvalues)
        cumulative_variance = np.cumsum(explained_variance)
        n_pcs = np.count_nonzero(cumulative_variance < eigenval_thresh) + 1
        num_pcs_list.append(n_pcs)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=num_targets_list,
        y=num_pcs_list,
        mode='lines+markers',
        line=dict(color='royalblue'),
        marker=dict(size=8),
        name=f"PCs ≥ {int(eigenval_thresh * 100)}% variance"
    ))

    fig.update_layout(
        title="Intrinsic Dimensionality vs. Number of Targets",
        xaxis_title="Number of Targets",
        yaxis_title=f"Number of PCs ≥ {int(eigenval_thresh*100)}% Variance",
        xaxis=dict(tickmode='linear'),
        template="plotly_white",
        height=500,
        width=600
    )

    return fig
