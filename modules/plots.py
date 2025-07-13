import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px # For advanced plotting

# ====================
# == MANIFOLD PLOTS ==
# ====================
def create_manifold_plot(manifold):

  # Manifold is of shape (time, components)
  t = np.arange(manifold.shape[0])  # time or index

  # Create a DataFrame for plotting
  df = pd.DataFrame({
      "PC1": manifold[:, 0],
      "PC2": manifold[:, 1],
      "PC3": manifold[:, 2],
      "time": t
  })

  # Plot with Plotly
  fig = px.scatter_3d(
      df,
      x="PC1", y="PC2", z="PC3",
      color="time",
      color_continuous_scale="Viridis",
      title="Activity Projected Onto Manifold (3D PCA)"
  )
  fig.update_traces(marker=dict(size=1))
  return fig


def create_scree_plot(eigenvalues):
  df_eigenvals = pd.DataFrame({
      'Principal Component Number': np.arange(1, len(eigenvalues) + 1),
      'Eigenvalue': eigenvalues
  })

  fig = px.line(
      df_eigenvals,
      x='Principal Component Number',
      y='Eigenvalue',
      markers=True,
      title='Scree Plot'
  )

  fig.update_layout(xaxis_title='Principal Component Number', yaxis_title='Eigenvalue')
  return fig


def plot_manifold_and_scree(manifold_activity, eigenvalues):
  # Get the individual figures
  manifold_fig = create_manifold_plot(manifold_activity)
  scree_fig = create_scree_plot(eigenvalues)

  # Create subplots
  fig = make_subplots(
      rows=1, cols=2, 
      specs=[[{'type': 'scatter3d'}, {'type': 'scatter'}]],
      subplot_titles=['Activity Projected Onto Manifold', 'Scree Plot'])

  # Add traces from the manifold plot
  for trace in manifold_fig.data:
      fig.add_trace(trace, row=1, col=1)

  # Add traces from the scree plot
  for trace in scree_fig.data:
      fig.add_trace(trace, row=1, col=2)

  return fig
