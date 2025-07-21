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


# ==============================
# == VISUALIZATION BASE CLASS ==
# ==============================

class Visualization():

    def __init__(self):
        self.figures = []
        
    def save(self, filepath, **kwargs):
        save_figs(self.figures, filepath, **kwargs)


# ============================
# == MANIFOLD VISUALIZATION ==
# ============================

class ManifoldViz(Visualization):

    fig_height = 500
   
    def __init__(self, manifold_data, eigenval_thresh=0.9):
        super().__init__()
        
        # Extract data from manifold dictionary
        eigenvalues = manifold_data['ev']
        proj = manifold_data['xi2'] # Reshaped projection
        targets_by_trial = manifold_data['order'] # Target indices, by trial

        # Load all plots
        traj_plot = self._plot_traj(proj, targets_by_trial)
        cumvar_plot = self._plot_cumulative_var(eigenvalues, eigenval_thresh)
        
        # Store plots in self.figures
        self.figures = [traj_plot, cumvar_plot]
        
    def _plot_cumulative_var(self, eigenvalues, eigenval_thresh):
        """
        Given a list of eigenvalues, compute and plot the cumulative 
        variance explained.

        """
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
    
    def _plot_traj(self, proj, targets_by_trial):
        """
        Plot the trajectories of the RNN's activity in each trial 
        onto the neural manifold. Two different views of the trajectories 
        are plotted  together in a subplot: 
        1) Trajectories, colored by time
        2) Trajectories, colored by target

        Parameters:
        - proj: ndarray of shape (n_trials, n_tsteps, n_pcs)
            A list containing the projected activity for each trial across 
            all timesteps.
        - targets_by_trial: list
            List of the target index for each trial. E.g., for 3 trials [0,1,5] would 
            mean that the target index of trial 1 was 0, for trial 2 was 1, etc.
            
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

        for trial_idx, target_idx in enumerate(targets_by_trial):

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
            height=self.fig_height 
        )
        return fig


# ========================
# == TASK VISUALIZATION ==
# ========================

class TaskViz(Visualization):
  
    pad = 0.01 # Y-axis padding for coordinate plot
  

    def __init__(self, task, activity, bci, targets_by_trial, dt=0.01):
        super().__init__()

        # Set member variables
        self.dt = dt

        # Get prediction from activity and BCI
        pred = activity @ bci.transformer.T
        
        # Create figures
        targets = task.targets[...,task.stim_length:, :]
        coords_fig = self._plot_coordinates(pred, targets, targets_by_trial)
        reaching_fig = self._plot_reaching(pred, targets, activity)

        # Store figures in self.figures
        self.figures = [coords_fig, reaching_fig]
    

    def _plot_coordinates(self, predicted, targets, targets_by_trial):
        """
        Plot predicted vs. target coordinates over time.

        Parameters:
        - predicted: Predicted coordinates of shape (n_trials, tsteps, 2)
        - targets: Target coordinates of shape (n_trials, tsteps, 2)
        - targets_per_trial: list
            List of the target index for each trial. E.g., for 3 trials [0,1,5] would 
            mean that the target index of trial 1 was 0, for trial 2 was 1, etc.
        - dt: time per timestep (e.g., 0.01s for 10ms)

        """
        # Intialize timesteps
        n_trials, tsteps, _ = predicted.shape
        time = np.arange(tsteps) * self.dt

        # Build dataframe structure
        data = {
            "time": [],
            "coord": [],
            "axis": [],
            "trial": [],
            "is_prediction":[],
        }

        # Populate dictionary for dataframe conversion
        for trial_idx in range(n_trials):
        
            is_prediction = [True, False]

            # Index of the target used in this trial
            trial_targ_idx = targets_by_trial[trial_idx]

            # Predicted and actual target coordinates 
            # for this trial, through time.
            trial_pred = predicted[trial_idx,...]
            trial_target = targets[trial_targ_idx]

            # Fill the dictionary with 
            for is_pred, dataset in zip(is_prediction,[trial_pred, trial_target]):
                for axis, name in enumerate(["x", "y"]):
                    data["time"].extend(time)
                    data["coord"].extend(dataset[:,axis])
                    data["axis"].extend([name] * tsteps)
                    data["trial"].extend([trial_idx] * tsteps)
                    data["is_prediction"].extend([is_pred] * tsteps)

        # Convert dictionary to dataframe
        df = pd.DataFrame(data)

        # Create the plot using the dataframe
        fig = px.line(
            df,
            x="time",
            y="coord",
            color="axis",
            animation_frame="trial",
            facet_col="is_prediction",
            labels={"coord": "Target Coordinate (a.u.)", "time": "Time (s)"},
            title="Target X/Y Coordinates Over Time (by Trial)"
        )

        # Update figure layout
        fig.update_layout(
            legend_title_text="Axis",
            yaxis_range=[predicted.min() - self.pad, predicted.max() + self.pad],
            transition={"duration": 0},
        )

        # Customize facet titles
        fig.layout.annotations[0].text = "Predicted Coords."
        fig.layout.annotations[1].text = "Target Coords."

        # Add horizontal line at origin
        fig.add_hline(y=0,line_dash='dash',line_color="gray")

        return fig


    def _plot_reaching(self, pred, targets, activity):

        # Reconstruct trajectories from velocities
        pos_original = np.zeros(pred.shape)
        for j in range(activity.shape[1]):
            pos_original[:,j,:] = pos_original[:,j-1,:] + pred[:,j,:]*self.dt
        timesteps = pos_original.shape[1]
        n_trials = pos_original.shape[0]

        # Compute min/max for all x and y positions (for FOV)
        all_x = pos_original[:, :, 0].flatten()
        all_y = pos_original[:, :, 1].flatten()
        x_min, x_max = all_x.min(), all_x.max()
        y_min, y_max = all_y.min(), all_y.max()
        x_pad = (x_max - x_min) * 0.05
        y_pad = (y_max - y_min) * 0.05
        x_range = [x_min - x_pad, x_max + x_pad]
        y_range = [y_min - y_pad, y_max + y_pad]

        # Get the final (x, y) position for each target
        target_x = targets[:, -1, 0] * self.dt * timesteps
        target_y = targets[:, -1, 1] * self.dt * timesteps

        # Create a frame for each time step, showing all trajectories up to that time
        frames = []
        for t in range(timesteps):
            frame_data = []
            for trial in range(n_trials):
                frame_data.append(go.Scatter(
                    x=pos_original[trial, :t+1, 0],
                    y=pos_original[trial, :t+1, 1],
                    mode='lines',
                    name=f'Trial {trial+1}',
                    line=dict(width=2)
                ))
            frames.append(go.Frame(data=frame_data, name=str(t)))

        # Initial plot: all trajectories at time 0
        init_data = []
        for trial in range(n_trials):
            init_data.append(go.Scatter(
                x=[pos_original[trial, 0, 0]],
                y=[pos_original[trial, 0, 1]],
                mode='lines',
                name=f'Trial {trial+1}',
                line=dict(width=2)
            ))
        # Add target markers (static, not animated)
        init_data.append(go.Scatter(
            x=target_x,
            y=target_y,
            mode='text',
            text=['🐵'] * len(target_x),
            textfont=dict(size=24),
            name='Targets',
            showlegend=False

        ))

        fig = go.Figure(
            data=init_data,
            layout=go.Layout(
                xaxis=dict(
                    title='x-position on screen',
                    title_standoff=30,  # <-- This moves the label down
                    range=x_range
                ),
                yaxis=dict(
                    title='y-position on screen',
                    range=y_range
                ),
                title="Simulated Reaching (interactive, all targets)",
                updatemenus=[
                    dict(
                        type="buttons",
                        buttons=[
                            dict(label="Play", method="animate", args=[None, {"frame": {"duration": 30, "redraw": True}, "fromcurrent": True}]),
                            dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
                        ],
                        direction="left",
                        x=0,
                        xanchor="left",
                        y=1.15,
                        yanchor="top"
                    )
                ]
            ),
            frames=frames
        )

        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.5, y=1.13,  # adjust y to place above the plot
            text="🐵 = Target",
            showarrow=False,
            font=dict(size=18)
        )

        # Add a time slider
        fig.update_layout(
            sliders=[dict(
                steps=[
                    dict(
                        method='animate',
                        args=[[str(t)], dict(mode='immediate', frame=dict(duration=30, redraw=True), fromcurrent=True)],
                        label=f'{t}'
                    ) for t in range(timesteps)
                ],
                transition=dict(duration=0),
                x=0, y=-0.15, currentvalue=dict(font=dict(size=12), prefix="Time (ms): ", visible=True, xanchor='center')
            )]
        )

        return fig


# ======================================
# == EXPERIMENT SUMMARY VISUALIZATION ==
# ======================================

class ExperimentSummaryViz(Visualization):
    
    fig_height = 500
    
    def __init__(self, experiments, eigenvalue_list, 
                 eigenval_thresh=0.9, max_k=10, default_k=2):
        
        super().__init__()

        # Set member variables
        self.eigenval_thresh = eigenval_thresh
        self.max_k = max_k
        self.default_k = default_k

        # Create plots
        dim_vs_targs_fig = self._plot_dim_vs_targs(experiments, eigenvalue_list)
        dim_vs_targs_bar_fig = self._plot_dim_vs_targs_bar(experiments, eigenvalue_list)

        # Store plots in self.figures
        self.figures = [dim_vs_targs_fig, dim_vs_targs_bar_fig]
    
    def _plot_dim_vs_targs(self, experiments, eigenval_list):
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
            n_pcs = np.count_nonzero(cumulative_variance < self.eigenval_thresh) + 1
            num_pcs_list.append(n_pcs)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=experiments,
            y=num_pcs_list,
            mode='lines+markers',
            line=dict(color='royalblue'),
            marker=dict(size=8),
            name=f"PCs ≥ {int(self.eigenval_thresh * 100)}% variance"
        ))

        fig.update_layout(
            title="Intrinsic Dimensionality vs. Number of Targets",
            xaxis_title="Number of Targets",
            yaxis_title=f"Number of PCs ≥ {int(self.eigenval_thresh*100)}% Variance",
            xaxis=dict(tickmode='linear'),
            template="plotly_white",
            height=self.fig_height,
        )

        return fig

    def _plot_dim_vs_targs_bar(self, experiments, eigenval_list):
        """
        Plot the amount of variance explained by the top-k principal components 
        as a function of the number of targets.

        Parameters:
        - num_targets_list: list of int  
            A list containing the number of targets used in each task.
        - eigenval_list: list of np.ndarray  
            A list of 1D numpy arrays, where each array contains the PCA eigenvalues
            corresponding to a given task.
        - max_k: int, optional (default=5)  
            The maximum number of top PCs to consider in the slider.
        - default_k: int, optional (default=2)  
            The default number of PCs displayed when the plot is first rendered.

        """
        x_labels = [str(n) for n in experiments]
        traces = []
        steps = []

        for k in range(1, self.max_k + 1):
            y_vals = []
            for eigenvalues in eigenval_list:
                explained_variance = eigenvalues / np.sum(eigenvalues)
                var_k = np.sum(explained_variance[:k])
                y_vals.append(var_k)

            visible = (k == self.default_k)

            trace = go.Bar(
                x=x_labels,
                y=y_vals,
                name=f"Top {k} PCs",
                visible=visible,
                marker_color='royalblue',
                width=0.6,
            )
            traces.append(trace)

            step = dict(
                method="update",
                args=[
                    {"visible": [i == (k - 1) for i in range(self.max_k)]},
                    {"title": f"Variance Explained by Top {k} PCs vs. Number of Targets",
                    "yaxis": {"title": f"Variance Explained (Top {k} PCs)"}}
                ],
                label=str(k)
            )
            steps.append(step)

        sliders = [dict(
            active=self.default_k - 1,
            currentvalue={"prefix": "Top-k PCs: "},
            pad={"t": 50},
            steps=steps
        )]

        fig = go.Figure(data=traces)

        fig.update_layout(
            sliders=sliders,
            title=f"Variance Explained by Top {self.default_k} PCs vs. Number of Targets",
            xaxis_title="Number of Targets",
            yaxis_title=f"Variance Explained (Top {self.default_k} PCs)",
            yaxis=dict(range=[0, 1]),
            template='plotly_white',
            height=self.fig_height,
        )

        return fig
