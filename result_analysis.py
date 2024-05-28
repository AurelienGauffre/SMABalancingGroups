
import wandb
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import plotly.graph_objects as go

DATASETS = {
    'a': 'dogs',
    'b': 'medical-leaf',
    'c': 'texture-dtd',
    'd': 'birds',
    'e': 'AWA',
    'f': 'plt-net',
    'g': 'resisc',
    'h': 'plt-doc',
    'i': 'airplanes',
    '_': 'ALL'
}

COLOR_DICT = {
    'erm': 'blue',
    'jtt': 'red',
    'suby': '#00CC96',  # green
    'subg': '#ff7f0e',  # orange
    'rwy': '#00CC96',
    'rwg': '#ff7f0e',
    'dro': '#DEA0FD'  # purple
}


def get_df():
    # Initialize wandb
    wandb.login()

    # Set your entity and project
    entity_name = "aureliengauffre"  # e.g., your username or team name
    project_name = "SMA_all_2_best"

    # Initialize the wandb API ||client
    api = wandb.Api()

    # Fetch all runs from the specified project
    runs = api.runs(f"{entity_name}/{project_name}")

    # Create an empty list to hold data for each run
    data = []

    # Loop through runs and ext/ract the data you're interested in
    for run in runs:
        # Extract both summary metrics and config (hyperparameters) for each run
        run_data = {
            "name": run.name,
            "summary_metrics": run.summary._json_dict,
            "config": run.config,
            # Add any other attributes you're interested in here
        }
        data.append(run_data)

    # Convert the list of data to a pandas DataFrame
    df = pd.DataFrame(data)

    # For summary metrics and config (hyperparameters), expand them into separate columns
    df_summary = pd.json_normalize(df['summary_metrics'])
    df_config = pd.json_normalize(df['config'])
    df = pd.concat([df.drop(['summary_metrics', 'config'],
                   axis=1), df_summary, df_config], axis=1)
    # Drop the first column containing the name of the runs (since there is also another name column)
    df = df.iloc[:, 1:]
    return df
    # Now we have a DataFrame `df` with all runs, their summary metrics, and hyperparameters
    # print(df.head())  # Print the first few rows of the DataFrame


def plot_rank(df, x_axis, y_axis, dataset_name, error_bars=None, normalize=False, methods=None):
    print(dataset_name)
    if methods is None or methods == 'ALL':
        methods = ['erm', 'jtt', 'suby', 'subg', 'rwy', 'rwg', 'dro']
    dashed_methods = ['rwy', 'rwg', 'dro']

    if dataset_name is None or dataset_name == 'ALL':
        df_dataset = df
    else:
        df_dataset = df[(df['name'] == dataset_name)]

    if normalize:
        df_dataset = normalize_df(df_dataset, metric=y_axis)

    fig = go.Figure()  # Initialize a Plotly figure
    all_stats = pd.DataFrame()

    # First, calculate means for all methods and combine them
    for method in methods:
        df_method = df_dataset[df_dataset['method'] == method]
        stats = df_method.groupby(x_axis)[y_axis].mean().reset_index()
        stats['method'] = method
        all_stats = pd.concat([all_stats, stats], ignore_index=True)

    # Now rank the combined data
    all_stats['rank'] = all_stats.groupby(x_axis)[y_axis].rank(ascending=False)

    # Plot each method's data
    for method in methods:
        stats = all_stats[all_stats['method'] == method]
        line_style = 'dash' if method in dashed_methods else 'solid'
        color = COLOR_DICT.get(method, 'grey')
        fig.add_trace(go.Scatter(x=stats[x_axis], y=stats['rank'], mode='lines+markers',
                                 name=method, line=dict(dash=line_style, color=color)))

    # Update the layout
    fig.update_layout(title=f'Rank of {y_axis} vs {x_axis}, dataset={dataset_name}',
                      xaxis_title=x_axis,
                      yaxis_title='Rank',
                      legend_title='Method',
                      # Reverse the y-axis to show the best rank at the top
                      yaxis=dict(autorange='reversed'),
                      width=800,  # Width of the figure in pixels
                      height=800  # Height of the figure in pixels
                      )

    # Show the figure
    fig.show()
    return all_stats.pivot_table(index=[x_axis], columns='method', values='rank', aggfunc='first')


def normalize_df(df, metric='best_acc_te'):
    # Copy the DataFrame to avoid modifying the original data
    result_df = df.copy()

    # Group by 'name', 'K', 'mu', and 'init_seed' and calculate the minimum and maximum accuracy
    grouped = df.groupby(['name', 'K', 'mu', 'init_seed'])[metric]
    min_acc = grouped.transform(np.min)
    max_acc = grouped.transform(np.max)

    # Apply the Min-Max normalization formula
    result_df[metric] = (df[metric] - min_acc) / (max_acc - min_acc)

    return result_df


def plot_graph(df, x_axis, y_axis, dataset_name, error_bars=None, normalize=False, methods=None):
    print(dataset_name)
    """
    Plot a line graph of the mean of y_axis vs x_axis for each method in methods.
    """
    if methods is None or methods == 'ALL':
        methods = ['erm', 'jtt', 'suby', 'subg', 'rwy', 'rwg', 'dro']
    dashed_methods = ['rwy', 'rwg', 'dro']

    if dataset_name is None or dataset_name == 'ALL':
        df_dataset = df
    else:
        df_dataset = df[(df['name'] == dataset_name)]

    if normalize:
        df_dataset = normalize_df(df_dataset, metric=y_axis)

    fig = go.Figure()  # Initialize a Plotly figure
    for method in methods:
        df_method = df_dataset[df_dataset['method'] == method]
        # Group by x_axis and calculate the mean, standard deviation, and count (for standard error calculation)
        stats = df_method.groupby(x_axis)[y_axis].agg(
            ['mean', 'std', 'count']).reset_index()

        # Calculate standard error (SEM)
        stats['sem'] = stats['std'] / np.sqrt(stats['count'])
        # print(method, stats) # print the number on which we average, interesting !
        # Determine line style based on whether method is in dashed_methods
        line_style = 'dash' if method in dashed_methods else 'solid'

        # Add a line with error bars to the plot for the current method
        color = COLOR_DICT.get(method, 'grey')
        if error_bars:
            fig.add_trace(go.Scatter(x=stats[x_axis], y=stats['mean'], mode='lines+markers',
                                     name=method, line=dict(
                                         dash=line_style, color=color),
                                     error_y=dict(type='data', array=stats['sem'], visible=True)))
        else:
            fig.add_trace(go.Scatter(x=stats[x_axis], y=stats['mean'], mode='lines+markers',
                                     name=method, line=dict(dash=line_style, color=color)))

    # Update the layout
    fig.update_layout(title=f'{y_axis} vs {x_axis}, dataset={dataset_name}',
                      xaxis_title=x_axis,
                      yaxis_title=y_axis,
                      legend_title='Method',
                      width=800,  # Width of the figure in pixels
                      height=800  # Height of the figure in pixels
                      )

    # Show the figure

    fig.show()


def plot_graph_all(df, x_axis, y_axis, error_bars=None, normalize=False, methods=None):
    """Unlike the original plot_graph function which calculates and plots error bars
    based on individual datasets directly, this version computes the standard error within each dataset first 
    and then averages these errors across all datasets for each method. This approach provides a generalized 
    view of method performance and variability across different datasets."""

    if methods is None or methods == 'ALL':
        methods = ['erm', 'jtt', 'suby', 'subg', 'rwy', 'rwg', 'dro']
    dashed_methods = ['rwy', 'rwg', 'dro']

    if normalize:
        df = normalize_df(df, metric=y_axis)

    fig = go.Figure()  # Initialize a Plotly figure
    all_stats = pd.DataFrame()

    # Process each method separately
    for method in methods:
        df_method = df[df['method'] == method]

        # Group data first by 'name' and then by x_axis and compute statistics
        grouped = df_method.groupby(['name', x_axis])
        stats = grouped[y_axis].agg(['mean', 'std', 'count']).reset_index()

        # Calculate standard error within each dataset
        stats['sem'] = stats['std'] / np.sqrt(stats['count'])

        # Append results to the all_stats DataFrame for later display
        stats['method'] = method
        all_stats = pd.concat([all_stats, stats], ignore_index=True)

        # Now group by x_axis and calculate the mean of the means and the mean of the SEMs
        final_stats = stats.groupby(x_axis).agg(
            {'mean': 'mean', 'sem': 'mean'}).reset_index()

        # Determine line style based on whether method is in dashed_methods
        line_style = 'dash' if method in dashed_methods else 'solid'

        # Add a line with error bars to the plot for the current method
        color = COLOR_DICT.get(method, 'grey')
        if error_bars:
            fig.add_trace(go.Scatter(x=final_stats[x_axis], y=final_stats['mean'], mode='lines+markers',
                                     name=method, line=dict(
                                         dash=line_style, color=color),
                                     error_y=dict(type='data', array=final_stats['sem'], visible=True)))
        else:
            fig.add_trace(go.Scatter(x=final_stats[x_axis], y=final_stats['mean'], mode='lines+markers',
                                     name=method, line=dict(dash=line_style, color=color)))

    # Update the layout
    fig.update_layout(title=f'{y_axis} vs {x_axis}, Average on all dataset (mean and std)',
                      xaxis_title=x_axis,
                      yaxis_title=y_axis,
                      legend_title='Method',
                      width=800,  # Width of the figure in pixels
                      height=800  # Height of the figure in pixels
                      )

    # Show the figure
    fig.show()

    #Display the all_stats DataFrame
    return all_stats.pivot_table(index=[x_axis, 'name'], columns='method', values=['count', 'sem'], aggfunc='first')



import plotly.graph_objects as go
import pandas as pd
import numpy as np

def normalize_within_dataset(df, y_axis,mode=None):
    """Normalize the performance within each dataset."""
    df_normalized = df.copy()
    #if the last letter of mode is not K
    if mode[-1] != 'K':
        for name in df['name'].unique():
            mask = df['name'] == name
            if mode == 'min-max':
                df_normalized.loc[mask, y_axis] = (df[mask][y_axis] - df[mask][y_axis].min()) / (df[mask][y_axis].max() - df[mask][y_axis].min())
            elif mode == 'z-score':
                df_normalized.loc[mask, y_axis] = (df[mask][y_axis] - df[mask][y_axis].mean()) / df[mask][y_axis].std()
            elif mode == 'max':
                df_normalized.loc[mask, y_axis] = (df[mask][y_axis] ) / (df[mask][y_axis].max())
            elif mode == 'mean':
                df_normalized.loc[mask, y_axis] = (df[mask][y_axis] ) / (df[mask][y_axis].mean())
    else :
        for name in df['name'].unique():
            for K in df['K'].unique():
                mask = (df['name'] == name) & (df['K'] == K)
                if mode == 'min-max-K':
                    df_normalized.loc[mask, y_axis] = (df[mask][y_axis] - df[mask][y_axis].min()) / (df[mask][y_axis].max() - df[mask][y_axis].min())
                elif mode == 'z-score-K':
                    df_normalized.loc[mask, y_axis] = (df[mask][y_axis] - df[mask][y_axis].mean()) / df[mask][y_axis].std()
                elif mode == 'max-K':
                    df_normalized.loc[mask, y_axis] = (df[mask][y_axis] ) / (df[mask][y_axis].max())
                elif mode == 'mean-K':
                    df_normalized.loc[mask, y_axis] = (df[mask][y_axis] ) / (df[mask][y_axis].mean())
                else :
                    raise ValueError(f"Invalid mode: {mode}")
    return df_normalized

def plot_bar(df, x_axis, y_axis, normalize_mode=False, methods=None, width=800, height=800):
    """Plots the performance of different methods across datasets and K values using bar plots."""
    
    if methods is None or methods == 'ALL':
        methods = ['erm', 'jtt', 'suby', 'subg', 'rwy', 'rwg', 'dro']

    if normalize_mode: #normalize_mode in ['min-max', 'z-score', 'max', 'min-max-K', 'z-score-K', 'max-K']
        df = normalize_within_dataset(df, y_axis,normalize_mode)

    fig = go.Figure()  # Initialize a Plotly figure
    all_stats = pd.DataFrame()

    # Define color palette
    COLOR_DICT = {
        'erm': 'blue', 'jtt': 'green', 'suby': 'red', 'subg': 'purple',
        'rwy': 'orange', 'rwg': 'brown', 'dro': 'pink'
    }

    for method in methods:
        df_method = df[df['method'] == method]

        # Group data by x_axis and compute statistics
        grouped = df_method.groupby([x_axis])
        stats = grouped[y_axis].agg(['mean', 'std', 'count']).reset_index()

        # Calculate standard error
        stats['sem'] = stats['std'] / np.sqrt(stats['count'])

        # Append results to the all_stats DataFrame for later display
        stats['method'] = method
        all_stats = pd.concat([all_stats, stats], ignore_index=True)

        # Add a bar for each value of x_axis
        fig.add_trace(go.Bar(
            x=stats[x_axis], 
            y=stats['mean'], 
            name=method,
            error_y=dict(type='data', array=stats['sem'], visible=True),
            marker_color=COLOR_DICT.get(method, 'grey')
        ))

    # Update the layout
    fig.update_layout(barmode='group',
                      title=f'{y_axis} vs {x_axis}, Method Performance',
                      xaxis_title=x_axis,
                      yaxis_title=y_axis,
                      legend_title='Method',
                      width=width,  # Width of the figure in pixels
                      height=height  # Height of the figure in pixels
                      )

    # Show the figure
    fig.show()

    # Display the all_stats DataFrame
    return all_stats.pivot_table(index=[x_axis], columns='method', values=['count', 'sem'], aggfunc='first')

# Example usage:
# Assuming `df` is your DataFrame with the necessary columns
# plot_bar_graph(df, x_axis='K', y_axis='worst_grp_acc_te', normalize=True)
