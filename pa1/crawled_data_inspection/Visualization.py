import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc


def get_n_colors(n, colorscale='Turbo', vmin=0.17, vmax=0.8):
    scale_range = np.linspace(vmin, vmax, n)
    return pc.sample_colorscale(colorscale, scale_range)


def visualize_clusters_interactive(points_2d, point_cluster_labels, titles, urls, ids, cluster_keywords):
    print("Loading clustering visualization")

    noise_ratio = sum(label == -1 for label in point_cluster_labels) / len(point_cluster_labels)

    df = pd.DataFrame(points_2d, columns=['x', 'y'])
    df['cluster'] = point_cluster_labels
    df['title'] = titles
    df['url'] = urls
    df['id'] = ids
    df['cluster'] = df['cluster'].astype(str)

    sorted_cluster_labels = sorted(
        df['cluster'].unique(),
        key=lambda x: int(x) if x.lstrip('-').isdigit() else float('inf')
    )

    # Assign colors
    n_clusters = sum(1 for c in sorted_cluster_labels if c != '-1')
    color_list = get_n_colors(n_clusters, colorscale='Turbo')
    cluster_colors = {}
    color_index = 0
    for cluster in sorted_cluster_labels:
        if cluster == '-1':
            cluster_colors[cluster] = 'black'
        else:
            cluster_colors[cluster] = color_list[color_index]
            color_index += 1

    # Axis range with padding
    x_min, x_max = df['x'].min(), df['x'].max()
    y_min, y_max = df['y'].min(), df['y'].max()
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05
    x_range = [x_min - x_pad, x_max + x_pad]
    y_range = [y_min - y_pad, y_max + y_pad]

    fig = go.Figure()
    label_positions = []

    for cluster_label in sorted_cluster_labels:
        cluster_df = df[df['cluster'] == cluster_label]
        x = cluster_df['x'].values
        y = cluster_df['y'].values

        is_outlier = cluster_label == '-1'
        point_size = 5 if is_outlier else 6
        point_opacity = 0.3 if is_outlier else 0.9
        point_color = cluster_colors[cluster_label]

        if cluster_keywords and cluster_label in cluster_keywords:
            keywords = cluster_keywords[cluster_label]
            if isinstance(keywords, list):
                legend_label = f"Cluster {cluster_label}: " + ", ".join(keywords)
            else:
                legend_label = str(keywords)
        else:
            legend_label = f"noise ({noise_ratio:.2%} of all points)"

        # Plot points
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='markers',
            name=legend_label,
            legendgroup=f"Cluster {cluster_label}",
            marker=dict(
                size=point_size,
                color=point_color,
                opacity=point_opacity
            ),
            customdata=np.stack((
                np.full(len(cluster_df), f"Cluster {cluster_label}"),
                cluster_df['title'],
                cluster_df['url'],
                cluster_df['id'],
            ), axis=-1),
            hovertemplate=(
                "Cluster: %{customdata[0]}<br>"
                "Title: %{customdata[1]}<br>"
                "ID: %{customdata[3]}<br>"
                "<a href='%{customdata[2]}' target='_blank'>Open article</a>"
                "<extra></extra>"
            ),
            showlegend=True
        ))

        if not is_outlier and len(cluster_df) >= 2:
            centroid_x = x.mean()
            centroid_y = y.mean()
            label_positions.append((centroid_x, centroid_y, cluster_label))

    for cx, cy, cluster_label in label_positions:
        offset = -(y_range[1] - y_range[0]) * 0.02
        fig.add_trace(go.Scatter(
            x=[cx],
            y=[cy + offset],
            mode='text',
            text=[f"{cluster_label}"],
            textposition='top center',
            textfont=dict(
                size=25,
                color='black',
                family='Verdana'
            ),
            showlegend=False,
            hoverinfo='skip',
            opacity=1.0,
            legendgroup=f"Cluster {cluster_label}"
        ))

    # Annotation
    fig.add_annotation(
        text=(
            "<b>How to use this visualization:</b><br>"
            "• Clicking on a cluster in the legend hides it, double clicking selects only the cluster. Double click again to show everything.<br>"
            "• Hover over points to see the article's title and click to open the URL.<br>"
            "• Each color represents a cluster of semantically similar articles. Black dots are outliers (noise)"
        ),
        xref="paper", yref="paper",
        x=0, y=-0.05,
        yanchor="top",
        xanchor="left",
        showarrow=False,
        align="left",
        font=dict(size=16),
        borderpad=10,
        bgcolor="rgba(255,255,255,0.9)",
    )

    fig.update_layout(
        title='UMAP projection of 24Ur.com article clusters',
        legend_title_text='Cluster Keyword Legend:',
        clickmode='event+select',
        xaxis=dict(fixedrange=True, range=x_range),
        yaxis=dict(fixedrange=True, range=y_range, scaleanchor='x', scaleratio=0.65),
        height=1200,
        width=2300,
        margin=dict(b=230),
    )

    fig.show()