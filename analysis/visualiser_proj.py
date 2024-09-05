import os

import dash
import flask
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output
from sklearn.manifold import MDS

from config import Config
from playground.environment import Environment
from tm_utils import ImageEmbeddings, SimilarityMeasure


def prepare_data():
    config = Config()
    config.OBJ_NUM = 51
    config.IMAGE_EMBEDDINGS = ImageEmbeddings.DINO_FULL
    config.SIMILARITY_MEASURE = SimilarityMeasure.COSINE
    config.IMAGE_PREPROCESSING = []
    env = Environment(config)
    latent_dist = np.ones((config.OBJ_NUM, config.OBJ_NUM))
    latent_sim = np.zeros((config.OBJ_NUM, config.OBJ_NUM))
    vis_dist = np.ones((config.OBJ_NUM, config.OBJ_NUM))
    object_names = []
    tasks = []
    images = []
    task_to_ind = {}
    closest_k = []
    similarities_k = []
    k = 5
    for i in range(config.OBJ_NUM):
        obj_i = env.get_objects()[i]
        if obj_i.task.value not in task_to_ind:
            task_to_ind[obj_i.task.value] = []
        task_to_ind[obj_i.task.value].append(i)
        images.append(os.path.join(obj_i.image_path, "image_0.png"))
        object_names.append(obj_i.image_path.split("/")[1])
        tasks.append(obj_i.task.value)

    object_names = np.array(object_names)

    for i in range(config.OBJ_NUM):
        obj_i = env.get_objects()[i]
        for j in range(config.OBJ_NUM):
            obj_j = env.get_objects()[j]
            if obj_i.task != obj_j.task:
                continue
            vis_dist[i, j] = 1 - env.storage._visual_similarities[i, j]
            latent_dist[i, j] = (
                2
                - (
                    env.storage._latent_similarities[i, j]
                    + env.storage._latent_similarities[j, i]
                )
            ) / 2
            latent_sim[i, j] = env.storage._latent_similarities[i, j]

        indices = np.argsort(latent_sim[i])[-k:][::-1]
        closest_k.append(object_names[indices])
        similarities_k.append(latent_sim[i, indices])

    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)

    embeddings_latent = []
    embeddings_vis = []

    for task, ind in task_to_ind.items():
        embeddings_latent.extend(mds.fit_transform(latent_dist[ind, :][:, ind]))
        embeddings_vis.extend(mds.fit_transform(vis_dist[ind, :][:, ind]))

    embeddings_latent = np.array(embeddings_latent)
    embeddings_vis = np.array(embeddings_vis)
    closest_k = np.array(closest_k)
    similarities_k = np.array(similarities_k)

    data = {
        "vis_dim_1": embeddings_vis[:, 0],
        "vis_dim_2": embeddings_vis[:, 1],
        "lat_dim_1": embeddings_latent[:, 0],
        "lat_dim_2": embeddings_latent[:, 1],
        "name": object_names,
        "task": tasks,
        "image": images,
    }

    for i in range(k):
        data[f"closest_{i}"] = closest_k[:, i]
        data[f"similarity_{i}"] = similarities_k[:, i]

    return pd.DataFrame(data), os.getcwd()


def create_fig(df, edges=None):
    fig = px.scatter(
        df,
        x="vis_dim_1",
        y="vis_dim_2",
        hover_name="name",
    )
    fig.update_traces(marker=dict(size=10))

    if edges is not None:
        for edge in edges:
            fig.add_trace(
                go.Scatter(
                    x=edge["x"],
                    y=edge["y"],
                    mode="lines",
                    line=dict(dash="dash", width=edge["width"], color="blue"),
                    showlegend=False,
                )
            )

    return fig


if __name__ == "__main__":
    df, image_folder = prepare_data()

    # Initialize the Dash app with a Flask server
    app = dash.Dash(__name__)
    server = app.server

    # Layout
    app.layout = html.Div(
        [
            dcc.Dropdown(
                id="task-selector",
                options=[
                    {"label": task, "value": task} for task in df["task"].unique()
                ],
                value=df["task"].unique()[0],  # Set default value to the first task
                style={"width": "50%", "margin": "auto"},
            ),
            dcc.Graph(id="scatter-plot"),
            html.Div(id="click-data"),
            html.Div(
                [
                    html.Div(
                        id="object-name",
                        style={
                            "textAlign": "center",
                            "marginBottom": "10px",
                            "marginRight": "15px",
                        },
                    ),
                ],
                style={"display": "flex", "justifyContent": "center"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Img(id="displayed-image", style={"display": "none"}),
                            html.Div(id="closest-objects-list"),
                        ],
                        style={
                            "display": "flex",
                            "justifyContent": "center",
                            "alignItems": "center",
                        },
                    ),
                ],
                style={"display": "flex", "justifyContent": "center"},
            ),
        ]
    )

    # Flask route to serve images
    @server.route("/images/<path:image_path>")
    def serve_image(image_path):
        return flask.send_from_directory(image_folder, image_path)

    @app.callback(
        Output("scatter-plot", "figure"),
        [Input("task-selector", "value"), Input("scatter-plot", "clickData")],
    )
    def update_scatter_plot(selected_task, click_data):
        filtered_df = df[df["task"] == selected_task]
        edges = []

        if click_data:
            point_name = click_data["points"][0]["hovertext"]
            point_info = filtered_df[filtered_df["name"] == point_name]
            if len(point_info) == 0:
                return create_fig(filtered_df, None)
            clicked_x = point_info["lat_dim_1"].iat[0]
            clicked_y = point_info["lat_dim_2"].iat[0]
            clicked_x_vis = point_info["vis_dim_1"].iat[0]
            clicked_y_vis = point_info["vis_dim_2"].iat[0]

            for _, row in filtered_df.iterrows():
                distance = np.sqrt(
                    (clicked_x - row["lat_dim_1"]) ** 2
                    + (clicked_y - row["lat_dim_2"]) ** 2
                )
                if distance != 0:  # Ignore self
                    edges.append(
                        {
                            "x": [clicked_x_vis, row["vis_dim_1"]],
                            "y": [clicked_y_vis, row["vis_dim_2"]],
                            "width": 0.5 / distance,
                        }
                    )

        return create_fig(filtered_df, edges)

    # Callback to display the image when a point is clicked
    @app.callback(
        [
            Output("displayed-image", "src"),
            Output("displayed-image", "style"),
            Output("object-name", "children"),
            Output("closest-objects-list", "children"),
        ],
        Input("scatter-plot", "clickData"),
    )
    def display_image(click_data):
        if click_data:
            point_name = click_data["points"][0]["hovertext"]
            point_info = df[df["name"] == point_name]
            image_path = point_info["image"].iat[0]
            image_url = f"/images/{image_path}"
            object_name = point_info["name"].iat[0]

            # Retrieve the precomputed closest neighbor indices
            closest_info = []
            for i in range(5):
                closest_name = point_info[f"closest_{i}"].iat[0]
                largest_similarity = point_info[f"similarity_{i}"].iat[0]
                closest_info.append(f"{closest_name}: {largest_similarity:.2f}")

            if closest_info:
                closest_objects = html.Div(
                    [
                        html.H4("Closest Objects", style={"marginLeft": "20px"}),
                        html.Ul([html.Li(item) for item in closest_info]),
                    ]
                )
            else:
                closest_objects = ""

            return (
                image_url,
                {"display": "inline-block", "width": "50%", "marginRight": "10px"},
                object_name,
                closest_objects,
            )
        return "", {"display": "none"}, "", ""

    app.run_server(debug=True)
