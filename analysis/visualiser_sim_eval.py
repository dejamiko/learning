import os

import dash
import flask
import pandas as pd
import plotly.express as px
from dash import dcc, html
from dash.dependencies import Input, Output

from config import Config
from playground.environment import Environment
from tm_utils import NNImageEmbeddings, NNSimilarityMeasure, ImagePreprocessing, ImageEmbeddings, SimilarityMeasure


def prepare_data():
    config = Config()
    config.OBJ_NUM = 51
    # best overall
    config.IMAGE_EMBEDDINGS = NNImageEmbeddings.SIAMESE
    config.SIMILARITY_MEASURE = NNSimilarityMeasure.TRAINED
    config.IMAGE_PREPROCESSING = [
        ImagePreprocessing.CROPPING,
        ImagePreprocessing.BACKGROUND_REM,
        ImagePreprocessing.GREYSCALE,
    ]
    config.USE_ALL_IMAGES = True
    # best non-siamese
    # config.IMAGE_EMBEDDINGS = ImageEmbeddings.DINO_FULL
    # config.SIMILARITY_MEASURE = SimilarityMeasure.COSINE
    # config.IMAGE_PREPROCESSING = []
    # config.USE_ALL_IMAGES = False
    # config.IMAGE_EMBEDDINGS = ImageEmbeddings.DINO_FULL
    # original metric
    # config.SIMILARITY_MEASURE = SimilarityMeasure.COSINE
    # config.IMAGE_PREPROCESSING = [
    #     ImagePreprocessing.CROPPING,
    #     ImagePreprocessing.BACKGROUND_REM,
    # ]
    # config.USE_ALL_IMAGES = True
    env = Environment(config)
    vis_sims = []
    latent_sims = []
    image_pairs = []
    object_names_1 = []
    object_names_2 = []
    tasks = []
    for i in range(config.OBJ_NUM):
        for j in range(config.OBJ_NUM):
            obj_i = env.get_objects()[i]
            obj_j = env.get_objects()[j]
            if obj_i.task != obj_j.task:
                continue
            vis_sims.append(env.storage.get_visual_similarity(i, j))
            latent_sims.append(env.storage._latent_similarities[i, j])
            image_pairs.append(
                (
                    os.path.join(obj_i.image_path, "image_0.png"),
                    os.path.join(obj_j.image_path, "image_0.png"),
                )
            )
            object_names_1.append(obj_i.image_path.split("/")[1])
            object_names_2.append(obj_j.image_path.split("/")[1])
            tasks.append(obj_i.task.value)

    data = {
        "visual_similarity": vis_sims,
        "transfer_success_rate": latent_sims,
        "image_pair_path": image_pairs,
        "object_name_1": object_names_1,
        "object_name_2": object_names_2,
        "task": tasks,
        "name": [f"{n1}, {n2}" for n1, n2 in zip(object_names_1, object_names_2)],
    }
    return pd.DataFrame(data), os.getcwd()


def get_relative_data(df):
    df["sum_score"] = df.groupby("object_name_1")["transfer_success_rate"].transform(
        "sum"
    )
    df["transfer_success_rate"] = df["transfer_success_rate"] / df["sum_score"]
    df = df.drop(columns=["sum_score"])

    return df


def get_dist_from_mean(df):
    df["mean_score"] = df.groupby("object_name_1")["transfer_success_rate"].transform(
        "mean"
    )
    df["transfer_success_rate"] = df["transfer_success_rate"] - df["mean_score"]
    df = df.drop(columns=["mean_score"])

    return df


def create_fig(df):
    fig = px.scatter(
        df,
        x="visual_similarity",
        y="transfer_success_rate",
        hover_name="name",
        trendline="ols",
    )
    fig.update_traces(marker=dict(size=10))
    return fig


if __name__ == "__main__":
    df, image_folder = prepare_data()

    # df = get_dist_from_mean(df)

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
                        id="object-name-1",
                        style={
                            "textAlign": "center",
                            "marginBottom": "10px",
                            "marginRight": "15px",
                        },
                    ),
                    html.Div(
                        id="object-name-2",
                        style={
                            "textAlign": "center",
                            "marginBottom": "10px",
                            "marginLeft": "15px",
                        },
                    ),
                ],
                style={"display": "flex", "justifyContent": "center"},
            ),
            html.Div(
                [
                    html.Img(id="displayed-image-1", style={"display": "none"}),
                    html.Img(id="displayed-image-2", style={"display": "none"}),
                ],
                style={"display": "flex", "justifyContent": "center"},
            ),
            html.Div(
                id="visual-similarity",
                style={"textAlign": "center", "marginTop": "10px"},
            ),
            html.Div(
                id="transfer-success-rate",
                style={"textAlign": "center", "marginTop": "10px"},
            ),
        ]
    )

    # Flask route to serve images
    @server.route("/images/<path:image_path>")
    def serve_image(image_path):
        return flask.send_from_directory(image_folder, image_path)

    @app.callback(
        Output("scatter-plot", "figure"),
        Input("task-selector", "value"),
    )
    def update_scatter_plot(selected_task):
        filtered_df = df[df["task"] == selected_task]
        return create_fig(filtered_df)

    # Callback to display the image when a point is clicked
    @app.callback(
        [
            Output("displayed-image-1", "src"),
            Output("displayed-image-1", "style"),
            Output("displayed-image-2", "src"),
            Output("displayed-image-2", "style"),
            Output("object-name-1", "children"),
            Output("object-name-2", "children"),
            Output("visual-similarity", "children"),
            Output("transfer-success-rate", "children"),
        ],
        Input("scatter-plot", "clickData"),
    )
    def display_image(click_data):
        if click_data:
            point_name = click_data["points"][0]["hovertext"]
            point_info = df[df["name"] == point_name]
            image_pair_path = point_info["image_pair_path"].iat[0]
            image1_path, image2_path = image_pair_path
            image_1_url = f"/images/{image1_path}"
            image_2_url = f"/images/{image2_path}"
            object_name_1 = point_info["object_name_1"].iat[0]
            object_name_2 = point_info["object_name_2"].iat[0]
            visual_similarity = point_info["visual_similarity"].iat[0]
            transfer_success_rate = point_info["transfer_success_rate"].iat[0]
            return (
                image_1_url,
                {"display": "inline-block", "width": "20%", "marginRight": "10px"},
                image_2_url,
                {"display": "inline-block", "width": "20%", "marginLeft": "10px"},
                object_name_1,
                object_name_2,
                f"Visual similarity: {visual_similarity}",
                f"Transfer success rate: {transfer_success_rate}",
            )
        return "", {"display": "none"}, "", {"display": "none"}, "", "", "", ""

    app.run_server(debug=True)
