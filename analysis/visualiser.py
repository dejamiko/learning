# import numpy as np
# import pandas as pd
# import plotly.express as px
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# from umap import UMAP
#
# from config import Config
# from optim.mh import EvolutionaryStrategy
# from playground.environment import Environment
# from tm_utils import get_object_indices, VisualisationMethod as VM
#
#
# class Visualiser:
#     def __init__(self, config):
#         self.config = config
#
#     def visualise_selection(
#         self, all_objects, selected_indices, reachable_indices_to_selected
#     ):
#         all_objects_lower_dim = self._prepare_data(all_objects)
#
#         # there is color, size, symbol, and hover_data to play with
#         # create a dataframe with the data
#         df = pd.DataFrame(all_objects_lower_dim, columns=["x", "y"])
#         df["selected"] = "not selected"
#         df.loc[selected_indices, "selected"] = "selected"
#         df["reachable"] = "not reachable"
#         df.loc[reachable_indices_to_selected.keys(), "reachable"] = "reachable"
#
#         # make all points larger
#         fig = px.scatter(df, x="x", y="y", color="selected", symbol="reachable")
#         fig.update_traces(marker=dict(size=10))
#
#         # discretise the plot space
#         x_range = df["x"].max() - df["x"].min()
#         y_range = df["y"].max() - df["y"].min()
#
#         width = 1000
#         height = 500
#
#         x_step = x_range / width
#         y_step = y_range / height
#
#         # create a 2d array of the discretised space
#         discretised_space = np.zeros((width, height))
#
#         # populate it with points
#         for i, row in df.iterrows():
#             x = int((row["x"] - df["x"].min()) / x_step)
#             y = int((row["y"] - df["y"].min()) / y_step)
#             x = min(width - 1, max(0, x))
#             y = min(height - 1, max(0, y))
#             discretised_space[x, y] = 1
#
#         # add lines between reachable points and the corresponding selected point
#         for o in reachable_indices_to_selected.keys():
#             for s, sim in reachable_indices_to_selected[o]:
#                 if o == s:
#                     continue
#                 # display the similarity as a label on the line
#                 fig.add_shape(
#                     type="line",
#                     x0=df.loc[o, "x"],
#                     y0=df.loc[o, "y"],
#                     x1=df.loc[s, "x"],
#                     y1=df.loc[s, "y"],
#                     line=dict(color="black", width=1, dash="dash"),
#                 )
#
#                 # find a position for the label where it doesn't overlap with the other points
#                 orig_x = (df.loc[o, "x"] + df.loc[s, "x"]) / 2
#                 orig_y = (df.loc[o, "y"] + df.loc[s, "y"]) / 2
#
#                 # find the cell this point corresponds to
#                 x = int((orig_x - df["x"].min()) / x_step)
#                 y = int((orig_y - df["y"].min()) / y_step)
#                 x = min(width - 1, max(0, x))
#                 y = min(height - 1, max(0, y))
#
#                 # find the nearest point that is not occupied in the discretised space
#                 pairs = [(i, j) for i in range(-5, 6) for j in range(-5, 6)]
#                 pairs.sort(key=lambda x: x[0] ** 2 + x[1] ** 2)
#                 for i, j in pairs:
#                     if (
#                         width >= x + i >= 0
#                         and 0 == discretised_space[x + i, y + j]
#                         and 0 <= y + j <= height
#                     ):
#                         break
#
#                 # map the cell back to the original space
#                 orig_x += i * x_step
#                 orig_y += j * y_step
#                 fig.add_annotation(
#                     x=orig_x,
#                     y=orig_y,
#                     text=f"{sim:.2f}",
#                     showarrow=False,
#                     font=dict(size=10),
#                 )
#
#         # save the figure to a file
#         fig.write_html(f"visualisation_{self.config.VISUALISATION_METHOD}.html")
#
#     def _prepare_data(self, all_objects):
#         all_objects = [o.visible_repr for o in all_objects]
#         all_objects = np.array(all_objects)
#         if self.config.VISUALISATION_METHOD == VM.PCA:
#             pca = PCA(n_components=2)
#             pca.fit(all_objects)
#             all_objects_lower_dim = pca.transform(all_objects)
#         elif self.config.VISUALISATION_METHOD == VM.TSNE:
#             all_objects_lower_dim = TSNE(n_components=2).fit_transform(all_objects)
#         elif self.config.VISUALISATION_METHOD == VM.UMAP:
#             umap = UMAP(n_components=2)
#             umap.fit(all_objects)
#             all_objects_lower_dim = umap.transform(all_objects)
#         else:
#             raise ValueError(
#                 f"Unknown visualisation method: {self.config.VISUALISATION_METHOD}"
#             )
#
#         return all_objects_lower_dim
#
#
# if __name__ == "__main__":
#     # TODO try to come up with something interactive - human driven, heuristic selection, and so on
#     # TODO on hover show the similarities between the points (maybe just line thickness)
#     # TODO add more information about the selection
#     config = Config()
#     environment = Environment(config)
#     objects = environment.get_objects()
#     heuristic = EvolutionaryStrategy(config, environment, [])
#     selected_indices = heuristic.solve()
#     selected_indices = get_object_indices(selected_indices)
#     reachable_indices_to_selection = environment.get_reachable_object_indices(
#         selected_indices
#     )
#
#     methods = [v for v in VM]
#     for m in methods:
#         config.VISUALISATION_METHOD = m
#         visualiser = Visualiser(config)
#         visualiser.visualise_selection(
#             objects, selected_indices, reachable_indices_to_selection
#         )
import os

import dash
import flask
import pandas as pd
import plotly.express as px
from dash import dcc, html
from dash.dependencies import Input, Output

from config import Config
from playground.environment import Environment


def prepare_data():
    config = Config()
    env = Environment(config)
    vis_sims = []
    latent_sims = []
    image_pairs = []
    for i in range(config.OBJ_NUM):
        for j in range(config.OBJ_NUM):
            vis_sims.append(env.storage.get_visual_similarity(i, j))
            latent_sims.append(env.storage._latent_similarities[i, j])
            image_pairs.append(
                (
                    os.path.join(env.get_objects()[i].image_path, "image_0.png"),
                    os.path.join(env.get_objects()[j].image_path, "image_1.png"),
                )
            )

    data = {
        "visual_similarity": vis_sims,
        "transfer_success_rate": latent_sims,
        "image_pair_path": image_pairs,
    }
    return pd.DataFrame(data)


def prepare_data_test():
    # Example DataFrame creation
    data = {
        "visual_similarity": [0.1, 0.2, 0.3],
        "transfer_success_rate": [0.5, 0.6, 0.7],
        "image_pair_path": ["image_0.png", "image_1.png", "image_2.png"],
    }
    image_folder = "/Users/mikolajdeja/Coding/learning/tests/_test_assets/"
    return pd.DataFrame(data), image_folder


if __name__ == "__main__":
    df, image_folder = prepare_data_test()

    # Initialize the Dash app with a Flask server
    app = dash.Dash(__name__)
    server = app.server

    # Scatter plot
    fig = px.scatter(
        df,
        x="visual_similarity",
        y="transfer_success_rate",
        hover_name="image_pair_path",
    )
    fig.update_traces(marker=dict(size=10))

    # Layout
    app.layout = html.Div(
        [
            dcc.Graph(id="scatter-plot", figure=fig),
            html.Div(id="click-data"),
            html.Div(
                html.Img(id="displayed-image", style={"display": "none"}),
                style={"textAlign": "center"},  # Center the image
            ),
        ]
    )

    # Flask route to serve images
    @server.route("/images/<path:image_path>")
    def serve_image(image_path):
        return flask.send_from_directory(image_folder, image_path)

    # Callback to display the image when a point is clicked
    @app.callback(
        Output("displayed-image", "src"),
        Output("displayed-image", "style"),
        Input("scatter-plot", "clickData"),
    )
    def display_image(click_data):
        if click_data:
            point_index = click_data["points"][0]["pointIndex"]
            image_pair_path = df.iloc[point_index]["image_pair_path"]
            image_pair_url = f"/images/{image_pair_path}"
            return image_pair_url, {
                "display": "block",
                "width": "20%",
                "margin": "auto",
            }
        return "", {"display": "none"}

    app.run_server(debug=True)
