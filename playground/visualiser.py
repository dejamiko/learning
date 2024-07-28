import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

from al.mh import EvolutionaryStrategy
from config import Config
from playground.environment import Environment
from utils import get_object_indices


class Visualiser:
    def __init__(self, config):
        self.config = config

    def visualise_selection(
        self, all_objects, selected_indices, reachable_indices_to_selected
    ):
        all_objects_lower_dim = self._prepare_data(all_objects)

        # there is color, size, symbol, and hover_data to play with
        # create a dataframe with the data
        df = pd.DataFrame(all_objects_lower_dim, columns=["x", "y"])
        df["selected"] = "not selected"
        df.loc[selected_indices, "selected"] = "selected"
        df["reachable"] = "not reachable"
        df.loc[reachable_indices_to_selected.keys(), "reachable"] = "reachable"

        # make all points larger
        fig = px.scatter(df, x="x", y="y", color="selected", symbol="reachable")
        fig.update_traces(marker=dict(size=10))

        # discretise the plot space
        x_range = df["x"].max() - df["x"].min()
        y_range = df["y"].max() - df["y"].min()

        width = 1000
        height = 500

        x_step = x_range / width
        y_step = y_range / height

        # create a 2d array of the discretised space
        discretised_space = np.zeros((width, height))

        # populate it with points
        for i, row in df.iterrows():
            x = int((row["x"] - df["x"].min()) / x_step)
            y = int((row["y"] - df["y"].min()) / y_step)
            x = min(width - 1, max(0, x))
            y = min(height - 1, max(0, y))
            discretised_space[x, y] = 1

        # add lines between reachable points and the corresponding selected point
        for o in reachable_indices_to_selected.keys():
            for s, sim in reachable_indices_to_selected[o]:
                if o == s:
                    continue
                # display the similarity as a label on the line
                fig.add_shape(
                    type="line",
                    x0=df.loc[o, "x"],
                    y0=df.loc[o, "y"],
                    x1=df.loc[s, "x"],
                    y1=df.loc[s, "y"],
                    line=dict(color="black", width=1, dash="dash"),
                )

                # find a position for the label where it doesn't overlap with the other points
                orig_x = (df.loc[o, "x"] + df.loc[s, "x"]) / 2
                orig_y = (df.loc[o, "y"] + df.loc[s, "y"]) / 2

                # find the cell this point corresponds to
                x = int((orig_x - df["x"].min()) / x_step)
                y = int((orig_y - df["y"].min()) / y_step)
                x = min(width - 1, max(0, x))
                y = min(height - 1, max(0, y))

                # find the nearest point that is not occupied in the discretised space
                pairs = [(i, j) for i in range(-5, 6) for j in range(-5, 6)]
                pairs.sort(key=lambda x: x[0] ** 2 + x[1] ** 2)
                for i, j in pairs:
                    if (
                        width >= x + i >= 0
                        and 0 == discretised_space[x + i, y + j]
                        and 0 <= y + j <= height
                    ):
                        break

                # map the cell back to the original space
                orig_x += i * x_step
                orig_y += j * y_step
                fig.add_annotation(
                    x=orig_x,
                    y=orig_y,
                    text=f"{sim:.2f}",
                    showarrow=False,
                    font=dict(size=10),
                )

        # save the figure to a file
        fig.write_html(f"visualisation_{self.config.VISUALISATION_METHOD}.html")

    def _prepare_data(self, all_objects):
        all_objects = [o.latent_repr for o in all_objects]
        all_objects = np.array(all_objects)
        if self.config.VISUALISATION_METHOD == "pca":
            pca = PCA(n_components=2)
            pca.fit(all_objects)
            all_objects_lower_dim = pca.transform(all_objects)
        elif self.config.VISUALISATION_METHOD == "tsne":
            all_objects_lower_dim = TSNE(n_components=2).fit_transform(all_objects)
        elif self.config.VISUALISATION_METHOD == "umap":
            umap = UMAP(n_components=2)
            umap.fit(all_objects)
            all_objects_lower_dim = umap.transform(all_objects)
        else:
            raise ValueError(
                f"Unknown visualisation method: {self.config.VISUALISATION_METHOD}"
            )

        return all_objects_lower_dim


if __name__ == "__main__":
    # TODO try to come up with something interactive - human driven, heuristic selection, and so on
    # TODO on hover show the similarities between the points (maybe just line thickness)
    # TODO add more information about the selection
    config = Config()
    environment = Environment(config)
    objects = environment.get_objects()
    heuristic = EvolutionaryStrategy(config, environment, [])
    selected_indices = heuristic.solve()
    selected_indices = get_object_indices(selected_indices)
    reachable_indices_to_selection = environment.get_reachable_object_indices(
        selected_indices
    )

    methods = ["pca", "tsne", "umap"]
    for m in methods:
        config.VISUALISATION_METHOD = m
        visualiser = Visualiser(config)
        visualiser.visualise_selection(
            objects, selected_indices, reachable_indices_to_selection
        )
