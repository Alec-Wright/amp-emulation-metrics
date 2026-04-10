import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from Utils.greedy_algo import greedy_vertex_selection
import os
from collections import Counter


def count_equal_elements_any_order(list1, list2):
    # count occurrences of each element in both lists
    counter1 = Counter(list1)
    counter2 = Counter(list2)

    # find common elements and sum the minimum occurrences
    common_elements = counter1 & counter2
    equal_count = sum(common_elements.values())

    return equal_count


def are_clips_same_paper(list1, list2, paper_ids):
    # check if the clips are from the same paper
    for key in paper_ids.keys():
        if list1 in range(paper_ids[key][0], paper_ids[key][1] + 1) and list2 in range(
            paper_ids[key][0], paper_ids[key][1] + 1
        ):
            return 1
    return 0


def prepare_data(df, log=False):
    """
    Reads a CSV file containing rating and loss values, extracts the last columns corresponding to loss values,
    removes the 'loss-MelSTFTLoss' column (which contains NaN values), normalizes the loss values, and returns the DataFrame and normalized losses.
    Args:
        - df (pandas.DataFrame): The DataFrame containing the clips data.
        - log (bool, optional): Whether to apply a log transformation to the loss values. Defaults to False.
    Returns:
        - df (pandas.DataFrame): The DataFrame read from the CSV file, with transformed values if log==True.
        - losses (pandas.DataFrame): The normalized loss values DataFrame.
    """

    # remove column MelSTFTLoss (contains NaN values)
    if "loss-MelSTFTLoss" in df.columns:
        df = df.drop(columns=["loss-MelSTFTLoss"])
        print("Warning: Column 'loss-MelSTFTLoss' has been dropped.")

    loss_indx = 12  # index of the first loss column
    # dictionary of transformation functions for each loss to log scale
    if log:
        transfroms = {

            "esr_basic": lambda x: 10 * np.log10(x),
            "mse": lambda x: 10 * np.log10(x),
            "mae": lambda x: 20 * np.log10(x),
            "bss_eval": lambda x: 1 * x,
            "specconv": lambda x: 20 * np.log10(x),
            "logstft": lambda x: 20 * np.log10(x),
            "linstft": lambda x: 20 * np.log10(x),
            "mrstft": lambda x: 20 * np.log10(x),
            "melstft": lambda x: 20 * np.log10(x),
            "JTFS": lambda x: 10 * np.log10(x),
            "psychloss": lambda x: 20 * np.log10(x),
        }  # TODO add missing losses to the dictionary
        missing_losses = set(df.columns[loss_indx:]) - set(transfroms.keys())
        if missing_losses:
            print(
                f"Warning: The following losses are missing from the log transformation dictionary: {missing_losses}"
            )
        # apply the log transformation to the respective columns
        for col in transfroms.keys():
            df[col] = df[col].apply(func=transfroms[col])

    # extract the last columns corresponding to loss values
    losses = df.iloc[:, loss_indx:]

    # normalize the loss values
    losses = (losses) / losses.std(ddof=1)
    return df, losses


def sample_clips(df, losses, n_pages=30):
    """
    Sample clips from a DataFrame based on a specified method.

    Args:
        - df (pandas.DataFrame): The DataFrame containing the clips data.
        - losses (pandas.DataFrame): The array of loss values.
        - n_pages (int, optional): The number of pages to sample. Defaults to 30.

    Returns:
        - select_clips (numpy.ndarray): The indices of the selected clips.
        - X (numpy.ndarray): The array of loss values for the selected clips.
    """

    # extract the unique values of the Folder Name column and relative paper names
    folders = df["Folder Name"].unique()
    papers = list(set([folder.split("-")[0] for folder in folders]))

    # remove folders with less than n_conditions unique models
    n_conditions = 3 # number of models in a folder, these will correspond to MUSHRA conditions
    for folder in df["Folder Name"].unique():
        # for each of unique Folder Name values, count the number of unique values of the Model column
        n_models = df[df["Folder Name"] == folder]["Model"].nunique()

        # remove folders that cannot be used due to lack of data
        if n_models < n_conditions:
            df = df[df["Folder Name"] != folder]
            folders = folders[folders != folder]
            print(f"Folder {folder} removed due to lack of data")
            # remove the corresponding paper from the list
            for paper in papers:
                if paper in key:
                    papers.remove(paper)

    # make all possible combinations of 3 clips from the same folder
    combinations_list = []  # list of all possible combinations of 3 clips identified by their index in df
    folder_ids = {} # dictionary of folder names and corresponding indices of the first and last clip 
    pntr = 0    # a pointer
    for folder in folders:
        clips_in_folder = df[df["Folder Name"] == folder]
        # loop over clips id
        pntr2 = pntr
        for id in clips_in_folder["clip_id"].unique():
            # create all possible combinations of 3 clips
            clips_combinations = list(
                combinations(clips_in_folder[clips_in_folder["clip_id"] == id].index, 3)
            )
            combinations_list.extend(clips_combinations)
            pntr2 += len(clips_combinations)
        folder_ids[folder] = [pntr, pntr2 - 1]
        pntr = pntr2  # update pointer

    # get the indices of the first and last clip of each paper
    paper_ids = (
        {}
    )  # dictionary of paper names and corresponding indices of the first and last clip (similar to folder_ids but on a higher level)
    for key in papers:
        paper_ids[key] = [float("inf"), 0]
    for key in folder_ids.keys():
        for paper in papers:
            if paper in key:
                paper_ids[paper] = [
                    min(paper_ids[paper][0], folder_ids[key][0]),
                    max(paper_ids[paper][1], folder_ids[key][1]),
                ]

    # convert the combinations to a DataFrame
    combinations_df = pd.DataFrame(combinations_list, columns=["C1", "C2", "C3"])
    n_combinations = len(combinations_df)

    # save the loss values into a matrix
    X = np.zeros((n_combinations, 3, losses.shape[-1]))
    for i, row in combinations_df.iterrows():
        for j, cond in enumerate(["C1", "C2", "C3"]):
            X[i, j, :] = losses.iloc[row[cond], :].values

    # GREEDY ALGORITHM based on CORRELATION
    # sample n_pages pages by maximizing the uncorrelation between losses in a pages and across pages
    # each vertex of a graph corresponds to a combination of 3 clips (page)

    # build adjacency matrix
    graph = {i: list(range(0, n_combinations)) for i in range(n_combinations)}

    # get vertex weights
    vertex_weights = {}
    for i in range(n_combinations):
        page_corr = np.triu(np.corrcoef(X[i, :]), 1)
        vertex_weights[i] = np.mean(page_corr[np.triu_indices(3, 1)])

    # get edge weights
    edge_weights = {}
    edge_matrix = np.zeros((n_combinations, n_combinations))
    for i in range(n_combinations):
        for j in range(i + 1, n_combinations):
            # add penality if the two pages have the same clips
            num_equal_samples = count_equal_elements_any_order(
                combinations_list[i], combinations_list[j]
            )
            clip_penalty = num_equal_samples * 0.3
            # add penality if the clips are part of the same paper
            paper_penalty = are_clips_same_paper(i, j, paper_ids) * 0.5
            # update the edge weights with the penality terms
            edge_weights[(i, j)] = (
                np.abs(np.corrcoef(X[i].flatten(), X[j].flatten())[0, 1])
                + clip_penalty
                + paper_penalty
            )
            edge_matrix[i, j] = edge_weights[(i, j)]

    # run greedy algorithm to select the vertices
    selected_vertices = greedy_vertex_selection(
        graph, vertex_weights, edge_weights, n_pages
    )

    select_clips = list(selected_vertices)

    return select_clips, X, combinations_df


def plot_pages(clips_id, X, headers, filename, limits=None, elev=None, azim=None):
    """
    Plot the selected clips in a 3D space.

    Args:
        - clips_id (list): A list of clip IDs to be plotted.
        - X (ndarray): A 3D array containing the coordinates of the clips in the losses' space.
        - headers (list): A list of strings representing the headers for each loss value.
        - filename (str): The filename to save the plot as.
        - limits (ndarray, optional): A 3D array containing the limits for each axis. Defaults to None.
        - elev (float, optional): The elevation angle of the plot view. Defaults to None.
        - azim (float, optional): The azimuth angle of the plot view. Defaults to None.
    """
    # Get the number of loss values
    n_losses = X.shape[-1]
    # Create a figure with subplots
    plt.figure(figsize=(15, 9))
    plt.suptitle("Loss Distribution", fontsize=16)
    # Iterate over each loss value
    for i in range(n_losses):
        ax = plt.subplot(
            n_losses // 3, n_losses // 3 + n_losses % 3, i + 1, projection="3d"
        )
        # Plot the selected clips in the 3D space
        for j in clips_id:
            ax.scatter(X[j, 0, i], X[j, 1, i], X[j, 2, i], label=headers[i])
        ax.set_xlabel("C1")
        ax.set_ylabel("C2")
        ax.set_zlabel("C3")
        ax.set_title(headers[i])

        # Set limits for the axes if provided
        if isinstance(limits, list):
            ax.set_xlim(limits[0, :, i])
            ax.set_ylim(limits[1, :, i])
            ax.set_zlim(limits[2, :, i])
        elif limits == "minmax" or limits == "std":
            axis_lim = get_limits(X, type=limits)
            ax.set_xlim(axis_lim[0, :, i])
            ax.set_ylim(axis_lim[1, :, i])
            ax.set_zlim(axis_lim[2, :, i])
        else:
            raise ValueError("Invalid value for limits.")
        # Set the view angle if provided
        if elev is not None and azim is not None:
            ax.view_init(elev=elev, azim=azim)


    plt.tight_layout()
    # Check if the folder in filename exists, otherwise create it
    folder_path = os.path.dirname(filename)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    plt.savefig(filename)


def get_limits(X, type="minmax"):
    """
    Calculate the limits to use in plot_pages.

    Parameters:
    X (ndarray): Input array of shape (n_samples, 3, n_features).
    type (str, optional): Type of limits to calculate. Default is 'minmax'.

    Returns:
    ndarray: Array of shape (3, 2, n_features) containing the limits.
    """

    X_limits = np.zeros((3, 2, X.shape[-1]))
    if type == "minmax":
        for i in range(3):
            for j in range(X.shape[-1]):
                X_limits[i, :, j] = [X[:, i, j].min(), X[:, i, j].max()]
    elif type == "std":
        for i in range(3):
            for j in range(X.shape[-1]):
                X_limits[i, :, j] = [
                    X[:, i, j].mean() - 3 * X[:, i, j].std(ddof=1),
                    X[:, i, j].mean() + 3 * X[:, i, j].std(ddof=1),
                ]
    return X_limits


def main(n_pages, losses_filename, sampled_clip_loc, plot=False):
    # Read the CSV file containing the losses
    df = pd.read_csv(losses_filename)

    df, losses = prepare_data(df)

    clips_id, X, combinations_df = sample_clips(df, losses, n_pages=n_pages)

    # save the filepaths to the selected clips in a csv file
    subset_clips_df = combinations_df.iloc[clips_id, :3]
    filenameC1, filenameC2, filenameC3 = [], [], []
    for indx in clips_id:

        rowC1, rowC2, rowC3 = (
            df.iloc[combinations_df.iloc[indx, 0]],
            df.iloc[combinations_df.iloc[indx, 1]],
            df.iloc[combinations_df.iloc[indx, 2]],
        )
        filenameC1.append(
            os.path.join(
                rowC1["Folder Name"], f"{rowC1['Model']}-{rowC1['clip_id']}.wav"
            )
        )
        filenameC2.append(
            os.path.join(
                rowC2["Folder Name"], f"{rowC2['Model']}-{rowC2['clip_id']}.wav"
            )
        )
        filenameC3.append(
            os.path.join(
                rowC3["Folder Name"], f"{rowC3['Model']}-{rowC3['clip_id']}.wav"
            )
        )
    (
        subset_clips_df["filenameC1"],
        subset_clips_df["filenameC2"],
        subset_clips_df["filenameC3"],
    ) = (filenameC1, filenameC2, filenameC3)

    subset_clips_df.to_csv(sampled_clip_loc, index=False)


    # plot 2D projection of the losses of the selected clips
    if plot:
        elevs, azims = [0, 90], [0, 90]
        for elev in elevs:
            for azim in azims:
                orientation = f"e{elev}a{azim}"
                # plot selected clips
                filename = os.path.join(
                    "figures/",
                    f"loss_distribution_of_selected_clips_{orientation}.png",
                )
                plot_pages(
                    clips_id,
                    X,
                    losses.columns.values,
                    filename,
                    limits="minmax",
                    elev=elev,
                    azim=azim,
                )
                filename = os.path.join(
                    "figures/", f"loss_distribution_of_all_clips_{orientation}.png"
                )
                plot_pages(
                    range(X.shape[0]),
                    X,
                    losses.columns.values,
                    filename,
                    limits="minmax",
                    elev=elev,
                    azim=azim,
                )
