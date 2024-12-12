"""
Utility functions for hierarchical bootstrap methods.
"""

# imports
import numpy as np


def hierarchical_resampling(df, variable, level_1, level_2, iterations):
    """
    Get distribution of resampled means for hierarchical bootstrap. This 
    function resamples the data and computes the mean for each iteration of the
    bootstrap.

    NOTE: the number of instances per cluster is computed as the average number
    of instances per cluster. This is done to avoid biasing the resampling
    towards clusters with more instances.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing data to resample.
    variable : str
        Variable to resample.
    level_1 : str
        First level of hierarchy to resample. This is the higher level i.e.
        level_2 is nested within level_1; e.g. electrodes within subjects.
    level_2 : str
        Second level of hierarchy to resample. This is the lower level i.e.
        level_2 is nested within level_1; e.g. electrodes within subjects.
    iterations : int
        Number of iterations for resampling.

    Returns
    -------
    distribution : numpy.ndarray
        Resampled distribution.
    """

    # get cluster info
    clusters = df[level_1].unique()
    n_clusters = len(clusters)

    # count number of instances per cluster
    instances_per_cluster = np.zeros(n_clusters)
    for i_cluster, cluster_i in enumerate(clusters):
        instances_per_cluster[i_cluster] = len(df.loc[df[level_1]==cluster_i, 
                                                      level_2].unique())
    n_instances = int(np.nanmean(instances_per_cluster)) # use average number of instances per cluster

    # Precompute unique instances for each cluster
    cluster_instance_map = {cluster: df.loc[df[level_1] == cluster, level_2].unique() for cluster in clusters}

    # loop through iterations
    distribution = np.zeros(iterations)
    for i_iteration in range(iterations):
        # resample level 1 (clusters)
        clusters_resampled = np.random.choice(clusters, size=n_clusters)

        # resample level 2 (instances) and get data for each cluster
        values = []
        for i_cluster, cluster_i in enumerate(clusters_resampled):
            # resample level 3
            instances = cluster_instance_map[cluster_i]
            instances_resampled = np.random.choice(instances, size=n_instances)

            # get data for each instance within cluster
            for _, instance_i in enumerate(instances_resampled):
                value = df.loc[(df[level_1]==cluster_i) & \
                               (df[level_2]==instance_i), variable].values[0]
                values.append(value)

        # compute average for iteration
        distribution[i_iteration] = np.nanmean(values)

    return distribution


def compute_p_value(distribution):    
    '''
    This function computes the p-value for the null hypothesis that the 
    distribution is centered around zero.
    '''

    # count values greater than 0
    n_greater = np.sum(distribution > 0)
    n_less = np.sum(distribution < 0)
    p_value = np.min([n_greater, n_less]) / len(distribution)
    sign = np.sign(n_greater - n_less)

    return p_value, sign


def check_input(df, variable, level_1, level_2, condition=None):
    """
    Check input data for paired hierarchical bootstrap. This function checks 
    that each instance has both conditions present, and that the data is
    structured as a nested, hierarchical dataset.
    """
    
    # check that 'variable', 'condition', 'level_1', and 'level_2' are in df
    if variable not in df.columns:
        raise ValueError(f"Variable '{variable}' not found in dataframe.")
    if (condition not in df.columns) and (condition is not None):
        raise ValueError(f"Condition '{condition}' not found in dataframe.")
    if level_1 not in df.columns:
        raise ValueError(f"Level 1 '{level_1}' not found in dataframe.")
    if level_2 not in df.columns:
        raise ValueError(f"Level 2 '{level_2}' not found in dataframe.")

    # check that each level_1-level_2 pair has data for both conditions
    # and drop cases of missing data
    if condition is not None:
        df = drop_missing_data(df, level_1, level_2, condition, variable)

    return df


def drop_missing_data(df, level_1, level_2, condition, variable):
    """
    Drop instances with missing data. This function checks that each instance
    has both conditions present, and that the data is structured as a nested,
    hierarchical dataset.
    """
    
    n_dropped = 0
    n_instances = 0
    clusters = df[level_1].unique()
    for cluster in clusters:
        instances = df.loc[df[level_1]==cluster, level_2].unique()
        for instance in instances:
            df_i = df.loc[(df[level_1]==cluster) & (df[level_2]==instance)]
            if (len(df_i) != 2) or (len(df_i[condition].unique()) != 2) or \
                (df_i[variable].isnull().any()):
                df = df.drop(df_i.index)
                n_dropped += 1
            n_instances += 1

    if n_dropped > 0:
        print(f"Warning: {n_dropped}/{n_instances} instances dropped due to missing data.")

    return df
