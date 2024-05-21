"""
This module contains functions for performing a hierarchical, repeated-measures 
analysis of variance. This procedure is a non-parametric test that assesses 
whether the centroid of two or more groups are equivalent. In principle, this 
statistical test is similar to permutational multivariate analysis of variance 
(PERMANOVA), but it is designed to handle hierarchical and repeated measures 
data. Here, this method has been implemented for the univariate case only. This 
procedure respects the hierarchical and repeated measures structure of the data 
by sequentially resampling each hierarchical level, then randomly permuting 
across conditions within each resampled instance. The F-statistic is computed 
for the empirical data and each resampled dataset; then the p-value is computed 
as the proportion of resampled F-statistics that are greater than the true 
F-statistic.

"""


# imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def hierarchical_rm_permanova(df, variable, condition, level_1, level_2, 
                              n_iterations=1000, verbose=True, plot=True):    
    """
    Perform hierarchical, repeated-measures analysis of variance.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing data to resample.
    variable : str
        Variable to resample. This is the dependent variable.
    condition : str
        Experimental condition of interest. This is the independent variable.
    level_1 : str
        First level of hierarchy to resample. This is the higher level i.e.
        level_2 is nested within level_1; e.g. electrodes within subjects.
    level_2 : str
        Second level of hierarchy to resample. This is the lower level i.e.
        level_2 is nested within level_1; e.g. electrodes within subjects.
    iterations : int
        Number of iterations for resampling.
    verbose : bool
        Whether to print results, including p-value and true differnce.
    plot : bool
        Whether to plot results, including empirical data and resampled
        distribution.
    **kwargs : dict
        Additional keyword arguments for plotting.

    Returns
    -------
    p_value : float
        p-value for difference between conditions.

    """

    # compute true F-statistic
    data = df.pivot(index=[level_1, level_2], columns=condition, 
                    values=variable).values
    f_statistic = _compute_f_statistic(data)

    # compute surrogate F-statistic distribution
    f_surrogate = np.zeros(n_iterations)
    for i_iteration in range(n_iterations):
        # perform hierarchical resampling
        resampled_data = _hierarchical_resampling(df, variable, condition, 
                                                  level_1, level_2)
        
        # compute F-statistic
        f_surrogate[i_iteration] = _compute_f_statistic(resampled_data)

    # compute p-value
    p_value = np.sum(f_surrogate > f_statistic) / n_iterations

    # print results    
    if verbose:
        _print_results(p_value, f_statistic, n_iterations)

    # plot results
    if plot:
        _plot_results(df, variable, condition, f_statistic, f_surrogate)

    return p_value, f_statistic, f_surrogate 


def _hierarchical_resampling(df, variable, condition, level_1, level_2):
    """
    Perform hierarchical resampling. Sequentially resample level 1 (clusters)
    and level 2 (instances) with replacement to generate a resampled dataset; 
    then randomly permute the conditions within each resampled instance.
    This procedure respects the hierarchical and repeated measures structure of 
    the data.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing data to resample.
    variable : str
        Variable to resample.
    condition : str
        Experimental condition of interest.
    level_1 : str
        First level of hierarchy to resample (higher level).
    level_2 : str
        Second level of hierarchy to resample (lower level).

    Returns
    -------
    resampled_data : numpy.ndarray
        Resampled data.

    """

    # get cluster and condition info
    clusters = df[level_1].unique()
    n_clusters = len(clusters)
    conditions = df[condition].unique()
    n_conditions = len(conditions)

    # count number of instances per cluster
    instances_per_cluster = np.zeros(n_clusters)
    for i_cluster, cluster_i in enumerate(clusters):
        instances_per_cluster[i_cluster] = len(df.loc[df[level_1]==cluster_i, 
                                                      level_2].unique())
    n_instances = int(np.nanmean(instances_per_cluster)) # use average number of instances per cluster

    # init
    resampled_data = np.zeros([int(n_clusters*n_instances), n_conditions])

    # Resample level 1 (clusters) 
    clusters_resampled = np.random.choice(clusters, size=n_clusters)

    # resample level 2 (instances) and get data 
    for i_cluster, cluster_i in enumerate(clusters_resampled):
        # resample level 2 (instances)
        instances = df.loc[df[level_1]==cluster_i, level_2].unique()
        instances_resampled = np.random.choice(instances, size=n_instances)

        # get resampled data 
        for instance_i in instances_resampled:
            values_ii = df.loc[(df[level_1]==cluster_i) & 
                               (df[level_2]==instance_i), variable].values
            resampled_data[i_cluster*n_instances + instance_i] = values_ii  

    # random permutation of conditions
    for ii in range(resampled_data.shape[0]):
        resampled_data[ii] = np.random.permutation(resampled_data[ii])

    return resampled_data


def _compute_f_statistic(data):
    """
    Compute F-statistic. F = ((SS_A / (p-1)) / (SS_W / N-p)),
    where SS_W is the sum of squares within groups, SS_A is the sum of squares
    between groups, p is the number of groups, and N is the total number of
    observations. Here, the sum of squares is defined as the sum of squared
    differences between all pairs of observations.

    """

    # count number of groups and observations
    p = data.shape[1] # number of groups
    n = data.shape[0] # observations per group
    N = np.size(data) # total number of observations

    # compute total sum of squares - measure difference between each observation
    data_f = np.ravel(data)
    indices = np.triu_indices(N, k=1)
    ss_total = np.sum((data_f[indices[0]] - data_f[indices[1]])**2)

    # compute within group sum of squares - measure difference between each observation (within same group only)
    group = np.tile(np.arange(p), n)
    delta = group[indices[0]] == group[indices[1]]
    ss_within = np.sum((data_f[indices[0]] - data_f[indices[1]])**2 * delta)

    # compute between group sum of squares
    ss_between = ss_total - ss_within

    # compute F-statistic
    f_statistic = (ss_between / (p-1)) / (ss_within / (N-p))

    return f_statistic


def _plot_results(df, variable, condition, f_statistic, f_surrogate):
    """
    Plot results. The first subplot is a violin plot of the empirical data, 
    split by condition. The second subplot is a histogram of the surrogate 
    F-statistics, with the true F-statistic annotated.

    """

    # set plotting params
    plotting_params = {
        'data'  :   df,
        'x'     :   condition,
        'y'     :   variable,
    }

    # create figure
    _, (ax0, ax1) = plt.subplots(1, 2, figsize=(12,4))

    # ax0: violin plot of empirical data
    sns.violinplot(**plotting_params, ax=ax0, palette='colorblind',
                   hue=condition)
    ax0.set_title('Empirical data')
    ax0.legend().remove()
    
    # ax1: histogram of resampled F-statistic
    ax1.hist(f_surrogate, color='k', alpha=0.5, label='surrogate values')
    ax1.axvline(f_statistic, color='k', linestyle='--', label='true value')
    ax1.set_xlabel('F-statistic')
    ax1.set_ylabel('count')
    ax1.set_title('Resampled F-statistic')
    ax1.legend()


def _print_results(p_value, true_f, n_iterations):

    # print p-value
    if p_value==0:
        print(f"p-value: <{1/n_iterations}")
    else:
        if p_value < 0.001:
            print(f"p-value: {p_value:.2e}")
        else:
            print(f"p-value: {p_value:.3f}")

    # print mean difference
    print(f"True F-statistic: {true_f:.2f}")

