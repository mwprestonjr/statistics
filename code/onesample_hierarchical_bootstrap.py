"""
This module contains functions for performing the hierarchical bootstrap. 

The hierarchical bootstrap is a non-parametric method for computing the 
uncertainty of statistics derived from nested data structure. For instance, this 
method can be used to test whether the means of two distributions are 
significantly different from one another (as implemented here). This method is 
an extension of the standard bootstrap method that accounts for 
non-independence in the dataset resulting from a nested, multi-level, or 
hierarchical structure, e.g. if data are recorded from multiple electrodes 
within each subject, within each experimental condition.

A great overview of the hierarchical bootstrap is provided in "Application of 
the hierarchical bootstrap to multi-level data in neuroscience" by Saravanan et 
al., 2020; available: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7906290/. The 
authors were generous enough to provide code for performing the hierarchical 
bootstrap here: https://github.com/soberlab/Hierarchical-Bootstrap-Paper. The 
present module extends the aforementioned implementation by providing support
for datasets in which data are not equally distributed across conditions and/or 
clusters. Furthermore, this module provides additional functionality for
plotting of the results.
"""

# imports
import numpy as np
import matplotlib.pyplot as plt
from utils import hierarchical_resampling, compute_p_value


def onesample_hierarchical_bootstrap(df, variable, level_1, level_2, n_iterations=1000, 
                           verbose=True, plot=True):    
    """
    Perform the one-sample hierarchical bootstrap. 

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
    verbose : bool
        Whether to print p-value.
    plot : bool
        Whether to plot results.

    Returns
    -------
    p_value : float
        p-value for one-sample test.
    distribution : numpy.ndarray
        Resampled distribution.
    """

    # perform hierarchical bootstrap
    distribution = hierarchical_resampling(df, variable, level_1, level_2, 
                                           n_iterations)

    # compute p-value
    p_value = compute_p_value(distribution)

    # print/plot results
    if verbose:
        print(f"p-value: {p_value:.3f}")
    if plot:
        _plot_results(df, variable, distribution)

    return p_value, distribution


def _plot_results(df, variable, distribution):
    """
    Plot bootstrap results. PLotting function for run_hierarchical_bootstrap().
    """

    # create figure
    _, (ax0, ax1) = plt.subplots(1,2, figsize=(12, 4))

    # ax0: plot orignal distributions
    data = df[variable].values
    bin_edges = np.linspace(np.nanmin(data), np.nanmax(data), 30)
    ax0.hist(data.ravel(), bins=bin_edges, color='grey')
    ax0.set_xlabel('value')
    ax0.set_ylabel('count')
    ax0.set_title('Original dataset')
    
    # ax1: plot reasmapled distributions
    ax1.hist(distribution, bins=bin_edges, color='grey')
    ax1.set_xlabel(variable)
    ax1.set_ylabel('count')
    ax1.set_title('Bootstrap results')

