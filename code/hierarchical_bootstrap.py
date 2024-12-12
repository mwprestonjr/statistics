"""
Wrapper function for hierarchical bootstrap tests.
"""

# imports
from twosample_hierarchical_bootstrap import twosample_hierarchical_bootstrap
from paired_hierarchical_bootstrap import paired_hierarchical_bootstrap
from onesample_hierarchical_bootstrap import onesample_hierarchical_bootstrap
from utils import check_input


def hierarchical_bootstrap(df, variable, level_1, level_2, condition=None,
                           paired=False, one_sample=False, n_iterations=1000, 
                           verbose=True, plot=True):
    """
    Wrapper function for hierarchical bootstrap tests.

    Parameters
    ----------
    df : pandas DataFrame
        Dataframe containing the data.
    variable : str
        Name of the variable to test.
    level_1 : str
        First level of hierarchy to resample. This is the higher level i.e.
        level_2 is nested within level_1; e.g. electrodes within subjects.
    level_2 : str
        Second level of hierarchy to resample. This is the lower level i.e.
        level_2 is nested within level_1; e.g. electrodes within subjects.
    condition : str, optional
        Name of the condition to test. Required for two-sample tests.
        Default is None.
    paired : bool, optional
        Whether to perform a paired test. Default is False.
    one_sample : bool, optional
        Whether to perform a one-sample test. Default is False.
    n_iterations : int, optional
        Number of bootstrap iterations. Default is 1000.
    verbose : bool, optional
        Whether to print results. Default is True.
    plot : bool, optional
        Whether to plot results. Default is True.

    Returns
    -------
    dict
        Dictionary containing the results of the test.

    Usage
    -----
    # two-sample test
    hierarchical_bootstrap(df, variable, level_1, level_2, condition=None,
                          paired=False, one_sample=False, n_iterations=1000, 
                          verbose=True, plot=True

    # paired two-sample test
    hierarchical_bootstrap(df, variable, level_1, level_2, condition=None,
                          paired=True, one_sample=False, n_iterations=1000, 
                          verbose=True, plot=True

    # one-sample test
    hierarchical_bootstrap(df, variable, level_1, level_2, condition=None,
                          paired=False, one_sample=True, n_iterations=1000, 
                          verbose=True, plot=True
    """
    
    # One-sample test
    if one_sample:
        return onesample_hierarchical_bootstrap(df, variable, level_1, level_2,
                                                n_iterations, verbose, plot)

    # Two-sample tests
    else:
        # check input data
        if condition is None:
            raise ValueError("Condition must be specified for two-sample test.")
        df = check_input(df, variable, condition, level_1, level_2)

        if paired:
            # paired two-sample test
            return paired_hierarchical_bootstrap(df, variable, condition, 
                                                 level_1, level_2, n_iterations,
                                                 verbose, plot)
        else:
            # two-sample test
            return twosample_hierarchical_bootstrap(df, variable, condition,
                                                    level_1, level_2, 
                                                    n_iterations, verbose, plot)
