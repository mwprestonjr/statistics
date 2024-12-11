"""
Wrapper function for hierarchical bootstrap tests.
"""

# imports
from twosample_hierarchical_bootstrap import twosample_hierarchical_bootstrap
from paired_hierarchical_bootstrap import paired_hierarchical_bootstrap
from onesample_hierarchical_bootstrap import onesample_hierarchical_bootstrap


def hierarchical_bootstrap(df, variable, level_1, level_2, condition=None,
                           paired=False, onesample=False, n_iterations=1000, 
                           verbose=True, plot=True):
    
    if onesample:
        return onesample_hierarchical_bootstrap(df, variable, level_1, level_2,
                                                n_iterations, verbose, plot)

    else:
        if condition is None:
            raise ValueError("Condition must be specified for two-sample test.")

        if paired:
            return paired_hierarchical_bootstrap(df, variable, condition, 
                                                 level_1, level_2, n_iterations,
                                                 verbose, plot)
        else:
            return twosample_hierarchical_bootstrap(df, variable, condition,
                                                    level_1, level_2, 
                                                    n_iterations, verbose, plot)
