# PC_Screen



The file PC_screen.py provides the codes of main functions for the paper "Model-free Feature Screening and FDR Control with Knockoff Features". The codes are written in Python 2.7

Main functions

1. projection_corr(X, Y) -- computes the projection correlation between X and Y, where X and Y are both 2d array. We also provide two different versions of this function for some special cases for faster computing.

1.1 projection_corr_1d(X, Y) --  X and Y are both one dimensional.

1.2 projection_corr_1dy(X, Y) -- Y is one dimensional.

2. get_equi_features(X) -- output the equicorrelated knockoff counterpart of X, where is a 2d array.

## If you use PC_Screen (Projection correlation for feature screening) in publications, we would appreciate citations of the following paper: Model-free Feature Screening and FDR Control with Knockoff Features
