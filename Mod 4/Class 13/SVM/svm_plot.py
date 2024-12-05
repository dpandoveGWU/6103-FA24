"""
Helper functions for SVMs
=========================

This module contains functions used for the SVM
lab of ISLP. Currently it contains just a simple function to plot
decision boundary and points through a two-dimensional
slice for an SVM classifier.

"""

import matplotlib.pyplot as plt
import numpy as np

def plot(X,
         Y,
         svm, 
         features=(0, 1),
         xlim=None,
         nx=300,
         ylim=None,
         ny=300,
         ax=None,
         decision_cmap=plt.cm.plasma,
         scatter_cmap=plt.cm.tab10,
         alpha=0.2):

   '''
   Graphical representation of fitted support vector classifier.

   Parameters
   ----------

   X : array-like of shape (n_samples, n_features)
       Features used in fitting `svm`. Assumed to have at least 2 columns.

   Y : array-like of shape (n_samples,)
       Labels used in fitting `svm`. Used to color
       points by class.

   svm : `sklearn.svm.SVC`
       Fitted support vector classifier. Assumed that svm has been fit on (X,Y).

   features : (int, int), default=(0, 1)
       Which columns of X used for plotting. If more then two
       features, remaining features are set at mean values for
       2-dimensional slice.

   xlim : (float, float), optional
       Range of values for x-axis of plot.

   nx : int, default=300
       Resolution of grid for x-axis.

   ylim : (float, float), optional
       Range of values for y-axis of plot.

   ny : int, default=300
       Resolution of grid for y-axis.

   ax : a matplotlib axis

   decision_cmap : a matplotlib colormap for coloring decision rule.

   scatter_cmap : a matplotlib colormap for coloring points by class.

   alpha : float
       Alpha level for opacity of decision rule.
   '''
   X = np.asarray(X)

   if X.shape[1] < 2:
      raise ValueError('expecting at least 2 columns to display decision boundary')
   
   X0, X1 = [X[:,i] for i in features]

   if xlim is None:
      xlim = (X0.min() - 0.5 * X0.std(),
              X0.max() + 0.5 * X0.std())

   if ylim is None:
      ylim = (X1.min() - 0.5 * X1.std(),
              X1.max() + 0.5 * X1.std())

   if ax is None:
      fig, ax = plt.subplots()
   else:
      fig = ax.figure

   # draw the points

   ax.scatter(X0, X1, c=Y, cmap=scatter_cmap)

   # add the contour

   xval, yval = np.meshgrid(np.linspace(xlim[0], xlim[1], nx),
                            np.linspace(ylim[0], ylim[1], ny))   

   # this will work well when labels are integers

   grid_val = np.array([xval.reshape(-1),
                        yval.reshape(-1)]).T
   X_pred = np.multiply.outer(np.ones(grid_val.shape[0]),
                              X.mean(0))
   X_pred[:,features[0]] = grid_val[:,0]
   X_pred[:,features[1]] = grid_val[:,1]
   
   prediction_val = svm.predict(X_pred)

   ax.contourf(xval,
               yval,
               prediction_val.reshape(yval.shape),
               cmap=decision_cmap,
               alpha=alpha)

   # add the support vectors    

   ax.scatter(X[svm.support_,features[0]], 
              X[svm.support_,features[1]], marker='+', c='k', s=200)