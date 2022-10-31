# stat_modeling
### Decision Boundary
##### Ref: http://luisvalesilva.com/datasimple/random_forests.html
```python
def compute_decision_boundaries(clf, x, y):
    mn_x1, mx_x1 = int(x[:,0].min())-1, int(x[:,0].max()+1)
    mn_x2, mx_x2 = int(x[:,1].min())-1, int(x[:,1].max()+1)
    x1s = np.linspace(mn_x1, mx_x1, 300)
    x2s = np.linspace(mn_x1, mx_x2, 300)
    x1, x2 = np.meshgrid(x1s, x2s)
    x_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(x_new).reshape(x1.shape)
    axes = [mn_x1, mx_x1, mn_x2, mx_x2]
    return x1, x2, y_pred, axes
```
```python
def plot_feature_space(clf, x, y):
    x1, x2, y_pred, axes = compute_decision_boundaries(clf, x, y)
    custom_cmap = ListedColormap(['y','b','r'])
    plt.figure(figsize=(8, 4))
    plt.contourf(x1, x2, y_pred, cmap=custom_cmap, alpha=0.1)
    plt.plot(x[:, 0][y==0], x[:, 1][y==0], "yo", marker='.')
    plt.plot(x[:, 0][y==1], x[:, 1][y==1], "bs", marker='.')
    plt.plot(x[:, 0][y==2], x[:, 1][y==2], "rd", marker='.')
    plt.axis(axes)
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)
    plt.show()
```
