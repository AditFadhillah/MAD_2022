import numpy as np

diatoms = np.loadtxt('diatoms.txt', delimiter=',').T
diatoms_classes = np.loadtxt('diatoms_classes.txt', delimiter=',')
print('Shape of diatoms:', diatoms.shape)
print('Shape of diatoms_classes:', diatoms_classes.shape)
#print('Classes:', diatoms_classes)

d,N = diatoms.shape
print('Dimension:', d)
print('Sample size:', N)


# Here's a function that will plot a given diatom. Let's try it on the first diatom in the dataset.

import matplotlib.pyplot as plt

def plot_diatom(diatom):
    xs = np.zeros(91)
    ys = np.zeros(91)
    for i in range(90):
        xs[i] = diatom[2 * i]
        ys[i] = diatom[2 * i + 1]
    
    # Loop around to first landmark point to get a connected shape
    xs[90] = xs[0]
    ys[90] = ys[0]
    
    plt.plot(xs, ys)    
    plt.axis('equal')   

plot_diatom(diatoms[:,0])


# Let's next compute the mean diatom and plot it.

mean_diatom = np.mean(diatoms, 1)
plot_diatom(mean_diatom)


# ### Task1: Implementing PCA
# 
# To implement PCA, please check the algorithm explaination from the lecture.
# Hits:
# 
# 1) Noramilize data subtracting the mean shape. No need to use Procrustes Analysis or other more complex types of normalization
# 
# 2) Compute covariance matrix (check np.cov)
# 
# 3) Compute eigenvectors and values (check np.linalg.eigh)

import numpy.matlib

def pca(data):
    data_cent = np.zeros((data.shape[0], data.shape[1]))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data_cent[i][j] = data[i][j] - mean_diatom[i]
    data_cent = np.cov(data_cent)
    PCevals, PCevecs = np.linalg.eigh(data_cent)
    PCevals = np.flip(PCevals, 0)
    PCevecs = np.flip(PCevecs, 1)
    return PCevals, PCevecs, data_cent

PCevals, PCevecs, data_cent = pca(diatoms)
# PCevals is a vector of eigenvalues in decreasing order. To verify, uncomment:
# PCevecs is a matrix whose columns are the eigenvectors listed in the order of decreasing eigenvectors


# ***Recall:***
# * The eigenvalues represent the variance of the data projected to the corresponding eigenvectors. 
# * Thus, the 2D linear subspace with highest projected variance is spanned by the eigenvectors corresponding to the two largest eigenvalues.
# * We extract these eigenvectors and plot the data projected onto the corresponding space.

# ### Compute variance of the first 10 components
# 
# How many components you need to cover 90%, 95% and 99% of variantion. Submit the resulting numbers for grading.

variance_explained_per_component = PCevals/np.sum(PCevals)
cumulative_variance_explained = np.cumsum(variance_explained_per_component)

plt.plot(cumulative_variance_explained)
plt.xlabel('Number of principal components included')
plt.ylabel('Proportion of variance explained')
plt.title('Proportion of variance explained as a function of number of PCs included')

# Let's print out the proportion of variance explained by the first 10 PCs
for i in range(10):
    print('Proportion of variance explained by the first '+str(i+1)+' principal components:', cumulative_variance_explained[i])


# ### Task2: Plot varianace accosiated with the first component
# 
# Please fill the gaps in the code to plot mean diatom shape with added FOURTH eigenvector mulitplied by [-3,-2,-1,0,1,2,3] standard deviations corresponding to this eigenvector.
# 
# Submit the resulting plot for grading.


e4 = PCevecs[:, 3] # gets the second eigenvector
lambda4 = PCevals[3] # gets the second eigenvalue
std4 = np.sqrt(lambda4) # In case the naming std is confusing -- the eigenvalues have a statistical interpretation
temp = [- 3, - 2, - 1, 0, 1, 2, 3]

diatoms_along_pc = np.zeros((7, 180))
for i in range(7):
    diatoms_along_pc[i] = mean_diatom + e4 * std4 * temp[i]
plt.figure()
for i in range(7):
    plot_diatom(diatoms_along_pc[i])

plt.title('Diatom shape along PC1')
plt.show()
