import numpy as np

# *** Mean Normalization ***

X = np.random.randint(5001, size=(1000, 20))
print(X.shape)

ave_cols = X.mean(axis=0)  # mean of the columns
std_col = X.std(axis=0)  # standard deviation of the columns

print(ave_cols.shape)
print(std_col.shape)

X_norm = (X-ave_cols)/std_col

# print the average of all the values of X_norm
print(np.mean(X_norm))

# print the average of the minimum value in each column of X_norm
print(np.mean(X_norm.min(axis=0)))

# print the average of the maximum value in each column of X_norm
print(np.mean(X_norm.max(axis=0)))


# *** Data Separation ***

# create a rank 1 ndarray that contains a random permutation of the row indices
# of `X_norm`
row_indices = np.random.permutation(X_norm.shape[0])

training_idx_end = int(len(row_indices) * .6)
cross_idx_end = int(training_idx_end + (len(row_indices) * .2))

# training set
X_train = X_norm[:training_idx_end, :]

# cross-validation set
X_crossVal = X_norm[training_idx_end:cross_idx_end, :]

# test set
X_test = X_norm[cross_idx_end:, :]


# print the shape of X_train
print(X_train.shape)

# print the shape of X_crossVal
print(X_crossVal.shape)

# print the shape of X_test
print(X_test.shape)
