import numpy as np


def check_vector_span(set_of_vectors, vector_to_check):
    # creates an empty vector of correct size
    vector_of_scalars = np.asarray([None]*set_of_vectors.shape[0])

    # Solves for the scalars that make the equation true if vector is within
    # the span
    try:
        # TODO: Use np.linalg.solve() function here to solve for
        # vector_of_scalars
        vector_of_scalars = np.linalg.solve(set_of_vectors, vector_to_check)
        if not (vector_of_scalars is None):
            print("\nVector is within span.\nScalars in s:", vector_of_scalars)
    # Handles the cases when the vector is NOT within the span
    except Exception as exception_type:
        if str(exception_type) == "Singular matrix":
            print("\nNo single solution\nVector is NOT within span")
        else:
            print("\nUnexpected Exception Error:", exception_type)
    return vector_of_scalars


t = np.array([4, 11])
vw = np.array([[1, 2], [3, 5]])

print("\nMatrix vw:", vw, "\nVector t:", t, sep="\n")
s = check_vector_span(vw, t)

vw2 = np.array([[1, 2], [2, 4]])
t2 = np.array([6, 12])
print("\nNew Vectors:\n Matrix vw2:", vw2, "\nVector t2:", t2, sep="\n")
# Call to check_vector_span
s2 = check_vector_span(vw2, t2)

vw3 = np.array([[1, 2], [1, 2]])
t3 = np.array([6, 10])
print("\nNew Vectors:\n Matrix vw3:", vw3, "\nVector t3:", t3, sep="\n")
# Call to check_vector_span
s3 = check_vector_span(vw3, t3)
