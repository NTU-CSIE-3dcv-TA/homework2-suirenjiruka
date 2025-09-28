A = []
    for xi in X.copy():
        temp = np.roots([(1 +xi**2 - 2*xi*cos_12),0,- dis12**2]) 
        if abs(np.imag(temp[0])) > 1e-19:
            X.remove(xi)
        else:
            A.append(abs(float(np.real(temp[0]))))

    Y = []
    for xi, ai in zip(X.copy(), A.copy()):
        temp = np.roots([ai**2, -2*(ai**2)*cos_13, ai**2 -dis13**2])
        if abs(np.imag(temp[0])) > 1e-19:
            X.remove(xi)
            A.remove(ai)
        else:
            Y.append(abs(float(np.real(temp[0]))))