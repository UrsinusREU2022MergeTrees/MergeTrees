import numpy as np
import matplotlib.pyplot as plt

def backtrace(backpointers, node, involved):
    optimal = False
    for P in backpointers[node[0]][node[1]]:
        if backtrace(backpointers, (P[0], P[1]), involved):
            P[2] = True
            optimal = True
            involved[node[0], node[1]] = 1
    if node[0] == 0 and node[1] == 0:
        return True #Reached the beginning
    return optimal

def dtw(X, Y):
    M = X.shape[0]
    N = Y.shape[0]
            
    backpointers = []
    for i in range(0, M+1):
        backpointers.append([])
        for j in range(0, N+1):
            backpointers[i].append([])

    D = np.zeros((M+1, N+1))
    D[1::, 0] = np.inf
    D[0, 1::] = np.inf
    for i in range(1, M+1):
        for j in range(1, N+1):
            d = np.abs(X[i-1] - Y[j-1])
            dul = d + D[i-1, j-1]
            dl = d + D[i, j-1]
            du = d + D[i-1, j]
            D[i, j] = min(min(dul, dl), du)
            if dul == D[i, j]:
                backpointers[i][j].append([i-1, j-1, False])
            if dl == D[i, j]:
                backpointers[i][j].append([i, j-1, False])
            if du == D[i, j]:
                backpointers[i][j].append([i-1, j, False])
    involved = np.zeros((M+1, N+1))
    backtrace(backpointers, (M, N), involved) #Recursive backtrace from the end
    return (D, backpointers, involved)