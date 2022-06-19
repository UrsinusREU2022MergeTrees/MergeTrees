import numpy as np
import matplotlib.pyplot as plt
from utils import draw_curve

class MergeNode(object):
    def __init__(self, y, x=None):
        """
        Parameters
        ----------
        y: float
            Height of node
        x: float
            x position of node (optional)
        """
        self.left = None
        self.right = None
        self.x = x
        self.y = y
        self.idx = -1 # Inorder index
        self.birth_death = []
    
    def inorder(self, idx):
        """
        Perform an inorder traversal

        Parameters
        idx: list[1]
            A count, by reference
        """
        if self.left:
            self.left.inorder(idx)
        self.idx = idx[0]
        idx[0] += 1
        if self.right:
            self.right.inorder(idx)
    
    def get_coords(self, use_inorder):
        """
        Return a list of the [x, y] coordinates of this node
        
        Parameters
        ----------
        use_inorder: boolean
            If True, use the inorder coordinate for x.  If false,
            use a prespecified x coordinate if it exists
        """
        coords = np.array([self.idx, self.y])
        if not use_inorder:
            if self.x:
                coords[0] = self.x
        return coords

    def render(self, use_inorder, params):
        """
        Recursive helper method for rendering

        Parameters
        ----------
        use_inorder: boolean
            If True, use the inorder coordinate for x.  If false,
            use a prespecified x coordinate if it exists
        
        params: dict {
            offset: [x, y]: Offset by which to plot this
            draw_curved: boolean: If true, draw parabolic curved lines between nodes
            linewidth: int: How thick to draw the edges
            pointsize: int: How big to draw the nodes
        }
        """
        offset = np.array([0, 0]) if not 'offset' in params else params['offset']
        draw_curved = True if not 'draw_curved' in params else params['draw_curved']
        linewidth = 3 if not 'linewidth' in params else params['linewidth']
        pointsize = 200 if not 'pointsize' in params else params['pointsize']
        X = np.array([self.x, self.y])
        X = self.get_coords(use_inorder) + offset
        plt.scatter(X[0], X[1], pointsize, 'k')
        if len(self.birth_death) > 0:
            plt.text(X[0], X[1], "{}".format(self.birth_death))
        if self.left:
            Y = self.left.get_coords(use_inorder) + offset
            if draw_curved:
                draw_curve(X, Y, linewidth)
            else:
                plt.plot([X[0], Y[0]], [X[1], Y[1]], 'k', lineWidth=linewidth)
            self.left.render(use_inorder, params)
        if self.right:
            Y = self.right.get_coords(use_inorder) + offset
            if draw_curved:
                draw_curve(X, Y, linewidth)
            else:
                plt.plot([X[0], Y[0]], [X[1], Y[1]], 'k', lineWidth=linewidth)
            self.right.render(use_inorder, params)


def unionfind_root(pointers, u):
    """
    Union find root operation with path-compression

    Parameters
    ----------
    pointers: list
        A list of pointers to representative nodes
    u: int
        Index of the node to find
    
    Returns
    -------
        Index of the representative of the component of u
    """
    if not (pointers[u] == u):
        pointers[u] = unionfind_root(pointers, pointers[u])
    return pointers[u]

def unionfind_union(pointers, u, v, idxorder):
    """
    Union find "union" with early birth-based merging
    (similar to rank-based merging...not sure if exactly the
    same theoretical running time)

    Parameters
    ----------
    pointers: list
        A list of pointers to representative nodes
    u: int
        Index of first node
    v: int
        Index of the second node
    idxorder: list
        List of order in which each point shows up
    """
    u = unionfind_root(pointers, u)
    v = unionfind_root(pointers, v)
    if u != v:
        [ufirst, usecond] = [u, v]
        if idxorder[v] < idxorder[u]:
            [ufirst, usecond] = [v, u]
        pointers[usecond] = ufirst

class MergeTree(object):
    def __init__(self):
        self.root = None
    
    def render(self, use_inorder, params={}):
        """
        Draw this tree

        Parameters
        ----------
        use_inorder: boolean
            If True, use the inorder coordinate for x.  If false,
            use a prespecified x coordinate if it exists
        
        params: dict {
            offset: [x, y]: Offset by which to plot this
            draw_curved: boolean: If true, draw parabolic curved lines between nodes
            linewidth: int: How thick to draw the edges
            pointsize: int: How big to draw the nodes
        }
        """
        if use_inorder:
            idx = [0]
            self.root.inorder(idx)
        self.root.render(use_inorder, params)
    
    def init_from_timeseries(self, x):
        """
        Uses union find to make a merge tree object from the time series x
        (NOTE: This code is pretty general and could work to create merge trees
        on any domain if the neighbor set was updated)

        Parameters
        ----------
        x: ndarray(N)
            1D array representing the time series
        
        Returns
        -------
        I: ndarray(N, 2)
            H0 persistence diagram for this merge tree
        """
        #Add points from the bottom up
        N = len(x)
        idx = np.argsort(x)
        idxorder = np.zeros(N)
        idxorder[idx] = np.arange(N)
        pointers = np.arange(N) #Pointer to oldest indices
        representatives = {} # Nodes that represent a connected component
        leaves = {} # Leaf nodes
        I = [] #Persistence diagram
        for i in idx: # Go through each point in the time series in height order
            neighbs = set([])
            #Find the oldest representatives of the neighbors that
            #are already alive
            for di in [-1, 1]: #Neighbor set is simply left/right
                if i+di >= 0 and i+di < N:
                    if idxorder[i+di] < idxorder[i]:
                        neighbs.add(unionfind_root(pointers, i+di))
            if len(neighbs) == 0:
                #If none of this point's neighbors are alive yet, this
                #point will become alive with its own class
                leaves[i] = MergeNode(x[i], i)
                representatives[i] = leaves[i]
                self.root = representatives[i]
            else:
                neighbs = list(neighbs)
                #Find the oldest class, merge earlier classes with this class,
                #and record the merge events and birth/death times
                oldest_neighb = neighbs[np.argmin([idxorder[n] for n in neighbs])]
                #No matter, what, the current node becomes part of the
                #oldest class to which it is connected
                unionfind_union(pointers, oldest_neighb, i, idxorder)
                if len(neighbs) == 2: #A nontrivial merge
                    for n in neighbs:
                        if not (n == oldest_neighb):
                            #Create node and record persistence event if it's nontrivial
                            if x[i] > x[n]:
                                # Record persistence information
                                I.append([x[n], x[i]])
                                leaves[n].birth_death = (x[n], x[i])
                                # Create new node
                                node = MergeNode(x[i], i)
                                node.left, node.right = [representatives[n] for n in neighbs]
                                self.root = node
                                #Change the representative for this class to be the new node
                                representatives[oldest_neighb] = node
                        unionfind_union(pointers, oldest_neighb, n, idxorder)
        #Add the essential class
        idx1 = np.argmin(x)
        idx2 = np.argmax(x)
        [b, d] = [x[idx1], x[idx2]]
        I.append([b, d])
        leaves[idx1].birth_death = (b, d)
        return np.array(I)


if __name__ == '__main__':
    from persim import plot_diagrams
    
    MT = MergeTree()
    x = [0, 1, -1, 3, 3, 0.5, 1, 1, 2, 1, 4, 0][::-1]
    I = MT.init_from_timeseries(x)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    MT.render(False)
    plt.subplot(122)
    plot_diagrams([I])
    plt.show()