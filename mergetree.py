import numpy as np
import matplotlib.pyplot as plt
from utils import draw_curve, plot_diagrams

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
        self.children = []
        self.x = x
        self.y = y
        self.idx = -1 # Inorder index
        self.birth_death = []

    
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
            if self.x or self.x == 0:
                coords[0] = self.x
        return coords

    def get_weight_sequence(self, val, seq):
        """
        Recursively obtain the non-redundant, height-based weight sequence
        by doing a generalized preorder traversal through the tree

        Parameters
        ----------
        val: float
            Cumulative weight sum
        seq: list
            Weight sequence being constructed
        """
        if len(self.children) == 0:
            # Node is a leaf node
            seq.append(val)
        else:
            for i, child in enumerate(sorted(self.children, key=lambda c: c.x)):
                if i == 0:
                    child.get_weight_sequence(val + self.y - child.y, seq)
                else:
                    child.get_weight_sequence(self.y - child.y, seq)


    def inorder(self, idx):
        """
        Perform a generalized inorder traversal
        NOTE: This will sort child nodes arbitrarily if 
        their x coordinates have not been specified

        Parameters
        idx: list[1]
            A count, by reference
        """
        for child in sorted(self.children+[self], key=lambda c: c.x):
            if self == child:
                self.idx = idx[0]
                idx[0] += 1
            else:
                child.inorder(idx)

    def get_rep_timeseries(self, ys):
        """
        Create a piecewise linear function that is 
        obtained from an inorder traversal of the y
        coordinates of the nodes in this tree

        Parameters
        ----------
        ys: list of float
            Time series that I'm building

        Returns
        -------
        ndarray(N): Time series representing piecewise linear function,
        with as many samples as there are nodes in the tree
        """
        if len(self.children) == 0:
            ys.append(self.y)
        for i, child in enumerate(sorted(self.children, key=lambda c: c.x)):
            child.get_rep_timeseries(ys)
            if i < len(self.children)-1:
                # Put the max in between every adjacent pair of children
                ys.append(self.y) 

    def get_eps_saddle_pairs(self, eps, pairs, depth):
        """
        Collect all saddles whose height difference to their
        children is under epsilon

        Parameters
        ----------
        eps: float
            Height difference below which to add a saddle pair
        pairs: list of (lower, upper, depth)
            Saddle pairs list that's being built
        depth: int
            Depth in tree
        """
        for child in self.children:
            # Children may mutate from lower
            if self.y - child.y <= eps:
                pairs.append((child, self, depth))
            child.get_eps_saddle_pairs(eps, pairs, depth+1)

    def plot(self, use_inorder, params):
        """
        Recursive helper method for plotting

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
            plot_birthdeath: boolean: Whether to plot (birth, death) at leaf nodes
        }
        """
        offset = np.array([0, 0]) if not 'offset' in params else params['offset']
        draw_curved = True if not 'draw_curved' in params else params['draw_curved']
        linewidth = 3 if not 'linewidth' in params else params['linewidth']
        pointsize = 200 if not 'pointsize' in params else params['pointsize']
        plot_birthdeath = False if not 'plot_birthdeath' in params else params['plot_birthdeath']
        X = np.array([self.x, self.y])
        X = self.get_coords(use_inorder) + offset
        plt.scatter(X[0], X[1], pointsize, 'k')
        if len(self.birth_death) > 0 and plot_birthdeath:
            plt.text(X[0], X[1], "{}".format(self.birth_death), c='r')
        for child in self.children:
            Y = child.get_coords(use_inorder) + offset
            if draw_curved:
                draw_curve(X, Y, linewidth)
            else:
                plt.plot([X[0], Y[0]], [X[1], Y[1]], 'k', lineWidth=linewidth)
            child.plot(use_inorder, params)


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
        self.PD = np.array([[]])

    def get_weight_sequence(self):
        """
        Recursively obtain the non-redundant, height-based weight sequence
        by doing a generalized preorder traversal through the tree

        Returns
        -------
        list: Weight sequence 
        """
        seq = []
        if self.root:
            self.root.get_weight_sequence(0, seq)
        return np.array(seq)

    def get_rep_timeseries(self):
        """
        Return a piecewise linear function that is 
        obtained from an inorder traversal of the y
        coordinates of the nodes in this tree
        
        Returns
        -------
        ndarray(N): Time series representing piecewise linear function,
        with as many samples as there are nodes in the tree
        """
        ys = []
        if self.root:
            self.root.get_rep_timeseries(ys)
        return np.array(ys)

    def collapse_saddles(self, eps):
        """
        Collapse saddles from the bottom up, as in section 4.4 of [1]

        [1] Sridharamurthy, R., Masood, T. B., Kamakshidasan, A., 
        & Natarajan, V. (2018). Edit distance between merge trees. 
        IEEE transactions on visualization and computer graphics, 
        26(3), 1518-1531.

        Parameters
        ----------
        eps: float
            Height difference below which to make a collapse
        """
        if self.root:
            pairs = []
            self.root.get_eps_saddle_pairs(eps, pairs, 0)
            pairs = sorted(pairs, key=lambda x: x[2], reverse=True)
            for (lower, upper, _) in pairs:
                children = [c for c in upper.children if c != lower]
                upper.children = sorted(children + lower.children, key=lambda c:c.x)

    def plot(self, use_inorder, params={}):
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
            plot_birthdeath: boolean: Whether to plot (birth, death) at leaf nodes
        }
        """
        if self.root:
            if use_inorder:
                idx = [0]
                self.root.inorder(idx)
            self.root.plot(use_inorder, params)
    
    def plot_with_pd(self, use_inorder, params={}):
        """
        Draw this tree alongslide its persistence diagram

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
            plot_birthdeath: boolean: Whether to plot (birth, death) at leaf nodes
            use_grid: boolean: Whether to draw grid lines
            show_merge_xticks: Whether to show the x ticks for the merge tree
        }
        """
        if self.root:
            use_grid = False if not 'use_grid' in params else params['use_grid']
            show_merge_xticks = False if not 'show_merge_xticks' in params else params['show_merge_xticks']
            yvals = np.sort(np.unique(self.get_rep_timeseries()))
            dy = yvals[-1] - yvals[0]
            plt.subplot(121)
            self.plot(use_inorder, params)
            plt.gca().set_yticks(yvals)
            plt.ylim(yvals[0]-0.1*dy, yvals[-1]+0.1*dy)
            if not show_merge_xticks:
                plt.gca().set_xticks([])
            if use_grid:
                plt.grid()
            plt.subplot(122)
            plot_diagrams([self.PD])
            plt.gca().set_yticks(np.unique(self.PD[:, 1]))
            plt.ylim(yvals[0]-0.1*dy, yvals[-1]+0.1*dy)
            plt.gca().set_xticks(np.unique(self.PD[:, 0]))
            plt.xlim(yvals[0]-0.1*dy, yvals[-1]+0.1*dy)
            if use_grid:
                plt.grid()

    def init_from_timeseries(self, y):
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
            H0 persistence diagram for this merge tree (also store locally
            as a side effect)
        """
        #Add points from the bottom up
        N = len(y)
        idx = np.argsort(y)
        idxorder = np.zeros(N)
        idxorder[idx] = np.arange(N)
        pointers = np.arange(N) #Pointer to oldest indices
        representatives = {} # Nodes that represent a connected component
        leaves = {} # Leaf nodes
        I = [] #Persistence diagram
        for i in idx: # Go through each point in the time series in height order
            neighbs = []
            #Find the oldest representatives of the neighbors that
            #are already alive
            for di in [-1, 1]: #Neighbor set is simply left/right
                if i+di >= 0 and i+di < N:
                    if idxorder[i+di] < idxorder[i]:
                        neighbs.append(unionfind_root(pointers, i+di))
            if len(neighbs) == 0:
                #If none of this point's neighbors are alive yet, this
                #point will become alive with its own class
                leaves[i] = MergeNode(y[i], i)
                representatives[i] = leaves[i]
                self.root = representatives[i]
            else:
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
                            if y[i] > y[n]:
                                # Record persistence information
                                I.append([y[n], y[i]])
                                leaves[n].birth_death = (y[n], y[i])
                                # Create new node
                                node = MergeNode(y[i], i)
                                left_right = [representatives[n] for n in neighbs]
                                if left_right[0].x > left_right[1].x:
                                    left_right = left_right[::-1]
                                node.children = left_right
                                self.root = node
                                #Change the representative for this class to be the new node
                                representatives[oldest_neighb] = node
                        unionfind_union(pointers, oldest_neighb, n, idxorder)
        #Add the essential class
        idx1 = np.argmin(y)
        idx2 = np.argmax(y)
        [b, d] = [y[idx1], y[idx2]]
        I.append([b, d])
        leaves[idx1].birth_death = (b, d)
        PD = np.array(I)
        self.PD = PD
        return PD