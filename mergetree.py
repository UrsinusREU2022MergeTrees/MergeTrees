import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from utils import draw_curve, plot_diagrams

@jit(nopython=True)
def delete_repeats(x, circular=False):
    """
    Delete repeated points in a sequence in a time series

    Parameters
    ----------
    x: ndarray(N)
        Original time series
    circular: bool
        Whether to assume the time series wraps around
    
    Returns
    -------
    xret: ndarray(M <= N)
        Filtered time series
    xret_idx: ndarray(M <= N)
        Indices in the original time series
    """
    xret = []
    xret_idx = []
    for i in range(len(x)):
        if i == 0:
            xret.append(x[i])
            xret_idx.append(i)
        elif x[i] != x[i-1]:
            xret.append(x[i])
            xret_idx.append(i)
    if circular and len(xret) > 1 and xret[0] == xret[-1]:
        xret.pop()
        xret_idx.pop()
    return np.array(xret), np.array(xret_idx)

@jit(nopython=True)
def get_crit_timeseries(x, circular=False):
    """
    Filter out regular points from a time series
    
    Parameters
    ----------
    x: ndarray(N)
        Input time series
    circular: bool
        Whether to assume the time series wraps around
    
    Returns
    -------
    y: ndarray(M <= N)
        A time series with only critical points
    signs: ndarray(M)
        A parallel array indicating local min (-1) or local max (+1)
    y_idx: ndarray(M)
        Indices of the critical points in the original time series
    """
    ## Step 1: Merge adjacent points that are equal
    x, x_idx = delete_repeats(x, circular=circular)

    ## Step 2: Find critical points
    y = []
    y_idx = []
    signs = []
    if x.size > 1:
        for i in range(x.size):
            neighbs = [n for n in [i-1, i+1] if circular or (n >= 0 and n < x.size)]
            if circular:
                neighbs = [n % x.size for n in neighbs]
            xn = [np.sign(x[i]-x[n]) for n in neighbs]
            # If two neighbors are both the same sign
            if len(xn) == 2 and np.prod(np.array(xn)) == 1: 
                y.append(x[i])
                y_idx.append(x_idx[i])
                signs.append(xn[0])
            elif not circular:
                if i == 0 and x[1] > x[0]:
                    y.append(x[i])
                    y_idx.append(x_idx[i])
                    signs.append(-1.0)
                if i == x.size - 1 and x[-2] > x[-1]:
                    y.append(x[-1])
                    y_idx.append(x_idx[-1])
                    signs.append(-1.0)
    return np.array(y), np.array(signs), np.array(y_idx)

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
        self.is_globalmin = False

    
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

    def gather_inorder(self, nodes):
        """
        Perform a generalized inorder traversal
        NOTE: This will sort child nodes arbitrarily if 
        their x coordinates have not been specified

        Parameters
        nodes: list of MergeNode
            List of the nodes in the merge tree, sorted inorder
        """
        for child in sorted(self.children+[self], key=lambda c: c.x):
            if self == child:
                nodes.append(self)
            else:
                child.gather_inorder(nodes)

    def get_rep_timeseries(self, xs, ys, signs):
        """
        Create a piecewise linear function that is 
        obtained from an inorder traversal of the y
        coordinates of the nodes in this tree

        Parameters
        ----------
        xs: list of float
            X coordinates of time series that I'm building
        ys: list of float
            Time series that I'm building
        signs: list of [-1, 1]
            A parallel list indicating local min (-1) or local max (+1)

        """
        if len(self.children) == 0:
            xs.append(self.x)
            ys.append(self.y)
            signs.append(-1)
        for i, child in enumerate(sorted(self.children, key=lambda c: c.x)):
            child.get_rep_timeseries(xs, ys, signs)
            if i < len(self.children)-1:
                # Put the max in between every adjacent pair of children
                xs.append(self.x)
                ys.append(self.y)
                signs.append(1)

    def persistence_simplify(self, eps):
        """
        Remove all leaves that are under a certain persistence threshold

        Parameters
        ----------
        eps: Persistence threshold
        """
        survived = True
        if not self.is_globalmin and len(self.birth_death) == 2: # Leaf node
            if self.birth_death[1] - self.birth_death[0] < eps:
                survived = False
        elif len(self.children) > 0:
            self.children = [c for c in self.children if c.persistence_simplify(eps)]
            if len(self.children) == 0:
                survived = False
        return survived
    
    def make_zss_tree(self, node):
        """
        Convert this into a zhang/shasha tree object

        Parameters
        ----------
        node: zss.simple_tree.Node
            zss node object
        """
        from zss.simple_tree import Node
        for c in self.children:
            c_node = Node(c)
            node.addkid(c_node)
            c.make_zss_tree(c_node)

    def delete_singletons(self):
        """
        Delete nodes with a single child
        """
        ret = self
        if len(self.children) == 1:
            ret = self.children[0].delete_singletons()
        else:
            for i, c in enumerate(self.children):
                self.children[i] = c.delete_singletons()
        return ret

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
            color: string: What color to draw the nodes and branches
        }
        """
        offset = np.array([0, 0]) if not 'offset' in params else params['offset']
        draw_curved = True if not 'draw_curved' in params else params['draw_curved']
        linewidth = 3 if not 'linewidth' in params else params['linewidth']
        pointsize = 200 if not 'pointsize' in params else params['pointsize']
        plot_birthdeath = False if not 'plot_birthdeath' in params else params['plot_birthdeath']
        color = 'k' if not 'color' in params else params['color']
        X = np.array([self.x, self.y])
        X = self.get_coords(use_inorder) + offset
        plt.scatter(X[0], X[1], pointsize, color)
        if len(self.birth_death) > 0 and plot_birthdeath:
            plt.text(X[0], X[1], "{:.2f}, {:.2f}".format(*self.birth_death), c='r')
        for child in self.children:
            Y = child.get_coords(use_inorder) + offset
            if draw_curved:
                draw_curve(X, Y, linewidth, color)
            else:
                plt.plot([X[0], Y[0]], [X[1], Y[1]], color, lineWidth=linewidth)
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
    def __init__(self, x=np.array([])):
        """
        Construct a new merge tree

        Parameters
        ----------
        x: ndarray(N)
            Time series with which to initialize a merge tree.
            If left blank, initialize an empty merge tree.

        """
        self.root = None
        if len(x) > 0:
            self.init_from_timeseries(x)
        else:
            self.root = None
            self.PD = np.array([[]])
            self.PDIdx = np.array([[]], dtype=int)

    def gather_inorder(self):
        """
        Gather all of the nodes, sorted inorder

        Returns
        -------
        nodes: list of MergeNode
            List of the nodes in the merge tree, sorted inorder
        """
        nodes = []
        if self.root:
            self.root.gather_inorder(nodes)
        return nodes

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
        coordinates of the nodes in this tree, as well as
        a parallel array that indicates whether the points
        are mins or maxes
        
        Returns
        -------
        {
            xs: ndarray(N): Coordinates of time series
            ys: ndarray(N): Time series representing piecewise linear function,
                with as many samples as there are nodes in the tree,
            signs: ndarray(N): A parallel array of signs
        }
        """
        ys = []
        xs = []
        signs = []
        if self.root:
            self.root.get_rep_timeseries(xs, ys, signs)
        return dict(xs=np.array(xs), ys=np.array(ys), signs=np.array(signs))

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

    def persistence_simplify(self, eps):
        """
        Remove all leaves that are under a certain persistence threshold

        Parameters
        ----------
        eps: Persistence threshold
        """
        if self.root:
            self.root.persistence_simplify(eps)
            self.root.delete_singletons()
            pers = self.PD[:, 1] - self.PD[:, 0]
            self.PDIdx = self.PDIdx[pers >= eps, :]
            self.PD = self.PD[pers >= eps, :]

    def make_zss_tree(self):
        """
        Convert this into a zhang/shasha tree object

        Returns
        ----------
        root: zss.simple_tree.Node
            The root of the tree
        """
        from zss.simple_tree import Node
        root = Node(self.root)
        if self.root:
            self.root.make_zss_tree(root)
        return root

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
            yvals = np.sort(np.unique(self.get_rep_timeseries()['ys']))
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

    def init_from_timeseries(self, y, include_essential=False, circular=False):
        """
        Uses union find to make a merge tree object from the time series x
        (NOTE: This code is pretty general and could work to create merge trees
        on any domain if the neighbor set was updated)

        Parameters
        ----------
        y: ndarray(N)
            1D array representing the time series
        include_essential: bool
            Whether to include the essential class
        circular: boolean
            Whether to assume that the domain wraps around circularly

        Returns
        -------
        I: ndarray(N, 2)
            H0 persistence diagram for this merge tree (also store locally
            as a side effect)
        IIdx: ndarray(N, 2)
            Indices in the original time series of the birth/death pairs
        """
        #Add points from the bottom up
        N = len(y)
        idxs = np.argsort(y)
        idxorder = np.zeros(N)
        idxorder[idxs] = np.arange(N)
        pointers = np.arange(N) #Pointer to oldest indices
        representatives = {} # Nodes that represent a connected component
        leaves = {} # Leaf nodes
        I = [] #Persistence diagram
        IIdx = [] # Paired indices
        for i in idxs: # Go through each point in the time series in height order
            neighbs = []
            #Find the oldest representatives of the neighbors that
            #are already alive
            for di in [-1, 1]: #Neighbor set is simply left/right
                if circular or (i+di >= 0 and i+di < N):
                    idx = i + di
                    if circular:
                        idx = idx % N
                    if idxorder[idx] < idxorder[i]:
                        neighbs.append(unionfind_root(pointers, idx))
            if len(neighbs) == 0:
                #If none of this point's neighbors are alive yet, this
                #point will become alive with its own class
                leaves[i] = MergeNode(y[i], i)
                representatives[i] = leaves[i]
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
                                IIdx.append([n, i])
                                leaves[n].birth_death = (y[n], y[i])
                                # Create new node
                                node = MergeNode(y[i], i)
                                self.root = node
                                left_right = [representatives[n] for n in neighbs]
                                if left_right[0].x > left_right[1].x:
                                    left_right = left_right[::-1]
                                node.children = left_right
                                node.birth_death = (y[n], y[i])
                                #Change the representative for this class to be the new node
                                representatives[oldest_neighb] = node
                        unionfind_union(pointers, oldest_neighb, n, idxorder)
        #Add the essential class
        leaves[idxs[0]].is_globalmin = True
        if include_essential:
            idx1 = idxs[0]
            idx2 = np.argmax(y)
            [b, d] = [y[idx1], y[idx2]]
            I.append([b, d])
            IIdx.append([idx1, idx2])
            leaves[idx1].birth_death = (b, d)
        self.PD = np.array(I)
        self.PDIdx = np.array(IIdx, dtype=int)
        self.crit = self.get_rep_timeseries()["ys"]
        return self.PD, self.PDIdx
    
    def convert_to_bdi(self):
        """
        Convert to the format required by BDI-ED
        """
        nodes = self.gather_inorder()
        for i, n in enumerate(nodes):
            n.idx = i
        nodeScalars = [n.y for n in nodes]
        children = [[c.idx for c in n.children] for n in nodes]
        rootID = -1
        if self.root:
            rootID = self.root.idx
        return nodeScalars, children, rootID


def edit_remove_cost(node):
    ret = 0
    if len(node.label.birth_death) == 2:
        (b, d) = node.label.birth_death
        ret = np.abs(d-b)/2
    return ret

def update_cost(nodep, nodeq):
    ret = 0
    nodep = nodep.label
    nodeq = nodeq.label
    p_saddle = len(nodep.children) > 0
    q_saddle = len(nodeq.children) > 0
    if len(nodep.birth_death) == 2 and len(nodeq.birth_death) == 2:
        if p_saddle == q_saddle: # Only match mins to mins and saddles to saddles
            (bp, dp) = nodep.birth_death
            (bq, dq) = nodeq.birth_death
            ret = max(np.abs(bq-bp), np.abs(dq-dp))
            ret = min(ret, (np.abs(dp-bp) + np.abs(dq-bq)))
        else:
            ret = np.inf
    elif len(nodep.birth_death) == 2 or len(nodeq.birth_death) == 2:
        # Don't update the cost of a global min
        ret = np.inf
    return ret


def get_zss_map(T1, T2, compute_map=False):
    """
    Parameters
    ----------
    T1: zss.simple_tree.Node
        The root of the first tree
    T2: zss.simple_tree.Node
        The root of the first tree
    compute_map: bool
        Whether to compute the map that achieves the optimal distance
    """
    from zss.simple_tree import Node
    from zss.compare import distance, zssBacktrace

    ret = distance(T1, T2, Node.get_children, edit_remove_cost, edit_remove_cost, update_cost, compute_map)

    if compute_map:
        (dist, KeyrootPtrs) = ret
        (Map, BsNotHit) = zssBacktrace(KeyrootPtrs)

        #Now copy over the returned map
        Matched = []
        BsNotHit = [N.label for N in BsNotHit]
        for AZSS in Map:
            A = AZSS.label
            B = Map[AZSS]
            if B:
                B = B.label
            Matched.append((A, B))
        ret = (dist, Matched, BsNotHit)
    return ret