"""
Programmer: Chris Tralie
Purpose: A simple vanilla version of a compresssed cover tree that allows
a variable contraction constant. Not designed for speed, but more for getting
the structure to be used on applications where cover tree is not the bottleneck
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import time

######################################
#        COVER TREE FUNCTIONS        #
######################################
class CoverNode(object):
    def __init__(self, parent, idx, level):
        self.parent = parent
        self.children = []
        self.idx = idx #Index into original point set
        self.level = level #First level at which this node occurs

class CoverTree(object):
    def __init__(self, theta = 0.5, Verbose = False):
        self.root = None
        self.theta = theta #Contraction constant (0 < theta < 1)
        self.D = np.array([[]]) #Distance function
        #Quick lookup to find a node object given a point index
        self.idxtonode = []
        self.R0 = 0
        self.DistQueries = 0
        self.Verbose = Verbose
        self.equalidx = {} #Store indices of points that are equal to points
        #which have already been added.

    def dist(self, idx1, idxs2):
        """
        A wrapper function around the distance matrix that helps record the
        number of distance queries made
        """
        DRet = self.D[idx1, idxs2]
        self.DistQueries += DRet.size
        return DRet

    def getDistSavingsStr(self):
        N = self.D.shape[0]
        NTotal = N*(N-1)/2
        return "DistQueries: %i/%i (%.3g"%(T.DistQueries, NTotal, 100*float(T.DistQueries)/NTotal) + "%)"

    def construct(self, D, seedidx = 0):
        """
        Construct the cover tree from a distance matrix
        :param D: NxN matrix of all pairwise similarities
        (TODO: Smarter to do this on demand)
        :param seedidx: The root of the cover tree
        """
        self.DistQueries = 0
        self.D = D
        self.idxtonode = [None for i in range(D.shape[0])]
        self.root = CoverNode(None, seedidx, 0)
        self.idxtonode[seedidx] = self.root
        #Find the maximum distance of the point at the root node
        #to all other points, and make this the initial radius
        self.R0 = np.max(self.dist(seedidx, range(D.shape[0])))
        for i in range(self.D.shape[0]):
            if i == seedidx:
                continue
            #Start at the top of the tree and insert each point recursively
            if i%(self.D.shape[0]/40 + 1) == 0:
                sys.stdout.write(".")
                sys.stdout.flush()
            dRoot = self.dist(seedidx, i)
            if dRoot == 0: #This point is the same as the root (TODO: More numerical tolerance?)
                if not seedidx in self.equalidx:
                    self.equalidx[seedidx] = []
                self.equalidx[seedidx].append(i)
            else:
                self.insertPoint(i, [self.root], [dRoot], 0)

    def addEqualNodes(self):
        """
        This function will add all of the node which were equal to some other node
        as children (usually used at the end of the construction).  The tree will
        no longer satisfy the packing condition, but these points aren't in general
        position so that's technically impossible anyway
        """
        #First figure out the maximum level
        maxLevel = 0
        for n in self.idxtonode:
            if n:
                if n.level > maxLevel:
                    maxLevel = n.level
        #Makes these nodes one greater
        maxLevel += 1
        count = 0
        for idx1 in self.equalidx:
            n1 = self.idxtonode[idx1]
            for idx2 in self.equalidx[idx1]:
                n2 = CoverNode(n1, idx2, maxLevel)
                n1.children.append(n2)
                self.idxtonode[idx2] = n2
                count += 1

    def insertPoint(self, pidx, Ql, dpQl, l):
        """
        #Recursive function for insert points into the tree, maintaining
        #cover tree invariants (covering/packing)
        :param pidx: The index of the point p to insert
        :param Ql: An array of the nodes at level l that are being considered potential parents of p
        :param dpQl: An array of distances from p to the elements in Q (passed along to save computation)
        """
        #First check to make sure this point isn't equal to some point that's already
        #been added
        imin = np.argmin(dpQl)
        if dpQl[imin] == 0:
            #Trying to add a point that's equal to a point already in the tree
            if not Ql[imin].idx in self.equalidx:
                self.equalidx[Ql[imin].idx] = []
            self.equalidx[Ql[imin].idx].append(pidx)
            return True
        #Proceed as normal otherwise
        R = self.R0*self.theta**l
        if self.Verbose:
            print("insertPoint ", pidx, ", Ql = ", [q.idx for q in Ql], " level ", l, ", R = ", R, ", dpQl = ", dpQl)
        #Look at all of the children at the next level
        Q = []
        for node in Ql:
            for n in node.children:
                #Add nodes that are added at the next level only (not further down)
                if n.level == l+1:
                    Q.append(n)
        if len(Q) > 0:
            dpQ = self.dist(pidx, [q.idx for q in Q])
        else:
            dpQ = np.array([])
        Q = Ql + Q #By nesting, nodes at this level carry onto the next
        dpQ = np.concatenate((dpQl, dpQ)) #Save computation by reusing distances
        #RCover is different from R to handle cases where theta is not 0.5
        #(a generalization of the Beygelzimer paper)
        RCover = R*self.theta/(1.0-self.theta)
        Q = [Q[i] for i in range(len(Q)) if dpQ[i] <= RCover]
        dpQ = dpQ[dpQ <= RCover]
        if len(Q) == 0:
            if self.Verbose:
                print("RETURN HERE (len(Q) = %i), np.min(dpQl) = %g > R=%g"%(len(Q), np.min(dpQl), R))
            return False
        if self.insertPoint(pidx, Q, dpQ, l+1):
            #Try inserting it at further down levels first
            return True
        for i in range(len(Ql)):
            if dpQl[i] <= R:
                N = Ql[i]
                newNode = CoverNode(N, pidx, l+1)
                N.children.append(newNode)
                self.idxtonode[pidx] = newNode
                if self.Verbose:
                    print("Inserting %i, child of %i, at level %i"%(pidx, N.idx, l+1))
                return True
        return False

    def getSubtree(self, N, level):
        """
        Return subtree rooted at N, with N as the first element in the array
        """
        subtree = [N]
        for c in N.children:
            if c.level > level:
                subtree = subtree + self.getSubtree(c, level)
        return subtree

    def getEdges(self, N):
        """
        Return an (NPoints-1) x 2 array of edges for the cover tree on NPoints
        where each row contains the indices into the point set that make up the edge
        """
        NChildren = len(N.children)
        edges = np.zeros((NChildren, 2), dtype='int')
        for i in range(NChildren):
            edges[i, 0] = N.idx
            edges[i, 1] = N.children[i].idx
        for c in N.children:
            edges = np.concatenate((edges, self.getEdges(c)), 0)
        return edges

    def getETETreeRec(self, names, node, enode):
        for c in node.children:
            d = self.dist(node.idx, c.idx)
            cnode = enode.add_child(name=names[c.idx], dist=d)
            self.getETETreeRec(names, c, cnode)

    def getETETree(self, names):
        """
        Wrap this tree into the ETE-3 library, where the name
        of each node is its index in the point cloud and the
        distance is the distance to its parent
        """
        import ete3
        t = ete3.Tree()
        root = t.add_child(name=names[self.root.idx], dist=0)
        self.getETETreeRec(names, self.root, root)
        return t

#A quad tree-esque structure is a very easy special case of the cover tree
class FlexQuadTree(CoverTree):
    def __init__(self):
        CoverTree.__init__(self, theta = 0.5, distfn = LInfDist)

######################################
#            UNIT TESTS              #
######################################

#Test the data structure on an example point set and plot and save all
#of the different levels with the covering radii, as well as the tree
#structure all to an HTML file
if __name__ == '__main__':
    N = 200
    theta = 0.5
    t = np.linspace(0, 2*np.pi, N+1)
    t = t[0:N]
    C = np.zeros((N, 2))
    C[:, 0] = np.cos(t)
    C[:, 1] = np.sin(t)
    #4 Random circles
    X = np.zeros((0, 2))
    np.random.seed(601)
    for ii in range(4):
        X = np.concatenate((X, 2*np.random.randn(1, 2) + np.random.rand(1)*C), 0)
    X = np.random.rand(N, 2)
    #X = C
    X = X - np.min(X)
    XSqr = np.sum(X**2, 1)
    D = XSqr[:, None] + XSqr[None, :] - 2*X.dot(X.T)
    D[D < 0] = 0
    D = np.sqrt(D)

    seedidx = np.argmin(np.mean(D, 1))
    T = CoverTree(theta)
    T.construct(D, seedidx = seedidx)
    print(T.getDistSavingsStr())
    count = 0

    #Now plot levels
    t = np.linspace(0, 2*np.pi, 100)
    UnitCircle = np.zeros((100, 2))
    UnitCircle[:, 0] = np.cos(t)
    UnitCircle[:, 1] = np.sin(t)
    levels = np.array([n.level for n in T.idxtonode])
    L = np.unique(levels)
    #Plot levels with balls
    for l in L:
        r = T.R0*(T.theta**(l))
        plt.clf()
        plt.plot(X[:, 0], X[:, 1], 'b.')
        P = [i for i in range(len(levels)) if levels[i] <= l]
        plt.scatter(X[P, 0], X[P, 1], 30, 'r')
        Pc = [i for i in range(len(levels)) if levels[i] == l+1]
        plt.scatter(X[Pc, 0], X[Pc, 1], 60, 'k', marker='x')
        Circle = r*UnitCircle
        for p in P:
            C = Circle + X[p, :]
            plt.plot(C[:, 0], C[:, 1], 'b')
        plt.xlim(-0.2*np.max(X[:, 0]), 1.2*np.max(X[:, 0]))
        plt.ylim(-0.2*np.max(X[:, 1]), 1.2*np.max(X[:, 1]))
        plt.title("Level %i (r = %g) %i Nodes"%(l, r, len(P)))
        plt.savefig("%i.png"%l)

    #Plot edges
    edges = T.getEdges(T.root)
    plt.clf()
    plt.plot(X[:, 0], X[:, 1], 'b.')
    for i in range(edges.shape[0]):
        e = edges[i, :].flatten()
        plt.plot(X[e, 0], X[e, 1], 'r')
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], "%i"%levels[i])
    plt.xlim(-0.2*np.max(X[:, 0]), 1.2*np.max(X[:, 0]))
    plt.ylim(-0.2*np.max(X[:, 1]), 1.2*np.max(X[:, 1]))
    plt.savefig("TreeEdges.png", dpi=200)

    #Plot subtrees at each level
    for l in L:
        P = [i for i in range(len(levels)) if levels[i] <= l]
        subtrees = [T.getSubtree(T.idxtonode[i], l) for i in P]
        NTotal = 0
        Colors = np.random.rand(len(subtrees), 3)
        plt.clf()
        for i in range(len(subtrees)):
            s = subtrees[i]
            idx = np.array([n.idx for n in s])
            plt.plot(X[idx, 0], X[idx, 1], '.', c=Colors[i, :].flatten())
            plt.scatter(X[idx[0], 0], X[idx[0], 1], 60, 'k', marker = 'x')
        plt.savefig("Clusters%i.png"%l)

    #Now make HTML file to organize all of the results
    fout = open("index.html", "w")
    fout.write("<html>\n<body>\n<h2>NPoints = %i<BR>theta = %g<BR>%i unique levels<BR>maxlevel = %i<BR>%s</h2><BR><img src = 'TreeEdges.png'><BR><BR>\n<table><tr>"%(X.shape[0], T.theta, len(L), L[-1], T.getDistSavingsStr()))
    for l in L:
        fout.write("<td><img src = '%i.png'></td>"%l)
    fout.write("</tr>\n<tr>")
    for l in L:
        fout.write("<td><img src = 'Clusters%i.png'></td>"%l)
    fout.write("</tr></table></body></html>")
    fout.close()
