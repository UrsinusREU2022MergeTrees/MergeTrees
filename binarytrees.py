import numpy as np
import matplotlib.pyplot as plt

class TreeNode(object):
    def __init__(self, key=""):
        self.key = key
        self.left = None
        self.right = None
        self.inorder_pos = 0
        self.origin = None
    
    def is_leaf(self):
        return (not self.left) and (not self.right)

    def inorder(self, num, key_list):
        """
        Parameters
        ----------
        num: list
            List of a single element which keeps 
            track of the number I'm at
        """
        if self.left:
            self.left.inorder(num, key_list)
        self.inorder_pos = num[0]
        key_list.append(self.key)
        num[0] += 1
        if self.right:
            self.right.inorder(num, key_list)
    
    def draw(self, y):
        y = self.key
        x = self.inorder_pos
        color = ('green' if self.origin == 'augment' else 'black')
        # plt.scatter([x], [y], 50, 'k')
        plt.scatter([x], [y], 50, c=color)
        plt.text(x+0.2, y, "{}".format(self.key))
        # y_next = y-1
        if self.left:
            y_next = self.left.key
            x_next = self.left.inorder_pos
            plt.plot([x, x_next], [y, y_next])
            self.left.draw(y_next)
        if self.right:
            y_next = self.right.key
            x_next = self.right.inorder_pos
            plt.plot([x, x_next], [y, y_next])
            self.right.draw(y_next)
        
        
class BinaryTree(object):
    def __init__(self):
        self.root = None
    
    def inorder(self):
        key_list = []
        if self.root:
            self.root.inorder([0], key_list)
        return key_list
    
    def draw(self):
        self.inorder()
        if self.root:
            self.root.draw(0)
    
    def get_quaternary_code(self):
        """
        Compute the Kirk sequence for this tree

        Returns
        -------
        list of length 2n-3 for leaf nodes
        """
        from collections import deque
        seq = []
        if self.root:
            # Do a breadth-first search to visit the nodes
            # level by level
            nodes = deque()
            nodes.append(self.root)
            while len(nodes) > 0:
                node = nodes.popleft()
                if not node.is_leaf():
                    if node.left.is_leaf():
                        seq.append(0)
                    else:
                        seq.append(1)
                        nodes.append(node.left)
                    if node.right.is_leaf():
                        seq.append(0)
                    else:
                        seq.append(1)
                        nodes.append(node.right)
            seq = seq[0:-2] # Chop off last two zeros
        return seq


def weightsequence_to_binarytree(pws):
    """
    Convert a weight sequence (as defined by Pallo) 
    into a binary tree object by pairing the appropriate nodes
    """
    ws = [w for w in pws]
    ws.append(len(ws)+1) # The last element is implied
    N = len(ws)
    nodes = [TreeNode() for i in range(N)]
    i = 0
    while i < len(ws):
        k = 0
        while k < ws[i]-1:
            # Pair n(i), n(i-1)
            parent = TreeNode()
            parent.left = nodes[i-1]
            nodes[i-1].parent = parent
            parent.right = nodes[i]
            nodes[i].parent = parent
            k += ws[i-1]
            # Coalesce two nodes
            # TODO: A more efficient way to do this would be
            # using a linked list
            ws = ws[0:i-1] + ws[i::]
            nodes = nodes[0:i-1] + [parent] + nodes[i+1::]
            i -= 1
        i += 1
    T = BinaryTree()
    T.root = nodes[0]
    return T

def enumerate_weightsequences(N):
    """
    Enumerate all of the weight sequences
    for a tree with N internal nodes
    Parameters
    ----------
    N: int
        Number of internal nodes
    """
    ws = [np.ones(N, dtype=int)]
    w = np.ones(N, dtype=int)
    finished = False
    while not finished:
        i = N-1
        while w[i] >= i+1 and i >= 0:
            i -= 1
        if i == -1:
            finished = True
        else:
            j = i - w[i]
            w[i] += w[j]
            for m in range(i+1, N):
                w[m] = 1
            ws.append(np.array(w))
    return ws

def enumerate_trees(N):
    """
    Enumerate all of the weight sequences
    for a tree with N internal nodes
    Parameters
    ----------
    N: int
        Number of internal nodes
    """
    return [weightsequence_to_binarytree(w) for w in enumerate_weightsequences(N)]
