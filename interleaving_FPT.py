'''
Interleaving FPT.py

This code is built using the algorithm presented in Touli and Wang's 2019 FPT-Algorithms paper.

Author: Zach Schlamowitz
Date: Jun 27 2022

Updates with Jose Arbelo: Aug 16, 2022
'''

import binarytrees as bt

def augment_trees(tree1, tree2, delta):
    '''
    This function augments two trees, adding all nodes in the super levels of each tree.
    '''
    critical_heights_T1 = get_criticalheights(tree1)
    critical_heights_T2 = get_criticalheights(tree2)
    # print(critical_heights_T1 == critical_heights_T2)  # debugging
    for height in critical_heights_T1:
        print("From Tree 1 node(s) at height", height, "adding super level at height", height + delta, "to Tree 2.")
        root_added, new_root = add_level(tree2.root, height + delta)
        if root_added:
            tree2.root = new_root

    for height in critical_heights_T2:
        print("From Tree 2 node(s) at height", height, "adding super level at height", height - delta, "to Tree 1.")
        root_added, new_root = add_level(tree1.root, height - delta)
        if root_added:
            tree1.root = new_root

    return tree1, tree2



def get_nodeheights(node, nodeheights):
    '''
    This function creates a dictionary mapping the heights of nodes in a tree to the nodes at that height
    '''
    # Recursive Case:
    if node.left is not None:
        nodeheights = get_nodeheights(node.left, nodeheights)
    
    if node.key not in nodeheights.keys():
        nodeheights[node.key] = list()  # array will preserve left-right order
    nodeheights[node.key].append(node)  # adding existing tree node to the set of nodes at that height

    if node.right is not None:
        nodeheights = get_nodeheights(node.right, nodeheights)

    return nodeheights



def get_criticalheights(tree):
    '''
    This function identifies the critical heights of a tree, adds all levels (nodes) at those heights (using add_level()), 
    and creates a dictionary mapping the heights of nodes in a tree to the nodes at that height (using get_nodeheights())
    '''
    nodeheights = get_nodeheights(tree.root, nodeheights={})
    # print("nodeheights in get_crit:", nodeheights)

    for height in nodeheights:
        root_added, new_root = add_level(tree.root, height)
        
        if root_added:
            tree.root = new_root
    
    criticalheights = get_nodeheights(tree.root, nodeheights={})
    # print("critheights in get_crit:", criticalheights)

    return criticalheights



def add_level(root, height):
    '''
    This function adds nodes to a tree at all intesection points along a horizontal line.
    The height of this horizontal line is specificed by <height>. 
    '''
    # Base cases:
    root_added = False
    new_root = None

    if root is None or root.key == height:
        return root_added, new_root
    
    if root.key < height: # this will only happen on the actual root, if at all
        new_root = bt.TreeNode(key=height)
        new_root.left = root
        new_root.origin = 'augment'
        root_added = True
        return root_added, new_root

    # Recursive cases:
    if root.left:
        if root.left.key < height:  # if desired height is between the two nodes in question
            new_node = bt.TreeNode(key=height)
            new_node.left = root.left
            root.left = new_node
            new_node.origin = 'augment'
        else:
            root_added, new_root = add_level(root.left, height)
        
    if root.right:
        if root.right.key < height:  # if desired height is between the two nodes in question
            new_node = bt.TreeNode(key=height)
            new_node.right = root.right
            root.right = new_node
            new_node.origin = 'augment'
        else:
            root_added, new_root = add_level(root.right, height)

    return root_added, new_root

'''
NEED:
# first need to get a structure containing ALL superlevels not just OGs --> call get_nodeheights again?. YES, resolved
# potential issue: (a) need to extend root infinitely upwards so that when augment can also add new nodes above OG root height; resolved in add_level base case
                   (b) need also deal with case when lowest node on one tree is more than delta below the lowest point on other tree. 
                       Here, a delta-good map is clearly not possible (violates property 1 of ε-good definition), so just eliminate it
                       immediately; sorta resolved in DPgoodmap_main()
- A function to get all valid pairs for a given level (i.e., a way to get all valid pairs)
- A way to get all possible partitions of children set of a set S in a valid pair (S,w)

''' 

#  def get_valid_pairs()







def DPgoodmap_main(tree1, tree2, delta):
    # if |(lowest height on tree1) - (lowest height on tree2)| > delta, immediately return "NO"

    # NEED: trees with the following strutures: 
    #   <-- valid pairs (S,w) 
    # IDEA: valid_pairs: i --> vps; that is, send the ordinal height rank to a set of valid pairs at that height. 
    #       Then, we can obtain ALL valid pairs by doing valid_pairs.values(), or just the ones at a desired height
    #       by keying in at that height. 
    #   <-- super-levels (ordered!)
    #   <-- critical heights 
    #   <-- all nodes in tree and their heights 
    #   also, depth() method in treenode class
    #   way to access children sets Ch(S) and Ch(w) and way to partition Ch(S)

    # Initialize feasbility values for each valid pair:
    # feas = dict()
    # for vp in valid_pairs.values():
    #   feas[vp] = 0

    # We now will move up the super levels i = 1,...,m, in order starting from the bottom
    ## BASE CASE (i=1):
    # for vp in valid_pairs[1]:
    #   if depth(w) <= 2*delta:
    #       feas[vp] = 1
    
    ## RECURSIVE CASE (i>1):
    # THEN call DPgoodmap_recursive with i=2
    return

def DPgoodmap_recursive(tree1, tree2, delta, i, feas):
    # Base case of recursion: If we've reached the root of both trees, draw conclusions. If not, continue building feasibility.
    if i == length(valid_pairs.values()):
        if feas(rootT1, rootT2) == 1:
            print("YES. The interleaving distance between these trees is no greater than ", delta)
            return
        else:
            print("NO. The interleaving distance between these trees is greater than ", delta)
            return
    
    # (If not at roots, continue building feasibility:)
    for vp in valid_pairs[i]:
        if Ch(w) is None: # Ch(w) is empty
            if Ch(S) is None:
                feas[vp] = 1
            else:
                continue

        # Ch(w) is NOT empty! So...
        # get partitions of Ch(S). We are going to check to see if any satisfy the necessary requirements to make (S,w) feasible.
        # Suppose there are p partitions... loop over them until either one works or all fail:
        # parition_works = False
        # l = 1
        # while partition_works is False and l <= p:
        #   # suppose partition l has form P_l = S1 union S2 union ... union Sk, or, as an array: P_l = [S1 S2 ... Sk]:        
        #   partition_works = True # we start by assuming the next partition works until proven otherwise
        #   for j in range (1,k):
        #         if (P_l[j] is not None and feas(P_l[j],w[j]) == 1) or (P_l[j] is None and depth(w[j]) <= 2*delta - (heightG[i]-heightG[j-1])) is False:
        #             parition_works = False
        #             l += 1
        #             break
        # if partition_works = True:
        #   feas(S,w) = 1
    
    # i += 1
    # DPgoodmap_recursive(tree1, tree2, delta, i, feas)      
    return



