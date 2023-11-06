# class Node:
#   def __init__(self, val):
#     self.val = val
#     self.left = None
#     self.right = None
from collections import deque


def breadth_first_values(root):
    if root is None:
        return []
    res = []
    q = deque()
    q.append(root)
    while q:
        cur = q.popleft()
        res.append(cur.val)
        if cur.left:
            q.append(cur.left)
        if cur.right:
            q.append(cur.left)
    return res


def tree_includes(root, target):
    if root is None:
        return False
    if root.val == target:
        return True

    return tree_includes(root.left, target) or tree_includes(root.right, target)


def tree_sum(root):
    if root is None:
        return 0
    return root.val + tree_sum(root.left) + tree_sum(root.right)


def tree_min_value(root):
    if root is None:
        return float('inf')
    left = tree_min_value(root.left)
    right = tree_min_value(root.right)
    smalest = min(root.val, left, right)
    return smalest
