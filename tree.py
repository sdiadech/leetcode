import collections
from typing import List
from typing import Optional


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        # base case
        if root is None:
            return

        # revert nodes
        tmp = root.left
        root.left = root.right
        root.right = tmp
        # dfs recurse
        self.invertTree(root.left)
        self.invertTree(root.right)
        return root

    def maxDepth(self, root: Optional[TreeNode]) -> int:
        res = 0

        def dfs(root):
            nonlocal res
            if not root:
                return 0

            left = dfs(root.left)
            right = dfs(root.right)
            res = max(res, left + right)
            return 1 + max(res, left + right)

        dfs(root)
        return res

    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        def dfs(root):
            if not root:
                return [True, 0]

            left, right = dfs(root.left), dfs(root.right)
            balanced = left[0] and right[0] and abs(left[1] - right[1]) <= 1
            return [balanced, 1 + max(left[1], right[1])]

        return dfs(root)[0]

    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if not p and not q:
            return True
        if not p or not q or p.val != q.val:
            return False
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)

    def isSubtreehelper(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        if not root and not subRoot:
            return True
        if root and subRoot and root.val == subRoot.val:
            return self.isSubtreehelper(root.left, subRoot.left) and self.isSubtreehelper(root.right, subRoot.right)
        return False

    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        if not subRoot:
            return True
        if not root:
            return False
        if self.isSubtreehelper(root, subRoot):
            return True
        return self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)

    def lowestCommonAncestor(
            self, root: "TreeNode", p: "TreeNode", q: "TreeNode"
    ) -> "TreeNode":
        cur = root
        while cur:
            if p.val > cur.val and q.val > cur.val:
                cur = cur.right
            elif p.val < cur.val and q.val < cur.val:
                cur = cur.left
            else:
                return cur

    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        res = []
        queue = collections.deque()
        if root:
            queue.append(root)
        while queue:
            level_list = []
            for _ in range(len(queue)):
                level = queue.popleft()
                if level:
                    level_list.append(level.val)
                if level.left:
                    queue.append(level.left)
                if level.right:
                    queue.append(level.right)
            res.append(level_list)
        return res

    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        res = []
        queue = collections.deque()
        if root:
            queue.append(root)
        while queue:
            right_side = None
            for _ in range(len(queue)):
                level = queue.popleft()
                if level:
                    right_side = level
                    queue.append(level.left)
                    queue.append(level.right)
            if right_side:
                res.append(right_side.val)
        return res

    def goodNodes(self, root: TreeNode) -> int:
        def dfs(node, max_val):
            if not node:
                return 0
            res = 1 if node.val >= max_val else 0
            max_val = max(max_val, node.val)
            res += dfs(node.left, max_val)
            res += dfs(node.right, max_val)
            return res

        return dfs(root, root.val)

    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def is_valid(node, left, right):
            if not node:
                return True
            if not(left < node.val < right):
                return False
            return is_valid(node.left, node.val, left) and is_valid(node.right, node.val, right)
        return is_valid(root, float("-inf"), float("inf"))


def right_side_view(root):
    if root is None:
        return []

    rside = []
    dfs(root, 0, rside)

    return rside


# Apply depth-first search
def dfs(node, level, rside):
    if level == len(rside):
        rside.append(node.data)

    for child in [node.right, node.left]:
        if child:
            dfs(child, level + 1, rside)

