from typing import List


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    @staticmethod
    def createTreeNode(list: List[int]):
        root = TreeNode(list[0])
        cur = [root]
        for i in range(1, list.__len__(), 2):
            index = cur.pop(0)
            if index is None:
                continue
            index.left = TreeNode(list[i]) if i < list.__len__() and list[i] is not None else None
            index.right = TreeNode(list[i+1]) if i+1 < list.__len__() and list[i+1] is not None else None
            cur.extend([index.left, index.right])
        return root


class Solution:
    # 二分一个一个找合适的天数
    def minDays(self, bloomDay: List[int], m: int, k: int) -> int:
        if m*k > bloomDay.__len__():
            return -1

        def check(day: int) -> bool:
            res, cur = 0, 0
            for bloom in bloomDay:
                cur += 1 if bloom <= day else -cur
                if cur == k:
                    cur = 0
                    res += 1
            return res >= m

        left, right = min(bloomDay), max(bloomDay)
        while left < right:
            mid = (left+right) >> 1
            if check(mid):
                right = mid
            else:
                left = mid+1
        return left

    # 872. 叶子相似的树
    def leafSimilar(self, root1: TreeNode, root2: TreeNode) -> bool:
        # def BFS(root: TreeNode) -> List[int]:
        #     stack, res = [root], []
        #     while stack.__len__() != 0:
        #         cur = stack.pop(0)
        #         if cur is not None:
        #             stack.extend([cur.left, cur.right])
        #         if cur is not None and (cur.left is None and cur.right is None):
        #             res.append(cur.val)
        #     return res
        def preorder(root: TreeNode) -> List[int]:
            if root is None:
                return []
            res = [root.val] if root.left is None and root.right is None else []
            res.extend(preorder(root=root.left))
            res.extend(preorder(root=root.right))
            return res
        a, b = preorder(root1), preorder(root2)
        return a.__eq__(b)


root1 = TreeNode.createTreeNode([1, 2, 3])
root2 = TreeNode.createTreeNode([2, 3, 2])
s = Solution()
print(s.leafSimilar(root1, root2))
