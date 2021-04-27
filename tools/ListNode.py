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
            index = cur.pop()
            if index is None:
                continue
            index.left = TreeNode(list[i]) if list[i] is not None else None
            index.right = TreeNode(list[i+1]) if list[i+1] is not None else None
            cur.extend([index.left, index.right])
        return root


if __name__ == '__main__':
    pass
