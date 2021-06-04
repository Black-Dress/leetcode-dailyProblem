from typing import List


class TreeNode():
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    @staticmethod
    def createTreeNode(list: List[int]) -> 'TreeNode':
        root = TreeNode(list[0])
        cur = [root]
        for i in range(1, list.__len__(), 2):
            index = cur.pop(0)
            if index is None:
                continue
            index.left = TreeNode(list[i]) if list[i] is not None else None
            index.right = TreeNode(list[i+1]) if list[i+1] is not None else None
            cur.extend([index.left, index.right])
        return root


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

    @staticmethod
    def createListNode(list: List[int]) -> 'ListNode':
        head = ListNode(0)
        index = head
        for i in list:
            index.next = ListNode(i)
            index = index.next
        return head.next
