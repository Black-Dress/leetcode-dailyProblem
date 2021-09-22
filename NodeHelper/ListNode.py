from typing import List


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
