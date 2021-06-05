from typing import List
from Modules.NodeHealper import ListNode
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

#     @staticmethod
#     def createListNode(list: List[int]) -> 'ListNode':
#         head = ListNode(0)
#         index = head
#         for i in list:
#             index.next = ListNode(i)
#             index = index.next
#         return head.next


class Solution:
    def findMaxLength(self, nums: List[int]) -> int:
        map, count = {0: -1}, 0
        res = 0
        for i in range(nums.__len__()):
            count += 1 if nums[i] == 1 else -1
            if map.get(count) is not None:
                res = max(i-map[count], res)
            else:
                map[count] = i
        return res

    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        table = dict()
        while headA is not None:
            table[headA] = 1
            headA = headA.next
        while headB is not None:
            if table.get(headB) is not None:
                return headB
            headB = headB.next
        return None
    # 203. 移除链表元素
    def removeElements(self, head: ListNode, val: int) -> ListNode:
        newHead = ListNode(-1)
        pre,index = newHead,head
        pre.next = index
        while index is not None:
            if index.val == val:
                pre.next = index.next
            else:
                pre = pre.next
            index = index.next
        return newHead.next
if __name__ == '__main__':
    s = Solution()
    headA = ListNode.createListNode([4, 1, 8, 4, 5])
    headB = ListNode.createListNode([5, 0, 1, 8, 4, 5])
    print(s.getIntersectionNode(headA, headB))
