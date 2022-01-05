from typing import List
from NodeHelper.ListNode import ListNode


class Solution:
    # 23. 合并K个升序链表
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        def merge(a: ListNode, b: ListNode) -> ListNode:
            res = ListNode(0)
            index = res
            while a is not None and b is not None:
                if a.val < b.val:
                    index.next = a
                    a = a.next
                else:
                    index.next = b
                    b = b.next
                index = index.next
            if a is not None:
                index.next = a
            else:
                index.next = b
            return res.next

        def merge_(l: int, r: int, lists: List[ListNode]) -> ListNode:
            if l == r:
                return lists[l]
            if l > r:
                return None
            mid = (l + r) >> 1
            return merge(merge_(l, mid, lists), merge_(mid + 1, r, lists))

        return merge_(0, len(lists) - 1, lists)

    # 32. 最长有效括号
    def longestValidParentheses(self, s: str) -> int:
        stack, n = [-1], len(s)
        res = 0
        for i in range(n):
            if s[i] == '(':
                stack.append(i)
            else:
                stack.pop()
                if len(stack) == 0:
                    stack.append(i)
                else:
                    res = max(res, i - stack[-1])
        return res

    # 33. 搜索旋转排序数组
    def search(self, nums: List[int], target: int) -> int:
        # 确定 target和nums[0]的大小关系，确定target在左半区还是右半区
        l, r = 0, len(nums) - 1
        # while l <= r:
        #     mid = (l + r) >> 1
        #     if nums[mid] == target:
        #         return mid
        #     # 如果在左半区
        #     if nums[0] <= target < nums[mid]:
        #         r = mid - 1
        #     if nums[0] <= nums[mid] < target:
        #         l = mid + 1
        #     if nums[mid] < nums[0] and target >= nums[0]:
        #         r = mid - 1
        #     # 如果在右半区
        #     if nums[mid] < target < nums[0]:
        #         l = mid + 1
        #     if target < nums[mid] < nums[0]:
        #         r = mid - 1
        #     if nums[mid] >= nums[0] and target < nums[0]:
        #         l = mid + 1
        # return -1


s = Solution()
a = ListNode.createListNode([1, 4, 5])
b = ListNode.createListNode([1, 3, 4])
c = ListNode.createListNode([2, 6])
ListNode.print(s.mergeKLists([a, b, c]))
