import collections
from typing import Collection, Dict, List, Literal
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
        while l <= r:
            mid = (l + r) >> 1
            if nums[mid] == target:
                return mid
            # 左半区
            if target >= nums[0]:
                if nums[0] <= nums[mid] < target:
                    l = mid + 1
                else:
                    r = mid - 1
            # 右半区
            else:
                if target < nums[mid] < nums[0]:
                    r = mid - 1
                else:
                    l = mid + 1
        return -1

    # 39. 组合总和
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        # cur 第一个位置存储和
        def DFS(candiates: List[int], target: int, index: int, cur: List[int], res: List[List[int]]):
            for i in range(index, len(candidates)):
                if cur[0] == target:
                    res.append(cur[1:].copy())
                    return
                if cur[0] < target:
                    cur[0] += candidates[i]
                    cur.append(candidates[i])
                    DFS(candiates, target, i, cur, res)
                    cur[0] -= candidates[i]
                    cur.pop()
                if cur[0] > target:
                    return

        candidates.sort()
        res = []
        DFS(candidates, target, 0, [0], res)
        return res

    # 40. 组合总和 II
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        # 题目要求不能选取重复数字，并且不能有重复结果
        # 需要记录 cur 的状态，如果 cur 状态相同那么就不需要再进入循环了
        # cur 第一个位置存储数组之和
        # def DFS(candiates: List[int], target: int, index: int, cur: List[int], res: List[List[int]]):
        #     # cur 状态已经走过，return
        #     if cnt.get("".join(str(i) for i in cur[1:])) is not None:
        #         return
        #     if cur[0] == target:
        #         res.append(cur[1:].copy())
        #         return
        #     for i in range(index, len(candidates)):
        #         if cur[0] < target:
        #             cur[0] += candidates[i]
        #             cur.append(candidates[i])
        #             # 从下一位开始
        #             DFS(candiates, target, i + 1, cur, res)
        #             # 进入一次循环之后更新，状态
        #             cnt["".join(str(j) for j in cur[1:])] = 1
        #             cur[0] -= candidates[i]
        #             cur.pop()
        #         if cur[0] > target:
        #             return

        # candidates.sort()
        # res, cnt = [], collections.defaultdict(str)
        # DFS(candidates, target, 0, [0], res)
        # return res

        def DFS(candiates: List[int], target: int, index: int, cur: List[int], res: List[List[int]]):
            if cur[0] == target:
                res.append(cur[1:].copy())
                return
            for i in range(index, candiates.__len__()):
                if i > index and candiates[i] == candiates[i - 1]:
                    continue
                if cur[0] < target:
                    cur[0] += candidates[i]
                    cur.append(candidates[i])
                    # 从下一位开始
                    DFS(candiates, target, i + 1, cur, res)
                    # 进入一次循环之后更新，状态
                    cur[0] -= candidates[i]
                    cur.pop()
                else:
                    break
        candidates.sort()
        res = []
        DFS(candidates, target, 0, [0], res)
        return res

    # 71. 简化路径
    def simplifyPath(self, path: str) -> str:
        stack, n = [], len(path)
        i, j = 0, 0
        while i < n and j < n:
            # 移动到目录
            while j < n and path[j] == '/':
                j += 1
            i = j
            # 移动到目录后一个斜杠
            while i < n and path[i] != '/':
                i += 1
            if j != i and path[j:i] == '..' and stack.__len__() != 0:
                stack.pop()
            if j != i and path[j:i] != '..' and path[j:i] != '.':
                stack.append("".join(path[j:i]))
            j = i
        res = "".join(['/' + i for i in stack])
        return "/" if res == "" else res


s = Solution()
a = ListNode.createListNode([1, 4, 5])
b = ListNode.createListNode([1, 3, 4])
c = ListNode.createListNode([2, 6])
print(s.combinationSum2([10, 1, 2, 7, 6, 1, 5], 8))
