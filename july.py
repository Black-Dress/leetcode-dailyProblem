from collections import defaultdict
from datetime import datetime
from email.policy import default
from math import gcd
import math
from textwrap import indent
from typing import Counter, List
from NodeHelper.ListNode import ListNode
from test import transform


class dsu:
    def __init__(self, nums: List[int]):
        self.parent = defaultdict(int)
        for i in nums:
            self.parent[i] = i

    def find(self, x: int) -> int:
        if x == self.parent[x]:
            return x
        self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: int, b: int):
        if a not in self.parent:
            self.parent[a] = a
        if b not in self.parent:
            self.parent[b] = b
        ap, bp = self.find(a), self.find(b)
        if ap != bp:
            self.parent[ap] = bp


class Solution:
    # 556. 下一个更大元素 III
    def nextGreaterElement(self, n: int) -> int:
        # 需要从后排找到一个大于k的最小值
        s = str(n)
        if s == "".join(sorted(str(n), reverse=True)):
            return -1
        r, nums, cnt = len(s), list(s), defaultdict(int)
        while r > 0:
            r -= 1
            cnt[int(nums[r])] = r
            if nums[r] <= nums[r - 1]:
                continue
            # 找到大于nums[r-1]的最小值
            i = int(nums[r - 1]) + 1
            while i <= 9 and i not in cnt:
                i += 1
            nums[r - 1], nums[cnt[i]] = nums[cnt[i]], nums[r - 1]
            break
        nums = nums[:r] + sorted(nums[r:])
        res = int("".join(nums))
        return res if res <= 2**31 - 1 else -1

    def largestComponentSize(self, nums: List[int]) -> int:
        # 并查集，每一次的查询都需要遍历已经存在的并查集，然后合并子项
        d = dsu(nums)
        for num in nums:
            i = 2
            while i * i <= num:
                if num % i == 0:
                    d.union(num, i)
                    d.union(num, num // i)
                i += 1
        return max(Counter(d.find(i) for i in nums).values())

    def exclusiveTime(self, n: int, logs: List[str]) -> List[int]:
        # 利用栈，每次start 入栈 end 就弹出,计算完时间之后往站栈内存储一个元素，表示中间的空余量
        # 每次弹栈的时候需要减去中间的空余量
        # map记录每一个程序的时间
        # list 按照 map 的顺序返回
        stack, cnt = list(), defaultdict(int)
        for log in logs:
            temp = log.split(":")
            index, time = int(temp[0]), int(temp[-1])
            if temp[1] == "start":
                stack.append([time, 0])
            else:
                d = time - stack[-1][0] + 1
                cnt[index] += d - stack[-1][1]
                stack.pop()
                if stack:
                    stack[-1][1] += d
        return [i[1] for i in sorted(cnt.items(), key=lambda x:x[0])]

    def transform(self, root: ListNode) -> ListNode:
        head = ListNode(-1)
        head.next = root
        l, r = head, head
        while r and r.next:
            l = l.next
            r = r.next
            if r:
                r = r.next
        stack = []
        l = l.next
        while l:
            stack.append(l)
            l = l.next
            stack[-1].next = None
        index = head.next
        while index and stack:
            nxt = stack.pop()
            nxt.next = index.next
            index.next = nxt
            index = index.next.next
        index.next = None
        return head.next

    def test():
        a = datetime.now()
        b = datetime.strptime("2022-9-15", '%Y-%m-%d')


s = Solution()
root = ListNode.createListNode([1, 2, 3, 4, 5])
ListNode.print(s.transform(root=root))
