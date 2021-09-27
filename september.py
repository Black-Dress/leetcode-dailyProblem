import math
from collections import defaultdict
from typing import Collection, List
from NodeHelper.ListNode import ListNode


class Node:
    def __init__(self, val, prev, next, child):
        self.val = val
        self.prev = prev
        self.next = next
        self.child = child

    def __init__(self, val):
        self.val = val
        self.prev = None
        self.next = None
        self.child = None


class Solution:
    # 回旋镖的数量，找到两个点的中点
    # 暴力法
    def numberOfBoomerangs(self, points: List[List[int]]) -> int:
        cnt, res = defaultdict(int), 0
        for i in range(points.__len__()):
            cnt.clear()
            for j in range(points.__len__()):
                distance = math.pow(points[i][0] - points[j][0], 2) + math.pow(points[i][1] - points[j][1], 2)
                cnt[distance] += 1
            for (k, v) in cnt.items():
                res += v * (v - 1)
        return res

    # 524. 通过删除字母匹配到字典里最长单词
    def findLongestWord(self, s: str, dictionary: List[str]) -> str:
        # 先判断 s 是否含有字典中的某一个值
        # 只要源字符串包含了匹配串的所有字符，并且按照匹配串的顺序 则成功匹配
        # 双指针匹配
        def match(s: str, ss: str) -> bool:
            i, j, p, q = 0, s.__len__() - 1, 0, ss.__len__() - 1
            while i <= j:
                while s[i] != ss[p] and i < j:
                    i += 1
                p += 1 if s[i] == ss[p] else 0
                i += 1
                while s[j] != ss[q] and i < j:
                    j -= 1
                q -= 1 if s[j] == ss[q] else 0
                j -= 1
                if p > q:
                    return True
            return False
        dictionary.sort()
        dictionary.sort(key=lambda x: len(x), reverse=True)
        for i in dictionary:
            if match(s, i):
                return i
        return ""

    # 162. 寻找峰值
    # 你必须实现时间复杂度为 O(log n) 的算法来解决此问题。
    def findPeakElement(self, nums: List[int]) -> int:
        l, r = 0, nums.__len__() - 1
        while l < r:
            mid = (l + r) >> 2
            if nums[mid] < nums[mid + 1]:
                l = mid + 1
            else:
                r = mid
        return l

    # 725. 分隔链表
    def splitListToParts(self, head: ListNode, k: int) -> List[ListNode]:
        index, cnt = head, 0
        nodes = []
        while index is not None:
            cnt += 1
            nodes.append(index)
            index = index.next
        mod = cnt % k
        cnt = 1 if cnt < k else cnt // k
        j = 0
        res = []
        cnt += 1 if mod > 0 else 0
        for i in range(k):
            cnt -= 1 if mod == 0 else 0
            if j < nodes.__len__():
                res.append(nodes[j:j + cnt])
            else:
                res.append([])
            j += cnt
            mod -= 1
        return res

    # 326. 3的幂
    def isPowerOfThree(self, n: int) -> bool:
        if n == 1:
            return True
        if n < 1:
            return False
        return self.isPowerOfThree(n / 3)

    # 430. 扁平化多级双向链表
    def flatten(self, head: 'Node') -> 'Node':
        def DFS(index: 'Node') -> Node:
            while index is not None and (index.next is not None or index.child is not None):
                next = index.next
                if index.child is not None:
                    index.next = index.child
                    index.child.prev = index
                    child = index.child
                    index.child = None
                    index = DFS(child)
                index.next = next
                if next is None:
                    return index
                next.prev = index
                index = index.next
            return index
        res = DFS(head)
        while res is not None and res.prev is not None:
            res = res.prev
        return res


s = Solution()
a = Node(1)
b = Node(2)
c = Node(3)
d = Node(4)
a.next = b
b.next = c
c.child = d
b.prev = a
c.prev = b
print(s.flatten(None))
