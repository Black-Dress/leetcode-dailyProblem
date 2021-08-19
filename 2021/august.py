from collections import defaultdict
import collections
from typing import Collection, List
import scipy.special


class Solution:
    def __init__(self):
        self.status_576 = collections.defaultdict()

    # 576
    def findPaths(self, m: int, n: int, maxMove: int, startRow: int, startColumn: int) -> int:
        # 普通dfs 超时，需要记忆状态
        mod = 10**9+7
        direction = [(0, 1), (0, -1), (-1, 0), (1, 0)]
        if startColumn < 0 or startColumn == n or startRow < 0 or startRow == m:
            return 1
        if maxMove == 0 or (m - maxMove > startRow > maxMove - 1 and n - maxMove > startColumn > maxMove - 1):
            return 0
        res = 0
        # 状态压缩
        key = startRow*2500 + startColumn*50 + maxMove
        if(self.status_576.get(key)):
            return self.status_576[key]
        for dx, dy in direction:
            res = (res + self.findPaths(m, n, maxMove-1, startRow+dx, startColumn+dy)) % mod
        self.status_576[key] = res
        return res

    # 526. 优美的排列
    def countArrangement(self, n: int) -> int:
        def dfs_526(cur: List[int]) -> int:
            if cur.__len__() == n:
                return 1
            res = 0
            for i in range(1, n+1):
                if cur.count(i) == 0 and (i % (cur.__len__()+1) == 0 or (cur.__len__()+1) % i == 0):
                    cur.append(i)
                    res += dfs_526(cur)
                    cur.pop()
            return res
        return dfs_526([])

    # 551. 学生出勤记录 I
    def checkRecord(self, s: str) -> bool:
        sl = list(s)
        a, l = 0, 0
        for i in range(0, sl.__len__()):
            if sl[i] == 'A':
                a += 1
            if sl[i] == 'L':
                if i > 0 and sl[i-1] == 'L':
                    l += 1
                else:
                    l = 1
            if a >= 2 or l >= 3:
                return False
        return True

    # 剑指 Offer II 119. 最长连续序列
    def longestConsecutive(self, nums: List[int]) -> int:
        table, res = set(nums), 0
        for num in nums:
            if num-1 not in table:
                next, cnt = num+1, 1
                while next in table:
                    next += 1
                    cnt += 1
                res = max(res, cnt)
        return res

    # 552. 学生出勤记录 II
    # 未完成
    def checkRecord2(self, n: int) -> int:
        mod = 10**9+7
        maxL = n//3*2+n % 3
        # 计算只有a和p的情况或者没有a
        res = 1+n
        for i in range(1, maxL+1):
            # 从maxl中选出i个位置用来替换成L
            a = scipy.special.comb(maxL, i) % mod
            # 将 一个 p 替换成 a 和不替换的情况数量
            b = n-i+1
            # 计算总数
            res += (a*b) % mod
        return res

    # 345. 反转字符串中的元音字母
    def reverseVowels(self, s: str) -> str:
        target = list(s)
        L, R = 0, target.__len__()-1
        vowels = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']
        while L < R:
            while L < R and vowels.count(target[L]) == 0:
                L += 1
            while L < R and vowels.count(target[R]) == 0:
                R -= 1
            if L < R:
                target[L], target[R] = target[R], target[L]
                L += 1
                R -= 1
        return "".join(target)


s = Solution()
print(s.reverseVowels("hello"))
