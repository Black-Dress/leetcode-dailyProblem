from collections import defaultdict
import collections
from typing import Collection, List


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

    # 787. K 站中转内最便宜的航班
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        maxcnt = 1000000*n
        dp = [[maxcnt]*n for i in range(k+2)]
        dp[0][src] = 0
        for t in range(k+2):
            for j, i, price in flights:
                dp[t][i] = min(dp[t][i], dp[t-1][j]+price)
        res = min(dp[t][dst] for t in range(k+2))
        return -1 if res == maxcnt else res


s = Solution()
print(s.findCheapestPrice(10, [[3, 4, 4], [2, 5, 6], [4, 7, 10], [9, 6, 5], [7, 4, 4], [6, 2, 10], [6, 8, 6], [7, 9, 4], [1, 5, 4], [1, 0, 4], [9, 7, 3], [
      7, 0, 5], [6, 5, 8], [1, 7, 6], [4, 0, 9], [5, 9, 1], [8, 7, 3], [1, 2, 6], [4, 1, 5], [5, 2, 4], [1, 9, 1], [7, 8, 10], [0, 4, 2], [7, 2, 8]], 6, 0, 7))
