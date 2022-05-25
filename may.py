from typing import Counter, List
from collections import defaultdict


class Solution:
    def findRightInterval(self, intervals: List[List[int]]) -> List[int]:
        # 记录所有的右边界
        max_l = max([l for l, r in intervals])
        index_r, index_l = defaultdict(list), defaultdict(int)
        res = [-1] * len(intervals)
        for i in range(len(intervals)):
            index_r[intervals[i][1]].append(i)
            index_l[intervals[i][0]] = i
        intervals.sort(key=lambda x: x[0])
        # 设置标记数组
        nums, index = defaultdict(int), 0
        for i in range(intervals[0][0], max_l + 1):
            if intervals[index][0] < i:
                index += 1
            nums[i] = intervals[index][0]
        # 找到对应结果
        for k, v in index_r.items():
            for i in v:
                res[i] = index_l[nums[k]] if k <= max_l else -1
        return res

    # 675. 为高尔夫比赛砍树
    def cutOffTree(self, forest: List[List[int]]) -> int:
        # 利用优先队列 存储下一个位置，并且计算从当前位置到下一个位置需要花费的步数
        m, n = len(forest), len(forest[0])

        def bfs(sx: int, sy: int, ex: int, ey: int) -> int:
            points, dx, dy = [(sx, sy)], [1, -1, 0, 0], [0, 0, -1, 1]
            index, d = 1, 0
            vist = {(sx, sy): 1}
            while points:
                cur = points.pop(0)
                index -= 1
                if cur[0] == ex and cur[1] == ey:
                    return d
                for i in range(4):
                    x, y = cur[0] + dx[i], cur[1] + dy[i]
                    if x >= 0 and x < m and y >= 0 and y < n and forest[x][y] >= 1 and (x, y) not in vist:
                        points.append((x, y))
                        vist[(x, y)] = 1
                if index == 0:
                    d += 1
                    index = len(points)
            return -1
        nxt = []
        pre = [0, 0]
        res = 0
        for i in range(m):
            for j in range(n):
                if forest[i][j] > 1:
                    nxt.append([i, j, forest[i][j]])
        nxt.sort(key=lambda x: x[2])
        for i in nxt:
            d = bfs(pre[0], pre[1], i[0], i[1])
            if d < 0:
                return -1
            forest[i[0]][i[1]] = 1
            res += d
            pre = i
        return res

    # 467. 环绕字符串中唯一的子字符串
    def findSubstringInWraproundString(self, p: str) -> int:
        # 找到p中按照字典序排列的子串
        def ord_(s: str) -> int:
            return ord(s) - ord('a')
        k, n = 1, len(p)
        dp = defaultdict(int)
        dp[p[0]] = 1
        for i in range(1, n):
            if ord_(p[i]) == (ord_(p[i - 1]) + 1) % 26:
                k += 1
            else:
                k = 1
            dp[p[i]] = max(k, dp[p[i]])
        return sum(dp.values())


s = Solution()
print(s.findSubstringInWraproundString("abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz"))
