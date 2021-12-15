from typing import List
from queue import PriorityQueue, Queue
import heapq
import sys


class Solution:
    # 1446. 连续字符
    def maxPower(self, s: str) -> int:
        maxnum, cur = 1, 1
        for i in range(1, len(s)):
            if s[i] == s[i - 1]:
                cur += 1
            else:
                cur = 1
            maxnum = max(cur, maxnum)
        return maxnum

    # 1005. K 次取反后最大化的数组和
    def largestSumAfterKNegations(self, nums: List[int], k: int) -> int:
        minqueue = PriorityQueue()
        for num in nums:
            minqueue.put(num)
        for i in range(k):
            item = minqueue.get()
            minqueue.put(-item)
        return sum(minqueue.queue)

    # 1816. 截断句子
    def truncateSentence(self, s: str, k: int) -> str:
        index = 0
        while k != 0:
            k -= 1 if s[index] == ' ' or index == len(s) - 1 else 0
            index += 1
        return s[:index].strip()

    # 1034. 边界着色
    def colorBorder(self, grid: List[List[int]], row: int, col: int, color: int) -> List[List[int]]:
        que, dx, dy = set(), [0, 0, 1, -1], [1, -1, 0, 0]
        m, n, origin = len(grid), len(grid[0]), grid[row][col]
        vist = [[0 for i in range(n)] for j in range(m)]
        que.add((row, col))
        vist[row][col] = 1
        while len(que) != 0:
            item = que.pop()
            # 判断是否需要变更颜色
            for i in range(4):
                x, y = item[0] + dx[i], item[1] + dy[i]
                # 在边界
                if x >= m or x < 0 or y >= n or y < 0:
                    grid[item[0]][item[1]] = color
                else:
                    # 在连通边界
                    if grid[x][y] != origin and vist[x][y] == 0:
                        grid[item[0]][item[1]] = color
                    else:
                        # 没有访问过，但是连通
                        if vist[x][y] == 0:
                            que.add((x, y))
                        vist[x][y] = 1
        return grid

    # 689. 三个无重叠子数组的最大和
    def maxSumOfThreeSubarrays(self, nums: List[int], k: int) -> List[int]:
        # sum[i] 指 nums[i,k] 的和
        # l[i] = k 表示 0~i 最大sum[k],0<k<i
        # r[i] = k 表示 i~n-1 最大sum[k],i<k<n-1
        # 枚举所有的 i 得到 max(sum[l[i-k]]+sum[i]+sum[r[i+k]])
        # 只有比maxnum大的时候才更新res，就能够维护最小的字典序
        # 需要利用滑动窗口计算和，不然会超时
        sum_ = [sum(nums[i] for i in range(k))]
        for i in range(1, len(nums) - k + 1):
            sum_.append(sum_[-1] - nums[i - 1] + nums[i + k - 1])
        l, r = [0 for i in range(len(sum_))], [len(nums) - k for i in range(len(sum_))]
        for i in range(1, len(sum_)):
            l[i] = i if sum_[i] > sum_[l[i - 1]] else l[i - 1]
        for i in range(len(sum_) - 2, -1, -1):
            r[i] = i if sum_[i] >= sum_[r[i + 1]] else r[i + 1]
        maxnum, res = 0, []
        for i in range(k, len(sum_) - k):
            cur = sum_[i] + sum_[l[i - k]] + sum_[r[i + k]]
            if cur > maxnum:
                maxnum = cur
                res = [l[i - k], i, r[i + k]]
        return res

    # 807. 保持城市天际线
    def maxIncreaseKeepingSkyline(self, grid: List[List[int]]) -> int:
        n, res = len(grid), 0
        row, col = [0 for i in range(n)], [0 for i in range(n)]
        for i in range(n):
            row[i] = max(grid[i])
            col[i] = max(grid[j][i] for j in range(n))
        for i in range(n):
            for j in range(n):
                res += min(row[i], col[j]) - grid[i][j]
        return res

    # 198. 打家劫舍
    def rob(self, nums: List[int]) -> int:
        # dp[i][0] 表示 0—i 且 i 不选能够偷盗的最大金额
        # dp[i][1] 表示 0-i 且 i 选择能够偷盗的最大金额
        # dp[i][0] = max(dp[i-1][1],dp[i-1][0])
        # dp[i][1] = dp[i-1][0]+nums[i]
        n = len(nums)
        # dp = [[0, 0] for i in range(n)]
        # dp[0][1] = nums[0]
        # for i in range(1, n):
        #     dp[i][0] = max(dp[i - 1][0], dp[i - 1][1])
        #     dp[i][1] = dp[i - 1][0] + nums[i]
        # return max(dp[n - 1][0], dp[n - 1][1])
        if n == 1:
            return nums[0]
        dp = [0 for i in range(n)]
        dp[0], dp[1] = nums[0], max(nums[0], nums[1])
        for i in range(2, n):
            dp[i] = max(dp[i - 2] + nums[i], dp[i - 1])
        return dp[n - 1]

    # 630. 课程表 III
    def scheduleCourse(self, courses: List[List[int]]) -> int:
        # 贪心：先关闭就先开始学习
        # 如果某一个东西不能学习，判断结果中的持续时间最长的和当前课程进行比较，替换较小持续时长即可
        # python 是小根堆
        courses.sort(key=lambda x: x[1])
        res, cur = [], 0
        for d, l in courses:
            if cur + d <= l:
                cur += d
                heapq.heappush(res, -d)
            elif len(res) != 0 and res[0] < -d:
                cur -= -res[0] - d
                heapq.heappop(res)
                heapq.heappush(res, -d)
        return len(res)

    # 851. 喧闹和富有
    def loudAndRich(self, richer: List[List[int]], quiet: List[int]) -> List[int]:
        # 利用dict 存储【直接】比 person i 更有钱的人的下标，或者n*n的矩阵进行存储
        # 最后遍历所有的人，在dict中进行深度遍历得到最小quite
        n = len(quiet)
        grid, res = [[0] * n for i in range(n)], [i for i in range(n)]
        vist = [0] * n
        for i, j in richer:
            grid[j][i] = 1

        # 深度优先遍历查询所有比index小的值,并且更新结果
        def DFS(grid: List[List[int]], index: int, res: List[int]) -> int:
            if vist[index] == 1:
                return res[index]
            for i in range(n):
                if grid[index][i] != 0:
                    var = DFS(grid, i, res)
                    res[index] = var if quiet[var] < quiet[res[index]] else res[index]
            vist[index] = 1
            return res[index]

        for i in range(n):
            DFS(grid, i, res)
        return res


s = Solution()
print(s.loudAndRich([], [0]))
