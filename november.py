import collections
from typing import Collection, List


class Solution:
    # 575. 分糖果
    def distributeCandies(self, candyType: List[int]) -> int:
        n, cnt = len(candyType), 0
        table = Collection.defaultdict(int)
        for i in candyType:
            if cnt >= n / 2:
                break
            if table.get(i) is None:
                table[i] += 1
                cnt += 1
        return table.__len__()

    # 367. 有效的完全平方数
    def isPerfectSquare(self, num: int) -> bool:
        l, r = 0, num
        while l <= r:
            mid = (l + r) // 2
            square = mid * mid
            if square < num:
                l = mid + 1
            elif square > num:
                r = mid - 1
            else:
                return True
        return False

    # 1218. 最长定差子序列
    def longestSubsequence(self, arr: List[int], difference: int) -> int:
        # dp[i] 表示 （0，i）的最长定差子序列的长度
        # 使用 map 记录 arr[i] 在 dp 中靠右的位置
        # dp, n = [1] * arr.__len__(), len(arr)
        # table = collections.defaultdict(int)
        # for i in range(n):
        #     if table.get(arr[i] - difference) is not None:
        #         j = table[arr[i] - difference]
        #         dp[i] = dp[j] + 1
        #     table[arr[i]] = i
        # return max(dp)
        dp = collections.defaultdict(int)
        for v in arr:
            # defaultdict 不存在的时候返回0
            dp[v] = dp[v - difference] + 1
        return max(dp.values())

    # 299. 猜数字游戏
    def getHint(self, secret: str, guess: str) -> str:
        # 如何找到 数值相同但是位置不同的数字
        bulls, cows = 0, 0
        table = collections.defaultdict(list)
        n = len(secret)
        isbull = [0] * n
        for i in range(n):
            table[secret[i]].append(i)
        # 更新bulls
        for i in range(n):
            if secret[i] == guess[i]:
                bulls += 1
                table[secret[i]].pop()
                isbull[i] = 1
        for i in range(n):
            if table[guess[i]].__len__() != 0 and isbull[i] == 0:
                cows += 1
                table[guess[i]].pop()

        return str(bulls) + "A" + str(cows) + "B"

    # 495. 提莫攻击
    def findPoisonedDuration(self, timeSeries: List[int], duration: int) -> int:
        begin, end = timeSeries[0], timeSeries[0] + duration
        res = 0
        for time in timeSeries[1:]:
            if time > end:
                res += end - begin
                begin, end = time, time + duration
            else:
                end = max(end, time + duration)
        return res + end - begin

    # 375. 猜数字大小 II
    def getMoneyAmount(self, n: int) -> int:
        # dp 状态转移方程
        # 从[i,j]猜测一个k，需要考虑最坏的情况 所以max
        # 但是需要最优解所以在k的循环中时提取最小解
        # dp[i][j] = k+max(dp[i][k-1],dp[k+1][j])
        dp = [[0] * (n + 1) for i in range(n + 1)]
        for i in range(n - 1, 0, -1):
            for j in range(i + 1, n + 1, 1):
                dp[i][j] = min(k + max(dp[i][k - 1], dp[k + 1][j]) for k in range(i, j))
        return dp[1][n]


s = Solution()
print(s.findPoisonedDuration([1], 2))
