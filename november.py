import collections
from typing import Collection, List
from math import sqrt
from functools import reduce
from NodeHelper.TreeNode import TreeNode


class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children


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

    # 319. 灯泡开关
    def bulbSwitch(self, n: int) -> int:
        # 分解质因数
        # 超时
        # def func(num: int) -> int:
        #     cnt, res = collections.defaultdict(int), 1
        #     for i in range(2, num + 1):
        #         while num != 0 and num % i == 0:
        #             cnt[i] += 1
        #             num /= i
        #     for k, v in cnt.items():
        #         res *= (v + 1)
        #     return res
        # res = 0
        # for i in range(1, n + 1):
        #     if func(i) % 2 != 0:
        #         res += 1
        # return res
        # 正解
        return int(sqrt(n + 0.5))

    # # 318. 最大单词长度乘积
    # def maxProduct(self, words: List[str]) -> int:
    #     # reduce(lambda,iterable,initial)
    #     # a 是累计值,b 是迭代值
    #     mask, res = [reduce(lambda a, b: a | (1 << (ord(b) - ord('a'))), i, 0) for i in words], 0
    #     for a in range(words.__len__()):
    #         for b in range(a + 1, words.__len__()):
    #             if(mask[a] & mask[b] == 0):
    #                 res = max(res, words[a].__len__() * words[b].__len__())
    #     return res

    # 152. 乘积最大子数组
    def maxProduct(self, nums: List[int]) -> int:
        # dp[i][0] 表示0~i的最大值
        # dp[i][1] 表示0~i的最小值
        dp, res = [[i, i] for i in nums], nums[0]
        for i in range(1, len(nums)):
            dp[i][0] = max(dp[i - 1][0] * nums[i], dp[i - 1][1] * nums[i], nums[i])
            dp[i][1] = min(dp[i - 1][0] * nums[i], dp[i - 1][1] * nums[i], nums[i])
            res = max(res, dp[i][0])
        return res

    # 563. 二叉树的坡度
    def findTilt(self, root: TreeNode) -> int:
        res = 0

        def solve(root: TreeNode) -> int:
            nonlocal res
            if root is None:
                return 0
            l, r = solve(root.left), solve(root.right)
            res += abs(l - r)
            return l + r + root.val
        solve(root)
        return res

    # 397. 整数替换
    def integerReplacement(self, n: int) -> int:
        res = 0
        while n != 1:
            if n & 1 != 0:
                n += 1 if (n >> 1) & 1 != 0 and n != 3 else -1
            else:
                n >>= 1
            res += 1
        return res

    # 559. N 叉树的最大深度
    def maxDepth(self, root: Node) -> int:
        if root.children is None:
            return 1
        return max(self.maxDepth(child) for child in root.children) + 1


s = Solution()
t = Node(1, None)
a = [Node(i, None) for i in range(2, 5)]
b = [Node(i, None) for i in [5, 6]]
t.children = a
a[1].children = b

print(s.maxDepth(t))
