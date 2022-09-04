from collections import defaultdict
import math
from sys import stdin
from typing import List


def jindong01() -> int:
    n = int(stdin.readline().strip("\n"))
    nums = list(map(int, stdin.readline().strip("\n").split(" ")))
    # 真品的品质是最高的，找到最多的赝品，找到最大值的个数，然后返回n-maxnum
    maxnum, size = nums[0], 1
    for i in nums[1:]:
        if i > maxnum:
            size = 1
            maxnum = i
        elif i == maxnum:
            size += 1
    return n - size


def jindong02() -> int:
    n = int(stdin.readline().strip("\n"))
    nums = list(map(int, stdin.readline().strip("\n").split(" ")))
    # res = 0
    # for i in nums:
    #     res += dfs(i)
    # return res

    maxnum = max(nums)
    dp = [i for i in range(maxnum + 1)]
    dp[1] = 0
    res = 0
    for i in range(2, maxnum + 1):
        dp[i] = min(dp[i], dp[i - 1] + 1)
        for j in range(2, int(math.sqrt(i)) + 1):
            if i % j == 0:
                dp[i] = min(dp[i // j] + dp[j] + 1, dp[i])
    for i in nums:
        res += dp[i]
    return res


def jindong03() -> int:
    # 需要知道所有的字串，在获取子串的时候，更新结果
    # 在遍历的时候更新子串的结果，总体结果存储在res中
    res = defaultdict(int)
    s = stdin.re1adline().strip("\n")
    for i in range(len(s) - 1):
        c = []
        if s[i] == '(':
            c.append(i)
        cnt = 0
        for j in range(i + 1, len(s)):
            if s[j] == ')' and c:
                c.pop()
                cnt += 1
            if s[j] == '(':
                c.append(j)
            res[cnt] += 1
    size = 0
    for k, v in res.items():
        size += k * 2 * v
    return size


print(jindong02())
