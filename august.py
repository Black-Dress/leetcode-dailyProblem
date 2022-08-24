from collections import defaultdict
from curses.ascii import isdigit
from email.policy import default
from operator import eq
from sys import stdin


class Solution:
    def solveEquation(self, equation: str) -> str:
        factor = val = 0
        i, n, sign = 0, len(equation), 1  # 等式左边默认系数为正
        while i < n:
            if equation[i] == '=':
                sign = -1
                i += 1
                continue

            s = sign
            if equation[i] == '+':  # 去掉前面的符号
                i += 1
            elif equation[i] == '-':
                s = -s
                i += 1

            num, valid = 0, False
            while i < n and equation[i].isdigit():
                valid = True
                num = num * 10 + int(equation[i])
                i += 1

            if i < n and equation[i] == 'x':  # 变量
                factor += s * num if valid else s
                i += 1
            else:  # 数值
                val += s * num

        if factor == 0:
            return "No solution" if val else "Infinite solutions"
        return f"x={-val // factor}"

    def solution1():
        a, b = stdin.readline().split(" ")
        # 大数是小数的倍数，大数可以删除数字小数可以删除数字
        # 最少的操作次数
        # 如何删除？永远删除大数的数字？之后再删除小数的数字？

    def solution4(self):
        n = int(stdin.readline())
        nums = list(map(int, stdin.readline().split(" ")))
        cnt = defaultdict(list)
        res = 0

        for i in range(n):
            cnt[nums[i]].append(i)
        for v in cnt.values():
            if len(v) >= 2:
                index = []
                for i in range(1, len(v)):
                    index.append(0)
                    for j in range(v[i - 1], v[i]):
                        if nums[j] < nums[v[0]]:
                            index[-1] += 1
                for i in range(len(index)):
                    res += (len(index) - i) * index[i]
                    if i != 0 and i != len(index) - 1:
                        res += (len(index) - i) * index[i]
        print(res)

    def solution3(self):
        n = int(stdin.readline())
        nums = list(map(int, stdin.readline().split(" ")))
        a, b = nums[0], nums[1]
        sum_a, sum_b = 0, 0
        res = 0
        for i, val in enumerate(nums):
            if (i + 1) % 2 == 0:
                sum_a += val
                if val > a:
                    a = val

            if (i + 1) % 2 != 0:
                sum_b += val
                if val > b:
                    b = val

        if a != b:
            res = a * (n // 2) - sum_a + b * ((n + 1) // 2) - sum_b
        else:
            res = min((a + 1) * (n // 2) - sum_a + b * ((n + 1) // 2) - sum_b, a * (n // 2) - sum_a + (b + 1) * ((n + 1) // 2) - sum_b)
        print(res)

    def solution2():
        s = stdin.readline()
        max_num = (s // 5) * 2 + (s // 3)
        i, res = 0, 0
        reds = s.split("red")
        for i in reds:
            if len(i) >= 2 and (i[:-2] == "de" or i[:2] == "er"):
                pass

    # 97. 交错字符串
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        # dp[i][j] 表示 s1[0:i],s2[0:j]能否够组成s3[0:i+j]
        if len(s3) != len(s1) + len(s2):
            return False
        dp = [[False] * (len(s2) + 1)for _ in range(len(s1) + 1)]
        dp[0][0] = True
        for i in range(len(s1) + 1):
            for j in range(len(s2) + 1):
                if i > 0:
                    dp[i][j] = dp[i][j] or dp[i - 1][j] and s1[i - 1] == s3[i + j - 1]
                if j > 0:
                    dp[i][j] = dp[i][j] or dp[i][j - 1] and s2[j - 1] == s3[i + j - 1]
        return dp[len(s1)][len(s2)]


s = Solution()
print(s.isInterleave("a",
                     "b",
                     "a"))
