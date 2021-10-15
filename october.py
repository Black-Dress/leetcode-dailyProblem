from collections import defaultdict
from typing import Collection, Counter, List, no_type_check


class Solution:
    # 187. 重复的DNA序列
    def findRepeatedDnaSequences(self, s: str) -> List[str]:
        target, table = s[0:10], defaultdict(int)
        table[target] = 1
        res = list()
        for i in range(10, s.__len__()):
            target = target[1:] + s[i]
            table[target] += 1
        for k, v in table.items():
            if v > 1:
                res.append(k)
        return res

    # 273. 整数转换英文表示
    def numberToWords(self, num: int) -> str:
        singles = ["", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
        teens = ["Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"]
        tens = ["", "Ten", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"]
        thousands = ["", "Thousand", "Million", "Billion"]
        if num == 0:
            return "Zero"

        def num_helper(num: int) -> str:
            s = ""
            if num == 0:
                return s
            elif num < 10:
                s += singles[num] + " "
            elif num < 20:
                s += teens[num - 10] + " "
            elif num < 100:
                s += tens[num // 10] + " " + num_helper(num % 10)
            else:
                s += singles[num // 100] + " " + "Hundred " + num_helper(num % 100)
            return s

        res, unit = "", int(1e9)
        for i in range(3, -1, -1):
            curNum = num // unit
            # unit 代表一个三位数字的元组，如果curNum不为零则代表进入到正常阶段
            if curNum:
                num -= curNum * unit
                res += num_helper(curNum) + thousands[i] + " "
            unit //= 1000
        return res.strip()

    # 29. 两数相除
    def divide(self, dividend: int, divisor: int) -> int:
        INT_MIN, INT_MAX = -2**31, 2**31 - 1
        res, sign = 0, 1 if (divisor > 0 and dividend > 0) or (divisor < 0 and dividend < 0) else -1
        if dividend == INT_MIN and divisor == -1:
            return INT_MAX
        dividend, divisor = abs(dividend), abs(divisor)
        while dividend >= divisor:
            temp, cnt = divisor, 1
            while (temp << 1) <= dividend:
                temp <<= 1
                cnt <<= 1
            dividend -= temp
            res += cnt
        return res if sign == 1 else -res

    # 412. Fizz Buzz
    def fizzBuzz(self, n: int) -> List[str]:
        res = []
        for i in range(1, n + 1):
            if i % 3 == 0 and i % 5 == 0:
                res.append("FizzBuzz")
            if i % 3 == 0 and i % 5 != 0:
                res.append("Fizz")
            if i % 3 != 0 and i % 5 == 0:
                res.append("Buzz")
            if i % 3 != 0 and i % 5 != 0:
                res.append(str(i))
        return res

    # 剑指 Offer II 069. 山峰数组的顶部
    def peakIndexInMountainArray(self, arr: List[int]) -> int:
        # 算法复杂度要求log(n) -> 二分查找顶端元素
        # 题目中存在的条件：
        # if i < res : arr[i] > arr[i-1]
        # if i >= res : arr[i] > arr[i+1]
        l, r, res = 1, arr.__len__() - 2, 0
        while l <= r:
            mid = (l + r) // 2
            if arr[mid] > arr[mid + 1]:
                res = mid
                r = mid - 1
            else:
                l = mid + 1
        return res

    # 38. 外观数列
    def countAndSay(self, n: int) -> str:
        if n == 1:
            return "1"
        s = self.countAndSay(n - 1)
        cnt, res = 1, ""
        for i in range(1, s.__len__()):
            if s[i] != s[i - 1]:
                res += str(cnt) + s[i - 1]
                cnt = 1
            else:
                cnt += 1
        res += str(cnt) + s[s.__len__() - 1]
        return res


s = Solution()
print(s.countAndSay(1))
