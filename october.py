from collections import defaultdict
from typing import Collection, List, no_type_check


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


s = Solution()
print(s.divide(10, 1))
