from collections import defaultdict
import collections
from typing import Collection, Counter, List, Tuple, no_type_check


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

    # 453. 最小操作次数使数组元素相等
    def minMoves(self, nums: List[int]) -> int:
        # 思路1：蛮力法
        # 可以每一次的行动让最小的n-1个数加一，直到所有的数字都相等
        # 思路2：数学
        # 因为n-1个数字加1等效于 1个数字减1 ，所以只要让所有的数字都等于那个最小值就行了
        minNum, res = min(nums), 0
        for i in nums:
            res += i - minNum
        return res

    # 229. 求众数 II
    def majorityElement(self, nums: List[int]) -> List[int]:
        # 求大于三分之一的元素，那么答案就只有[0,2] 个
        # 摩尔投票从 n/2 推广到 n/3
        # n/3 的情况下结果最多两个，所以过程中记录两个元素，并且统计两个元素的票数，抵消的时候两个都抵消
        element1, element2 = 0, 0
        vote1, vote2 = 0, 0
        cnt1, cnt2 = 0, 0
        res = []
        # 摩尔投票
        for num in nums:
            # 优先统计计数情况，而不是变动情况，不然会出现element1和element2相同
            if vote1 > 0 and num == element1:
                vote1 += 1
                continue
            if vote2 > 0 and num == element2:
                vote2 += 1
                continue
            if vote1 == 0:
                element1 = num
                vote1 += 1
                continue
            if vote2 == 0:
                element2 = num
                vote2 += 1
                continue
            if num != element2 and num != element1:
                vote2 -= 1
                vote1 -= 1
        # 统计结果，至少要统计那些有票的元素
        for num in nums:
            if vote1 > 0 and element1 == num:
                cnt1 += 1
            if vote2 > 0 and element2 == num:
                cnt2 += 1
        # 统计那些有票数的元素，以及票数是否达标
        if vote1 > 0 and cnt1 > nums.__len__() / 3:
            res.append(element1)
        if vote2 > 0 and cnt2 > nums.__len__() / 3:
            res.append(element2)
        return res

    # 496. 下一个更大元素 I
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        stack, table = [nums2[0]], collections.defaultdict(int)
        res = list()
        for i in nums2[1:]:
            while stack.__len__() > 0 and i > stack[-1]:
                table[stack[-1]] = i
                stack.pop()
            stack.append(i)
        for i in nums1:
            target = -1 if table.get(i) is None else table[i]
            res.append(target)
        return res

    # 869. 重新排序得到 2 的幂
    def reorderedPowerOf2(self, n: int) -> bool:
        # 因为题目的范围在  [0,10^9]
        def countDigit(n: int) -> Tuple[int]:
            res = [0] * 10
            while n:
                res[n % 10] += 1
                n //= 10
            return tuple(res)
        # 得到所有可能的2的幂次
        table = {countDigit(1 << i) for i in range(30)}
        return countDigit(n) in table

    # 335. 路径交叉
    def isSelfCrossing(self, distance: List[int]) -> bool:
        n = len(distance)
        for i in range(3, n):
            # 第 1 类路径交叉的情况
            if (distance[i] >= distance[i - 2]
                    and distance[i - 1] <= distance[i - 3]):
                return True

            # 第 2 类路径交叉的情况
            if i == 4 and (distance[3] == distance[1]
                           and distance[4] >= distance[2] - distance[0]):
                return True

            # 第 3 类路径交叉的情况
            if i >= 5 and (distance[i - 3] - distance[i - 5] <= distance[i - 1] <= distance[i - 3]
                           and distance[i] >= distance[i - 2] - distance[i - 4]
                           and distance[i - 2] > distance[i - 4]):
                return True
        return False


s = Solution()
print(s.isSelfCrossing([1, 1, 2, 2, 3, 3, 4, 4, 10, 4, 4, 3, 3, 2, 2, 1, 1]))
