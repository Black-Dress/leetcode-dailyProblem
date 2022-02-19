from cmath import pi
from heapq import *
from typing import List


class Solution:
    def medianSlidingWindow(self, nums: List[int], k: int) -> List[float]:
        # l 比 r 多
        l, r = [], []

        def mid(l: List[int], r: List[int]) -> int:
            return -l[0] if len(l) > len(r) else (-l[0] + r[0]) / 2

        def inStack(l: List[int], r: List[int], m: int):
            if not l or m <= -l[0]:
                if len(l) == len(r):
                    heappush(l, -m)
                else:
                    heappush(r, -heappushpop(l, -m))
            else:
                if len(l) == len(r):
                    heappush(l, -heappushpop(r, m))
                else:
                    heappush(r, m)

        for i in range(k):
            inStack(l, r, nums[i])

        res = [mid(l, r)]
        for i in range(k, len(nums)):
            # 弹栈
            if nums[i - k] <= -l[0]:
                l.remove(-nums[i - k])
                heapify(l)
                while len(l) - len(r) < 0:
                    heappush(l, -heappop(r))
            else:
                r.remove(nums[i - k])
                heapify(r)
                while len(l) - len(r) > 1:
                    heappush(r, -heappop(l))
            # 入栈
            inStack(l, r, nums[i])
            res.append(mid(l, r))
        return res

    def dicesProbability(self, n: int) -> List[float]:
        # dp[i][j] 表示i颗骰子时 总和为 j 的组合数
        dp = [[0] * (n * 6 + 1) for i in range(n + 1)]
        for i in range(1, 7):
            dp[1][i] = 1
        for i in range(2, n + 1):
            for j in range(i, i * 6 + 1):
                # dp[i][j] = sum([dp[i - 1][j - k] for k in range(1, min(7, j // 2 + 1))])
                dp[i][j] = sum(dp[i - 1][j - k] if j > k else 0 for k in range(1, 7))

        total = sum(dp[n][:])
        return sorted([dp[n][i] / total for i in range(n, n * 6 + 1)])

    # 剑指 Offer 14- II. 剪绳子 II
    def cuttingRope(self, n: int) -> int:
        if n <= 3:
            return n - 1
        a, b, mod, res, x = n // 3 - 1, n % 3, 1000000007, 1, 3
        # 快速幂求余
        while a:
            if a % 2:
                res = (res * x) % mod
            x = (x**2) % mod
            a //= 2
        if b == 0:
            return (res * 3) % mod
        if b == 1:
            return (res * 4) % mod
        return (res * 6) % mod

    # 540. 有序数组中的单一元素
    def singleNonDuplicate(self, nums: List[int]) -> int:
        def dfs(l: int, r: int, nums: List[int]) -> int:
            if l > r:
                return -1
            mid = (l + r) >> 1
            # 和前面相等
            a = mid > 0 and nums[mid] == nums[mid - 1]
            # 和后面相等
            b = mid < r and nums[mid] == nums[mid + 1]
            # 奇偶
            c = (mid + 1) % 2 == 0
            if not a and not b:
                return nums[mid]
            if (a and c) or (b and not c):
                return dfs(mid + 1, r, nums)
            if (b and c) or (a and not c):
                return dfs(l, mid - 1, nums)
        return dfs(0, len(nums) - 1, nums)

    # 快排
    def quick_sort(self, nums: List[int]) -> List[int]:
        if not nums:
            return []
        i, j = 0, len(nums) - 1
        while i < j:
            while i < j and nums[i] <= nums[-1]:
                i += 1
            while i < j and nums[j] >= nums[-1]:
                j -= 1
            nums[i], nums[j] = nums[j], nums[i]
        nums[i], nums[-1] = nums[-1], nums[i]
        return self.quick_sort(nums[:i]) + [nums[i]] + self.quick_sort(nums[i + 1:])

    def lengthOfLongestSubstring(self, s: str) -> int:
        # 双指针，区间内部是没有重复数字的子串
        # map 记录区间内部所有元素的位置
        # 每次右指针向右移动，判断是否重复
        #   若无重复或者重复元素的位置小于左指针，更新元素位置，更新最大长度
        #   若重复元素在区间内，更新左指针位置到重复元素位置的右侧
        i, res = -1, 0
        cnt = dict()
        for j in range(len(s)):
            # 移动右指针
            if s[j] in cnt:
                i = max(i, cnt[s[j]])
            res = max(res, j - i)
            cnt[s[j]] = j
        return res

    # 440. 字典序的第K小数字
    def findKthNumber(self, n: int, k: int) -> int:

        # 前缀为cur的时候子树大小
        def getCount(n: int, cur: int) -> int:
            res, next = 0, cur + 1
            while cur <= n:
                res += min(next, n + 1) - cur
                cur *= 10
                next *= 10
            return res

        res = 1
        k -= 1
        while k > 0:
            nums = getCount(n, res)
            if k >= nums:
                k -= nums
                res += 1
            else:
                res *= 10
                k -= 1
        return res

    # 969. 煎饼排序
    def pancakeSort(self, arr: List[int]) -> List[int]:
        def help(arr: List[int], r: int) -> List[int]:
            arr[0:r + 1] = arr[r::-1]
            res = [r + 1]
            if r < 2:
                return res
            while r > 0 and arr[0] < arr[1]:
                i = 1
                while i <= r and arr[i] > arr[0]:
                    i += 1
                arr[0:i] = arr[i - 1::-1]
                res.append(i)
            arr[0:r + 1] = arr[r::-1]
            return res + [r + 1]
        res = []
        for i in range(len(arr) - 1):
            if arr[i] > arr[i + 1]:
                res.extend(help(arr, i + 1))
        return res


s = Solution()
print(s.pancakeSort([1, 4, 2, 3]))
# print(sorted([1, 2, 3, 4, 5, 6], key=lambda x: (x == 1, x - 1)))
# print(min([1, 2, 3, 4, 5, 6], key=lambda x: (x == 1, x - 1)))
