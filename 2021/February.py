import bisect
from typing import List


class Solution:
    # 480 滑动窗口中位数
    def medianSlidingWindow(self, nums: List[int], k: int) -> List[float]:
        def median(a): return (a[(len(a)-1)//2] + a[len(a)//2]) / 2
        a = sorted(nums[:k])
        res = [median(a)]
        for i, j in zip(nums[:-k], nums[k:]):
            # 二分法得到删除和插入位置的索引
            a.pop(bisect.bisect_left(a, i))
            a.insert(bisect.bisect_left(a, j), j)
            res.append(median(a))
        return res

    # 643 子数组最大平均数 I
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        cur = [nums[i] for i in range(k)]
        temp = res = sum(cur)
        for i in nums[k:]:
            temp = temp-cur[0]+i
            cur.pop(0)
            cur.append(i)
            res = max(res, temp)
        return res/k

    # 832. 翻转图像
    def flipAndInvertImage(self, A: List[List[int]]) -> List[List[int]]:
        for cur in A:
            i, j = 0, len(cur)-1
            while(i < j):
                cur[i], cur[j] = cur[j], cur[i]
                cur[i] ^= 1
                cur[j] ^= 1
                i += 1
                j -= 1
            if i == j:
                cur[i] ^= 1
        return A

    # 395. 至少有K个重复字符的最长子串
    def longestSubstring(self, s: str, k: int) -> int:
        if not s:
            return 0
        for i in set(s):
            if s.count(i) < k:
                return max(self.longestSubstring(t, k) for t in s.split(i))
        return len(s)

    # 896. 单调数列
    def isMonotonic(self, A: List[int]) -> bool:
        i = 1
        while i < len(A) and A[i]-A[i-1] == 0:
            i += 1
        if i >= len(A):
            return True
        flag = (A[i]-A[i-1] > 0)
        while i < len(A):
            if A[i]-A[i-1] == 0:
                i += 1
                continue
            if (A[i]-A[i-1] > 0) != flag:
                return False
            i += 1
        return True


s = Solution()
print(s.isMonotonic([1, 2, 3]))
