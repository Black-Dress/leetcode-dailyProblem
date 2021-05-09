from typing import List


class Solution:
    # 二分一个一个找合适的天数
    def minDays(self, bloomDay: List[int], m: int, k: int) -> int:
        if m*k > bloomDay.__len__():
            return -1

        def check(day: int) -> bool:
            res, cur = 0, 0
            for bloom in bloomDay:
                cur += 1 if bloom <= day else -cur
                if cur == k:
                    cur = 0
                    res += 1
            return res >= m

        left, right = min(bloomDay), max(bloomDay)
        while left < right:
            mid = (left+right) >> 1
            if check(mid):
                right = mid
            else:
                left = mid+1
        return left
