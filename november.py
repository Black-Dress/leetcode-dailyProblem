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
