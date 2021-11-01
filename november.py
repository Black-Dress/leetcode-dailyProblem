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
