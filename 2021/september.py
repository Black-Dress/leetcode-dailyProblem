from collections import defaultdict
from typing import Collection, List
import math


class Solution:
    # 回旋镖的数量，找到两个点的中点
    # 暴力法
    def numberOfBoomerangs(self, points: List[List[int]]) -> int:
        cnt, res = defaultdict(int), 0
        for i in range(points.__len__()):
            cnt.clear()
            for j in range(points.__len__()):
                distance = math.pow(points[i][0]-points[j][0], 2)+math.pow(points[i][1]-points[j][1], 2)
                cnt[distance] += 1
            for (k, v) in cnt.items():
                res += v*(v-1)
        return res


s = Solution()
print(s.numberOfBoomerangs([[0, 0], [1, 0], [2, 0]]))
