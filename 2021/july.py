from typing import List
from sys import maxsize
import bisect


class Solution:
    def getSkyline(self, buildings: List[List[int]]) -> List[List[int]]:
        n = len(buildings)
        res, right = [], []
        preRightMax = 0
        # 先判断左边界,还需要判断是否前一个与自己断开了
        for building in buildings:
            if res.__len__() == 0:
                res.append([building[0], building[2]])
            else:
                if building[2] > res[-1][1] or building[0] > preRightMax:
                    if building[0] == res[-1][0]:
                        res[-1][1] = building[2]
                    else:
                        res.append([building[0], building[2]])
            preRightMax = max(preRightMax, building[1])
        # 右边界倒序
        buildings.reverse()
        preHeight, preLeftMin = 0, maxsize
        for building in buildings:
            if right.__len__() == 0:
                right.append([building[1], preHeight])
            else:
                if building[2] > preHeight or building[1] < preLeftMin:
                    data = [building[1], 0] if building[1] < preLeftMin else [building[1], preHeight]
                    if building[1] == right[-1][0]:
                        right[-1][1] = building[2]
                    else:
                        right.append(data)
            preLeftMin = min(preLeftMin, building[0])
            preHeight = building[2]
        res.extend(right)
        return sorted(res, key=lambda x: x[0])

    # 1818. 绝对差值和
    def minAbsoluteSumDiff(self, nums1: List[int], nums2: List[int]) -> int:
        nums_sorted = sorted(nums1)
        n = nums1.__len__()
        maxgap, res = 0, 0
        for i in range(n):
            index = bisect.bisect_left(nums_sorted, nums2[i])
            a = abs(nums_sorted[index-1] - nums2[i]) if index > 0 else maxsize
            b = abs(nums_sorted[index] - nums2[i]) if -1 < index < n else maxsize
            c = abs(nums_sorted[index+1]-nums2[i]) if index < n-1 else maxsize
            cur = abs(nums1[i]-nums2[i])
            minval = min(a, b, c)
            maxgap = max(abs(cur-minval), maxgap)
            res += cur
        res -= maxgap
        return res % (10**9+7)


s = Solution()
print(s.minAbsoluteSumDiff([1, 10, 4, 4, 2, 7],  [9, 3, 5, 1, 7, 4]))
