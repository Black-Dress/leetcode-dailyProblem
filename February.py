import bisect


class Solution:
    # 480 滑动窗口中位数
    def medianSlidingWindow(self, nums: [int], k: int) -> [float]:
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
    def findMaxAverage(self, nums: [int], k: int) -> float:
        cur = [nums[i] for i in range(k)]
        temp = res = sum(cur)
        for i in nums[k:]:
            temp = temp-cur[0]+i
            cur.pop(0)
            cur.append(i)
            res = max(res, temp)
        return res/k


s = Solution()
print(s.findMaxAverage([0, 4, 0, 3, 2], 1))
