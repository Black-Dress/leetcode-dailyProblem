from re import A
from typing import Counter, List


class Solution:
    # 954. 二倍数对数组
    def canReorderDoubled(self, arr: List[int]) -> bool:
        # arr[2 * i + 1] = 2 * arr[2 * i] 表示在偶数位置上能够满足这样的条件
        # 有n//2个这样的数队据能够构成这一数组，dict判断是否存在能够对应的条件
        # 如果针对于 arr[i] 存在 2*arr[i] 和 arr[i]//2 ,优先找大的解，直到所有的位置遍历完成，构成数对的数量是否为n//2来进行判断
        cnt, n, res = Counter(arr), len(arr), 0
        if cnt[0] % 2 != 0:
            return False
        for i in sorted(cnt.keys(), key=lambda x: abs(x)):
            if i != 0 and cnt[2 * i] < cnt[i]:
                return False
            cnt[2 * i] -= cnt[i]
        return True


s = Solution()
print(s.canReorderDoubled([0, 1]))
