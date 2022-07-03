from collections import defaultdict


class Solution:
    # 556. 下一个更大元素 III
    def nextGreaterElement(self, n: int) -> int:
        # 需要从后排找到一个大于k的最小值
        s = str(n)
        if s == "".join(sorted(str(n), reverse=True)):
            return -1
        r, nums, cnt = len(s), list(s), defaultdict(int)
        while r > 0:
            r -= 1
            cnt[int(nums[r])] = r
            if nums[r] <= nums[r - 1]:
                continue
            # 找到大于nums[r-1]的最小值
            i = int(nums[r - 1]) + 1
            while i <= 9 and i not in cnt:
                i += 1
            nums[r - 1], nums[cnt[i]] = nums[cnt[i]], nums[r - 1]
            break
        nums = nums[:r] + sorted(nums[r:])
        res = int("".join(nums))
        return res if res <= 2**31 - 1 else -1


s = Solution()
print(s.nextGreaterElement(2147483486))
