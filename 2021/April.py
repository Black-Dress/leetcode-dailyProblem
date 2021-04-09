class Solution:
    # 1006. 笨阶乘
    def clumsy(self, N: int) -> int:
        def caculator(a: int, b: int, oper: str) -> int:
            if oper == "*":
                return a*b
            elif oper == "/":
                return a//b
            elif oper == "+":
                return a+b
            elif oper == "-":
                return a-b
            else:
                return 0
        operator = ["*", "/", "+", "-"]
        stack1, stack2 = [N], []
        index = 0
        for i in range(N-1, 0, -1):
            if operator[index % 4] == "*" or operator[index % 4] == "/":
                stack1.append(caculator(stack1.pop(), i, operator[index % 4]))
            else:
                stack1.append(i)
                stack2.append(operator[index % 4])
            index += 1
        for i in range(len(stack2)):
            a, b = stack1.pop(0), stack1.pop(0)
            stack1.insert(0, caculator(a, b, stack2[i]))
        return stack1.pop()

    # 80. 删除有序数组中的重复项 II
    def removeDuplicates(self, nums: [int]) -> int:
        count, i = 1, 1
        while i < len(nums):
            count = count+1 if nums[i] == nums[i-1] else 1
            if count > 2:
                nums.pop(i)
                i -= 1
            i += 1
        return len(nums)

    # 81. 搜索旋转排序数组 II
    def search(self, nums: [int], target: int) -> bool:
        def binary_search(nums: list, target: int, L: int, r: int) -> bool:
            if L > r:
                return False
            mid = (L+r)//2
            if target < nums[mid]:
                return binary_search(nums, target, L, mid-1)
            elif nums[mid] == target:
                return True
            else:
                return binary_search(nums, target, mid+1, r)
        index, n = 0, len(nums)
        for i in range(1, n):
            if nums[i] < nums[i-1]:
                index = i
                break
        if target >= nums[0]:
            return binary_search(nums, target, 0, index-1 if index > 0 else n-1)
        return binary_search(nums, target, index if index > 0 else 0, n-1)

    # 153. 寻找旋转排序数组中的最小值
    def findMin(self, nums: [int]) -> int:
        L, r = 0, len(nums)-1
        while L < r:
            if nums[L] <= nums[r]:
                return nums[L]
            else:
                mid = (L+r) >> 1
                if nums[mid] > nums[r]:
                    L = mid+1
                else:
                    r = mid
        return nums[L]

    # 154. 寻找旋转排序数组中的最小值 II
    def findMin2(self, nums: [int]) -> int:
        pass


s = Solution()
# nums = [1, 2, 3, 4, 5, 6]
# print(bisect.bisect(nums, 5, 0, 3))
print(s.findMin([1, 2]))
