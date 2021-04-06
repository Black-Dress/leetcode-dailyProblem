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


s = Solution()
print(s.removeDuplicates([0, 0, 1, 1, 1, 1, 2, 3, 3]))
