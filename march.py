class Solution:
    # 165. 比较版本号
    def compareVersion(self, version1: str, version2: str) -> int:
        a, b, i = version1.split('.'), version2.split('.'), 0
        while i < len(a) and i < len(b):
            if int(a[i]) > int(b[i]):
                return 1
            if int(a[i]) < int(b[i]):
                return -1
            i += 1
        while i < len(a):
            if int(a[i]) > 0:
                return 1
            i += 1
        while i < len(b):
            if int(b[i]) > 0:
                return -1
            i += 1
        return 0


s = Solution()
print(s.compareVersion("1.00", "1"))
