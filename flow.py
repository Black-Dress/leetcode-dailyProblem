from collections import defaultdict
from typing import List


class solution:
    def __init__(self):
        file = open("data.txt", "r")
        self.m, self.n = list(map(int, file.readline().strip("\n").split(" ")))
        self.supply = []
        self.needs = []
        self.borad = []
        self.direction = []
        self.orientation = {
            1 << 0: [0, 1],
            1 << 1: [1, 1],
            1 << 2: [1, 0],
            1 << 3: [1, -1],
            1 << 4: [0, -1],
            1 << 5: [-1, -1],
            1 << 6: [0, -1],
            1 << 7: [1, -1]
        }
        file.readline()
        for i in range(self.m):
            self.supply.append(list(map(int, file.readline().strip("\n").split(" "))))
        file.readline()
        for i in range(self.m):
            self.needs.append(list(map(int, file.readline().strip("\n").split(" "))))
        file.readline()
        for i in range(self.m):
            self.borad.append(list(map(int, file.readline().strip("\n").split(" "))))
        file.readline()
        for i in range(self.m):
            self.direction.append(list(map(int, file.readline().strip("\n").split(" "))))

    def cal(self, i: int, j: int) -> dict:
        res = dict()
        stack = [[i, j]]
        while stack:
            i, j = stack.pop()
            if self.supply[i][j] > self.needs[i][j]:
                nxt = self.supply[i][j] - self.needs[i][j]
                self.supply[i][j] = self.needs[i][j]
                x, y = i + self.orientation[self.direction[i][j]][0], j + self.orientation[self.direction[i][j]][1]
                if 0 <= x < self.m and 0 <= y < self.n and self.borad[x][y] != 1:
                    self.supply[x][y] += nxt
                    res[(i, j)] = [x, y, nxt]
                    stack.append((x, y))
        return res


s = solution()
print(s.cal(1, 1))
print(s.supply)
