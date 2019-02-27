class Singleton(object):
    def __init__(self, _idx):
        self.idx = _idx
        self.parent = _idx
        self.rank = 0


class UnionFind(object):
    def __init__(self):
        self.sets = {}

    def MakeSet(self, idx):
        if idx not in self.sets:
            self.sets[idx] = Singleton(idx)

    def Find(self, idx):
        if idx not in self.sets:
            self.MakeSet(idx)

        if self.sets[idx].parent == idx:
            return idx
        else:
            self.sets[idx].parent = self.Find(self.sets[idx].parent)
            return self.sets[idx].parent

    def Union(self, x, y):
        xRoot = self.Find(x)
        yRoot = self.Find(y)
        if xRoot == yRoot:
            return

        if self.sets[xRoot].rank < self.sets[yRoot].rank or (
            self.sets[xRoot].rank == self.sets[yRoot].rank and xRoot < yRoot
        ):
            self.sets[xRoot].parent = yRoot
            self.sets[yRoot].rank = self.sets[yRoot].rank + 1
        else:
            self.sets[yRoot].parent = xRoot
            self.sets[xRoot].rank = self.sets[xRoot].rank + 1

    def GetComponents(self):
        collections = {}
        for key, item in self.sets.iteritems():
            root = self.Find(key)
            if root not in collections:
                collections[root] = []
            collections[root].append(key)
        return collections.values()

