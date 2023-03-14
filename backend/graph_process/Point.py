class Point:
    def __init__(self, name, nodeId, feature, infoObj):
        self.name = name
        self.nodeId = nodeId
        self.infoObj = infoObj
        self.feature = feature

    def __hash__(self):
        return self.nodeId

    def __str__(self):
        return 'name：%s  nodeId：%d  feature：%s' % (self.name, self.nodeId, str(self.feature))