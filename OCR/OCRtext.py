class OCRtext:
    def __init__(self, left, top, width, height, text):
        self.leftX = left
        self.rightX = left+width
        self.topY = top
        self.bottomY = top + height
        self.text = text

    def __getstate__(self):
        state = dict(self.__dict__)
        del state['cachedBar']
        return state

    def getLeftX(self):
        return self.leftX

    def getRightX(self):
        return self.rightX

    def getTopY(self):
        return self.topY

    def getBottomY(self):
        return self.bottomY

    def getText(self):
        return self.text



