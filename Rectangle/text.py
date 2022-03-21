class Text:
    def __init__(self, left, top, width, height, text):
        self.leftX = left
        self.rightX = left+width
        self.topY = top
        self.bottomY = top + height
        self.text = text

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




