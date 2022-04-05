class OCRtext:
    def __init__(self, x, y, w, h, text):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.text = text

    def getLeftX(self):
        return self.x

    def getRightX(self):
        return self.x + self.w

    def getTopY(self):
        return self.y

    def getBottomY(self):
        return self.y + self.h

    def getText(self):
        return self.text


# class OCRtext:
#     def __init__(self, left, top, width, height, text):
#         self.x = left
#         self.rightX = left+width
#         self.topY = top
#         self.bottomY = top + height
#         self.text = text
#
#     def getLeftX(self):
#         return self.leftX
#
#     def getRightX(self):
#         return self.rightX
#
#     def getTopY(self):
#         return self.topY
#
#     def getBottomY(self):
#         return self.bottomY
#
#     def getText(self):
#         return self.text




