from torchstat import stat
from torchsummary import summary
from Model.DJImodule import DJI2D

model = DJI2D()
# summary(model, (1, 121, 121))
print(stat(model, (1, 121, 121)))

# class ListNode:
#     def __init__(self, val):
#         self.val = val
#         self.next = None

