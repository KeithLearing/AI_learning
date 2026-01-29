import pandas as pd

# data = {'name': ['zhangsan', 'lisi', 'wangwu'], 'age': [25, 40, 38]}
# df = pd.DataFrame(data)
# print(df)
#
# index = [1, 2, 3, 4]
# series = pd.Series([1, 2, 3, 4],index=index, name='A')
# print(series)

# a = [1, 2, 3]
# my = pd.Series(a)
# print(my)
# print(my[2])

b = ['series', 'max', 'min']
my1 = pd.Series(b, index=['x', 'y', 'z'])
print(my1)
