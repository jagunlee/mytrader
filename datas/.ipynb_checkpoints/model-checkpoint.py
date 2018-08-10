import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

'''
3d그래프 확인 가능
'''
df=pd.read_csv('checkdata.csv')
fig = plt.figure()						
ax = fig.gca(projection='3d')
ax.plot(df['buy'],df['sell'],df['pv'],'o') #처움 두개를 바꿔가면서 그래프 따로 확인 가능
plt.show()