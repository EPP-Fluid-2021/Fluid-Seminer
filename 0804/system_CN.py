#システム方程式 クランクニコルソン法
import numpy as np
import matplotlib.pyplot as plt

plt.subplots_adjust(wspace=1, hspace=1) #余白

fig = plt.figure()  #プロットの準備
fig_1 = fig.add_subplot(2,2,1)#密度グラフ
fig_2 = fig.add_subplot(2,2,2)#速度グラフ
fig_3 = fig.add_subplot(2,2,3)#圧力グラフ
fig_4 = fig.add_subplot(2,2,4)#エネルギーグラフ
fig.subplots_adjust(wspace=0.4, hspace=0.4) #余白


jmax = 101#格子の数
dt = 0.0005
nmax = 1000  #ステップ数
display_interval = 100 #表示間隔
gamma = 1.4 #比熱比　二原子分子

PI = 1.0


RHOI = 1.0
RHOE = 0.1

UI = 1.0
UE = 1.0


xmin,xmid,xmax = 0,0.5,1.0
x = np.linspace(xmin,xmax,jmax)
QTMP= np.zeros([jmax,3])
dx = (xmax-xmin)/(jmax-1)  #格子間隔

def init(): #初期値
    Q = np.zeros([jmax,3])
    Q[:,0]=RHOI*np.sin(x*np.pi)+RHOE
    Q[:,1]=(RHOI*np.sin(x*np.pi)+RHOE)*(UE)
    Q[:,2]=(PI/(gamma-1)+(RHOI*np.sin(x*np.pi)+RHOE)*(UE)**2/2)
    
    return Q


def compute(Q):
    E = np.zeros([jmax,3]) #整数地点での数値流速
    
    cnt = -1
    while(cnt<nmax):
        E[:,0]=Q[:,1]
        E[:,1]=(3-gamma)*Q[:,1]**2/Q[:,0]/2+(gamma-1)*Q[:,2]
        E[:,2]=(gamma*Q[:,2]-(gamma-1)*Q[:,1]**2/Q[:,0]/2)*Q[:,1]/Q[:,0]

        for i in range(0,jmax):
            QTMP[i]=Q[i]+dt/dx/2*(E[i-1]-E[(i+1)%jmax])
            Q[i]+=dt/dx/4*(E[i-1]-E[(i+1)%jmax])

        E[:,0]=QTMP[:,1]
        E[:,1]=(3-gamma)*QTMP[:,1]**2/QTMP[:,0]/2+(gamma-1)*QTMP[:,2]
        E[:,2]=(gamma*QTMP[:,2]-(gamma-1)*QTMP[:,1]**2/QTMP[:,0]/2)*QTMP[:,1]/QTMP[:,0]

        
        for i in range(0,jmax):
            Q[i]+=dt/dx/4*(E[i-1]-E[(i+1)%jmax])
       
        cnt+=1
        if(cnt%display_interval==0):
            fig_1.plot(x,Q[:,0])
            fig_2.plot(x,Q[:,1]/Q[:,0])
            fig_3.plot(x,(gamma-1)*(Q[:,2]-Q[:,1]**2/Q[:,0]/2))
            fig_4.plot(x,Q[:,2])
        
    fig_1.set_xlabel("x")
    fig_1.set_ylabel("density")
    fig_2.set_xlabel("x")
    fig_2.set_ylabel("velocity")
    fig_3.set_xlabel("x")
    fig_3.set_ylabel("pressure")
    fig_4.set_xlabel("x")
    fig_4.set_ylabel("energy")
    fig.show()

    

            
Q=init()
compute(Q)
