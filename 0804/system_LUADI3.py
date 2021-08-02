#システム方程式 LUSGS


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
nmax = 700  #ステップ数
display_interval = 100 #表示間隔
gamma = 1.4 #比熱比　二原子分子


xmin,xmid,xmax = 0,0.5,1.0
x = np.linspace(xmin,xmax,jmax)
dx = (xmax-xmin)/(jmax-1)  #格子間隔



PI = 1.0


RHOI = 1.0
RHOE = 1.0


UE = 1.0

VE = 1.0

def init(): #初期値
    Q = np.zeros([jmax,3])
    Q[:,0]=RHOI*np.exp(-(x-xmid)**2*100)+RHOE
    Q[:,1]=(RHOI*np.exp(-(x-xmid)**2*100)+RHOE)*(UE)
    Q[:,2]=(PI/(gamma-1)+(RHOI*np.exp(-(x-xmid)**2*100)+RHOE)*(UE)**2/2)
    
    return Q

def compute(Q):
    G = np.zeros([jmax,3])
    E = np.zeros([jmax,3])#E
    
    
    A = np.zeros([jmax,3,3])#A
    A[:,0,1]=1
    A[:,1,2]=gamma-1
    sigma=np.zeros(jmax)#abs(u)+c
    

    
    
    DeltaQ = np.zeros([jmax,3]) 
    QTMP = np.zeros([jmax,3]) 
    I=np.eye(3)
    
    for n in range(nmax):
        E[:,0]=Q[:,1]
        E[:,1]=(3-gamma)*Q[:,1]**2/Q[:,0]/2+(gamma-1)*Q[:,2]
        E[:,2]=(gamma*Q[:,2]-(gamma-1)*Q[:,1]**2/Q[:,0]/2)*Q[:,1]/Q[:,0]
        
        for i in range(1,jmax-1):#中心差分
                G[i]=dt/dx/2*(E[i-1]-E[i+1])
                

        u=Q[:,1]/Q[:,0]
        c=E[:,2]/E[:,0]-Q[:,2]/Q[:,0]
        c=np.sqrt(c*gamma)

        A[:,1,0] = (gamma-3)*u**2/2
        A[:,1,1] = -(gamma-3)*u
    

        A[:,2,0] = -gamma*u*Q[:,2]/Q[:,0]+(gamma-1)*u**3
        A[:,2,1] = gamma*Q[:,2]/Q[:,0]-3/2*(gamma-1)*u**2
        A[:,2,2] = gamma*u

        sigma=abs(u)+c
       
        for i in range(1,jmax):#第一スイープ
                DeltaQ[i]=(G[i]+dt/dx*np.dot((A[i-1]+sigma[i-1])/2,DeltaQ[i-1]))/(1+np.ones(3)*dt/dx*sigma[i])

        #第二スイープ
        for i in range(3):
            DeltaQ[:,i]*=1+dt/dx*sigma
        
        for i in range(jmax-2,-1,-1):#第三スイープ
                QTMP[i]=(DeltaQ[i]-dt/dx*np.dot((A[i+1]-sigma[i+1])/2,QTMP[i+1]))/(1+np.ones(3)*dt/dx*sigma[i])


        Q+=QTMP

        if(n%display_interval==0):
            fig_1.plot(x,Q[:,0])
            fig_2.plot(x,Q[:,1]/Q[:,0])
            fig_3.plot(x,(gamma-1)*(Q[:,2]-Q[:,1]**2/Q[:,0]/2))
            fig_4.plot(x,Q[:,2])
            
            
    

    fig_1.set_xlabel("x")
    fig_1.set_ylabel("density")
    fig_2.set_xlabel("x")
    fig_2.set_ylabel("velocity")
    fig_2.set_ylim(0.5,1.5)
    fig_3.set_xlabel("x")
    fig_3.set_ylabel("pressure")
    fig_3.set_ylim(0.5,1.5)
    fig_4.set_xlabel("x")
    fig_4.set_ylabel("energy")
    
    fig.show()


  

    

            
Q=init()

compute(Q)



















