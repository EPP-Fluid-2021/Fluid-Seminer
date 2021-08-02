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






jxmax = 50#格子の数x
jymax = 50#格子の数y
dt = 0.0005
nmax = 200  #ステップ数
display_interval = 20 #表示間隔
gamma = 1.4 #比熱比　二原子分子

ktemp=jymax//2 #表示するy方向の位置 0<ktemp<jymax



xmin,xmid,xmax = 0,0.5,1.0
x = np.linspace(xmin,xmax,jxmax)
dx = (xmax-xmin)/(jxmax-1)  #格子間隔

ymin,ymid,ymax = 0,0.5,1.0
y = np.linspace(ymin,ymax,jymax)
dy = (ymax-ymin)/(jymax-1)  #格子間隔

PI = 1.0


RHOI = 1.0
RHOE = 1.0


UE = 1.0

VE = 1.0

def init(): #初期値
    Q = np.zeros([jxmax,jymax,4])
    for i in range(jxmax):
        for k in range(jymax):
            Q[i,k,0]=RHOI*np.exp(-(x[i]+y[k]-xmid-ymid)**2*100)+RHOE
            Q[i,k,1]=(RHOI*np.exp(-(x[i]+y[k]-xmid-ymid)**2*100)+RHOE)*(UE)
            Q[i,k,2]=(RHOI*np.exp(-(x[i]+y[k]-xmid-ymid)**2*100)+RHOE)*(VE)
            Q[i,k,3]=(PI/(gamma-1)+(RHOI*np.exp(-(x[i]+y[k]-xmid-ymid)**2*100)+RHOE)*(UE**2+VE**2)/2)
    return Q

def compute(Q):
    G = np.zeros([jxmax,jymax,4])
    E = np.zeros([jxmax,jymax,4])#E
    F = np.zeros([jxmax,jymax,4])#F
    
    A = np.zeros([jxmax,jymax,4,4])#A
    A[:,:,0,1]=1
    A[:,:,1,3]=gamma-1
    sigmax=np.zeros(jxmax)#abs(u)+c
    

    B = np.zeros([jxmax,jymax,4,4])#B
    B[:,:,0,1]=1
    B[:,:,2,3]=gamma-1
    sigmay=np.zeros(jymax)#abs(v)+c
    
    DeltaQ = np.zeros([jxmax,jymax,4]) 
    QTMP = np.zeros([jxmax,jymax,4]) 
    I=np.eye(4)
    
    for n in range(nmax):
        E[:,:,0] = Q[:,:,1]
        F[:,:,0] = Q[:,:,2]
        E[:,:,1] = (3-gamma)*Q[:,:,1]**2/Q[:,:,0]/2+(gamma-1)*Q[:,:,3]
        F[:,:,2] = (3-gamma)*Q[:,:,2]**2/Q[:,:,0]/2+(gamma-1)*Q[:,:,3]
        E[:,:,2] = Q[:,:,1]*Q[:,:,2]/Q[:,:,0]
        F[:,:,1] = E[:,:,2]
        E[:,:,3] = (gamma*Q[:,:,3]-(gamma-1)*(Q[:,:,1]**2+Q[:,:,2]**2)/Q[:,:,0]/2)*Q[:,:,1]/Q[:,:,0]
        F[:,:,3] = E[:,:,3]*Q[:,:,2]/Q[:,:,1]
        
        for i in range(1,jxmax-1):
            for k in range(1,jymax-1):#中心差分
                G[i,k]=dt/dx/2*(E[i-1,k]-E[i+1,k])
                G[i,k]+=dt/dy/2*(F[i,k-1]-F[i,k+1])

        u=Q[:,:,1]/Q[:,:,0]
        v=Q[:,:,2]/Q[:,:,0]
        q2=u**2+v**2
        c=E[:,:,3]/E[:,:,0]-Q[:,:,3]/Q[:,:,0]
        c=np.sqrt(c*gamma)

        A[:,:,1,0] = (gamma-3)*u**2/2+(gamma-1)*v**2/2
        A[:,:,1,1] = -(gamma-3)*u
        A[:,:,1,2] = -(gamma-1)*v

        A[:,:,2,0] = -u*v
        A[:,:,2,1] = v
        A[:,:,2,2] = u

        A[:,:,3,0] = -gamma*u*Q[:,:,3]/Q[:,:,0]+(gamma-1)*u*q2
        A[:,:,3,1] = gamma*Q[:,:,3]/Q[:,:,0]-(gamma-1)*(2*u**2*q2)/2
        A[:,:,3,2] = -(gamma-1)*u*v
        A[:,:,3,3] = gamma*u

        B[:,:,1,0] = -u*v
        B[:,:,1,1] = v
        B[:,:,1,2] = u

        B[:,:,2,0] = (gamma-3)*v**2/2+(gamma-1)*u**2/2
        B[:,:,2,1] = -(gamma-1)*u
        B[:,:,2,2] = -(gamma-3)*v

        

        B[:,:,3,0] = -gamma*v*Q[:,:,3]/Q[:,:,0]+(gamma-1)*v*q2
        B[:,:,3,1] = -(gamma-1)*u*v
        B[:,:,3,2] = gamma*Q[:,:,3]/Q[:,:,0]+(gamma-1)*(2*v**2*q2)/2
        B[:,:,3,3] = gamma*v

        sigmax=abs(u)+c
        sigmay=abs(v)+c

        
        
        
    

        for i in range(1,jxmax):#第一スイープ
            for k in range(1,jymax):
                DeltaQ[i,k]=(G[i,k]+dt/dx*np.dot((A[i-1,k]+sigmax[i-1,k])/2,DeltaQ[i-1,k])+dt/dy*np.dot((B[i,k-1]+sigmay[i,k-1])/2,DeltaQ[i,k-1]))/(1+np.ones(4)*dt/dx*sigmax[i,k]+dt/dy*sigmay[i,k])

        #第二スイープ
        for i in range(4):
            DeltaQ[:,:,i]*=1+dt/dx*sigmax+dt/dy*sigmay
        
        for i in range(jxmax-2,-1,-1):#第三スイープ
            for k in range(jymax-2,-1,-1):
                QTMP[i,k]=(DeltaQ[i,k]-dt/dx*np.dot((A[i+1,k]-sigmax[i+1,k])/2,QTMP[i+1,k])-dt/dy*np.dot((B[i,k+1]-sigmay[i,k-+1])/2,QTMP[i,k+1]))/(1+np.ones(4)*dt/dx*sigmax[i,k]+dt/dy*sigmay[i,k])


        Q+=QTMP

        if(n%display_interval==0):
            fig_1.plot(x,Q[:,ktemp,0])
            fig_2.plot(x,Q[:,ktemp,1]/Q[:,ktemp,0])
            fig_3.plot(x,(gamma-1)*(Q[:,ktemp,3]-(Q[:,ktemp,1]**2+Q[:,ktemp,2]**2)/Q[:,ktemp,0]/2))
            fig_4.plot(x,Q[:,ktemp,3])
            
            
    

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



















