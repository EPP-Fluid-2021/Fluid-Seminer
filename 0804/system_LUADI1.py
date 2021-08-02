#システム方程式
#LUADI1

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
nn=10


PI = 1.0


RHOI = 1.0
RHOE = 1.0

UI = 1.0
UE = 1.0



xmin,xmid,xmax = 0,0.5,1.0
x = np.linspace(xmin,xmax,jmax)




dx = (xmax-xmin)/(jmax-1)  #格子間隔



def init(): #初期値
    Q = np.zeros([jmax,3])
    Q[:,0]=RHOI*np.exp(-(x-xmid)**2*100)+RHOE
    Q[:,1]=(RHOI*np.exp(-(x-xmid)**2*100)+RHOE)*(UE)
    Q[:,2]=(PI/(gamma-1)+(RHOI*np.exp(-(x-xmid)**2*100)+RHOE)*(UE)**2/2)
    
    return Q


def compute(Q):
    E = np.zeros([jmax,3]) #E
    F = np.zeros([jmax,3]) #Eの微分を差分化
    


    Lp = np.zeros([jmax,3,3])#lambda+
    Lm = np.zeros([jmax,3,3])#lambda-

    Rp = np.ones([jmax,3,3])#R 右固有行列
    Rm = np.zeros([jmax,3,3])#R-

    I=np.eye(3) #単位行列
    
    cnt = -1
    while(cnt<nmax):
        I=np.eye(3)
        QTMP=np.zeros([jmax,3])#deltaQ*
        RDeltaQ=np.zeros([jmax,3])#deltaQ
        E[:,0]=Q[:,1]
        E[:,1]=(3-gamma)*Q[:,1]**2/Q[:,0]/2+(gamma-1)*Q[:,2]
        E[:,2]=(gamma*Q[:,2]-(gamma-1)*Q[:,1]**2/Q[:,0]/2)*Q[:,1]/Q[:,0]

        

        for i in range(1,jmax-1):
                F[i]=dt/dx/2*(E[i-1]-E[(i+1)])
        #F[0]=dt/dx/2*(-E[1])
       # F[-1]=dt/dx/2*(E[-2])
        
        for i in range(jmax):
            
            u=Q[i,1]/Q[i,0]#rho*u/rho=u
            H=E[i,2]/E[i,0]#(e+p)*u/rho/u=(e+p)/rho=H
            c=H-Q[i,2]/Q[i,0]
            
            c=np.sqrt(c*gamma)#c=sqrt(p*gamma/rho) 局所音速
            
            Lp[i,0,0]=(u-c+abs(u-c))/2
            Lm[i,0,0]=(u-c-abs(u-c))/2
            
            Lp[i,1,1]=(u+abs(u))/2
            Lm[i,1,1]=(u-abs(u))/2
            
            Lp[i,2,2]=(u+c+abs(u+c))/2
            Lm[i,2,2]=(u+c-abs(u+c))/2

            Rp[i,1,0]=u-c
            Rp[i,1,1]=u
            Rp[i,1,2]=u+c

            Rp[i,2,0]=H-u*c
            Rp[i,2,1]=u**2/2
            Rp[i,2,2]=H+u*c

            Rm[i]=np.linalg.inv(Rp[i])#逆行列

            
        for i in range(1,jmax-1):
                F[i]=np.dot(Rm[i],F[i]) #右辺
        
        for i in range(1,jmax):
            QTMP[i]=np.dot(np.linalg.inv(I+dt/dx*Lp[i]),F[i]+dt/dx*np.dot(Lp[i-1],QTMP[i-1]))


        
        #DeltaQ[jmax-1]=np.dot(np.linalg.inv(I-dt/dx*Am[jmax-1]),QTMP[jmax-1]-dt/dx*np.dot(Am[0],DeltaQ[0]))
        for i in range(jmax-2,-1,-1):
            RDeltaQ[i]=np.dot(np.linalg.inv(I-dt/dx*Lm[i]),QTMP[i]-dt/dx*np.dot(Lm[i+1],RDeltaQ[i+1]))
            


        for i in range(0,jmax):
            Q[i]+=np.dot(Rp[i],RDeltaQ[i])
            
                
  
            
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
