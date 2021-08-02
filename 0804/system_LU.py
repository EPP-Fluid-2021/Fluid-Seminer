#システム方程式


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
    Ap = np.zeros([jmax,3,3])#A+
    Am = np.zeros([jmax,3,3])#A-


    Lp = np.zeros([3,3])#lambda+
    Lm = np.zeros([3,3])#lambda-

    Rp = np.ones([3,3])#R 右固有行列
    Rm = np.zeros([3,3])#R-

    I=np.eye(3) #単位行列
    
    cnt = -1
    while(cnt<nmax):
        
        QTMP=np.zeros([jmax,3])#deltaQ*
        DeltaQ=np.zeros([jmax,3])#deltaQ
        E[:,0]=Q[:,1]
        E[:,1]=(3-gamma)*Q[:,1]**2/Q[:,0]/2+(gamma-1)*Q[:,2]
        E[:,2]=(gamma*Q[:,2]-(gamma-1)*Q[:,1]**2/Q[:,0]/2)*Q[:,1]/Q[:,0]

        

        for i in range(1,jmax-1):
                F[i]=dt/dx/2*(E[i-1]-E[(i+1)])
        F[0]=dt/dx/2*(-E[1])
        F[-1]=dt/dx/2*(E[-2])
        
        for i in range(jmax):
            
            u=Q[i,1]/Q[i,0]#rho*u/rho=u
            H=E[i,2]/E[i,0]#(e+p)*u/rho/u=(e+p)/rho=H
            c=H-Q[i,2]/Q[i,0]
            
            c=np.sqrt(c*gamma)#c=sqrt(p*gamma/rho) 局所音速
            
            Lp[0,0]=(u-c+abs(u-c))/2
            Lm[0,0]=(u-c-abs(u-c))/2
            
            Lp[1,1]=(u+abs(u))/2
            Lm[1,1]=(u-abs(u))/2
            
            Lp[2,2]=(u+c+abs(u+c))/2
            Lm[2,2]=(u+c-abs(u+c))/2

            Rp[1,0]=u-c
            Rp[1,1]=u
            Rp[1,2]=u+c

            Rp[2,0]=H-u*c
            Rp[2,1]=u**2/2
            Rp[2,2]=H+u*c

            Rm=np.linalg.inv(Rp)#逆行列

            Ap[i]=np.dot(Rp,np.dot(Lp,Rm))#行列の積で計算
            Am[i]=np.dot(Rp,np.dot(Lm,Rm))
            
        
        for i in range(1,jmax):
            QTMP[i]=np.dot(np.linalg.inv(I+dt/dx*Ap[i]),F[i]+dt/dx*np.dot(Ap[i-1],QTMP[i-1]))

        
        #DeltaQ[jmax-1]=np.dot(np.linalg.inv(I-dt/dx*Am[jmax-1]),QTMP[jmax-1]-dt/dx*np.dot(Am[0],DeltaQ[0]))
        for i in reversed(range(0,jmax-1)):
            DeltaQ[i]=np.dot(np.linalg.inv(I-dt/dx*Am[i]),QTMP[i]-dt/dx*np.dot(Am[i+1],DeltaQ[i+1]))
            

        
       
        
        
        Q+=DeltaQ
            
            
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
