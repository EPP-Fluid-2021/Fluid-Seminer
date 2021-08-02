#システム方程式 AF


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
    
    Ap = np.zeros([jxmax,jymax,4,4])#A+
    Am = np.zeros([jxmax,jymax,4,4])#A-

    Bp = np.zeros([jxmax,jymax,4,4])#B+
    Bm = np.zeros([jxmax,jymax,4,4])#B-
    
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

        
        for k in range(jymax): #kごとにA計算
            Lp = np.zeros([4,4])#lambda+
            Lm = np.zeros([4,4])#lambda-

            Rp = np.ones([4,4])#R 右固有行列
            Rm = np.zeros([4,4])#R-

            for i in range(jxmax):
            
                u=Q[i,k,1]/Q[i,k,0]#rho*u/rho=u
                v=Q[i,k,2]/Q[i,k,0]#rho*v/rho=v
                H=E[i,k,3]/E[i,k,0]#(e+p)*u/rho/u=(e+p)/rho=H
                c=H-Q[i,k,3]/Q[i,k,0]
                
                c=np.sqrt(c*gamma)#c=sqrt(p*gamma/rho) 局所音速
                
                Lp[0,0]=(u-c+abs(u-c))/2
                Lm[0,0]=(u-c-abs(u-c))/2
                
                Lp[1,1]=(u+abs(u))/2
                Lm[1,1]=(u-abs(u))/2
                
                Lp[2,2]=(u+c+abs(u+c))/2
                Lm[2,2]=(u+c-abs(u+c))/2
                
                Lp[3,3]=(u+abs(u))/2
                Lm[3,3]=(u-abs(u))/2
                
                Rp[0,3]=0
                
                Rp[1,0]=u-c
                Rp[1,1]=u
                Rp[1,2]=u+c
                Rp[1,3]=0

                Rp[2,0]=v
                Rp[2,1]=v
                Rp[2,2]=v
                

                Rp[3,0]=H-u*c
                Rp[3,1]=(u**2+v**2)/2
                Rp[3,2]=H+u*c
                Rp[3,3]=v
                

                Rm=np.linalg.inv(Rp)#逆行列

                Ap[i,k]=np.dot(Rp,np.dot(Lp,Rm))#行列の積で計算
                Am[i,k]=np.dot(Rp,np.dot(Lm,Rm))

    
        for i in range(jxmax): #iごとにB計算
        
            Lp = np.zeros([4,4])#lambda+
            Lm = np.zeros([4,4])#lambda-

            Rp = np.ones([4,4])#R 右固有行列
            Rm = np.zeros([4,4])#R-

            for k in range(jymax):
            
                u=Q[i,k,1]/Q[i,k,0]#rho*u/rho=u
                v=Q[i,k,2]/Q[i,k,0]#rho*v/rho=v
                H=E[i,k,3]/E[i,k,0]#(e+p)*u/rho/u=(e+p)/rho=H
                c=H-Q[i,k,3]/Q[i,k,0]
                
                c=np.sqrt(c*gamma)#c=sqrt(p*gamma/rho) 局所音速
                
                Lp[0,0]=(v-c+abs(v-c))/2
                Lm[0,0]=(v-c-abs(v-c))/2
                
                Lp[1,1]=(v+abs(v))/2
                Lm[1,1]=(v-abs(v))/2
                
                Lp[2,2]=(v+c+abs(v+c))/2
                Lm[2,2]=(v+c-abs(v+c))/2
                
                Lp[3,3]=(v+abs(v))/2
                Lm[3,3]=(v-abs(v))/2
                
                Rp[0,3]=0
                
                Rp[1,0]=u
                Rp[1,1]=u
                Rp[1,2]=u
                

                Rp[2,0]=v-c
                Rp[2,1]=v
                Rp[2,2]=v+c
                Rp[2,3]=0
                

                Rp[3,0]=H-v*c
                Rp[3,1]=(u**2+v**2)/2
                Rp[3,2]=H+v*c
                Rp[3,3]=u
                

                Rm=np.linalg.inv(Rp)#逆行列

                Bp[i,k]=np.dot(Rp,np.dot(Lp,Rm))#行列の積で計算
                Bm[i,k]=np.dot(Rp,np.dot(Lm,Rm))


        for i in range(1,jxmax):
            for k in range(1,jymax):
                DeltaQ[i,k]=np.dot(np.linalg.inv(I+dt/dx*Ap[i,k]+dt/dy*Bp[i,k]),G[i,k]+dt/dx*np.dot(Ap[i-1,k],DeltaQ[i-1,k])+dt/dy*np.dot(Bp[i,k-1],DeltaQ[i,k-1]))
                
        for i in range(jxmax-2,-1,-1):
            for k in range(jymax-2,-1,-1):
                QTMP[i,k]=np.dot(np.linalg.inv(I-dt/dx*Am[i,k]-dt/dy*Bm[i,k]),DeltaQ[i,k]-dt/dx*np.dot(Am[i+1,k],QTMP[i+1,k])-dt/dy*np.dot(Bm[i,k+1],QTMP[i,k+1]))
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



















