## Hovey Hu 7/16/2018
## OD function differential correction
## input: <FILE>    a1 d1 t1 Rx1 Ry1 Rz1    h deg day AU AU AU
##                  a2 d2 t2 Rx2 Ry2 Rz2
##                  a3 d3 t3 Rx3 Ry3 Rz3 *5

## output: orbital elements a,e,i,O,w,M0
#!!! all coordinates in equatorial unless noted _ec

from HoveyToolbar200 import OD_data_extractor,OD,ephemeris,BABY_OD
from math import radians,degrees
import HoveyToolbar200 as ht
import numpy as np




# total: N observations, n = 2*N position data point

def O_C(Olist,Clist):
    n=len(Olist)
    if n<=6:
        raise ValueError
    s=0
    for i in range(n):
       s+=(Olist[i]-Clist[i])**2
    RMS=(s/(n-6))**0.5
    return RMS



#tool functions
def status_unpacker(status):
    x,y,z,u,v,w,t0=status
    r=np.array([[x],[y],[z]])
    v=np.array([[u],[v],[w]])
    return r,v,t0

def update_OE(oOE,t0):
    a,e,i,O,w,M0=oOE
    return a,e,i,O,w,M0,t0

def data_unpacker(data):
    #input unit in hr,deg,JD,AU
    #outpu unit in deg,deg,JD,AU
    observeA=ht.hr_to_deg(data[0])
    observeD=data[1]
    t=data[2]
    R_vec=data[3:6]
    R_vec.shape=(3,1)
    return observeA,observeD,t,R_vec

def twich_status(matx,status):
    tstatus=[]
    for i in range(6):
        tstatus.append(status[i]+matx[i,0]) #!sign with doubt
    tstatus.append(status[6])
    return tstatus

#func predictor
def predictor(status,tf,Rf):
    r,v,t0=status_unpacker(status)
    OE=update_OE(ht.BABY_OD(r,v,t0),t0)
    a,d=ephemeris(OE,tf,Rf)
    return a,d

#func calculate individual matA
def calc_a(status,data):
    RATIO=1/1000
    observeA,observeD,t,R_vec=data_unpacker(data)
    predictA,predictD=predictor(status,t,R_vec)
    matA=np.zeros((6,1))
    matD=np.zeros((6,1))
    for i in range(6):
        deltaA=observeA-predictA
        deltaD=observeD-predictD



        #partial derivitive calculation
        d=status[i]*RATIO
        #plus
        status_plus=status.copy()
        status_plus[i]+=d
        plusA,plusD=predictor(status_plus,t,R_vec)

        #minus
        status_minu=status.copy()
        status_minu[i]-=d
        minuA,minuD=predictor(status_minu,t,R_vec)

        papi=(plusA-minuA)/(2*d)
        pdpi=(plusD-minuD)/(2*d)

        matA[i,0]=(deltaA*papi)
        matD[i,0]=(deltaD*pdpi)
        
    mata=matA+matD
    return mata 



#func calculate individual matJ
def calc_J(status,data):
    RATIO=1/1000
    observeA,observeD,t,R_vec=data_unpacker(data)
    matA=np.zeros((6,1))
    matD=np.zeros((6,1))
    for i in range(6):
        #partial derivitive calculation
        d=status[i]*RATIO
        #plus
        status_plus=status.copy()
        status_plus[i]+=d
        plusA,plusD=predictor(status_plus,t,R_vec)

        #minus
        status_minu=status.copy()
        status_minu[i]-=d
        minuA,minuD=predictor(status_minu,t,R_vec)
        papi=(plusA-minuA)/(2*d)
        pdpi=(plusD-minuD)/(2*d)
        matA[i,0]=papi
        matD[i,0]=pdpi
    horizMatA=matA.copy()
    horizMatA.shape=(1,6)
    matJa=np.matmul(matA,horizMatA)
    
    horizMatD=matD.copy()
    horizMatD.shape=(1,6)
    matJd=np.matmul(matD,horizMatD)
    return matJa+matJd



def dif_OD(rawdata,dcflag):

    #get OD done

    ODdata=rawdata[0:3,:]
    ODresult=OD(ODdata)

    [a,e,i,O,w,M0],[r_vec,v_vec],t0=ODresult
    OE=[a,e,i,O,w,M0,t0]
    print("Orignal OD Orbital Elements:",ODresult[0],"at epoch",t0)
    print("Using Observation 1 2 3 ")
    if dcflag==False:
        return OE
    
    rx=r_vec[0,0]
    ry=r_vec[1,0]
    rz=r_vec[2,0]
    vx=v_vec[0,0]
    vy=v_vec[1,0]
    vz=v_vec[2,0]
    status=[rx,ry,rz,vx,vy,vz,t0]

    testR=ODdata[1,3:6]
    testR.shape=(3,1)

    Olist=[]
    Plist=[]
    for data in rawdata:
        observeA,observeD,t,R_vec=data_unpacker(data)
        predictA,predictD=predictor(status,t,R_vec)
        Olist.append(observeA)
        Olist.append(observeD)
        Plist.append(predictA)
        Plist.append(predictD)
    print("Old O-C:",O_C(Olist,Plist))

    N=len(rawdata)
    mata=np.zeros((6,1))
    matJ=np.zeros((6,6))
    for i in range(N):
        mata+=calc_a(status,rawdata[i])
        matJ+=calc_J(status,rawdata[i])

    invJ=np.linalg.inv(matJ)
    matx=np.matmul(invJ,mata)



    status=twich_status(matx,status)

    Olist=[]
    Plist=[]
    for data in rawdata:
        observeA,observeD,t,R_vec=data_unpacker(data)
        predictA,predictD=predictor(status,t,R_vec)
        Olist.append(observeA)
        Olist.append(observeD)
        Plist.append(predictA)
        Plist.append(predictD)
        


    r,v,t0=status_unpacker(status)
    oOE=ht.BABY_OD(r,v,t0)
    dOE=update_OE(oOE,t0)

    
    print("r,v,t",r,v,t0)
    ec_r=ht.equ_to_ecl(r)
    ec_v=ht.equ_to_ecl(v)
    ec_day_v=0.01720209895*ec_v
    print("r,v,t,in ecliptic",ec_r,ec_v,ec_day_v,t0)
    print("\nModified Orbital Elements:",dOE,"at epoch",t0)
    print("New O-C:",O_C(Olist,Plist))
    return dOE

rawdata=OD_data_extractor("OD Standard Input.txt")

#Main
dcflag=True #differencial correction flag
print(dif_OD(rawdata,dcflag))
print(dif_OD(rawdata,False))





