#Hovey's tool bar 2.0
#unify input OE
#all vectors expressed in vertical matrix

import numpy as np
import math
from math import pi,sqrt, sin, cos, asin, acos, tan, atan, radians, degrees,atan2
import copy
from astropy.io import fits
import matplotlib.pyplot as plt

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
    observeA=hr_to_deg(data[0])
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
    OE=update_OE(BABY_OD(r,v,t0),t0)
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
    oOE=BABY_OD(r,v,t0)
    dOE=update_OE(oOE,t0)

    return dOE

def findQuadrant(sine, cosine):
    if cosine > 0 and sine > 0: #1
        return asin(sine)

    if cosine < 0 and sine > 0: #2
        return acos(cosine)

    if cosine < 0 and sine < 0: #3
        return pi - asin(sine)

    if cosine > 0 and sine < 0: #4
        return 2*pi + asin(sine)

def ephemeris(OE,it,ivector_R):
    ia,ie,ii,iO,iw,iM0,it0=OE
    #input au,degrees,julian days, au
    #R is the vector from earth to Sun in equatorial coordinate
    #unit conversion
    a=ia
    e=ie
    i=radians(ii)
    O=radians(iO)
    w=radians(iw)
    t0=it0
    t=it
    M0=radians(iM0)
    vector_R=equ_to_ecl(ivector_R)
    #t in Julian date
    miu=0.01720209895**2
    n=sqrt(miu/(a**3))
    #calculate M
    M=M0+n*(t-t0)
    E=solve_kep(M,e)
    phycoo=np.array(calc_phy_coor(E,a,e))
    phycoo.shape=(3,1)
    ecccoo=ecliptic_coordinates(O,i,w,phycoo)
    vector_r=ecccoo
    vector_p=vector_R+vector_r
    equcoo=ecl_to_equ(vector_p)
    px=equcoo[0]
    py=equcoo[1]
    pz=equcoo[2]
    p=sqrt(px**2+py**2+pz**2)
    d=asin(pz/p)
    sina=py/p/cos(d)
    cosa=px/p/cos(d)
    a=findQuadrant(sina,cosa)
    
    dec=degrees(d)
    ra=degrees(a)
    return ra,dec

def calc_phy_coor(E,a,e):
    x=a*(cos(E)-e)
    y=a*sqrt(1-e**2)*sin(E)
    z=0
    return x,y,z

def OD(data):
    ## OD function
    ## input: <matrix>  a1 d1 t1 Rx1 Ry1 Rz1    h deg day AU AU AU
    ##                  a2 d2 t2 Rx2 Ry2 Rz2
    ##                  a3 d3 t3 Rx3 Ry3 Rz3

    ## output: orbital elements [a,e,i,O,w,M0],[r_ecl,v_ecl],t2

    ## Notation
    ##            roi     :   mag(roi_vec)
    ##            roi_hat :   unit vector
    ##            roi_vec :   earth-asteroid vector
    ##            ri_vec  :   position vector of obs i
    ##            ri      :   mag(ri_vec)
    ##            Ri      :   vector from earth to sun
    ##            vi_vec  :   velocity vector of obs i

    #!!! all coordinates in equatorial unless noted _ec
    #return v_vec in AU/Gaussian Day
    OElist=[]
    RVlist=[]
    ERRRAG=10**(-12) #ERRRAG error range
    miu=0.01720209895**2
    c=173.144633 #AU/day
    newtonflag=False
    testflag=False
    lcflag=False #light correction flag
    a1=radians(hr_to_deg(data[0,0]))
    a2=radians(hr_to_deg(data[1,0]))
    a3=radians(hr_to_deg(data[2,0]))
    d1=radians(data[0,1])
    d2=radians(data[1,1])
    d3=radians(data[2,1])

    t1=data[0,2]
    t2=data[1,2]
    t3=data[2,2]
    R1=data[0:1,3:]
    R2=data[1:2,3:]
    R3=data[2:3,3:]
    R1.shape=(3,1)
    R2.shape=(3,1)
    R3.shape=(3,1)

    #2.1 coordinate calc&trans: ro1~3 <- a1~3,d1~3===================

    eq1=get_hat(a1,d1)
    eq2=get_hat(a2,d2)
    eq3=get_hat(a3,d3)

    ro1_hat=eq1
    ro2_hat=eq2
    ro3_hat=eq3

    if testflag==True:
        print("ro1_hat,ro2_hat,ro3_hat",ro1_hat,ro2_hat,ro3_hat)

    #2.2 Gaussian time transformation: tao1~3 <- t1~3=================
    sqrtmiu=0.01720209895
    k=sqrtmiu
    tao3=k*(t3-t2)
    tao1=k*(t1-t2)
    tao=tao3-tao1


    #3 calc r2_vec, <- ro1~3,R1,R2,R3=================================
    A1=tao3/tao
    A3=-tao1/tao
    B1=1/6*A1*(tao**2-tao3**2)
    B3=1/6*A3*(tao**2-tao1**2)
    D0=dot(ro1_hat,cross(ro2_hat,ro3_hat))
    D21=dot(cross(ro1_hat,R1),ro3_hat)
    D22=dot(cross(ro1_hat,R2),ro3_hat)
    D23=dot(cross(ro1_hat,R3),ro3_hat)
    E=-2*(dot(ro2_hat,R2))
    F=(mag(R2))**2
    A=-(A1*D21-D22+A3*D23)/D0
    B=-(B1*D21+B3*D23)/D0

    a=-(A**2+A*E+F)
    b=-(2*A*B+B*E) # ?miu==1 or not
    c=-B**2

    if testflag==True:
        print("a,b,c",a,b,c)

    p8=[1,0,a,0,0,b,0,0,c]
    roots=np.roots(p8)

        #root loop
    r2list=root_test(roots,A,B)
    Nroot=len(r2list)
    if Nroot == 0:
        print("ERROR: no appropriate root for r2")
        return None
    for rootchoice in range(Nroot):
        r2=r2list[rootchoice]
        ro2=A+B/r2**3
        r2_vec=ro2*ro2_hat-R2
        ro2_vec=ro2*ro2_hat
        if testflag==True:
            print("ro2",ro2)


        #4 calc fi,gi <- taoi,r2_vec(r2),miu(=1)=========================
        f1=1-(tao1**2)/(2*r2**3)
        f3=1-(tao3**2)/(2*r2**3)
        g1=tao1-(tao1)**3/(6*r2**3)
        g3=tao3-(tao3)**3/(6*r2**3)

        v2_vec=np.array([[1],[1],[1]])

        r2_old=0
        r2_new=100
        v2_old=0
        v2_new=100

        MainCount=0
        while (abs(r2_old-r2_new)>=ERRRAG or abs(v2_old-v2_new)>=ERRRAG) and MainCount<300:
            #5 calc r1_vec,r3_vec <- ro1~3,R1~3,fi,gi========================
                #now scalar equation of range
            c1=g3/(f1*g3-g1*f3)
            c2=-1 #by definition
            c3=-g1/(f1*g3-g1*f3) #CHECKED
            D11=dot(cross(R1,ro2_hat),ro3_hat)
            D12=dot(cross(R2,ro2_hat),ro3_hat)
            D13=dot(cross(R3,ro2_hat),ro3_hat)
            D31=dot(ro1_hat,cross(ro2_hat,R1))
            D32=dot(ro1_hat,cross(ro2_hat,R2))
            D33=dot(ro1_hat,cross(ro2_hat,R3))
            ro1=(c1*D11+c2*D12+c3*D13)/(c1*D0)
            ro2=(c1*D21+c2*D22+c3*D23)/(c2*D0)
            ro3=(c1*D31+c2*D32+c3*D33)/(c3*D0)
                #light travel correction
            if lcflag==True:
                t1r=t1-ro1/c
                t2r=t2-ro2/c
                t3r=t3-ro3/c
                tao3=k*(t3r-t2r)
                tao1=k*(t1r-t2r)
                tao=tao3-tao1
            if testflag==True:
                print("ro2",ro2)
            ro1_vec=ro1*ro1_hat
            ro2_vec=ro2*ro2_hat
            ro3_vec=ro3*ro3_hat
            r1_vec=ro1_vec-R1
            r2_vec=ro2_vec-R2
            r3_vec=ro3_vec-R3
            r2_old=r2
            r1=mag(r1_vec)
            r2=mag(r2_vec)
            r3=mag(r3_vec)
            r2_new=r2
            
            if testflag==True:
                print("ro1_vec,ro2_vec,ro3_vec",ro1_vec,ro2_vec,ro3_vec)
            
            #6 calc v2_vec <- r1_vec,r3_vec,fi,gi============================
            d1=-f3/(f1*g3-f3*g1)
            d3=f1/(f1*g3-f3*g1)
            v2_old=mag(v2_vec)
            v2_vec=d1*r1_vec+d3*r3_vec
            v2_new=mag(v2_vec)

            #7 calc a,dEi <- r2_vec,v2_vec
            a,e,E2=MINI_OD(r2_vec,v2_vec)
            n=(miu/a**3)**0.5

            if newtonflag==True:
                    #newton's method calculate dEi
                    #x-(1-r2/a)sin(x)+e*sin(E2)*(1-cos(x))-n*(dtao)
                dtao=tao1 #! possible error: unit in gaussian day?
                xg=n*dtao
                xn=0
                i=0
                while abs(xg-xn)>=ERRRAG and i <1000000 :
                    fx=xg-(1-r2/a)*sin(xg)+e*sin(E2)*(1-cos(xg))-n*(dtao)
                    fx_=1-(1-r2/a)*cos(xg)+dot(r2_vec,v2_vec)/n/a**2*sin(xg)
                    xn=xg-fx/fx_
                    x=xn
                    xn=xg
                    xg=x
                    i+=1
                dE1=xn

                dtao=tao3 #! possible error: unit in gaussian day?
                xg=n*dtao
                xn=0
                i=0
                while abs(xg-xn)>=ERRRAG and i <1000000 :
                    fx=xg-(1-r2/a)*sin(xg)+e*sin(E2)*(1-cos(xg))-n*(dtao)
                    fx_=1-(1-r2/a)*cos(xg)+dot(r2_vec,v2_vec)/n/a**2*sin(xg)
                    xn=xg-fx/fx_
                    x=xn
                    xn=xg
                    xg=x
                    i+=1
                dE3=xn
                #8 calc newfi,newgi and go to step 5=============================
                f1=1-a/r2*(1-cos(dE1))
                f3=1-a/r2*(1-cos(dE3))
                g1=tao1+1/n*(sin(dE1)-dE1)
                g3=tao3+1/n*(sin(dE3)-dE3)
            else:
                u=1/r2**3
                z=dot(r2_vec,v2_vec)/r2**2
                q=dot(v2_vec,v2_vec)/r2**2-u
                f1=1-u/2*tao1**2+u*z/2*tao1**3+(3*u*q-15*u*z**2+u**2)/24*tao1**4
                g1=tao1-u/6*tao1**3+u*z/4*tao1**4
                f3=1-u/2*tao3**2+u*z/2*tao3**3+(3*u*q-15*u*z**2+u**2)/24*tao3**4
                g3=tao3-u/6*tao3**3+u*z/4*tao3**4
            MainCount+=1
        #LOOP END HERE
        
        r2_vec_ec=equ_to_ecl(r2_vec)
        v2_vec_ec=equ_to_ecl(v2_vec)
        
        orbital_elements=BABY_OD(r2_vec_ec,v2_vec_ec,t2)
        a=orbital_elements[0]
        e=orbital_elements[1]
        
            #sainity check: Near Earth Asteroid
        if a+a*e<0.7 or a-a*e>1.3:
            continue
        OElist.append(orbital_elements)
        RVlist.append([r2_vec_ec,v2_vec_ec])
    if len(OElist)!=1:
        print("Choose a set of orbital elements (0~"+str(len(OElist)-1)+"):",)
        print(OElist,"\n")
        print(RVlist,"\n")
        OEchoice=input()
        return OElist[OEchoice],RVlist[OEchoice],t2
    else:
        return OElist[0],RVlist[0],t2

def root_test(roots,A,B): #!!!MAY REQUIRE INPUT
    result=[]
    for root in roots:
        if root.imag==0:
            if root.real>=0.5:
                ro2=A+B/(root.real)**3
                if ro2.real>=0:
                    result.append(root.real)
    return result

def get_hat(a,d):
    X=cos(d)*cos(a)
    Y=cos(d)*sin(a)
    Z=sin(d)
    return np.array([[X],[Y],[Z]])

def OD_data_handler(filename):
    iSIF=np.loadtxt(filename)
    N=len(iSIF)
    HIFlist=[]
    for m in range(N-2):
        for n in range(m+1,N-1):
            for k in range(n+1,N):
                SIF=iSIF[[m,n,k],:]#FLAG           
                #IF want to choose data, slice SIF here
                HIF=np.zeros((3,6))
                for i in range(3):
                    #Calc RA,Dec
                    ra=SIF[i,6]+SIF[i,7]/60+SIF[i,8]/3600
                    dec=SIF[i,9]+SIF[i,10]/60+SIF[i,11]/3600
                    HIF[i,0]=ra
                    HIF[i,1]=dec
                    #Calc JD <- UT
                    Y=SIF[i,0]
                    M=SIF[i,1]
                    D=SIF[i,2]
                    DecHr=SIF[i,3]+SIF[i,4]/60+SIF[i,5]/3600
                    J0=367*Y-int(7*(Y+int((M+9)/12))/4)+int(275*M/9)+D+1721013.5
                    JD=J0+DecHr/24
                    HIF[i,2]=JD
                    #Calc Rx,Ry,Rz
                    Rx=SIF[i,12]
                    Ry=SIF[i,13]
                    Rz=SIF[i,14]
                    R=np.array([[Rx],[Ry],[Rz]])
                    
                    HIF[i,3]=R[0,0]
                    HIF[i,4]=R[1,0]
                    HIF[i,5]=R[2,0]
                HIFlist.append([HIF,m,n,k])
    return HIFlist

def OD_data_extractor(filename):
    SIF=np.loadtxt(filename)
    #IF want to choose data, slice SIF here
    HIF=np.zeros((len(SIF),6))
    for i in range(len(SIF)):
        #Calc RA,Dec
        ra=SIF[i,6]+SIF[i,7]/60+SIF[i,8]/3600
        dec=SIF[i,9]+SIF[i,10]/60+SIF[i,11]/3600
        HIF[i,0]=ra
        HIF[i,1]=dec
        #Calc JD <- UT
        Y=SIF[i,0]
        M=SIF[i,1]
        D=SIF[i,2]
        DecHr=SIF[i,3]+SIF[i,4]/60+SIF[i,5]/3600
        J0=367*Y-int(7*(Y+int((M+9)/12))/4)+int(275*M/9)+D+1721013.5
        JD=J0+DecHr/24
        HIF[i,2]=JD
        #Calc Rx,Ry,Rz
        Rx=SIF[i,12]
        Ry=SIF[i,13]
        Rz=SIF[i,14]
        R=np.array([[Rx],[Ry],[Rz]])
        
        HIF[i,3]=R[0,0]
        HIF[i,4]=R[1,0]
        HIF[i,5]=R[2,0]
    return HIF

def BABY_OD(r,v,t):
    #input:r np.array[[],[],[]],v np.array[[],[],[]],t
    #units in AU, AU, Julian date
    #!!!:ecliptic coordinate system
    #output:a,e,i,O,w,M0
    #units in AU,N/A,deg,deg,deg,deg
    c=173.144633    #speed of light in AU per Day
    miu=0.01720209895**2
    X=r[0,0]
    Y=r[1,0]
    Z=r[2,0]
    a=1/(2/mag(r)-dot(v,v))
    
    e=(1-(mag(cross(r,v)))**2/a)**0.5
    #next calculate h vector
    h=cross(r,v)
    i=acos(h[2,0]/mag(h))
    sinO=h[0,0]/(mag(h)*sin(i))
    cosO=-h[1,0]/(mag(h)*sin(i))
    O=atan2(sinO,cosO)
    O=norm_ang(O)
    sinwf=Z/(mag(r)*sin(i))
    coswf=1/cos(O)*(X/mag(r)+cos(i)*sinwf*sin(O))
    wf=atan2(sinwf,coswf)
    cosf=1/e*(a*(1-e**2)/mag(r)-1)
    sinf=dot(r,v)/e/mag(r)*sqrt(a*(1-e**2))
    
    f=atan2(sinf,cosf)
    w=wf-f
    cosE0=1/e*(1-mag(r)/a)
    sinE0=abs(sinf)/sinf*sqrt(1-cosE0**2)
    E0=atan2(sinE0,cosE0)
    M0=E0-e*sin(E0)

    testflag=True
    #now consider the speed of light
    if testflag==True:
        n=sqrt(miu/a**3)
        M0=M0-n*(mag(r)/c)
    return a,e,degrees(i),degrees(O),degrees(w),degrees(norm_ang(M0))

def MINI_OD(r,v):
    #input:r np.array[[],[],[]],v np.array[[],[],[]],t
    #units in AU, AU, Julian date
    #!!!:ecliptic coordinate system
    #output:a,e,i,O,w,M0
    #units in AU,N/A,deg,deg,deg,deg

    a=1/(2/mag(r)-dot(v,v))
    e=(1-(mag(cross(r,v)))**2/a)**0.5
    cosE0=1/e*(1-mag(r)/a)
    E0=acos(cosE0)
    return a,e,E0

def circ_ap(recAP,r):
    #circular aperture generation function
    #input: rectangular aperture of roof(r)*roof(r) in matrix
    #output: circular matrix, and Nap
        
    #recAP is a matrix, with odd number(in pix) side length N
    #central pix coordinate is (N-1)/2
    n=11 #n is the sample number in each side of a pixel
    N=len(recAP[0])
    cirAP=np.copy(recAP)
    #cirAP.dtype = 'float32'
    Nap=0
    for j in range(N):
        for i in range(N):
            xi=-N/2+1/2+i
            yj=-N/2+1/2+j
            #considering recAP[j,i]
            fracIn=0
            d=(xi**2+yj**2)**0.5
            #if all in
            if d<r-0.7072:
                fracIn=1
            #if all out
            elif d>r+0.7072:
                fracIn=0
            #if not all in/out
            else:
                for q in range(n):
                    for p in range(n):
                        ap=xi-1/2+1/2/n+p/n
                        bq=yj-1/2+1/2/n+q/n
                        d2=(ap**2+bq**2)
                        if(d2<=r**2):
                            fracIn+=1
                fracIn = fracIn / n**2
            
            cirAP[j,i]=fracIn*recAP[j,i]
            Nap+=fracIn
    return cirAP,Nap

def solve_kep(M,e):
    Eguess = M
    Mguess = Eguess - e*sin(Eguess)
    Mtrue = M
    while abs(Mguess - Mtrue) > 1e-004:                                                                 #CHANGE: < to >
        Eguess = Eguess - (Eguess - e*sin(Eguess) - Mtrue) / (1 - e*cos(Eguess))
        Mguess = Eguess - e*sin(Eguess)                                                                                                                        #CHANGE: changed order here
    return Eguess

def cartesian_coordinates(a,E,e):
    cc=np.array([
        [a*cos(E)-a*e],
        [a*sqrt(1-e**2)*sin(E)],
        [0]])
    return cc

def ecliptic_coordinates(o,i,w,cc): #cc is the cartesian coordinates
    matw=np.array([
        [cos(w),-sin(w),0],
        [sin(w),cos(w),0],
        [0,0,1]
        ])
    mati=np.array([
        [1,0,0],
        [0,cos(i),-sin(i)],
        [0,sin(i),cos(i)]
        ])
    mato=np.array([
        [cos(o),-sin(o),0],
        [sin(o),cos(o),0],
        [0,0,1],
        ])
    result=np.matmul(mato,np.matmul(mati,np.matmul(matw,cc)))
    return result

def norm_ang(rad):
    b=rad
    while b<0:
        b+=2*math.pi
    while b>2*math.pi:
        b-=2*math.pi
    return b
def ecl_to_equ(ec):
    e=radians(23.4352)
    mate=np.array([
        [1,0,0],
        [0,cos(e),-sin(e)],
        [0,sin(e),cos(e)]
        ])
    return np.matmul(mate,ec)

def equ_to_ecl(ec):
    e=radians(-23.4352)
    mate=np.array([
        [1,0,0],
        [0,cos(e),-sin(e)],
        [0,sin(e),cos(e)]
        ])
    return np.matmul(mate,ec)

def cal_coordinates(M0,time,a,e,oprime,iprime,wprime):
    M=M0+n*(time)
    E=solvekep(M)
    cc=cartesian_coordinates(a,E,e)
    ec=ecliptic_coordinates(oprime,iprime,wprime,cc)
    eqc=equatorial_coordinates(ec)
    r.x=eqc[0,0]
    r.y=eqc[1,0]
    r.z=eqc[2,0]
    return r

def cross(v1,v2):
    v1.shape=(1,3)
    v2.shape=(1,3)
    c=np.cross(v1,v2)
    c.shape=(3,1)
    v1.shape=(3,1)
    v2.shape=(3,1)
    return c

def mag(vector):
    #3D vector magnitude calculation only!
    s=0
    for i in range(3):
        s+=vector[i,0]**2
    return sqrt(s)

def dot(v1,v2):
    
    s=0
    for i in range(3):
        s+=v1[i,0]*v2[i,0]
    return s

def rtd_list(radlist):
    deglist=[]
    for i in radlist:
        deglist.append(rad_to_deg(i))
    return deglist
  
def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

def read(f):
    copy = []
    for line in f:
        for c in line:
            copy.append(c)
    return copy

def read_str(filename,f):
    
    copy=""
    for line in f:
        for c in line:
            copy+=c
        #copy+="\n"
    return copy

def filt_cln(data):
    for i in range(len(data)):
        if data[i]==':':
            data[i]=' ';
    return None

def write(filename,data):
    f = open(filename, 'w')
    f.write(data)

def deg_to_hr(degree,decimal_place):
    hour=int(degree/360*24)
    minute=int(degree/360*24*60)-hour*60
    second=round(float(degree/360*24*3600-hour*60*60-minute*60),decimal_place)
    return str(hour)+":"+str(minute)+":"+str(second)

def decdeg_to_deg(degree,decimal_place):
    deg=int(degree)
    minute=int(degree*60)-deg*60
    second=round(float(degree*3600-deg*60*60-minute*60),decimal_place)
    return str(deg)+":"+str(minute)+":"+str(second)

#read file function
#!!! it returns a dictionary with string like data !!!
#finished
def smart_read(f):      #smart, but format limited
                        #format: x y R.A. Dec.\n
    buffer=""
    paracount=1             #count 1 x, 2 y, 3 RA, 4 Dec
    stardata={}
    stardata["x"]=[]
    stardata["y"]=[]
    stardata["ra"]=[]
    stardata["dec"]=[]
    for line in f:
        for c in line:
            if c == " ":
                if paracount==1:
                    stardata["x"].append(buffer)
                    buffer=""
                    paracount+=1
                    continue
                if paracount==2:
                    stardata["y"].append(buffer)
                    buffer=""
                    paracount+=1
                    continue
                if paracount==3:
                    stardata["ra"].append(buffer)
                    buffer=""
                    paracount+=1
                    continue
            if c == "\n":
                stardata["dec"].append(buffer)
                buffer=""
                paracount=1
                continue
            buffer+=c
    return stardata

#solving plate contants
#finished and checked
def solv_plat_const(stardata):
    matA=np.zeros((3,1))
    #astro matrix:[[sum a],[sum a*x],[sum a*y]]
    for i in range(len(stardata["x"])):
        matA[0,0]+=float(stardata["ra"][i])
        matA[1,0]+=float(stardata["ra"][i])*float(stardata["x"][i])
        matA[2,0]+=float(stardata["ra"][i])*float(stardata["y"][i])
    matP=np.zeros((3,3))
    for i in range(len(stardata["x"])):
        matP[0,0]+=1
        matP[0,1]+=float(stardata["x"][i])
        matP[0,2]+=float(stardata["y"][i])
        matP[1,0]+=float(stardata["x"][i])
        matP[1,1]+=float(stardata["x"][i])**2
        matP[1,2]+=float(stardata["x"][i])*float(stardata["y"][i])
        matP[2,0]+=float(stardata["y"][i])
        matP[2,1]+=float(stardata["x"][i])*float(stardata["y"][i])
        matP[2,2]+=float(stardata["y"][i])**2
    invmatP=np.linalg.inv(matP)
    result1=np.dot(invmatP,matA)
    #result1 format:[[b1],[a11],[a12]]

    matA=np.zeros((3,1))
    #astro matrix:[[sum a],[sum a*x],[sum a*y]]
    for i in range(len(stardata["x"])):
        matA[0,0]+=float(stardata["dec"][i])
        matA[1,0]+=float(stardata["dec"][i])*float(stardata["x"][i])
        matA[2,0]+=float(stardata["dec"][i])*float(stardata["y"][i])
    result2=np.dot(invmatP,matA)
    #result2 format:[[b2],[a21],[a22]]
    result=np.zeros((3,2))
    result[:,0:1]=result1
    result[:,1:]=result2
    #result format:[[b1,b2],[a11,a21],[a12,a22]]
    return result

#astrometry for (x,y) calculation
#finished and checked
def pin_on_sky(matC,matA,matB):
    #matC(matrix coordinate) format: [[x],[y]]
    matS=matB+np.dot(matA,matC)
    #sky matrix format: [[R.A.],[Dec.]]
    return matS

#flatten function
#finished and check to the 8th decimal places
def flatten(stardata,L):
    flattened={}
    flattened["x"]=[]
    flattened["y"]=[]
    flattened["ra"]=[]
    flattened["dec"]=[]

    #calculate A,D
    A=0
    D=0
    N=len(stardata["x"])
    for i in range(N):
        A+=float(stardata["ra"][i])
        D+=float(stardata["dec"][i])
    A=radians(A/N)
    D=radians(D/N)

    #generate flattendstardata
    for i in range(N):
        d=radians(float(stardata["dec"][i]))
        a=radians(float(stardata["ra"][i]))
        x=float(stardata["x"][i])
        y=float(stardata["y"][i])
        H=sin(d)*sin(D)+cos(d)*cos(D)*cos(a-A)
        flata=(cos(d)*sin(a-A)/H-x/L)
        flatd=(sin(d)*cos(D)-cos(d)*sin(D)*cos(a-A))/H-y/L
        flattened["x"].append(x)
        flattened["y"].append(y)
        flattened["ra"].append(flata)
        flattened["dec"].append(flatd)
    return flattened

#unflatten function
#this function seems to accidentally change inputs
def unflatten(stardata,L,x,y,flata,flatd):
    #calculate A,D
    A=0
    D=0
    flata+=x/L
    flatd+=y/L
    N=len(stardata["x"])
    for i in range(N):
        A+=float(stardata["ra"][i])
        D+=float(stardata["dec"][i])
    A=radians(A/N)
    D=radians(D/N)
    
    #unflatten
    delta=cos(D)-flatd*sin(D)
    lalala=sqrt(flata**2+delta**2)
    a=degrees(A+atan(flata/delta))
    d=degrees(atan((sin(D)+flatd*cos(D))/lalala))
    return a,d

#uncertainty calculation
def uncertainty(stardata,matA,matB,flatflag,orgstardata):
    sa=0
    sd=0
    N=len(stardata["x"])
    for i in range(N):
        matC=np.zeros((2,1))
        matC[0,0]=float(stardata["x"][i])
        matC[1,0]=float(stardata["y"][i])
        matS=pin_on_sky(matC,matA,matB)
        if flatflag==True:
            coo=unflatten(orgstardata,3911/(24/1000),matC[0,0],matC[1,0],matS[0,0],matS[1,0])
            matS[0,0]=coo[0]
            matS[1,0]=coo[1]
        deltaA=float(orgstardata["ra"][i])-matS[0,0]
        deltaD=float(orgstardata["dec"][i])-matS[1,0]
        sa+=deltaA**2
        sd+=deltaD**2

    cigmaA=sqrt(sa/(N-3))
    cigmaD=sqrt(sd/(N-3))
    return cigmaA,cigmaD

def LSPR(stardata,flatflag,decpla,astroX,astroY):
    orgstardata=copy.deepcopy(stardata)
    if flatflag==True:
        stardata=flatten(stardata,3911/(24/1000))
    ConstMat=solv_plat_const(stardata)    #constants
    matB=ConstMat[0:1,:]
    matB.shape=(2,1)
    matA=np.zeros((2,2))
    #ConstMat format: [[b1,b2],[a11,a21],[a12,a22]]
    #matA format: [[a11,a12],[a21,a22]]
    matA[0,0]=ConstMat[1,0]
    matA[0,1]=ConstMat[2,0]
    matA[1,0]=ConstMat[1,1]
    matA[1,1]=ConstMat[2,1]

    #mattest=[[432.4],[466.6]]

    c=uncertainty(stardata,matA,matB,flatflag,orgstardata)
    cA=c[0]*3600
    cD=c[1]*3600

    astro=pin_on_sky([[astroX],[astroY]],matA,matB)
    astroA=astro[0,0]
    astroD=astro[1,0]
    if flatflag==True:
        stardata=flatten(stardata,3911/(24/1000))
        coo=unflatten(orgstardata,3911/(24/1000),astroX,astroY,astroA,astroD)
        astroA=coo[0]
        astroD=coo[1]
        astro=np.array([[astroA],[astroD]])
    """
    Interprating return value of this function
    LSPRresult=LSPR(stardata,flatflag,decpla,astroX,astroY)
    ConstMat=LSPRresult[0]
    c=LSPRresult[1]
    cA=c[0]*3600
    cD=c[1]*3600
    astro=LSPRresult[2]
    astroA=astro[0,0]
    astroD=astro[1,0]
    """
    return ConstMat,c,astro     #c is uncertainty

#------------------------------------------------------------------------------
#photometry functions

def fits_centroid(filename,x,y,d): #!!!central function
    #d is diameter of the object
    #odd interger, unit in pix
    d0=int(np.ceil(d))
    im=readfits(filename)
    
    aperture=im[int(y-(d0-1)/2):int(y+(d0+1)/2),int(x-(d0-1)/2):int(x+(d0+1)/2)]
    cirAP=circ_ap(aperture,d/2)[0]
    result=rect_centroid(aperture)
    xc=x-(d0-1)/2+result[0]
    yc=y-(d0-1)/2+result[1]
    uncerX=result[2]
    uncerY=result[3]
    return xc,yc,uncerX,uncerY

def rec_ap(image,x,y,d):
    aperture=image[int(y-(d-1)/2):int(y+(d+1)/2),int(x-(d-1)/2):int(x+(d+1)/2)]
    return aperture

def rect_centroid(data):     #data is a numpy matrix
    #this function performs rectangular rect_centroiding
    
    N=data.sum()
    xc=0                #centre x coordinate
    yc=0                #centre y coordinate
    
    #initialize sums
    xsum=[]
    ysum=[]
    for i in data:
        ysum.append(0)
    for i in data[0]:
        xsum.append(0)

    #calculate sums
    y=0
    for i in data:
        x=0
        for j in i:
            ysum[y]+=j
            xsum[x]+=j
            x+=1
        y+=1
    """We can also use slice here to calculate sum, quicker"""

    
    #calculate centre
    buffer=0
    for i in range(len(ysum)):
        buffer+=ysum[i]*i
    yc=buffer/N

    buffer=0
    for i in range(len(xsum)):
        buffer+=xsum[i]*i
    xc=buffer/N

    #calculate error
    buffer=0
    for i in range(len(xsum)):
        buffer+=xsum[i]*(i-xc)**2
                                #now buffer=sum((xi-xc)^2)
    ex=math.sqrt(buffer/N/(N-1))
    buffer=0
    for i in range(len(ysum)):
        buffer+=ysum[i]*(i-yc)**2
    ey=math.sqrt(buffer/N/(N-1))
    
    return xc,yc,ex,ey          #x coordinate, y coordinate
                                #x error, y error

"""Checked"""
def display(mat):               #display a matrix
    plt.imshow(mat)
    plt.gray()#haha, I like colorful ones, just put a brackets if want grey
    plt.show()
    return 0

"""Checked"""
def readfits(filename):
    return fits.getdata(filename)

def hr_to_deg(dechr):
    decdeg=dechr*360/24
    return decdeg

def coorsys_rot_mat(angle,axis):
    #rotate about z axis for now
    #angle in radians
    rot_mat=np.zeros((3,3))
    rot_mat[0,0]=cos(angle)
    rot_mat[0,1]=sin(angle)
    rot_mat[0,2]=0
    rot_mat[1,0]=-sin(angle)
    rot_mat[1,1]=cos(angle)
    rot_mat[1,2]=0
    rot_mat[2,0]=0
    rot_mat[2,1]=0
    rot_mat[2,2]=1
    return rot_mat

    
    







