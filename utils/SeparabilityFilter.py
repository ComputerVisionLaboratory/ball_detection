import numpy as np

def cvtIntegralImage(X):
    H, W = X.shape
    Z = np.zeros((H+1, W+1), np.float64)
    Z[1:,1:] = np.cumsum(np.cumsum(X,0),1)
    return Z

def cvtIntegralImage45_old(X):
    H, W = X.shape
    Z = np.zeros((H+2, W+1), np.float64)
    Z[1:-1, 1:] = X
    tmpX = Z.copy()
    for J in range(2, Z.shape[1]):
        Z[0,J] = Z[0,J] + Z[1,J-1] + tmpX[0, J-1]
        for I in range(1, Z.shape[0]-1):
            Z[I,J] = Z[I,J] + Z[I-1,J-1] + Z[I+1,J-1] - Z[I,J-2] + tmpX[I,J-1]
        Z[-1,J] = Z[-1,J] + Z[-2,J-1]  + tmpX[-1,J-1]
    Z=Z[1:,:]
    return Z

#Optimized Code
import numpy as np
import ctypes as ct

#.soファイルを指定
lib_cvtII45 = np.ctypeslib.load_library("lib_cvtIntegralImage45.so","utils")
#flaot64のポインタのポインタ型を用意
_FLOAT64_PP = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
#SetCHLAC()関数の引数の型を指定(ctypes)　
lib_cvtII45.cvtIntegralImage45.argtypes = [_FLOAT64_PP, ct.c_int, ct.c_int, _FLOAT64_PP]
#SetCHLAC()関数が返す値の型を指定(今回は返り値なし)
lib_cvtII45.cvtIntegralImage45.restype = None

def cvtIntegralImage45(X):
    H, W = X.shape
    Z = np.zeros((H+2, W+1), np.float64)
    Z[1:-1, 1:] = X
    tmpX = Z.copy()
    
    Z = Z.flatten()
    
    lib_cvtII45.cvtIntegralImage45(Z, int(H+2), int(W+1), tmpX.flatten())
    
    Z = Z.reshape(H+2, W+1)
    Z=Z[1:,:]
    
    return Z

def cvtCombSimpRectFilter(I,P,sh):
    bh = sh*2
    bw = np.ceil(sh/3).astype(np.int64)
    sw = np.ceil(sh/3).astype(np.int64)
    dh = 0
    dw = 0
    
    MAP   = np.zeros((I.shape[0]-1, I.shape[1]-1, 2), np.float64)
    
    MAP[:,:,0] = tmpFnc(I,P,bh,bw,sh,sw,dh,dw)
    MAP[:,:,1] = tmpFnc(I,P,bw,bh,sw,sh,dh,dw)
    
    return MAP

def cvtCombSimpRectFilter45(I,P,sh):
    
    r = np.round(sh/np.sqrt(2)).astype(np.int64)
    w = np.ceil(sh/3/np.sqrt(2)).astype(np.int64)
    br = 2*r
    
    MAP   = np.zeros((I.shape[0]-1, I.shape[1]-1, 2), np.float64)

    MAP[:,:,0] = tmpFnc45(I,P,r,br,w,w)
    MAP[:,:,1] = tmpFnc45(I,P,w,w,r,br)
    
    
    return MAP#

def tmpFnc(I,P,bh,bw,sh,sw,dh,dw):
    MAP   = np.zeros((I.shape[0]-1, I.shape[1]-1), np.float64)
    H,W = MAP.shape
    r = np.max([bh,bw])
    N  = (2*bh+1)*(2*bw+1)
    N1 = (2*sh+1)*(2*sw+1)
    N2 = N-N1

    S = (
        I[r -bh  :H-r -bh   ,r -bw  :W-r -bw  ]
      + I[r +bh+1:H-r +bh+1 ,r +bw+1:W-r +bw+1]
      - I[r -bh  :H-r -bh   ,r +bw+1:W-r +bw+1]
      - I[r +bh+1:H-r +bh+1 ,r -bw  :W-r -bw  ]
    )
    T = (
        P[r -bh  :H-r -bh   ,r -bw  :W-r -bw  ]
      + P[r +bh+1:H-r +bh+1 ,r +bw+1:W-r +bw+1]
      - P[r -bh  :H-r -bh   ,r +bw+1:W-r +bw+1]
      - P[r +bh+1:H-r +bh+1 ,r -bw  :W-r -bw  ]
    )
    M = S/N
    Y = T/N
    St = Y - np.power(M, 2)
    S1 = (
         I[r -sh+dh  :H-r -sh+dh   ,r -sw+dw  :W-r -sw+dw]
       + I[r +sh+dh+1:H-r +sh+dh+1,r +sw+dw+1:W-r +sw+dw+1]
       - I[r -sh+dh  :H-r -sh+dh  ,r +sw+dw+1:W-r +sw+dw+1]
       - I[r +sh+dh+1:H-r +sh+dh+1,r -sw+dw  :W-r -sw+dw]
    )
    S2=S-S1
    M1=S1/N1
    M2=S2/N2

    Sb = ((N1*(np.power(M1-M, 2))) + (N2*(np.power(M2-M, 2))))/N
    MAP[r:H-r,r:W-r] = (Sb/St)*np.sign(M2-M1)
    MAP[np.isnan(MAP)]=0
    MAP[np.isinf(MAP)]=0

    return MAP

def tmpFnc45(I,P,r,br,w,bw):
    MAP   = np.zeros((I.shape[0]-1, I.shape[1]-1), np.float64)
    H,W = MAP.shape
    
    h = bw+br+2
    N =(2*bw+1)*(2*(1+2*br))
    N1=(2*w+1)*(2*(1+2*r))
    N2=N-N1
    
    
    HH1 = bw + br +1+1 -1
    HH2 = H-(bw + br +2)
    WW1 = bw + br +3    -1     
    WW2 = W-(bw + br+1)
    
    P1 = I[HH1 - bw - br -1  :HH2 - bw - br -1  ,WW1 + bw - br - 1   :WW2 + bw - br - 1  ]
    P2 = I[HH1 + bw - br -1+1:HH2 + bw - br -1+1,WW1 - bw - br - 1-1:WW2 - bw - br - 1-1 ]
    P3 = I[HH1 + bw + br +1  :HH2 + bw + br +1  ,WW1 - bw + br - 1  :WW2 - bw + br - 1   ]
    P4 = I[HH1 - bw + br     :HH2 - bw + br      ,WW1 + bw + br      :WW2 + bw + br      ]

    S = (P4+P2-P3-P1)
    

    P1 = P[HH1 - bw - br -1  :HH2 - bw - br -1  ,WW1 + bw - br - 1   :WW2 + bw - br - 1  ]
    P2 = P[HH1 + bw - br -1+1:HH2 + bw - br -1+1,WW1 - bw - br - 1-1:WW2 - bw - br - 1-1 ]
    P3 = P[HH1 + bw + br +1  :HH2 + bw + br +1  ,WW1 - bw + br - 1  :WW2 - bw + br - 1   ]
    P4 = P[HH1 - bw + br     :HH2 - bw + br      ,WW1 + bw + br      :WW2 + bw + br      ]
    T= (P4+P2-P3-P1)
    
    

    M = S/N
    Y = T/N
    St = Y - np.power(M, 2)

    P1 = I[HH1 - w - r -1  :HH2 - w - r -1  ,WW1 + w - r - 1    :WW2 + w - r - 1]
    P2 = I[HH1 + w - r -1+1:HH2 + w - r -1+1,WW1 - w - r - 1-1  :WW2 - w - r - 1-1  ]
    P3 = I[HH1 + w + r +1  :HH2 + w + r +1  ,WW1 - w + r - 1    :WW2 - w + r - 1]
    P4 = I[HH1 - w + r     :HH2 - w + r     ,WW1 + w + r         :WW2 + w + r    ]
    S1= (P4+P2-P3-P1);

    S2=S-S1
    M1=S1/N1
    M2=S2/N2

    Sb = ((N1*(np.power(M1-M, 2))) + (N2*(np.power(M2-M, 2))))/N;

    MAP[h-1:-h,h-1:-h]=(Sb/St)*np.sign(M2-M1)
    MAP[np.isnan(MAP)]=0
    MAP[np.isinf(MAP)]=0
    
    return MAP

def cvtFindLocalPeakX(X, Flg, Thres):
    
    B = np.ones((X.shape), np.int64)
    
    if Flg == 1:
      B = (X > (Thres))
    elif Flg == -1:
      B = (X < (Thres))

    N = 1 # number of neighborhood peak

    B[ 0,:] = 0
    B[-1,:] = 0
    B[:, 0] = 0
    B[:,-1] = 0

    a, b = np.nonzero(B)
    #Candinate = [a,b]
    Candinate = np.stack([a,b], 0)
    #print(Candinate.shape)
    PeakList = np.zeros([3,Candinate.shape[1]])
    cnt=0

    if Flg == 1:
        for l in range(0, Candinate.shape[1]):
            y = Candinate[0,l]
            x = Candinate[1,l]
            tmp = X[y-1:y+2,x-1:x+2] #  consider 8 neighbor pixel
            tmp = tmp.reshape(-1)
            ind = np.argsort(tmp)[::-1]
            val = tmp[ind]

            for n in range(0, N):
                if ind[n] == 4:
                    PeakList[:,cnt] = [Candinate[0,l], Candinate[1,l], val[n]]
                    cnt = cnt+1
                
        PeakList = PeakList[:,0:cnt]
        
        ind = np.argsort(PeakList[2,:])[::-1]
        PeakList = PeakList[:,ind]

    elif Flg == -1:
        for l in range(0, Candinate.shape[1]):
            y = Candinate[0,l]
            x =  Candinate[1,l]
            tmp = X[y-1:y+2,x-1:x+2] #  consider 8 neighbor pixel
            tmp = tmp.reshape(-1)
            ind = np.argsort(tmp)#[::-1]
            val = tmp[ind]
            
            for n in range(0, N):
                if ind[n] == 4:
                    PeakList[:,cnt] = [Candinate[0,l], Candinate[1,l], val[n]]
                    cnt = cnt+1
                    
                    
        PeakList = PeakList[:,0:cnt]
        
        ind = np.argsort(PeakList[2,:])#[::-1]
        PeakList = PeakList[:,ind]
    return PeakList