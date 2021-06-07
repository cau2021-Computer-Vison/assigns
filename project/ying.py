import numpy as np
import matplotlib.pyplot as plt
import imageio
import scipy, scipy.misc, scipy.signal
import cv2
import sys

def computeTextureWeights(fin, sigma, sharpness):
    # 매개변수=> fin: np data(img -> shape: (50,50)), sigma: 3.0, sharpness: 0.001

    # 넘파이 diff를 이용한 1차 차분 가로쪽으로 이미지를 차분하고, [1차차분, 0행, 마지막행]을 쌓는다.
    dt0_v = np.vstack( ( np.diff(fin, n=1, axis=0), fin[0,:]-fin[-1,:] ) )
    # 위와 같은방식으로 세로에 대해서 하고 [1차차분, 0열, 마지막열]을 쌓는다.
    dt0_h = np.vstack((np.diff(fin, n=1, axis=1).conj().T, fin[:,0].conj().T-fin[:,-1].conj().T)).conj().T

    # 행, 열에 대해 만든 넘파이 배열에 대해 2차원 배열을 변환하여 다시 만들어줍니다.
    gauker_h = scipy.signal.convolve2d(dt0_h, np.ones((1,sigma)), mode='same')
    gauker_v = scipy.signal.convolve2d(dt0_v, np.ones((sigma,1)), mode='same')


    # 아래의 두값은 결국 요소끼리의 연산으로 shape은 그대로 (50,50)입니다.
    W_h = 1/(np.abs(gauker_h)*np.abs(dt0_h)+sharpness)
    W_v = 1/(np.abs(gauker_v)*np.abs(dt0_v)+sharpness)

    return  W_h, W_v
    
def solveLinearEquation(IN, wx, wy, lamda):
    # 매개변수 => IN: 이미지의 넘파이값 (50,50)의 shape , wx,wy 둘다 (50,50)의 넘파이 배열, 0.01

    # r: 50, c: 50, k: 2500
    [r, c] = IN.shape
    k = r * c

    # flatten을 통해서 벡터로 만들어주고, wx, wy을 해주고 -람다 곱하기
    dx =  -lamda * wx.flatten('F')
    dy =  -lamda * wy.flatten('F')

    # wx는 axis 0 대한 미분, wy는 axis 1에 대한 미분이므로 각 축에 맞게 roll(shift)해주고 temp값을 만듭니다.
    tempx = np.roll(wx, 1, axis=1)
    tempy = np.roll(wy, 1, axis=0)

    # (2500,)의 shape의 dxa, dya를 만듭니다.
    dxa = -lamda *tempx.flatten('F')
    dya = -lamda *tempy.flatten('F')
    

    # (50,50)의 temp 다시 초기화합니다.
    tmp = wx[:,-1]
    tempx = np.concatenate((tmp[:,None], np.zeros((r,c-1))), axis=1)
    tmp = wy[-1,:]
    tempy = np.concatenate((tmp[None,:], np.zeros((r-1,c))), axis=0)


    dxd1 = -lamda * tempx.flatten('F')
    dyd1 = -lamda * tempy.flatten('F')
    
    wx[:,-1] = 0
    wy[-1,:] = 0
    dxd2 = -lamda * wx.flatten('F')
    dyd2 = -lamda * wy.flatten('F')

    # 각각 (2, 2500)의 shape이 됩니다.
    # print(np.concatenate((dxd1[:,None], dxd2[:,None]), axis=1).T.shape)
    # print(np.concatenate((dyd1[None,:], dyd2[None,:]), axis=0).shape)
    
    # (2,2500)의 데이터에서 희소행렬을 -대각선의 성분?- 뽑아서 (50,50)으로 반환합니다.
    Ax = scipy.sparse.spdiags(np.concatenate((dxd1[:,None], dxd2[:,None]), axis=1).T, np.array([-k+r,-r]), k, k)
    Ay = scipy.sparse.spdiags(np.concatenate((dyd1[None,:], dyd2[None,:]), axis=0), np.array([-r+1,-1]), k, k)
    
    # D -> (2500,) 의 벡터
    D = 1 - ( dx + dy + dxa + dya)
    # A -> (2500, 25000) 의 넘파이
    A = ((Ax+Ay) + (Ax+Ay).conj().T + scipy.sparse.spdiags(D, 0, k, k)).T
    
    tin = IN[:,:]

    tout = scipy.sparse.linalg.spsolve(A, tin.flatten('F'))

    # (50,50) shape 넘파이 반환
    OUT = np.reshape(tout, (r, c), order='F')
    return OUT
    

def tsmooth(img, lamda=0.01, sigma=3.0, sharpness=0.001):
    # 화질 개선을 위해 정규화를 진행합니다. 밑의 행 진행해도 I의 shape: (50,50)
    I = cv2.normalize(img, None, 0.0, 1.0, cv2.NORM_MINMAX)

    x = np.copy(I)

    # wx, wy 각각 (50,50)의 shape
    wx, wy = computeTextureWeights(x, sigma, sharpness)

    # S도 (50,50) np array
    S = solveLinearEquation(I, wx, wy, lamda)
    return S

def rgb2gm(I):
    if (I.shape[2] == 3):
        I = cv2.normalize(I.astype('float64'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        I = np.abs((I[:,:,0]*I[:,:,1]*I[:,:,2]))**(1/3)

    return I

def applyK(I, k, a=-0.3293, b=1.1258):
    f = lambda x: np.exp((1-x**a)*b)
    beta = f(k)
    gamma = k**a
    J = (I**gamma)*beta
    return J

def entropy(X):
    tmp = X * 255
    tmp[tmp > 255] = 255
    tmp[tmp<0] = 0
    tmp = tmp.astype(np.uint8)
    _, counts = np.unique(tmp, return_counts=True)
    pk = np.asarray(counts)
    pk = 1.0*pk / np.sum(pk, axis=0)
    S = -np.sum(pk * np.log2(pk), axis=0)
    return S

def maxEntropyEnhance(I, isBad, a=-0.3293, b=1.1258):
    # Esatimate k
    tmp = cv2.resize(I, (50,50), interpolation=cv2.INTER_AREA)
    tmp[tmp<0] = 0
    tmp = tmp.real
    Y = rgb2gm(tmp)
    
    isBad = isBad * 1
    isBad = cv2.resize(isBad.astype('float64'), (50,50), interpolation=cv2.INTER_CUBIC)
    isBad = cv2.resize(isBad, (50,50), interpolation=cv2.INTER_CUBIC)
    isBad[isBad<0.5] = 0
    isBad[isBad>=0.5] = 1
    Y = Y[isBad==1]
    
    if Y.size == 0:
       J = I
       return J
    
    f = lambda k: -entropy(applyK(Y, k))
    opt_k = scipy.optimize.fminbound(f, 1, 7)
    
    # Apply k
    J = applyK(I, opt_k, a, b) - 0.01
    return J
    

def Ying_2017_CAIP(img, mu=0.5, a=-0.3293, b=1.1258):
    lamda = 0.5
    sigma = 5
    # 현재 img는 imageio로 읽은 파일입니다. cv2로 읽기 위해서 astype('float64')를 붙힙니다.
    I = cv2.normalize(img.astype('float64'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    # 현재 I의 shape : (크기,크기,3 -> 보통 rgb 여서)
    

    # Weight matrix estimation
    # 각 행마다의 최대값만 남기기(각 픽셀들의 rgb중 최대값만 남기기) -> (크기,크기)만 남게됩니다. -> 채널이 없어졌음!!!
    t_b = np.max(I, axis=2)
    
    # 현재 t_b는 넘파이 데이터입니다. cv2로 읽기 위해서 astype('float64')를 붙힙니다.
    # 우선 t_b를 50,50으로 16개의 픽셀을 이용한 3차회선 보간법을 통해서 재지정합니다.
    # 이후 위의 tsmooth함수를 호출합니다.
    # 해당 배열에 영상축소에 유용한 inter_area 보간을 이용합니다.
    t_our = cv2.resize(tsmooth(cv2.resize(t_b.astype('float64'), (50,50), interpolation=cv2.INTER_CUBIC), lamda, sigma), (t_b.shape[1], t_b.shape[0]), interpolation=cv2.INTER_AREA)
    # t_our 보면 그냥 보라색 화면...
    plt.imshow(t_our)
    plt.show()

    # Apply camera model with k(exposure ratio)
    # 0.5 미만이면 True 저장하는 아니면 False 저장하는 (50,50)의 넘파이
    isBad = t_our < 0.5
    
    # J의 shape은 -> 원본의 크기, 원본의 크기, 3 입니다.
    # 원본보다 많이 밝은 친구 만들어내는 과정!
    J = maxEntropyEnhance(I, isBad)
    plt.imshow(J)
    plt.show()

    # W: Weight Matrix
    
    t = np.zeros((t_our.shape[0], t_our.shape[1], I.shape[2]))
    for i in range(I.shape[2]):
        t[:,:,i] = t_our
    
    # t를 0.5만큼 제곱해준것이 W
    # W도 원본과 같은 shape 입니다.
    W = t**mu
    
    plt.imshow(W)
    plt.show()
    plt.imshow(1-W)
    plt.show()

    # 모두 shape이 원본과 같은 크기여서 요소의 연산이 가능합니다.
    I2 = I*W
    J2 = J*(1-W)
    result = I2 + J2

    # RGB 값에 맞춰서 255로 맞추기
    result = result * 255
    result[result > 255] = 255
    result[result<0] = 0
    return result.astype(np.uint8)

def main():
    img_name = "project/05.jpg"
    img = imageio.imread(img_name)
    plt.imshow(img)
    plt.show()
    result = Ying_2017_CAIP(img)
    plt.imshow(result)
    plt.show()

if __name__ == '__main__':
    main()