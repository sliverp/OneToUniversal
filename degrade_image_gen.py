import math
import imagecorruptions as ic
import numpy as np
from tqdm import tqdm
import skimage as sk
import cv2
import numpy as np
SCALE = 0.2


def gaussian_noise(x, severity=0.5):
    # severity0-1分布
    # 0表示没有，1表示最严重
    severity = severity * SCALE
    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=severity), 0, 1) * 255

def shift(image, dx, dy):
    if(dx < 0):
        shifted = np.roll(image, shift=image.shape[1]+dx, axis=1)
        shifted[:,dx:] = shifted[:,dx-1:dx]
    elif(dx > 0):
        shifted = np.roll(image, shift=dx, axis=1)
        shifted[:,:dx] = shifted[:,dx:dx+1]
    else:
        shifted = image

    if(dy < 0):
        shifted = np.roll(shifted, shift=image.shape[0]+dy, axis=0)
        shifted[dy:,:] = shifted[dy-1:dy,:]
    elif(dy > 0):
        shifted = np.roll(shifted, shift=dy, axis=0)
        shifted[:dy,:] = shifted[dy:dy+1,:]
    return shifted

def _motion_blur(x, radius, sigma, angle):

    def gauss_function(x, mean, sigma):
        a = np.exp(- x**2 / (2 * (sigma**2)))
        return a / (np.sqrt(2 * np.pi) * sigma)
    
    def getOptimalKernelWidth1D(radius, sigma):
        return radius * 2 + 1
    
    def getMotionBlurKernel(width, sigma):
        k = gauss_function(np.arange(width), 0, sigma)
        Z = np.sum(k)
        return k/Z

    width = getOptimalKernelWidth1D(radius, sigma)
    kernel = getMotionBlurKernel(width, sigma)
    point = (width * np.sin(np.deg2rad(angle)), width * np.cos(np.deg2rad(angle)))
    hypot = math.hypot(point[0], point[1])
    blurred = np.zeros_like(x, dtype=np.float32)
    for i in range(width):
        dy = -math.ceil(((i*point[0]) / hypot) - 0.5)
        dx = -math.ceil(((i*point[1]) / hypot) - 0.5)
        if (np.abs(dy) >= x.shape[0] or np.abs(dx) >= x.shape[1]):
            # simulated motion exceeded image borders
            break
        shifted = shift(x, dx, dy)
        blurred = blurred + kernel[i] * shifted
    return blurred

    
def motion_blur(x, severity=0):
    # severity0-1分布
    # 0表示没有，1表示最严重
    severity = severity * SCALE * 2
    shape = np.array(x).shape
    # c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]
    if severity == 0:
        return x
    c = [int(50*severity), int(30*severity)]
    x = np.array(x)
    if c[0] == 0:
        c[0] = 1
    angle = np.random.uniform(-45, 45)
    x = _motion_blur(x, radius=c[0], sigma=c[1], angle=angle)

    if len(x.shape) < 3 or x.shape[2] < 3:
        gray = np.clip(np.array(x).transpose((0, 1)), 0, 255)
        if len(shape) >= 3 or shape[2] >=3:
            return np.stack([gray, gray, gray], axis=2)
        else:
            return gray
    else:
        return np.clip(x, 0, 255)
    

def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

def plasma_fractal(mapsize=256, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble,
                                                      array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize,
                 stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize,
        0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()

def haze(x, severity=1):
    # c = [(1.5, 2), (2., 2), (2.5, 1.7), (2.5, 1.5), (3., 1.4)][severity - 1]
    # severity0-1分布
    # 0表示没有，1表示最严重
    # c[1]印象雾块大小
    severity = severity * SCALE
    if severity == 0:
        return x
    # c = (1,1.8-0.7*severity)
    c = (1+3*severity,1.8-0.7*severity)

    shape = np.array(x).shape
    max_side = np.max(shape)
    map_size = next_power_of_2(int(max_side))

    x = np.array(x) / 255.
    max_val = x.max()

    x_shape = np.array(x).shape
    if len(x_shape) < 3 or x_shape[2] < 3:
        x += c[0] * plasma_fractal(mapsize=map_size, wibbledecay=c[1])[
                    :shape[0], :shape[1]]
    else:
        x += c[0] * \
             plasma_fractal(mapsize=map_size, wibbledecay=c[1])[:shape[0],
             :shape[1]][..., np.newaxis]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255


def low_light(x, severity=1):
    # severity0-1分布
    # 0表示没有，1表示最严重
    severity = severity * SCALE
    c = -severity * 0.65

    x = np.array(x) / 255.

    if len(x.shape) < 3 or x.shape[2] < 3:
        x = np.clip(x + c, 0, 1)
    else:
        x = sk.color.rgb2hsv(x)
        x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
        x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255




 
 
def get_noise(img, severity):
    '''
    #生成噪声图像
    >>> 输入： img图像
        value= 大小控制雨滴的多少 
    >>> 返回图像大小的模糊噪声图像
    '''
    value = 10+500 * severity
    noise = np.random.uniform(0, 256, img.shape[0:2])
    # 控制噪声水平，取浮点数，只保留最大的一部分作为噪声
    v = value * 0.01
    noise[np.where(noise < (256 - v))] = 0
 
    # 噪声做初次模糊
    k = np.array([[0, 0.1, 0],
                  [0.1, 8, 0.1],
                  [0, 0.1, 0]])
 
    noise = cv2.filter2D(noise, -1, k)
    return noise

def rain_blur(image, length=80, angle=0,w=11,severity=0.8):
    #这里由于对角阵自带45度的倾斜，逆时针为正，所以加了-45度的误差，保证开始为正
    trans = cv2.getRotationMatrix2D((length/2, length/2), angle-45, 1-length/100.0)  
    dig = np.diag(np.ones(length))   #生成对焦矩阵
    k = cv2.warpAffine(dig, trans, (length, length))  #生成模糊核
    k = cv2.GaussianBlur(k,(w,w),0)    #高斯模糊这个旋转后的对角核，使得雨有宽度
    
    #k = k / length                         #是否归一化
    
    blurred = cv2.filter2D(get_noise(image,severity=severity), -1, k)    #用刚刚得到的旋转后的核，进行滤波
    
    #转换到0-255区间
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred

def rainy(img,severity = 0.8):
    
    #输入雨滴噪声和图像
    #beta = 0.8   #results weight
    #显示下雨效果
    
    #expand dimensin
    #将二维雨噪声扩张为三维单通道
    #并与图像合成在一起形成带有alpha通道的4通道图像
    severity = severity * SCALE
    rain = rain_blur(img,severity=severity)
    rain = np.expand_dims(rain,2)
    rain_effect = np.concatenate((img,rain),axis=2)  #add alpha channel
 
    rain_result = img.copy()    #拷贝一个掩膜
    rain = np.array(rain,dtype=np.float32)     #数据类型变为浮点数，后面要叠加，防止数组越界要用32位
    rain_result[:,:,0]= rain_result[:,:,0] * (255-rain[:,:,0])/255.0 + 0.7*rain[:,:,0]
    rain_result[:,:,1] = rain_result[:,:,1] * (255-rain[:,:,0])/255 + 0.7*rain[:,:,0] 
    rain_result[:,:,2] = rain_result[:,:,2] * (255-rain[:,:,0])/255 + 0.7*rain[:,:,0]
    #对每个通道先保留雨滴噪声图对应的黑色（透明）部分，再叠加白色的雨滴噪声部分（有比例因子）
    return rain_result

if __name__ == '__main__':
    img = ic.cv2.imread("1.png")
    for i in tqdm(range(0,100, 10)):
        corrupted_image = rainy(img, i/100)
        ic.cv2.imwrite(f"./degrade/rain/{i}.jpg", corrupted_image)
