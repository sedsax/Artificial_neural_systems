import random

def relu(x):
    return max(0, x)

def relu_deriv(x):
    return 1 if x > 0 else 0

def conv2d(image, kernel):
    h, w = 28, 28
    kh, kw = 3, 3
    output = [[0]*(w-2) for _ in range(h-2)]
    for i in range(h-2):
        for j in range(w-2):
            s = 0
            for ki in range(kh):
                for kj in range(kw):
                    s += image[i+ki][j+kj] * kernel[ki][kj]
            output[i][j] = relu(s)
    return output

def maxpool2x2(feature):
    h, w = len(feature), len(feature[0])
    pooled = [[0]*(w//2) for _ in range(h//2)]
    for i in range(0, h, 2):
        for j in range(0, w, 2):
            pooled[i//2][j//2] = max(
                feature[i][j],
                feature[i+1][j],
                feature[i][j+1],
                feature[i+1][j+1]
            )
    return pooled

def flatten(feature):
    return [x for row in feature for x in row]

def extract_features(img_flat, kernel):
    img = [img_flat[i*28:(i+1)*28] for i in range(28)]
    conv = conv2d(img, kernel)
    pool = maxpool2x2(conv)
    return flatten(pool)
