import cv2
import numpy as np

def haarDwt1D(signal):
  half_length = int(len(signal)/2)
  output_signal = np.empty_like(signal)[:half_length]
  for i in range(half_length):
    output_signal[i] = signal[2*i] * 0.5 - 0.5 * signal[2*i+1]
    if output_signal[i] <0:
      output_signal[i] = 0
  return output_signal 

def haarDwt2D(img, dim):
  # get each row
  if dim == 0: 
    wavelet_res = np.empty_like(img)[:, :int(img.shape[1]/2)]
    for i in range(img.shape[0]):
      row = img[i, :]
      wavelet_res[i,:] = haarDwt1D(row)
  # get each column
  else:
    wavelet_res = np.empty_like(img)[:int(img.shape[0]/2), :]
    for i in range(img.shape[1]):
      col = img[:, i]
      wavelet_res[:, i] = haarDwt1D(col)
  return wavelet_res

def haarDwt3D(colorImg, dim):
  channel_count = colorImg.shape[2]
  if dim == 0:
    half_col_count = int(colorImg.shape[1]/2)
    wavelet_res = np.empty_like(colorImg)[:, :half_col_count, :]
  else:
    half_row_count = int(colorImg.shape[0]/2)
    wavelet_res = np.empty_like(colorImg)[:half_row_count, :, :]
  for channel in range(channel_count):
    wavelet_res[:, :, channel] = haarDwt2D(colorImg[:, :, channel], dim)
  return wavelet_res

def create_wavelet(img_src):
  img_trans_0 = haarDwt3D(img_src, dim=0)
  img_trans_1 = haarDwt3D(img_src, dim=1)
  return [cv2.resize(img_trans_0, img_src.shape[:2]), cv2.resize(img_trans_1, img_src.shape[:2])]

def create_input(img_path, shape):
  img = cv2.imread(img_path)
  img = cv2.resize(img, shape)
  [wavelet1, wavelet2] = create_wavelet(img)
  return [wavelet1, wavelet2]

if __name__ == "__main__":
  file_name = 'positiveImages/z5404705670526_32d0e89a0b107041f0a847e41915d7ef.jpg'
  img = cv2.imread(file_name)
  imgs_wavelet = create_wavelet(img)
  print(np.max(imgs_wavelet[0]))
  cv2.imwrite('img0.jpg', imgs_wavelet[0])
  cv2.imwrite('img1.jpg', imgs_wavelet[1])