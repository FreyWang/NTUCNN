function demo = readFromMatlab()
   close all
   norm_train = h5read('norm_train.mat', '/norm_train'); % (25, 32, 3, 39889)
   num = round(rand(1)*size(norm_train,4))
   data = norm_train(:,:,1,num);
   FFT = interp_fft(data,224,224);
   imshow(FFT,[])
   figure
   imshow(FFT(74:148,74:148),[])
end