function FFT2()
w = 224
h = 224;
norm_train = h5read('norm_train.mat', '/norm_train'); % (25, 32, 3, 39889)
norm_test = h5read('norm_test.mat', '/norm_test');
FFT2_train = process(norm_train, w, h);
size(FFT2_train)
FFT2_test = process(norm_test, w, h);
size(FFT2_test)
save('FFT2_train.mat','FFT2_train','-v7.3')
save('FFT2_test.mat','FFT2_test','-v7.3')
end 
