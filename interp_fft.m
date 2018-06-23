function result = interp_fft(data, w, h)
% resize 2D matrix data to (w,h)
image = imresize(data, [w, h]);
F = fft2(double(image));
F = fftshift(F);
F = abs(F);
F = log(F+1);
%imshow(F,[])
result = F;
end