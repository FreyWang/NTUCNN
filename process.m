function result = process(train,w,h)
FFT = zeros(w,h,size(train, 3), size(train, 4));
n = 0;
for i = 1:size(train,4)
    for j = 1:size(train,3)
        data = train(:,:,j,i)';
        FFT(:,:,j,i) = interp_fft(data, w, h);
        
        fprintf(repmat('\b',1,n));  
        msg = sprintf('Processed %d/%d,%d/3',i,size(train, 4),j);
        fprintf(msg);
        n = numel(msg);
    end
end
result = FFT;
end
