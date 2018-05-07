function imgs_norm = ContrastNorm(imgs)
N = length(imgs)
cutoff = 0.01;
for i = 1:N    
    i
   img = imgs{i};
   img = imadjust(img,[cutoff 1-cutoff]);
   img = histeq(img);
   for j=1:3
       img(:,:,j) =adapthisteq(img(:,:,j));
   end
   imgs{i} = img;
end
imgs_norm = imgs;
end

