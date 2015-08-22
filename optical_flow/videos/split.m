clear
clc

obj = VideoReader('outfile.avi');
video = obj.read();

for k=1:200
    im1=video(:,:,:,k);
    im2=video(:,:,:,k);
    imshow(im);
end