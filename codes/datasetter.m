%clc; clear;

function [face_data, mean_vector, labels] = datasetter()
% this script reads out all files in a directory into images that are
% stored lexicographically one large row, row by row.
cd '../imgs';

% get a list of all the files in the current directory
[~, flist] = fileattrib('*');

% Read in jpg and pgm files only; Assume that all files are the same size.
% for the yale face repository all files are gifs
% find number of files
nfiles = max(size(flist));

% Initialize structure

m = 100; n = 100; k = 1;

face_data = zeros(nfiles,(m/k)*(n/k));
labels = zeros(nfiles, 1);
mean_face = zeros(m/k, n/k);
count = 0;

for i = 1:nfiles
    fi = flist(i).Name;
    x = double(imread(fi,'bmp'));
    x = imresize(x, [m/k,n/k]);
    mean_face = mean_face + x;
    count = count+1;
    face_data(i,:) = reshape(x,1,size(x,1)*size(x,2));
    labels(i) = ceil(i/11);
end

cd '../codes'  % get out of images data directory

face_data = face_data';


% figure(2); title('Different faces');
% subplot(2,2,1); imshow(mat2gray(reshape(face_data(:,2),243,320)));
% subplot(2,2,2); imshow(mat2gray(reshape(face_data(:,13),243,320)));
% subplot(2,2,3); imshow(mat2gray(reshape(face_data(:,24),243,320)));
% subplot(2,2,4); imshow(mat2gray(reshape(face_data(:,35),243,320)));

% save imagesall face_data;

%Find image mean
mean_face = mean_face/count;
mean_vector = reshape(mean_face, [(m/k)*(n/k),1]);
% figure(3); imshow((mat2gray(mean_face))); title('Image Mean');

% subtract mean from all images
%data_mean = face_data - reshape(mean_face,[(m/k)*(n/k),1]);    % data_mean = [d, n]
end
