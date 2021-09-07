%clc; clear;

function [face_data, mean_vector, labels] = dsharvard(num)

str1 = '../Subsets/subset';
str2 = num2str(num);
str = strcat(str1,str2);
cd(str);

% get a list of all the files in the current directory
[~, flist] = fileattrib('*');

nfiles = max(size(flist));

m = 192; n = 168; k = 1;

face_data = zeros(nfiles,(m/k)*(n/k));
labels = zeros(nfiles, 1);
mean_face = zeros(m/k, n/k);
count = 0;

num_per_person = [6,9,13,17,19];
num_per = num_per_person(num);

for i = 1:nfiles
    fi = flist(i).Name;
    x = double(imread(fi,'pgm'));
    x = imresize(x, [m/k,n/k]);
    mean_face = mean_face + x;
    count = count+1;
    face_data(i,:) = reshape(x,1,size(x,1)*size(x,2));
    labels(i) = ceil(i/num_per);
end

cd '../../codes'  % get in codes directory

face_data = face_data';

mean_face = mean_face/count;
mean_vector = reshape(mean_face, [(m/k)*(n/k),1]);

end