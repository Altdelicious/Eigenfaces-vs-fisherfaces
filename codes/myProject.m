clc; clear;

tic;

%% PCA on harvard database

[X1, ~, y_X1] = dsharvard(1);
[X2, ~, y_X2] = dsharvard(2);
[X3, ~, y_X3] = dsharvard(3);
[X4, ~, y_X4] = dsharvard(4);
[X5, ~, y_X5] = dsharvard(5);

% figure(1); title('Different faces');
% subplot(2,2,1); imshow(mat2gray(reshape(mean_X5,192,168)));
% subplot(2,2,2); imshow(mat2gray(reshape(X5(:,10),192,168)));
% subplot(2,2,3); imshow(mat2gray(reshape(X5(:,11),192,168)));
% subplot(2,2,4); imshow(mat2gray(reshape(X5(:,12),192,168)));

a = 0;
waitbar(a/8);
%% Part-1 : Extrapolation

k_range = [4,10];
[error2_extra3, error2_extra] = mypca(X1, X2, y_X1, y_X2, k_range);
[error3_extra3, error3_extra] = mypca(X1, X3, y_X1, y_X3, k_range);

fprintf('PCA extrapolation on Harvard done!\n')
a = 1;
waitbar(a/8);

%% Part-2 : Interpolation

X = [X1 X4];
y = [y_X1; y_X4];

[error2_inter, error2_inter3] = mypca(X, X2, y, y_X2, k_range);
[error3_inter, error3_inter3] = mypca(X, X3, y, y_X3, k_range);
[error5_inter, error5_inter3] = mypca(X, X5, y, y_X5, k_range);

% [error4_inter, error4_inter3] = mypca(X, X3, y, y_X4, k_range);

fprintf('PCA interpolation on Harvard done!\n')

a = 2;
waitbar(a/8);

%% LDA on harvard database - extrapolation

k2 = 5;

[X1, mean_X1, y_X1] = dsharvard(1);
[X2, mean_X2, y_X2] = dsharvard(2);
[X3, mean_X3, y_X3] = dsharvard(3);
[X4, mean_X4, y_X4] = dsharvard(4);
[X5, mean_X5, y_X5] = dsharvard(5);

errorlda2_extra = ffharvard(X1',X2',y_X1,y_X2,k2);
errorlda3_extra = ffharvard(X1',X3',y_X1,y_X3,k2);

fprintf('LDA extrapolation on Harvard done!\n')

a = 3;
waitbar(a/8);

%% LDA on harvard database - interpolation

X = [X1 X4];
y = [y_X1; y_X4];

errorlda2_inter = ffharvard(X',X2',y,y_X2,k2);
errorlda3_inter = ffharvard(X',X3',y,y_X3,k2);
errorlda5_inter = ffharvard(X',X3',y,y_X3,k2);
%errorlda4_inter = ffharvard(X',X4',y,y_X4,k2);

fprintf('LDA interpolation on Harvard done!\n')

a = 4;
waitbar(a/8);

%% Plotter - extrapolation

x = [1,2,3];
y1 = [0, error2_extra(2), error3_extra(2)];
y2 = [0, error2_extra3(2), error3_extra3(2)];
y3 = [0, errorlda2_extra, errorlda3_extra];

figure(1);
plot(x,y1,x,y2,'--',x,y3,'b:','LineWidth',4);
title('Extrapolation')
xlabel('Test subset no.')
ylabel('Error rate')
legend('Eigenface','Eigenface w/o first 3 components','fisherface')
xticks([1,2,3])
xticklabels({'sub-1','sub-2','sub-3'})

%% Plotter - interpolation

x = [2,3,5];
y1 = [error2_inter(2), error3_inter(2), error5_inter(2)];
y2 = [error2_inter3(2), error3_inter3(2), error5_inter3(2)];
y3 = [errorlda2_inter, errorlda3_inter, errorlda5_inter];

figure(2);
plot(x,y1,x,y2,'--',x,y3,'b:','LineWidth',4);
title('Interpolation')
xlabel('Test subset no.')
ylabel('Error rate')
legend('Eigenface','Eigenface w/o first 3 components','fisherface')
xticks([2,3,5])
xticklabels({'sub-2','sub-3','sub-5'})



%% PCA on yale dataset
%
[X, ~, y] = datasetter();  % [d,n]

% figure(1); title('Different faces');
% subplot(2,2,1); imshow(mat2gray(reshape(X_mean,100,100)));

k_range = [10,30];

[Wpca, error_rate, error_rate_3, mean_img] = pca(X, y, k_range);

a = 5;
waitbar(a/8);

fprintf('PCA on cropped yale done!\n')

[X, ~, y] = datasetterfull();  % [d,n]

figure(3);
subplot(3,2,1); imshow(mat2gray(reshape(Wpca(:,1),100,100)));
subplot(3,2,2); imshow(mat2gray(reshape(Wpca(:,2),100,100)));
subplot(3,2,3); imshow(mat2gray(reshape(Wpca(:,3),100,100)));
subplot(3,2,4); imshow(mat2gray(reshape(Wpca(:,45),100,100)));
subplot(3,2,5); imshow(mat2gray(reshape(Wpca(:,60),100,100)));
subplot(3,2,6); imshow(mat2gray(reshape(Wpca(:,75),100,100)));

[Wpca, error_ratefull, error_rate_3full, mean_imgfull] = pca(X, y, k_range);

a = 6;
waitbar(a/8);

fprintf('PCA on full-face yale done!\n')

%% LDA on yale dataset

[X, X_mean, y] = datasetter();  % [d,n], [d,1]
% 
X = X';                          % [n,d]
% 
k1 = 15;

[W_yale, error_yale] = fisherfaces(X,y,k1);

a = 7;
waitbar(a/8);

fprintf('LDA on cropped yale done!\n')
% 
figure(4); title('Different fisher-faces');
subplot(3,2,1); imshow(mat2gray(reshape(W_yale(:,1),100,100)));
subplot(3,2,2); imshow(mat2gray(reshape(W_yale(:,4),100,100)));
subplot(3,2,3); imshow(mat2gray(reshape(W_yale(:,7),100,100)));
subplot(3,2,4); imshow(mat2gray(reshape(W_yale(:,9),100,100)));
subplot(3,2,5); imshow(mat2gray(reshape(W_yale(:,11),100,100)));
subplot(3,2,6); imshow(mat2gray(reshape(W_yale(:,13),100,100)));


[X, ~, y] = datasetterfull();  % [d,n]
X = X';                         % [n,d]
[Wfull, errorlda_full] = fisherfaces(X,y,k1);

a = 8;
waitbar(a/8);

fprintf('LDA on full-face yale done!\n')


%% Plotter

x = k_range;
y1 = error_rate;
y2 = error_rate_3;
y3 = repmat(error_yale,1,length(k_range));

figure(5);
plot(x,y1,x,y2,'--',x,y3,'b:','LineWidth',4);
title('Facial Recognition')
xlabel('Number of principal components')
ylabel('Error rate')
legend('Eigenface','Eigenface w/o first 3 components','fisherface - 0.6%')


toc;
