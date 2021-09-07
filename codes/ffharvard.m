function [error_rate] = ffharvard(X, X_test, y, y_test,k)

% number of samples
n = size(X, 1);

% number of classes
classes = unique(y);
c = length(classes);

if(nargin < 3)
    k = c-1;
end

k = min(k,(c-1));
%disp(k)

training_set = X;

% get (N-c) principal components
[Wpca, mu] = ldapca(training_set, (n-c));             % Wpca = dx(n-c)
Y = (training_set-repmat(mu, n, 1))*Wpca;             % Y = nx(n-c)
[Wlda, ~] = lda(Y, y, k);
W = Wpca*Wlda;                                        % W = dx(n-c)*(n-c)x(c-1)=dx(c-1)
% disp(size(W))

num_incorrect = 0;

coeff_matrix = training_set*W;                        % nx(c-1)

for j=1:size(X_test,1)
    test_coeffs = (X_test(j,:) - mu)*W;               % 1x(c-1);
    % calculate squared dist. in projection domain
    dists = sum((coeff_matrix-test_coeffs).^2,2);
    [~,idx] = min(dists(:));
    predicted_label = y(idx);
    %fprintf('%d ', predicted_label); fprintf('%d\n', y_test(j));
    if predicted_label ~= y_test(j)
        num_incorrect = num_incorrect + 1;
    end
end

error_rate = (num_incorrect/size(X_test,1))*100;

end