function [W, error_rate] = fisherfaces(X,y,k)

X_orig = X;                         % nxd
true_labels = y;

% number of samples
n = size(X, 1);

% number of classes
classes = unique(y);
c = length(classes);

if(nargin < 3)
    k = c-1;
end

k = min(k,(c-1));

for i=1:n
    X = X_orig; y = true_labels;
    test_image = X(i,:);
    X(i,:) = [];
    y(i,:) = [];
    training_set = X;
    ni = size(training_set,1);
    
    % get (N-c) principal components
    [Wpca, mu] = ldapca(training_set, (ni-c));                        % Wpca = dx(ni-c)
    Y = (training_set-repmat(mu, ni, 1))*Wpca;                   % Y = nix(ni-c)
    [Wlda, ~] = lda(Y, y, k);
    W = Wpca*Wlda;                                        % W = dx(ni-c)*(ni-c)x(c-1)=dx(c-1)
    
    num_incorrect = 0;
    
    coeff_matrix = training_set*W;             % nix(c-1)
    
    test_coeffs = (test_image - mu)*W;    % 1x(c-1)
    
    dists = sum((coeff_matrix - test_coeffs).^2,2);
    [~,idx] = min(dists(:));
    predicted_label = y(idx);
    % fprintf('%d ', i); fprintf('%d ', predicted_label); fprintf('%d\n', true_labels(i));
    if predicted_label ~= true_labels(i)
        num_incorrect = num_incorrect + 1;
    end
    
end

error_rate = (num_incorrect/n)*100;

end