function [W, mu] = lda(X,y,k)
  
  [~,d] = size(X);
  
  classes = unique(y);
  c = length(classes);
  
  % allocate scatter matrices
  Sw = zeros(d,d);
  Sb = zeros(d,d);
  
  % total mean
  mu = mean(X);
  
  % calculate scatter matrices
  for i = 1:c
    Xi = X(y == classes(i),:); % samples for current class     %'find' was used originally
    n = size(Xi,1);
    mu_i = mean(Xi); % mean vector for current class
    Xi = Xi - repmat(mu_i, n, 1);
    Sw = Sw + Xi'*Xi;
    Sb = Sb + n * (mu_i - mu)'*(mu_i - mu);
  end
  
  % solve general eigenvalue problem
  [W, D] = eig(Sb, Sw);
  
  % sort eigenvectors
  [~, i] = sort(diag(D), 'descend');
  W = W(:,i);
  
  % keep at most (c-1) eigenvectors
  W = W(:,1:k);                     % (n-c)x(c-1)
end