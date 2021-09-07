function [W, mu] = ldapca(X, k)         % y was used in parameters
  [n,d] = size(X);
  
  mu = mean(X);
  Xm = X - repmat(mu, size(X,1), 1);
  
  if(n>d)
    C = Xm'*Xm;
    [W,D] = eig(C);
    
    % sort eigenvalues and eigenvectors
    [~, i] = sort(diag(D), 'descend');
    W = W(:,i);
    % keep k components
    W = W(:,1:k);
  else
    C = Xm*Xm';
    %C = cov(Xm');
    [W,D] = eig(C);
    % multiply with data matrix
    W = Xm'*W;
    % normalize eigenvectors
    for i=1:n
      W(:,i) = W(:,i)/norm(W(:,i));
    end
    % sort eigenvalues and eigenvectors
    [~, i] = sort(diag(D), 'descend');
    W = W(:,i);
    % keep k components
    W = W(:,1:k);
  end
end