%clc; clear;

%[A, A_test, training_labels, test_labels] = myorlDataset();
function [error_eig, error_eig3] = mypca(A, A_test, training_labels, test_labels, k_range)
    
    mean_img = mean(A, 2);
    
    for i = 1:size(A,2)
        A(:,i) = A(:,i)-mean_img;
    end
    %figure; imshow(reshape(mean_img,[112,92]),[])
    
    C = A'*A;
    
    [V,D] = eig(C);
    eigVals = diag(D);

    eigVals = eigVals(end:-1:1);
    V = V(:,end:-1:1);
    %plot(eigVals);xlabel('indexes'); ylabel('eigen values'); title('eigen values')
    
    for i = 1:size(V,2)
        V(:,i) = V(:,i)./sqrt(eigVals(i));
    end
    
    eigFaces = A*V;
    
    % figure; imshow(reshape(eigFaces(:,10),[112,92]),[])
    % title('the first eigenface')
    
    error_eig = zeros(size(k_range,1), 1);
    error_eig3 = zeros(size(k_range,1), 1);
    %x = waitbar(0, 'Progress bar:');
    
    for a=1:size(k_range,2)
        %waitbar(a/size(k_range, 2));
        k = k_range(a);
        eigFaces_sub = eigFaces(:,1:k);
        eigFaces_sub3 = eigFaces(:,4:k+3);
        training_set = zeros(k, size(A, 2));
        training_set3 = zeros(k, size(A, 2));
        for i= 1:size(A, 2)
            wt=A(:,i)'*eigFaces_sub; % weighting
            training_set(:,i) = wt';
            wt=A(:,i)'*eigFaces_sub3; % weighting
            training_set3(:,i) = wt';
        end
        
        score = 0;
        score3 = 0;
        
        for j=1:size(A_test,2)
            test_image = A_test(:,j) - mean_img;
            test_wt = test_image'*eigFaces_sub;
            test_wt3 = test_image'*eigFaces_sub3;
            % calculate squared dist. in projection domain
            dists = sum((training_set-test_wt').^2);
            [~,idx] = min(dists(:));
            predicted_label = training_labels(idx);
            score = score + (predicted_label ~= test_labels(j));
            
            dists3 = sum((training_set3-test_wt3').^2);
            [~,idx3] = min(dists3(:));
            predicted_label3 = training_labels(idx3);
            score3 = score3 + (predicted_label3 ~= test_labels(j));
        end
        error_eig(a) = (score/size(A_test,2))*100;
        error_eig3(a) = (score3/size(A_test,2))*100;
    end
    % 
    % plot(k_range, accuracy);xlabel('k'); ylabel('recognition-rate'); title('Recognition Rate')
    
    
    %close(x)
    %plot(k_range, accuracy);xlabel('k'); ylabel('recognition-rate'); title('Recognition Rate')
end