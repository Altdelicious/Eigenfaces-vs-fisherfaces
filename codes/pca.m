%clc; clear;

function [eigvectors, error_rate, error_rate_3, mean_img] = pca(A, y, k_range)

A_orig = A;
true_labels = y;

error_rate = zeros(1,length(k_range));
error_rate_3 = zeros(1,length(k_range));

for j=1:length(k_range)
    %waitbar(j/length(k_range));
    k = k_range(j);
    num_incorrect = 0;
    num_incorrect_3 = 0;
    for i=1:165
        A = A_orig; y = true_labels;
        test_image = A(:,i);
        A(:,i) = [];
        y(i,:) = [];
        training_set = A;
        
        mean_img = mean(training_set, 2);
        for l = 1:size(training_set,2)
            training_set(:,l) = training_set(:,l)-mean_img;
        end
        %figure; imshow(reshape(mean_img,[112,92]),[])
        
        C = training_set'*training_set;
        
        [V,D] = eig(C);
        eigVals = diag(D);
        
        eigVals = eigVals(end:-1:1);
        V = V(:,end:-1:1);
        %plot(eigVals);xlabel('indexes'); ylabel('eigen values'); title('eigen values')

        for l = 1:size(V,2)
            V(:,l) = V(:,l)./sqrt(eigVals(l));
        end
        
        eigvectors = A*V;
        eig_vectors = eigvectors(:,1:k);
        eigvectors_3 = eigvectors(:,4:k+3);
        
        coeff_matrix = zeros(k, size(training_set, 2));
        coeff_matrix_3 = zeros(k, size(training_set, 2));
        for m=1:size(coeff_matrix,2)
            wt = (training_set(:,m))'*eig_vectors; % weighting
            coeff_matrix(:,m) = wt';
            wt_3 = (training_set(:,m))'*eigvectors_3; % weighting
            coeff_matrix_3(:,m) = wt_3';
        end
        
        test_coeffs = ((test_image - mean_img)'*eig_vectors)';
        test_coeffs_3 = ((test_image - mean_img)'*eigvectors_3)';
        
        dists = sum((coeff_matrix - test_coeffs).^2);
        [~,idx] = min(dists(:));
        predicted_label = y(idx);
        % fprintf('%d ', i); fprintf('%d ', predicted_label); fprintf('%d\n', true_labels(i));
        if predicted_label ~= true_labels(i)
            num_incorrect = num_incorrect + 1;
        end
        
        dists_3 = sum((coeff_matrix_3 - test_coeffs_3).^2);
        [~,idx_3] = min(dists_3(:));
        predicted_label_3 = y(idx_3);
        % fprintf('%d ', i); fprintf('%d ', predicted_label); fprintf('%d\n', true_labels(i));
        if predicted_label_3 ~= true_labels(i)
            num_incorrect_3 = num_incorrect_3 + 1;
        end
    end
    error = (num_incorrect/165)*100;
    error_rate(j) = error;
    
    error_3 = (num_incorrect_3/165)*100;
    error_rate_3(j) = error_3;
end

end