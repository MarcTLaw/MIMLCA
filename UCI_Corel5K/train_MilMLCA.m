function [ L, meanX, new_H, centroids, assigned_centroids] = train_MilMLCA( X, U, Y, instances_per_bag, center_descriptor )
%TRAIN_MILMLCA Summary of this function goes here
%   Detailed explanation goes here
% Input:
% - X is the dataset representation:
%     each of its rows represents a training instance
% - U is a matrix containing left singular vectors of X:
%     if X is full rank, U is obtained with the command [U,~,~] = svd(X,'econ');
% - Y is the bag assignment matrix:
%     it is a logical matrix of size m x k
%     where m is the number of bags and
%     k is the number of clusters  
% - instances_per_bag is a vector of size m
%     it indicates the number of instances per bag
% - center_descriptor is a logical value indicating 
%     it indicates whether the training dataset is mean centered
%
%
% Output:
% - L is the learned linear transformation
% - meanX is the mean vector of the selected training instances
% - new_H is the learned assignment matrix
% - centroids is the set of learned centroids
% - assigned_centroids is a boolean indicated whether cluster are nonempty


if nargin < 5
    center_descriptor = false;
end
one_instance_bags = (instances_per_bag == 1);
cumsum_instances = cumsum(instances_per_bag);


one_instance_docs = (sum(Y,2) == 1);

initial_docs = (one_instance_bags & one_instance_docs);
one_instance_bags_but_multiple_docs = one_instance_bags & ~one_instance_docs;
one_detected_doc_but_multiple_instances = ~one_instance_bags & one_instance_docs;
the_rest = ~one_instance_bags & ~one_instance_docs;
%tic;
initial_H = zeros(size(U,1),size(Y,2));
for i=find(initial_docs')
    initial_H(cumsum_instances(i),Y(i,:)) = 1;
end
initialize_H = initial_H;
for i=find(one_instance_bags_but_multiple_docs')
    initialize_H(cumsum_instances(i),Y(i,:)) = 1;
end
for i=find(one_detected_doc_but_multiple_instances')
    for j=0:(instances_per_bag(i)-1)
        initialize_H(cumsum_instances(i)-j,Y(i,:)) = 1;
    end
end
for i=find(the_rest')
    for j=0:(instances_per_bag(i)-1)
        initialize_H(cumsum_instances(i)-j,Y(i,:)) = 1;
    end
end
%toc;
initialize_H = sparse(initialize_H);
a = max(1,sum(initialize_H,1));
%assigned_centroids = (a >= 0.5);
%a(a <= 1) = 1;
%tic;
pinvY = (bsxfun(@rdivide,(initialize_H),a))';
%toc;
%clear initialize_H;
%disp('centroid computation');
%tic;
Z = pinvY * U;
%toc;
%clear initialize_H;
clear pinvY;
old_H = 0;
new_H = initial_H;

for iter=1:1000
    %tic;
    %iter
    if isequal(new_H,old_H)
        %disp('breaking at iteration');
        %iter
        break;
    end
    old_H = new_H;
    new_H = initial_H;
    %toc;
    for i=find(one_instance_bags_but_multiple_docs')
        e = find(Y(i,1:end));
        centroids = Z(e,:);
        %d = bsxfun(@minus, centroids, U(cumsum_instances(i),:));
        d = sum((bsxfun(@minus, centroids, U(cumsum_instances(i),:))).^2,2);
        %u = repmat(U(cumsum_instances(i),:), size(centroids,1),1);
        %d = sum((centroids - u).^2,2);
        [~,b] = min(d);
        new_H(cumsum_instances(i),e(b)) = 1;
    end
    %toc;
    for i=find(one_detected_doc_but_multiple_instances')
        e = find(Y(i,1:end));
        u = U((cumsum_instances(i) - (0:(instances_per_bag(i)-1))),:);
        %centroid = repmat(Z(e,:),size(u,1),1);
        %d = sum((centroid - u).^2,2);
        d = sum((bsxfun(@minus, u, Z(e,:))).^2,2);
        [~,b] = min(d);
        new_H(cumsum_instances(i)-(b-1),e) = 1;        
    end
    %toc;
    cpt_empty = 0;
    for i=find(the_rest')
        e = find(Y(i,1:end));
        centroids = Z(e,:);
        v = U((cumsum_instances(i) - (0:(instances_per_bag(i)-1))),:);
        s1 = size(v,1);
        s2 = length(e);
        if ~min(s1,s2)
            cpt_empty = cpt_empty + 1;
            continue;
        end
        if s1 >= s2
            d = zeros(s1, s2);
            for j=1:s1
                %u = repmat(v(j,:), size(centroids,1),1);
                %d(j,:) = sum((centroids - u).^2,2);
                d(j,:) = sum((bsxfun(@minus, centroids, v(j,:))).^2,2);
            end
            cpt = -1;
            for optimal_assignment = assignmentoptimal(d)';
                cpt = cpt + 1;
                if optimal_assignment
                    new_H(cumsum_instances(i)-(cpt),e(optimal_assignment)) = 1;
                end
            end
        else
            d = zeros(s2, s1);
            for j=1:s1
                %u = repmat(v(j,:), size(centroids,1),1);
                %d(j,:) = sum((centroids - u).^2,2);
                d(:,j) = sum((bsxfun(@minus, centroids, v(j,:))).^2,2);
            end
            cpt = 0;
            for optimal_assignment = assignmentoptimal(d)';
                cpt = cpt + 1;
                if optimal_assignment
                    new_H(cumsum_instances(i)-(optimal_assignment-1),e(cpt)) = 1;
                end
            end
        end
    end
    %toc;
    %disp('end of loop');
    if cpt_empty
        fprintf('%d empty bags\n', cpt_empty);
    end
    new_H = sparse(new_H);
    a = max(1,sum(new_H,1));
    %a(a <= 1) = 1;
    pinvY = (bsxfun(@rdivide,new_H,a))';
    Z = pinvY * U;
    %toc;
end

assigned_centroids = (sum(new_H,1) ~= 0);
a = max(1,sum(new_H,1));
keep_instance = sum(new_H,2) >= 0.5;  
new_Y_for_metric = sparse(new_H(keep_instance,:));
J = sparse(bsxfun(@rdivide,new_Y_for_metric,sqrt(a))); % identified_faces ./ repmat(sqrt(sum(identified_faces)), size(identified_faces,1),1);

Z = sparse(bsxfun(@rdivide,new_Y_for_metric,a));
X_MLCA = X(keep_instance,:);
centroids = Z' * X_MLCA;

meanX = mean(X_MLCA);
if center_descriptor
    X_MLCA = bsxfun(@minus, X_MLCA, meanX);
end

%disp('learning model');
%tic;
L = pinv(X_MLCA) * J;
%toc;
end
