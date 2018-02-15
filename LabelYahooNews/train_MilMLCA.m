function [ L, meanX, new_Y, centroids, assigned_centroids] = train_MilMLCA( X, U, docs, one_instance_bags, cumsum_instances,count_instances_bags, center_descriptor )
%TRAIN_MILMLCA Summary of this function goes here
%   Detailed explanation goes here

one_instance_docs = (sum(docs,2) == 1);

initial_docs = (one_instance_bags & one_instance_docs);
one_instance_bags_but_multiple_docs = one_instance_bags & ~one_instance_docs;
one_detected_doc_but_multiple_instances = ~one_instance_bags & one_instance_docs;
the_rest = ~one_instance_bags & ~one_instance_docs;
%tic;
initial_Y = zeros(size(U,1),size(docs,2)-3);
for i=find(initial_docs')
    initial_Y(cumsum_instances(i),docs(i,4:end)) = 1;
end
initialize_Y = initial_Y;
for i=find(one_instance_bags_but_multiple_docs')
    initialize_Y(cumsum_instances(i),docs(i,4:end)) = 1;
end
for i=find(one_detected_doc_but_multiple_instances')
    for j=0:(count_instances_bags(i)-1)
        initialize_Y(cumsum_instances(i)-j,docs(i,4:end)) = 1;
    end
end
for i=find(the_rest')
    for j=0:(count_instances_bags(i)-1)
        initialize_Y(cumsum_instances(i)-j,docs(i,4:end)) = 1;
    end
end
%toc;
initialize_Y = sparse(initialize_Y);
a = max(1,sum(initialize_Y,1));
%assigned_centroids = (a >= 0.5);
%a(a <= 1) = 1;
%tic;
pinvY = (bsxfun(@rdivide,(initialize_Y),a))';
%toc;
clear initialize_Y;
% disp('centroid computation');
% tic;
Z = pinvY * U;
% toc;
%clear initialize_Y;
clear pinvY;
old_Y = 0;
new_Y = initial_Y;

for iter=1:120
    if isequal(new_Y,old_Y)
        break;
    end
    old_Y = new_Y;
    new_Y = initial_Y;
    for i=find(one_instance_bags_but_multiple_docs')
        e = find(docs(i,4:end));
        centroids = Z(e,:);
        %d = bsxfun(@minus, centroids, U(cumsum_instances(i),:));
        d = sum((bsxfun(@minus, centroids, U(cumsum_instances(i),:))).^2,2);
        %u = repmat(U(cumsum_instances(i),:), size(centroids,1),1);
        %d = sum((centroids - u).^2,2);
        [~,b] = min(d);
        new_Y(cumsum_instances(i),e(b)) = 1;
    end
    for i=find(one_detected_doc_but_multiple_instances')
        e = find(docs(i,4:end));
        u = U((cumsum_instances(i) - (0:(count_instances_bags(i)-1))),:);
        %centroid = repmat(Z(e,:),size(u,1),1);
        %d = sum((centroid - u).^2,2);
        d = sum((bsxfun(@minus, u, Z(e,:))).^2,2);
        [~,b] = min(d);
        new_Y(cumsum_instances(i)-(b-1),e) = 1;        
    end
    cpt_empty = 0;
    for i=find(the_rest')
        e = find(docs(i,4:end));
        centroids = Z(e,:);
        v = U((cumsum_instances(i) - (0:(count_instances_bags(i)-1))),:);
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
                    new_Y(cumsum_instances(i)-(cpt),e(optimal_assignment)) = 1;
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
                    new_Y(cumsum_instances(i)-(optimal_assignment-1),e(cpt)) = 1;
                end
            end
        end
    end
%     if cpt_empty
%         fprintf('%d empty bags\n', cpt_empty);
%     end
    new_Y = sparse(new_Y);
    a = max(1,sum(new_Y,1));
    %a(a <= 1) = 1;
    pinvY = (bsxfun(@rdivide,new_Y,a))';
    Z = pinvY * U;
end

assigned_centroids = (sum(new_Y,1) ~= 0);
a = max(1,sum(new_Y,1));
keep_instance = sum(new_Y,2) >= 0.5;  
new_Y_for_metric = sparse(new_Y(keep_instance,:));
J = sparse(bsxfun(@rdivide,new_Y_for_metric,sqrt(a))); % identified_faces ./ repmat(sqrt(sum(identified_faces)), size(identified_faces,1),1);

Z = sparse(bsxfun(@rdivide,new_Y_for_metric,a));
X_MLCA = X(keep_instance,:);
centroids = Z' * X_MLCA;

meanX = mean(X_MLCA);
if center_descriptor
    X_MLCA = bsxfun(@minus, X_MLCA, meanX);
end

% disp('learning model');
%tic;
L = pinv(X_MLCA) * J;
%toc;
end

