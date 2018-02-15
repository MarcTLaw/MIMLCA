function [] = train(set_indices)
if nargin < 1
    set_indices = 1:10;
end
mkdir('created_data')
disp('starting training')
addpath('assignment');
if ~exist('data','var')
    disp('loading data');
    tic;
    load('lyn_train-test.mat');
    toc;
    disp('data loaded');
end
load('ten_fold_set.mat');
load_xu = false;

learn_MLCA = true;
learn_MIMLCA_automatic = true;
learn_MIMLCA_human = true;


for set_index=set_indices
    disp(strcat('training on split ',int2str(set_index)));
    center_descriptor = false;
    
    set_1_bags = (fold_set_bags ~= set_index);
    set_1_instances = (fold_set_instances ~= set_index);
    
    
    if load_xu
        %load('x_seta.mat');
        load(strcat('created_data/training_set_',int2str(set_index), '.mat'));
        disp('Data loaded');
    else
        X = double(data.face_desc(:,~set_1_instances))';
        save(strcat('created_data/test_set_',int2str(set_index), '.mat'),'X', '-v7.3');
        X = double(data.face_desc(:,set_1_instances))';
        save(strcat('created_data/training_set_',int2str(set_index), '.mat'),'X', '-v7.3');
    end
    
    
    face_id = data.face_idx(set_1_instances,:);
    x = face_id(:,1);
    [count_instances_bags,~] = histc(x,unique(x));
    cumsum_instances = cumsum(count_instances_bags);
    
    one_instance_bags = (count_instances_bags == 1);
    nb_one_instance_bags = sum(one_instance_bags);
    
    docs = data.doc_nameid(set_1_bags,:);
    
    faces_id = data.face_id(set_1_instances,:);
    faces_id(faces_id(:,1:3)) = 0;
    nb_instances = sum(faces_id,2)';
    unique_x = unique(x);
    W_human = false(sum(set_1_bags),size(faces_id,2));
    for k = find(nb_instances)
        W_human((x(k)==unique_x),(faces_id(k,:))) = true;
    end
    W_human = sparse(W_human);
    
    if learn_MLCA
        disp('starting training of MLCA');
        identified_faces = data.face_id(set_1_instances,4:end);
        assigned_centroids = (sum(identified_faces) ~= 0);
        a = max(1,sum(identified_faces,1));
        Z = sparse(bsxfun(@rdivide,identified_faces,a));
        centroids = Z' * X;
        identified_faces = identified_faces(:,(sum(identified_faces) ~= 0));
        sum_Y = sum(identified_faces);
        sum_Y(~sum_Y) = 1;
        J = bsxfun(@rdivide,identified_faces,sqrt(sum_Y));
        keep_instance = sum(identified_faces,2) >= 1;
        
        X_MLCA = X(keep_instance,:);
        meanX = mean(X_MLCA,1);
        
        if center_descriptor
            meanX = mean(X_MLCA,1);
            X_MLCA = bsxfun(@minus, X_MLCA, meanX);
        end
        J = sparse(J(keep_instance,:));
        disp('learning MLCA');
        L = pinv(X_MLCA) * J;
        disp('MLCA learned');
        Y = identified_faces;
        save(strcat('created_data/metric_mlca_',int2str(set_index), '.mat'), 'L', 'center_descriptor', 'meanX', 'centroids', 'assigned_centroids','Y','J','-v7.3');
        disp('MLCA Metric saved');
    end
    
    
    if load_xu
        load(strcat('created_data/u_matrix_',int2str(set_index), '.mat'));
    else
        disp('Applying SVD');
        
        
        tic;
        [U,~,~] = svd(bsxfun(@minus, X, mean(X)),'econ');
        toc;
        disp('matrix U computed');
        save(strcat('created_data/u_matrix_',int2str(set_index), '.mat'), 'U', '-v7.3');
        
        disp('matrix U saved');
    end
    if learn_MIMLCA_human
        disp('training model using annotations provided by humans (scenario b)');
        
        [L, meanX, new_Y_human, centroids, assigned_centroids] = train_MilMLCA( X, U, W_human, one_instance_bags, cumsum_instances,count_instances_bags,center_descriptor  );
        disp('saving metric');
        save(strcat('created_data/metric_human_',int2str(set_index), '.mat'), 'L', 'center_descriptor', 'meanX', 'centroids', 'assigned_centroids', '-v7.3');
        disp('MIMLCA Metric saved (scenario b)');
    end
    
    if learn_MIMLCA_automatic
        disp('training model using automatic annotations (scenario c)');
        [L, meanX, new_Y_automatic, centroids, assigned_centroids] = train_MilMLCA( X, U, docs, one_instance_bags, cumsum_instances,count_instances_bags,center_descriptor );
        disp('saving metric');
        save(strcat('created_data/metric_automatic_',int2str(set_index), '.mat'), 'L', 'center_descriptor', 'meanX', 'centroids', 'assigned_centroids', '-v7.3');
        disp('MIMLCA Metric saved (scenario c)');
    end
    
end


end
