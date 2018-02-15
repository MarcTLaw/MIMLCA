function [] = test(set_indices)
if nargin < 1
    set_indices = 1:10;
end

accuracy_list = [];
precision_list = [];

test_MLCA = false;
test_MIMLCA_human = true;

if test_MLCA
    disp('testing MLCA (scenario a)')
elseif test_MIMLCA_human
    disp('testing MIMLCA with human supervision (scenario b)');
else
    disp('testing MIMLCA with automatic supervision (scenario c)');
end



for set_index = set_indices
    
    disp(strcat('test on split ',int2str(set_index))'');
    
    if ~exist('data','var')
        disp('loading data');
        tic;
        load('lyn_train-test.mat');
        toc;
        disp('data loaded');
    end
    disp('loading test set');
    
    load('ten_fold_set.mat');
    
    
    
    
    set_1_bags = (fold_set_bags ~= set_index);
    set_1_instances = (fold_set_instances ~= set_index);
    
    load(strcat('created_data/test_set_',int2str(set_index), '.mat'));
    B = data.face_id(~set_1_instances,:);
    B(:,1:3) = 0;
    
    
    nb_bags = sum(set_1_bags);
    nb_instances = sum(set_1_instances);
    
    
    disp('loading metric');
    
    if test_MLCA
        LL = load(strcat('created_data/metric_mlca_',int2str(set_index), '.mat'));
    elseif test_MIMLCA_human
        LL = load(strcat('created_data/metric_human_',int2str(set_index), '.mat'));
    else
        LL = load(strcat('created_data/metric_automatic_',int2str(set_index), '.mat'));
    end
    
    
    
    
    J = X *  LL.L;
    
    
    h1 =  LL.centroids * LL.L;
    
    accuracy = 0;
    
    minimum_nb_images_per_class_in_train = 5;
    minimum_nb_images_per_class_in_test = 5;
    
    
    C = data.face_id(set_1_instances,4:end);
    selected_categories = sum(C,1) >= minimum_nb_images_per_class_in_train;
    assigned_centroids = selected_categories;
    D = B;
    D(:,sum(D,1)< minimum_nb_images_per_class_in_test) = 0;
    test_identified_faces = (sum(D,2) == 1)';
    
    good = zeros(1,size(C,2));
    seen_categories = zeros(1,size(C,2));
    
    nb_test_samples = 0;
    cpt = 0;
    %
    %     disp('creating training set')
    %
    %     Xtraining = double(data.face_desc(:,set_1_instances))';
    %     YYtraining = data.face_id(set_1_instances,4:end);
    %     Ytraining = zeros(size(YYtraining,1),1);
    %     kept_training_knn = false(size(YYtraining,1),1);
    %     for ppp = 1:size(YYtraining,1)
    %         uuuu = find(YYtraining(ppp,:));
    %         if uuuu
    %             Ytraining(ppp) = uuuu;
    %             kept_training_knn(ppp) = true;
    %         end
    %     end
    %     Xtraining = Xtraining(kept_training_knn,:);
    %     Ytraining = Ytraining(kept_training_knn,:);
    %
    
    for a = find(test_identified_faces)
        cpt = cpt + 1;
        chosen_index = find(B(a,:));
        chosen_index2 = chosen_index-3;
        if ~assigned_centroids(chosen_index2)
            %disp('not assigned');
            continue;
        end
        seen_categories(chosen_index2) = seen_categories(chosen_index2) + 1;
        nb_test_samples = nb_test_samples + 1;
        
        
        [i,j] = min(sum((bsxfun(@minus, h1, J(a,:))).^2,2));
        good(chosen_index2) = good(chosen_index2) + (j == chosen_index2);
        accuracy = accuracy + (j == chosen_index2);
        
        
    end
    
    
    
    nb_tested_categories = sum(seen_categories > 0);
    
    
    accuracy_by_class = good ./ max(1, seen_categories);
    accuracy_by_class = mean(accuracy_by_class(seen_categories > 0));
    
    accuracy_list = [accuracy_list, accuracy_by_class];
    precision_list = [precision_list, (accuracy ./ nb_test_samples)];
    
    disp('end');
end

if test_MLCA
    disp('results for MLCA (scenario a)')
elseif test_MIMLCA_human
    disp('results for MIMLCA with human supervision (scenario b)');
else
    disp('results for MIMLCA with automatic supervision (scenario c)');
end



mean(accuracy_list) * 100
std(accuracy_list) * 100

mean(precision_list) * 100
std(precision_list) * 100
end
