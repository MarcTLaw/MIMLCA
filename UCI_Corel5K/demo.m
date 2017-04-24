clear all;
addpath('assignment');

training= true;


disp('loading files');
document_words = load('dataset/document_words');
document_blobs = load('dataset/document_blobs');
blobs = load('dataset/blobs');
blob_counts = load('dataset/blob_counts');
test_1_document_words = load('dataset/test_1_document_words');
test_1_blobs = load('dataset/test_1_blobs');
test_1_blob_counts = load('dataset/test_1_blob_counts');

nb_words = 499;
keep_top = 20;


R = 20;

disp('files loaded');

doc_words = document_words(:);

counts = zeros(1,nb_words);
for i=1:nb_words
    counts(i) = sum(doc_words==i);
end

[sorted_words, indices_top_words] = sort(counts,'descend');

sorted_words = sorted_words(1:keep_top);
indices_top_words = indices_top_words(1:keep_top);


keep_image = false(size(document_words,1),1);
for i=1:size(document_words,1)
    image_word = document_words(i,:);
    if sum(ismember(image_word,indices_top_words))
        keep_image(i) = true;
    end
end

keep_training_instance = false(size(blobs,1),1);
cpt = 1;
for i=1:size(blob_counts,1)
    blobi = blob_counts(i);
    if keep_image(i)
        keep_training_instance(cpt:(cpt+blobi-1)) = true;
    end
    cpt = cpt + blobi;
end

training_blobs = blobs(keep_training_instance,:);


keep_test_image = false(size(test_1_document_words,1),1);
for i=1:size(test_1_document_words,1)
    image_word = test_1_document_words(i,:);
    if sum(ismember(image_word,indices_top_words))
        keep_test_image(i) = true;
    end
end

nb_training_images = sum(keep_image);
nb_test_images = sum(keep_test_image);

keep_test_instance = false(size(test_1_blobs,1),1);
cpt = 1;
for i=1:size(test_1_blob_counts,1)
    blobi = test_1_blob_counts(i);
    if keep_test_image(i)
        keep_test_instance(cpt:(cpt+blobi-1)) = true;
    end
    cpt = cpt + blobi;
end


Y_training_images = false(nb_training_images,keep_top);
document_training_words = document_words(keep_image,:);
training_blob_counts = blob_counts(keep_image);


for i=1:size(document_training_words,1)
    image_word = document_training_words(i,:);
    a = ismember(image_word,indices_top_words);

    for j=image_word(a)
        Y_training_images(i,find(j==indices_top_words)) = true;
    end
end
center_descriptor = false;

disp('training model');
%size(Y_training_images)
if training
    
tStart = tic;
[U,~,~] = svd(training_blobs,'econ');
[L, meanX, new_Y_human, centroids, assigned_centroids] = train_MilMLCA(training_blobs, U, Y_training_images, training_blob_counts );
tElapsed = toc(tStart);
fprintf('training complete in %f seconds\n', tElapsed);
%save(strcat('mimlca_metric_.mat'), 'L', 'center_descriptor', 'meanX', 'centroids', 'assigned_centroids', '-v7.3');
else
load('mimlca_metric_.mat');
end
%L = L(:,1:(end-2));

training_blobs = training_blobs * L;
test_blobs = test_1_blobs(keep_test_instance,:) * L;
test_blob_counts = test_1_blob_counts(keep_test_image)';
training_blob_counts = training_blob_counts';
test_document_words = test_1_document_words(keep_test_image,:);
C = R;

cpt_big = 1;

if true
Hausdorff_dist = zeros(length(training_blob_counts),length(test_blob_counts));
for i=1:length(test_blob_counts)
    if ~mod(i,100)
        fprintf('computing Hausdorff distances iteration: %d/%d\n', i, length(test_blob_counts))
    end
    test_blob = test_blobs(cpt_big:(cpt_big+test_blob_counts(i)-1),:);
    D = zeros(size(training_blobs,1),size(test_blob,1));
    for j=1:size(test_blob,1)
        a = bsxfun(@minus, training_blobs, test_blob(j,:));
        D(:,j) = sum(a.^2,2);
    end
    cpt = 1;
    cpt_h = 1;
    for j = training_blob_counts
        d = D(cpt:(cpt + j - 1),:);
        %Hausdorff_dist(cpt_h,i) = max(min(max(d)),min(max(d')));
        Hausdorff_dist(cpt_h,i) = min(min(d));
        cpt_h = cpt_h + 1;
        cpt = cpt + j;
    end
    cpt_big = cpt_big + test_blob_counts(i);
end

%save('haus_dist.mat','Hausdorff_dist');
else
    load('haus_dist.mat');
end

R_matrix = zeros(length(test_blob_counts),R);
for i=1:length(test_blob_counts)
    h = Hausdorff_dist(:,i);
    [a,b] = sort(h);
    b = b(1:R);
    R_matrix(i,:) = b;
end


C_matrix = zeros(length(training_blob_counts),C);
for i=1:size(Hausdorff_dist,1)
    h = Hausdorff_dist(i,:);
    [a,b] = sort(h);
    b = b(1:C);
    C_matrix(i,:) = b;
end
one_error = 0;
coverage = 0;

average_precision = 0;

for i=1:length(test_blob_counts)
    if ~mod(i,100)
        fprintf('test evaluation: iteration %d/%d\n', i, length(test_blob_counts))
    end
    count_score_classes = zeros(1,keep_top);
    for j=R_matrix(i,:)
        for k=document_training_words(j,:)
            if ismember(k,indices_top_words)
                count_score_classes(find(k==indices_top_words)) = count_score_classes(find(k==indices_top_words)) + 1;
            end
        end
    end
    for j=1:size(C_matrix,1)
        if ismember(i, C_matrix(j,:))
            for k=document_training_words(j,:);
                if ismember(k,indices_top_words)
                    count_score_classes(find(k==indices_top_words)) = count_score_classes(find(k==indices_top_words)) + 1;
                end
            end
        end
    end
    [max_score, max_label] = max(count_score_classes);
    if ~ismember(indices_top_words(max_label),test_document_words(i,:))
        one_error  = one_error + 1;
    end
    
    [sort_score, sort_label] = sort(count_score_classes,'descend');
    rrr = 0;
    cptr = 0;
    for lab = sort_label
        cptr = cptr + 1;
        if ismember(indices_top_words(lab),test_document_words(i,:))
            rrr = cptr;
        end
    end
    coverage = coverage + rrr-1;
    
    rrr = 0;
    cptr = 0;
    nb_labels = 0;
    ap_score = 0;
    for lab = sort_label
        cptr = cptr + 1;
        if ismember(indices_top_words(lab),test_document_words(i,:))
            nb_labels = nb_labels + 1;
            rrr = cptr;
            ap_score = ap_score + nb_labels / rrr;
        end
    end
    ap_score = ap_score / nb_labels;
    
    average_precision = average_precision + ap_score;
    
end
%disp('one error');
oe = one_error/ length(test_blob_counts);
%disp('coverage');
cover = coverage / length(test_blob_counts);
%disp('average precision');
ap = average_precision / length(test_blob_counts);
if training
    fprintf('training time: %f seconds\n',tElapsed);
end
fprintf('one error: %f\n',oe)
fprintf('coverage: %f\n',cover)
fprintf('average precision: %f\n',ap)