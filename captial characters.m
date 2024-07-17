clc;
clear all;
close all;
imdsTrain=imageDatastore("Train","IncludeSubfolders",true,"LabelSource","foldernames")
imdsTest=imageDatastore("Test","IncludeSubfolders",true,"LabelSource","foldernames")
%imdstestchar=imageDatastore("testchar","IncludeSubfolders",true,"LabelSource","foldernames")
%im=imread("character_5.png")
%imshow(im)
load("cap_netw.mat");
net=trainedNetwork_2
out=classify(net,imdsTest)
out1=predict(net,imdsTest)
accuracy=sum(out==imdsTest.Labels)/numel(imdsTest.Labels)
plotconfusion(out,imdsTest.Labels)
co = confusionmat(imdsTest.Labels, out)
% Calculate precision
precision = diag(co) ./ sum(co, 1)';
% Calculate recall
recall = diag(co) ./ sum(co, 2);
% Calculate F1-score
f1_score = 2 * (precision .* recall) ./ (precision + recall);
% Display precision, recall, and F1-score
disp(['Precision: ', num2str(precision')])
disp(['Recall: ', num2str(recall')])
disp(['F1-score: ', num2str(f1_score')])
overall_precision = sum(diag(co)) / sum(co(:));
overall_recall = sum(diag(co)) / sum(co(:));
overall_f1_score = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall);
% Display overall precision, recall, and F1-score
disp(['Overall Precision: ', num2str(overall_precision)]);
disp(['Overall Recall: ', num2str(overall_recall)]);
disp(['Overall F1-score: ', num2str(overall_f1_score)]);
