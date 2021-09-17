
%-----------------------------------------------------
%
%   Pre-Processing
%
%-----------------------------------------------------

% Make all images same type format
% %Get all images in same data Object
imds = imageDatastore('C:\Users\sunsu\Downloads\images', 'IncludeSubFolders',true,'FileExtensions','.jpg','LabelSource','foldernames');

% img = readimage(imds,1);
% sz = size(img);

% numberOfImages = length(imds.Files);

% I Checked maxRows = 1600
% maxRows = 0;

% %Make All RGB
% for k = 1 : numberOfImages
%   % Get the input filename. 
%   inputFileName = imds.Files{k};
%   grayImg = imread(inputFileName);
%   [rows, columns, numberOfColorChannels] = size(grayImg);
% %   if rows > maxRows
% %       maxRows = rows;
% %   end
%   if numberOfColorChannels ~= 3
%     % It's gray so need to convert to rgb.
%     rgbImage = grayImg(:,:,[1 1 1]);
%     imwrite(rgbImage, inputFileName);
%   end
% end
% 
% % Resize All
% for k = 1 : numberOfImages
%   % Get the input filename. 
%   inputFileName = imds.Files{k};
%   img = imread(inputFileName);
%   img = imresize(img, [250 250]);
%   imwrite(img, inputFileName);
% end




% This function didnt really work, but might be useful later 
% if we want to play around with coloring

% function rgbImage = gray2rgb(grayImage) 
% cmap = jet(256); % Or whatever one you want.
% rgbImage = ind2rgb(grayImage, cmap); % Convert gray scale image into an RGB image.
% end



% Custom Neural Network Code, maybe useful later (gave like 40% accuracy
% earlier)



% numTrainingFiles = 5;
% [imdsTrain,imdsTest] = splitEachLabel(imds,numTrainingFiles,'randomize');


% augimds = augmentedImageDatastore([600 600], imds, 'ColorPreprocessing', 'gray2rgb');
% 
% imageSize = [600 600 3];
% 
% layers = [
%     imageInputLayer(imageSize)
%     
%     convolution2dLayer(3,8,'Padding','same')
%     batchNormalizationLayer
%     reluLayer   
%     
%     maxPooling2dLayer(2,'Stride',2)
%     
%     convolution2dLayer(3,16,'Padding','same')
%     batchNormalizationLayer
%     reluLayer   
%     
%     maxPooling2dLayer(2,'Stride',2)
%     
%     convolution2dLayer(3,32,'Padding','same')
%     batchNormalizationLayer
%     reluLayer   
%     
%     fullyConnectedLayer(7)
%     softmaxLayer
%     classificationLayer];
% 
% opts = trainingOptions('sgdm', ...
%     'MaxEpochs',15, ...
%     'Shuffle','every-epoch', ...
%     'Plots','training-progress', ...
%     'Verbose',false);
% 

% net = trainNetwork(augimds,layers,opts);

% Most useful network plotting and analyzing functions List:

% plotconfusion Plot classification confusion matrix
% ploterrcorr Plot autocorrelation of error time series
% ploterrhist Plot error histogram
% plotfit Plot function fit
% plotperform Plot network performance
% plotregression Plot linear regression
% plotroc Plot receiver operating characteristic
% plottrainstate Plot training state values



% Save Networks
% networkName = net;
% save networkName;





%-----------------------------------------------------
%
%   Feature Extraction
%
%-----------------------------------------------------


% Need to optimize to return predicted label
% of most incorrect not just last one, if that matters?

% All images
% 1 = file name, 2 = num appear, 3 = num incorrect,
% 4 = actual label, 5 = predicted label, 6 = prob of actual,
% 7= prob of predicted
images = cell(114,7);
for i = 1 : 114
    images(i,1) = imds.Files(i);
    images{i,2} = 0;
    images{i,3} = 0;
    images{i,6} = 0;
    images{i,7} = 0;
end

% Accuracy of model
accuracy = zeros(1,10);

% Run 10 times on specific layer
for i = 1 : 10
    % Split Data 70% training, 30% test
    % Change to .8/.2 later
    [imdsTrain,imdsTest] = splitEachLabel(imds,0.7,'randomized');   
    
    % Load some pretrained net
    net = googlenet;

    % Analyze Network to see the features 
    inputSize = net.Layers(1).InputSize;
    %analyzeNetwork(net)

    % Resize if need and other augmentation here
    augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
    augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);
    
    layer = 'inception_3a-relu_1x1';
    featuresTrain = activations(net,augimdsTrain,layer);
    featuresTest1 = activations(net,augimdsTest,layer);

    featuresTrain = squeeze(mean(featuresTrain,[1 2]))';
    featuresTest = squeeze(mean(featuresTest1,[1 2]))';

    % Get class labels
    YTrain = imdsTrain.Labels;
    YTest = imdsTest.Labels;
    
    % Use extracted  features to fit a SVM
    % Work on ScoreTransform to logit, or method to bad.
    classifier = fitcecoc(featuresTrain,YTrain, 'FitPosterior',true);

    % Use the classifier to predict results
    [YPred, ~,~,scores] = predict(classifier,featuresTest);
    
    accuracy(i) = mean(YPred == YTest);
    
    %categories of images
    order_cat = categories(YPred(1));
    
    % Get which ones are new test ones
    % Get labels
    for j = 1 : 33
        for ii = 1:114
            if (strcmp(imdsTest.Files{j,1},images{ii,1}))
                val = images{ii,2};
                images{ii,2} = val + 1;
                if (YPred(j) ~= YTest(j))
                    val = images{ii,3};
                    images{ii,3} = val + 1;
                    % Correct label
                    images{ii,4} = YTest(j);
                    % Predicted label
                    images{ii,5} = YPred(j);
                    % Probs
                    catIndx = find(order_cat == YTest(j));
                    images{ii,6} = scores(j, catIndx);
                    catIndxInc = find(order_cat == YPred(j));
                    images{ii,7} = scores(j, catIndxInc);
                end
            end
        end
    end
    
    % show layer features 
%     if (i == 1)
%         layer = 13;
%         name = net.Layers(layer).Name;
% 
%         channels = 1:64;
%         I = deepDreamImage(net,name,channels, ...
%                 'Verbose',false, ...
%                 'PyramidLevels',1);
% 
%         figure
%         I = imtile(I,'ThumbnailSize',[128 128]);
%         imshow(I)
%         title(['Layer ',name,' Features'],'Interpreter','none')
%     end
    
%     % show one image before and after layer
%     if (i == 1)
%         figure
%         I = imread(augimdsTest.Files{1,1});
%         imshow(I)
%         title("Image Before")
%         figure
%         act1 = featuresTest1(:,:,:,1);
%         %limit to 64 filters
%         act1 = act1(:,:,1:64);
%         sz = size(act1);
%         act1 = reshape(act1,[sz(1) sz(2) 1 sz(3)]);
%         I = imtile(act1,'GridSize',[8 8]);
%         imshow(I)
%         title("Image After")
%         figure
%         I = imtile(mat2gray(act1),'GridSize',[8 8]);
%         imshow(I)
%         title("Image After (Normalized GrayScale)")
%     end
    
    % For just one feature
%     if (i == 1)
%         figure
%         channels = 1:1;
%         I = deepDreamImage(net,name,channels, ...
%                 'Verbose',false, ...
%                 'PyramidLevels',1);
% 
%         subplot(2,2,1);
%         imshow(I)
%         title(['Layer ',name,' Feature 1'],'Interpreter','none')
%         
%         subplot(2,2,2);
%         I = imread(augimdsTest.Files{1,1});
%         imshow(I)
%         title("Image Before")
%         
%         subplot(2,2,3);
%         act1 = featuresTest1(:,:,:,1);
%         %limit to 64 filters
%         act1 = act1(:,:,1);
%         sz = size(act1);
%         act1 = reshape(act1,[sz(1) sz(2)]);
%         imshow(act1)
%         title("Image After")
%         
%         subplot(2,2,4);
%         I = mat2gray(act1);
%         imshow(I)
%         title("Image After (Normalized GrayScale)")
%     end
    
    
end


% This was for the top worst

% Find top 4 worst
% Prob do this in data struct/cell, if printing all then no need to yet
% worst4 = zeros(1,4);
% numAppears = zeros(1,4);
% correctLabel = ["" "" "" ""];
% incorrectLabel = ["" "" "" ""];
% worstImgs = ["" "" "" ""];
% probabilities = zeros(1,4);
% probabilitiesInc = zeros(1,4);
% for i = 1:114
%     if (images{i,2} > worst4(1))
%         worst4(1) = images{i,2};
%         numAppears(1) = images{i,3};
%         worstImgs(1) = convertCharsToStrings(images{i,1});
%         correctLabel(1) = images{i,4};
%         incorrectLabel(1) = images{i,5};
%         probabilities(1) = images{i,6};
%         probabilitiesInc(1) = images{i,7};
%     elseif (images{i,2} <= worst4(1) && images{i,2} > worst4(2))
%         worst4(2) = images{i,2};
%         numAppears(2) = images{i,3};
%         worstImgs(2) = images{i,1};
%         correctLabel(2) = images{i,4};
%         incorrectLabel(2) = images{i,5};
%         probabilities(2) = images{i,6};
%         probabilitiesInc(2) = images{i,7};
%     elseif (images{i,2} <= worst4(2) && images{i,2} > worst4(3))
%         worst4(3) = images{i,2};
%         numAppears(3) = images{i,3};
%         worstImgs(3) = images{i,1};
%         correctLabel(3) = images{i,4};
%         incorrectLabel(3) = images{i,5};
%         probabilities(3) = images{i,6};
%         probabilitiesInc(3) = images{i,7};
%     elseif (images{i,2} <= worst4(3) && images{i,2} > worst4(4))
%         worst4(4) = images{i,2};
%         numAppears(4) = images{i,3};
%         worstImgs(4) = images{i,1};
%         correctLabel(4) = images{i,4};
%         incorrectLabel(4) = images{i,5};
%         probabilities(4) = images{i,6};
%         probabilitiesInc(4) = images{i,7};
%     end
% end


% Sort by most incorrect
oldimages = images;
images = sortrows(images,3, 'descend'); 

% Writing to excel file
filename = 'resultsData1.xlsx';
accCell = cell(1,2);
accCell{1,1} = 'Average Accuracy = ';
accCell{1,2} = mean(accuracy);

writecell(accCell,filename,'Sheet',1,'Range','C5');

% for new do for all
letter = ["C10" "C30" "C50" "C70"];
letter1 = letter(1);

for i = 1:4
    figure
    I = imread(convertCharsToStrings(images{i,1}));
    numWrong = num2str(images{i,3});
    numAppear = num2str(images{i,2});
    probImg = num2str(images{i,6}, '%.3f');
    probImgInc = num2str(images{i,7}, '%.3f');
    numImgWrong = ['Num Wrong = ' numWrong '/' numAppear];
    imgLabel = ['Actual Cat./Prob: ' char(images{i,4}) ' / ' probImg];
    imgLabel2 = ['Predicted Cat./Prob: ' char(images{i,5}) ' / ' probImgInc];
    imshow(I)
    title({numImgWrong, imgLabel, imgLabel2})
    
    xlsPasteTo(filename,'Sheet1', 400, 300, letter(i));
    close
end





function xlsPasteTo(filename,sheetname,width, height,varargin)
%Paste current figure to selected Excel sheet and cell
%
%
% xlsPasteTo(filename,sheetname,width, height,range)
%Example:
%xlsPasteTo('File.xls','Sheet1',200, 200,'A1')
% this will paset into A1 at Sheet1 at File.xls the current figure with
% width and height of 200
%
% tal.shir@hotmail.com
%https://www.mathworks.com/matlabcentral/fileexchange/21158-paste-a-matlab-figure-to-excel
options = varargin;
    range = varargin{1};
    
[fpath,file,ext] = fileparts(char(filename));
if isempty(fpath)
    fpath = pwd;
end
Excel = actxserver('Excel.Application');
set(Excel,'Visible',0);
Workbook = invoke(Excel.Workbooks, 'open', [fpath filesep file ext]);
sheet = get(Excel.Worksheets, 'Item',sheetname);
invoke(sheet,'Activate');
    ExAct = Excel.Activesheet;
   
  ExActRange = get(ExAct,'Range',range);
    ExActRange.Select;
    pos=get(gcf,'Position');
    set(gcf,'Position',[ pos(1:2) width height])
    print -dmeta
invoke(Excel.Selection,'PasteSpecial');
invoke(Workbook, 'Save');
invoke(Excel, 'Quit');
delete(Excel);
end









