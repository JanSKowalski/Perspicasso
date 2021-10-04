
%-----------------------------------------------------
%
%   Pre-Processing
%
%-----------------------------------------------------

% Make all images same type format
% %Get all images in same data Object
% for Seawulf '/gpfs/home/sushahid/ESE 440/images'
% for local 'C:\Users\sunsu\Downloads\images'
imds = imageDatastore('/gpfs/home/sushahid/ESE 440/images', 'IncludeSubFolders',true,'FileExtensions','.jpg','LabelSource','foldernames');

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

% All images
% 1 = file name, 2 = num appear, 3 = num incorrect, *OLD METHOD* 4-10 = specific cat wrong,
% 11= prob of actual (make this 11-17), 12-18 prob of predicted
% Key for cats (1-7): Elliot,Hiv,OF,PhotoLA,Saul,Garden,Unknown
images = cell(114,4);

% Probability matrix (cell struct), 
%(3 columns, 1st misclassified cat, 2nd prob pred, 3rd prob act)
probMatrix = cell(114,1);

for i = 1 : 114
    images(i,1) = imds.Files(i);
    images{i,2} = 0;
    images{i,3} = 0;
    % index number
    images{i,4} = i;
    probMatrix{i,1} = zeros(1,3);
end

% amount of runs
% 100 RUNS = 27 MIN RUNTIME************
numRuns = 100;

% Accuracy of model
accuracy = zeros(1,numRuns);

disp("Starting running loops");

% Run 10 times on specific layer
for i = 1 : numRuns
    % Split Data 70% training, 30% test
    % Change to .8/.2 later
    % read more on the split to make sure random and unique seed (print
    % seed)
    [imdsTrain,imdsTest] = splitEachLabel(imds,0.7,'randomized');   
    
    % Load some pretrained net
    net = googlenet;

    % Analyze Network to see the features 
    inputSize = net.Layers(1).InputSize;
    % analyzeNetwork(net)

    % Resize if need and other augmentation here
    augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
    augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);
    
    % GoogleNet layers: conv1-7x7_s2, conv2-3x3, inception_3a-1x1
    % inception_3b-1x1, inception_3b-3x3, inception_3b-5x5,    
    % inception_4e-1x1, inception_4e-3x3, inception_4e-5x5, 
    % inception_5b-1x1, inception_5b-3x3, inception_5b-5x5
    layer = 'conv2-3x3';
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
                % inc numAppear
                val = images{ii,2};
                images{ii,2} = val + 1;
                if (YPred(j) ~= YTest(j))
                    % inc numIncorrect
                    val = images{ii,3};
                    images{ii,3} = val + 1;
                    
                    newVec = zeros(1,3);
                    % add wrong cat prob
                    catIndxInc = find(order_cat == YPred(j));
                    newVec(1) = catIndxInc;
                    newVec(2) = scores(j, catIndxInc);
                    % add actual prob
                    catIndx = find(order_cat == YTest(j));
                    newVec(3) = scores(j, catIndx);
                    %if first vec
                    if (sum(probMatrix{ii,1}) == 0)
                        probMatrix{ii,1} = newVec;
                    else
                        probMatrix{ii,1} = [probMatrix{ii,1};newVec];
                    end
                    
                    % *OLD Method for avgs
                    % Probs
%                     catIndx = find(order_cat == YTest(j));
%                     val = images{ii,11};
%                     images{ii,11} = val + scores(j, catIndx);
%                     catIndxInc = find(order_cat == YPred(j));
%                     val = images{ii,11+catIndxInc};
%                     images{ii,11+catIndxInc} = val + scores(j, catIndxInc);
%                     %which cat is misclassified as
%                     val = images{ii,3+catIndxInc};
%                     images{ii,3+catIndxInc} = val+1;
                end
                break;
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
    
    disp(strcat("Loop ", num2str(i), " done."));
    
end

disp("Done running loops");
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
images = sortrows(images,3, 'descend'); 

% Writing to excel file
filename = 'resultsGoogleNet.xlsx';
% print accuracy of each run
% Fix padding of info
acc = {['Average Accuracy = ' num2str(mean(accuracy)) ' over '... 
    num2str(numRuns) ' runs'],['Layer: ' layer]};

% Change for each new layer
sheetName = 'Sheet2';
sheetNum = 2;

writecell(acc',filename,'Sheet',sheetNum,'Range','C3');
writecell({'Accuracies:'}, filename, 'Sheet', sheetNum, 'Range', 'K5');
writematrix(accuracy', filename, 'Sheet', sheetNum, 'Range', 'K6');

inc = 10;

for i = 1:114
    if (images{i,3} <= 0)
        break;
    end
    %figure('visible','off');
    
    % * OLD METHOD
%     numWrong = num2str(images{i,3});
%     numAppear = num2str(images{i,2});
%     numImgWrongStr = ['Num Wrong = ' numWrong '/' numAppear];
%     % Probabilities written as predicted img prob/ actual img prob
%     catsAndProbs = ["", "", "", "", "", "", ""];
%     for j = 1:7
%         if (~strcmp(char(order_cat{j,1}),char(actCat)) && images{i,3+j} > 0)
%            predLabel = char(order_cat(j));
%            numWrong = images{i,3+j};
%            numWrongStr = num2str(numWrong);
%            probInc = num2str(images{i,11+j}/numWrong, '%.3f');
%            probCor = num2str(images{i,11}/numWrong, '%.3f');
%            catsAndProbs(j) = [predLabel ': Num Wrong = ' numWrongStr ', Prob: ' probInc ' / ' probCor];
%         end
%     end
%     catsAndProbsStr = "";
%     for j = 1:7
%         if (catsAndProbs(j) ~= "")
%             catsAndProbsStr = sprintf('%s %s \n',catsAndProbsStr,catsAndProbs(j));
%         end
%     end

    %I = imread('');
    
    %index for prob matrix
    pmIndx = images{i,4};
    
    % Write actual category
    % for local use \ for seawulf /
    actCat = split(images{i,1},'/');
    imPath = char(strcat('C:\Users\sunsu\Downloads\images\images\', actCat(7), '\', actCat(8)));
    % for local use 7, seawulf use 6
    actCat = actCat(7);
    imgLabel = {'Actual Cat.: ' char(actCat)};
    letter = strcat("E",num2str(inc));
    writecell(imgLabel, filename, 'Sheet', sheetNum, 'Range', letter);
    % Write header
    catLabels = {'Category', 'Pred.', 'True'};   
    letter = strcat("E",num2str(inc+1));
    writecell(catLabels, filename, 'Sheet', sheetNum, 'Range', letter);
    % Write probabilities
    sz = size(probMatrix{pmIndx,1});
    letter = strcat("E",num2str(inc+2));
    letter1 = strcat("F",num2str(inc+2));
    % Convert prob matrix first column to categories
    catsChars = {};
    catsNums = probMatrix{pmIndx,1};
    for j = 1:sz(1)
        catsChars = [catsChars;char(order_cat(catsNums(j,1)))];
    end
    writecell(catsChars,filename, 'Sheet', sheetNum, 'Range', letter);
    probMatrixprobs = probMatrix{pmIndx,1};
    writematrix(probMatrixprobs(:,2:3), filename, 'Sheet', sheetNum, 'Range', letter1);
    % Write total mistakes per category
    letter = strcat("E",num2str(inc+sz(1)+3));
    writecell({'Mistakes per Cat.:'}, filename, 'Sheet', sheetNum, 'Range', letter);
    letter = strcat("E",num2str(inc+sz(1)+4));
    letter1 = strcat("F",num2str(inc+sz(1)+4));
    
    mistakesVector = {};
    mistakesCats = {};
    for j = 1:7
        if (nnz(probMatrix{pmIndx,1}==j) > 0)
            mistakesVector = [mistakesVector; nnz(probMatrix{pmIndx,1}==j)];
            mistakesCats = [mistakesCats;char(order_cat(j))];
        end
    end
    writecell(mistakesCats, filename, 'Sheet', sheetNum, 'Range', letter);
    writecell(mistakesVector, filename, 'Sheet', sheetNum, 'Range', letter1);
    % Write total mistakes / num appear
    sz2 = size(mistakesVector);
    letter = strcat("E",num2str(inc+sz(1)+5+sz2(1)));
    writecell({'Total Mistakes', 'Num Appear'}, filename, 'Sheet', sheetNum, 'Range', letter);
    letter = strcat("E",num2str(inc+sz(1)+6+sz2(1)));
    totalMis = images{i,3};
    totalAppear = images{i,2};
    writematrix([totalMis, totalAppear], filename, 'Sheet', sheetNum, 'Range', letter);
    
    %imshow(I)
    
    %truesize([200,200]);
    
    %title({numImgWrongStr, imgLabel, catsAndProbsStr})
    letter = strcat("A",num2str(inc));
    inc = inc+60;
    if (inc-60+sz(1)+8+sz2(1) >= inc)
        inc = inc+20;
    end
    writecell({imPath}, filename, 'Sheet', sheetNum, 'Range', letter);
    %xlsPasteTo(filename,sheetName, 200, 200, letter);
    %close
end

disp("Completed.");


% Preprocessing in seperate file.
% Save probability stuff in seperate.

% Metrics to be collected for Feature Extraction:
% accuracy, precision, (overall and class-wise)
% Collect this for all layers of 4 DNN's.

% Repeat and change image size and then datasets?




% Augmentation, random crop, rotation, other options, do a good amount.

% Input size (50x50) to (225x225) (intervals of 25 or something) once (1600x1600)
% 500 runs each (5 scripts of 100 runs) 
% 10 epochs each run

% Varying Epochs 
% 1 to like 50 (if need more)

% Vary dataset

% Vary batch size








