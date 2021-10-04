
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%       DNN Models
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% begin timer
tic;

disp("Beginning program...");

% writing to excel
filename = 'resultsDNN1GoogleNet.xlsx';

% Change for each new run
sheetName = 'Sheet1';
sheetNum = 1;

writecell({'Network:', 'GoogleNet'}, filename, 'Sheet', sheetNum, 'Range', 'C3');
writecell({'Params.:', 'Setup A'}, filename, 'Sheet', sheetNum, 'Range', 'C4');
titles = {'Run#', 'Accuracy', 'Class-wise', 'Accuracy'};
writecell(titles, filename, 'Sheet', sheetNum, 'Range', 'C7');
writecell({'Class-wise', 'Precision'}, filename, 'Sheet', sheetNum, 'Range', 'L7');
writecell({'Class-wise', 'Sensitivity'}, filename, 'Sheet', sheetNum, 'Range', 'S7');
writecell({'Class-wise', 'Specificity'}, filename, 'Sheet', sheetNum, 'Range', 'Z7');
catNames = {'Elliot W.',	'Hivernacle',	'Old Future',	'Photo Con.',...
    'S Painted',	'The Gard.', 'The Unk.'};
writecell(catNames, filename, 'Sheet', sheetNum, 'Range', 'E8');
writecell(catNames, filename, 'Sheet', sheetNum, 'Range', 'L8');
writecell(catNames, filename, 'Sheet', sheetNum, 'Range', 'S8');
writecell(catNames, filename, 'Sheet', sheetNum, 'Range', 'Z8');



% %Get all images in same data Object
% for Seawulf '/gpfs/home/sushahid/ESE 440/images'
% for local 'C:\Users\sunsu\Downloads\images'
imds = imageDatastore('/gpfs/home/sushahid/ESE 440/images', 'IncludeSubFolders',true,'LabelSource','foldernames');

% preprocess here if need (imgSize)

%verify images 


% number of runs
numRuns = 100;
inc = 9;

for i = 1:numRuns
    
    % clear network data from memory each run just in case
    clearvars -except filename sheetName sheetNum imds numRuns inc i;
    
    % split data (80/20) train/test
    % validation = .9 of training
    % validation = testing data
    [imdsTrain,imdsTest] = splitEachLabel(imds,0.8,'randomized');
    [imdsTrain,imdsValidation] = splitEachLabel(imdsTrain,0.9,'randomized');
    
    
    % thought about moving this net design out of loop, to save time..
    % but at same time maybe it is better and more safe to load
    % everything again, instead of reusing in case something doesn't 
    % get changed
    
    
    % load pretrained net
    net = googlenet;
    
    % replace final classification layer for dataset
    lgraph = layerGraph(net);
    numClasses = numel(categories(imdsTrain.Labels));
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    lgraph = replaceLayer(lgraph,'loss3-classifier',newLearnableLayer);
    newClassLayer = classificationLayer('Name','new_classoutput');
    lgraph = replaceLayer(lgraph,'output',newClassLayer);
    
    % augmentation options
    % resizes image too
    inputSize = net.Layers(1).InputSize;
    pixelRange = [-30 30];
    rotRange = [-45 45];
    imageAugmenter = imageDataAugmenter( ...
        'RandXReflection',true,...
        'RandRotation', rotRange,...
        'RandYReflection',true, ...
        'RandXTranslation',pixelRange, ...
        'RandYTranslation',pixelRange);
    augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
        'DataAugmentation',imageAugmenter);
    % resize validation data
    augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
    augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);
    
    % training options
    % slow learning rate for pretrained layers
    %'Plots','training-progress' for visual plot
    options = trainingOptions('sgdm', ...
        'MiniBatchSize',128, ...
        'MaxEpochs',10, ...
        'InitialLearnRate',1e-4, ...
        'Shuffle','every-epoch', ...
        'ValidationData',augimdsValidation, ...
        'ValidationFrequency',3, ...
        'Verbose',false);
    
    disp("Training Data...");
    
    % train network
    netTransfer = trainNetwork(augimdsTrain,lgraph,options);
    
    disp("Training complete.");
    
    % classify on test set
    [YPred] = classify(netTransfer,augimdsTest);
    YTest = imdsTest.Labels;
    
    disp("Calculating Metrics...");
    % calculate metrics

    % confusion matrix
    figure('visible', 'off')
    cm = confusionchart(YTest,YPred);
    cmVals = cm.NormalizedValues;
    
    [Result,ReferenceResult]=confusion.getValues(cmVals);
    
    % Overall Accuracy
    ovAcc = Result.Accuracy;
    
    % CW Accuracy
    cwAcc = ReferenceResult.Accuracy2;
    
    % CW Precision
    cwPrec = ReferenceResult.Precision';
    
    % CW Sensitivity = Recall
    cwSens = ReferenceResult.Sensitivity';
    
    % CW Specificity
    cwSpec = ReferenceResult.Specificity';
    
    % write to excel
    letter = strcat("C",num2str(inc));
    writecell({i, ovAcc, cwAcc, cwPrec, cwSens, cwSpec}, filename, 'Sheet', sheetNum, 'Range', letter);
    inc = inc+1;

    disp(strcat("Done with Run# ", num2str(i)));
end

% done
disp("Completed.");

% end timer
elapsedTime = toc;
writecell({'Wall Time:', elapsedTime, 'seconds'}, filename, 'Sheet', sheetNum, 'Range', 'C5');

disp(['Wall time is: ' num2str(elapsedTime) ' seconds.']);






