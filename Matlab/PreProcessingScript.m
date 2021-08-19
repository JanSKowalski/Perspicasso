
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

numberOfImages = length(imds.Files);

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
