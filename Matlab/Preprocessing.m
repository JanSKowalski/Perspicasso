
%-----------------------------------------------------
%
%   Pre-Processing
%
%-----------------------------------------------------

% Make all images same type format
% %Get all images in same data Object
% for Seawulf '/gpfs/home/sushahid/ESE 440/images'
% for local 'C:\Users\sunsu\Downloads\images'
imds = imageDatastore('C:\Users\sunsu\Downloads\images', 'IncludeSubFolders',true,'LabelSource','foldernames');

% img = readimage(imds,1);
% sz = size(img);

% numberOfImages = length(imds.Files);

% % I Checked maxRows = 1600
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
% % Resize All, this writes image again to file
% for k = 1 : numberOfImages
%   % Get the input filename. 
%   inputFileName = imds.Files{k};
%   img = imread(inputFileName);
%   img = imresize(img, [250 250]);
%   imwrite(img, inputFileName);
% end

