%% NOTES
% The CUDA algorithm manage the object borders at the map border as they
% were countorned by non-object pixels. This mean that a 1-pixel object on
% the map border has perimeter 4 and not 3. This should be the same
% approach used by the MatLab "bwperim" built-in function.
% This choice was taken to solve the current trouble, where the *1 must be
% accounted properly:
% 
%   | 0 0 0
%   | 1 1 1
%   |*1 1 1
%   | 1 1 1
%   | 0 0 0
% 
% The *1 pixel would contribute to perimeter with "zero" value, while the
% most accurate for my purpose is a value equal to "one". This can happen
% only considering that outside the map we assume every pixel is zero, as
% the following:
% 
% 0 | 0 0 0
% 0 | 1 1 1
% 0 |*1 1 1
% 0 | 1 1 1
% 0 | 0 0 0
% 
% This pattern is valid for both East–Ovest and North–South searching
% directions.

%% PARs
WDIR        = '/home/giuliano/git/cuda/perimeter';
print_intermediate_arrays = false;

% **IN**
% FIL_ROI		= '/home/giuliano/git/cuda/perimeter/data/imp_mosaic_char_2006_cropped_64kpixels_roi.tif';
% FIL_BIN		= '/home/giuliano/git/cuda/perimeter/data/imp_mosaic_char_2006_cropped_64kpixels.tif';
FIL_ROI     = '/home/giuliano/git/cuda/fragmentation/data/lodi1954_roi.tif';
FIL_BIN 	= '/home/giuliano/git/cuda/fragmentation/data/lodi1954.tif';
%  **OUT**
FIL_PERI	= '/home/giuliano/git/cuda/perimeter/data/PERI-cuda.tif';

% **kernels**
kern{1} 	= 'gtranspose';
kern{2} 	= 'tidx2_ns';
kern{3}		= 'gtranspose';
kern{4} 	= 'tidx2_ns';
kern{5}     = 'reduce6_nvidia';
%% set ROI as all "ones"
% info = geotiffinfo( FIL_BIN );
% ROI = logical(ones( info.Height, info.Width )); %#ok<LOGL>
% geotiffwrite(FIL_ROI,ROI,info.RefMatrix, ...
%     'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag)
%% load
ROI = geotiffread( FIL_ROI );
BIN = geotiffread( FIL_BIN );
%% Size required on GPU device
Double_x1 = 8;%bytes
% (64*10^6 * Double_x1) / 1024^3    %GB
Char_1x = 1;%bytes
% (64*10^6 * Char_1x) / 1024^3      %GB

% Perimeter cuda kernels require 2*Dobles + 2*uChar maps of size=numel(BIN)
TOT_bytes_cuda_perimeter = (numel(BIN) * Char_1x)*2 + (numel(BIN) * Double_x1)*2;
% TOT_bytes_cuda_perimeter = (64*10^6 * Char_1x)*2 + (64*10^6 * Double_x1)*2

fprintf('Size required on GPU:%10.3f MB\n',TOT_bytes_cuda_perimeter/1024^2)
%% matlab perimeter
tic
mlPERI = bwperim(BIN, 4);
% mlPERI = sum(peri_BIN(:));
myTOC = toc;
%% load cuda PERIMETER
cuPERI = geotiffread( FIL_PERI );
%% check for differences
borders(1) = sum(mlPERI(:));
borders(2) = sum(logical(cuPERI(:)));
DIFF = logical(cuPERI)-mlPERI;

fprintf('Number of border pixels:\tml[%5d], cu[%5d]\n',borders)
fprintf('pixel-by-pixel difference:\tdiff[%5d]\n',sum(DIFF(:)))
fprintf('Cuda computed perimeter:\t%d*CELLSIZE\n',sum(cuPERI(:)))
% 

%% understand geometry of tidx2_ns kernel
[HEIGHT,WIDTH] = size(BIN);
THREADS = 1024;
mask_len = 40;

gdx_kTidx2NS_t 	= (mod(HEIGHT, THREADS)>0) + (HEIGHT / (THREADS));
gdy_kTidx2NS_t 	= (mod(WIDTH, mask_len)>0) + floor(WIDTH/ mask_len);


%% check intermediate maps
if print_intermediate_arrays

% _______PARs__________
chkKERN = 3;% 1-4
% _______PARs__________

% compute border pixels in MatLab:
mlPERI = bwperim(BIN, 4);

ImapFIL = fullfile( WDIR,'data',['-',num2str(chkKERN),'-',kern{chkKERN},'.tif'] );
Imap = geotiffread( ImapFIL );

switch chkKERN
    case 1
        Imap = Imap';
        DIFF = BIN-logical(Imap);
        sDiff= sum(DIFF(:));
        fprintf('pixel-by-pixel difference:\tdiff[%5d]\n',sDiff)
    case 2
        Imap = Imap';
        DIFF = logical(Imap) & ~mlPERI;
        sDiff= sum(DIFF(:));
        fprintf('Border pixels found in cuda but not in MatLab:\tdiff[%5d]\n',sDiff)
    case 3
        ImapFIL = fullfile( WDIR,'data',['-',num2str(2),'-',kern{2},'.tif'] );
        Imap2 = geotiffread( ImapFIL );
        DIFF = Imap2'-Imap;
        sDiff= sum(DIFF(:));
        fprintf('pixel-by-pixel difference:\tdiff[%5d]\n',sDiff)
    case 4
    otherwise
        error('Error :: set chkKERN properly!')
end

end

