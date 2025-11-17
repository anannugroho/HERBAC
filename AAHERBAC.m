function [phi,BnW,g,Beta,initial,time,error] = AAHERBAC(varargin)
%% Autoadaptive Hybrid Edge and Region Based Active Contour %%
% by          : Anan Nugroho
% last edited : July, 12-2020
% email       : anannugroho@mail.unnes.ac.id
%-------------------------------------------------------------------------%
% syntax      = [phi,g,Beta,initial,time,error] = AAHERBAC(img,flag,point);
% img         = Input image
% flag        = 'box','circle','freehand','poly'
% point       = fixed pixel location input
% phi         = Final level set
% g           = Binary zero level set
% Beta        = Tracking of evolution mode
% initial     = intial local area for edge-based GAC
% time        = Computation cost
% error       = Convergence error evolution
%-------------------------------------------------------------------------%
%-Default customization
maxs    = 250;       %limit evolution
dt      = 5;         %time step
teta    = 0.1;       %for convergence parameter
beta    = 0;         %balloon force ->(+)shrinking & (-)expanding
n       = 10;        %evolution sampling display
ishape  = 'ocircle'; %initial shape('ibox' =inner box,'obox' =outner box,'icircle','ocircle')
rec     = 'show';    %record evolution('show','gif','avi')
%-Regularization parameter
win     = 5;              %window size of binary Gaussian filter
sigma   = 3;              %standar deviation of binary Gaussian filter
% Nb=[0 1 0; 1 1 1;0 1 0];  %window size for median filter
% med=3;                    %median center point
semorf  = strel('disk',2);%structuring element for morphological regularization
%-Check minimal input argument
narginchk(1,5);
%-Gray_double image convertion
img = varargin{1};
validateattributes(img, ...
    {'uint8','uint16','uint32','int8','int16','int32','single','double'},...
    {'real','nonsparse'}, mfilename,'img',1);
%-Gray_double image convertion
img = im2graydouble(img);
%% -Initial LSF
if nargin ==1
    [mask,location] = maskBW(img,'whole');
elseif nargin ==2
    flag = varargin{2};
    [mask,location] = maskBW(img,flag);
elseif nargin ==3
    flag = varargin{2};point = varargin{3};
    if length(point)<=2
        mask = false(size(img));
        for all=1:point
            [msk,location] = maskBW(img,flag);
            mask = mask | msk;
        end
    else
        [mask,location] = maskBW(img,flag,point);
    end
end
phi0 = -2.*mask+1;
%% -Initial alocation
phi = phi0;
g = zeros(size(img));
error = zeros(2,1);Beta = zeros(1,1);
preLength = 0;preArea=0;i=1;
pixel= zeros(1,length(location));
t=cputime;tic
%% -Level set evolution
while i>=0
    %-Save beta during evolution
    Beta(i)= beta;
    %Neumann boundary condition
    phi = NeumannBound(phi);
    [~,absR] = Curvature(phi,g);
    [c1,c2]=FittingAverage(img,phi);
    %% -Hybrid Edge & Region-based Active Contour
    AAHERBAC = (1-abs(beta)).*(img-(c1+c2)/2) + beta*g.*absR;
    %     AAHERBAC = div.*absR + (1-abs(beta)).*(img-(c1+c2)/2) + beta*g.*absR;
    phi = phi + dt*AAHERBAC;
    %% -Gaussian Regularization
    phi = (phi>0)-(phi<0);%making phi [1,-1]
    kernel=fspecial('gaussian',win,sigma);
    phi=conv2(phi,kernel,'same');
    %% -Median Filter Regularization
    %     phi = ordfilt2(phi,med,Nb);% medfilt2(phi,[3 3]) for alternative
    %% -Morphological Regularization
    phi = phi>0;% making phi [0,1]
    phi = imclose(phi,semorf);%closing
    phi = imopen(phi,semorf);%opening
    phi = double(2.*phi-1);%making phi [1,-1]
    %% -Check Convergency
    if beta==0
        [Converge,preArea,preLength,Error]= Convergence(phi,i,absR,preArea,preLength,teta,maxs);
        if Converge
            BnW = phi<=0;
            if nargin==1
                [phi0,Obj,initial] = ObDetection(img,phi,ishape);
                if sum(strcmp(ishape,{'obox','ocircle'}))==1
                    g = 1-Obj;
                    beta = 1;
                elseif sum(strcmp(ishape,{'ibox','icircle'}))==1
                    g = Obj;
                    beta =-1;
                end
            else
                if strcmp(flag,'box')==1
                    loc(1,2) = location(1);            loc(1,1) = location(2);
                    loc(2,2) = location(1)+location(3);loc(2,1) = location(2);
                    loc(3,2) = location(1)+location(3);loc(3,1) = location(2)+location(4);
                    loc(4,2) = location(1);            loc(4,1) = location(2)+location(4);
                    loc      = floor(loc);
                else
                    loc = floor(location);
                end
                initial = location;
                for j=1:length(location)
                    pixel(j) = phi(loc(j,2),loc(j,1));
                end
                if sum(pixel)>=0
                    g = phi>=0;
                    beta =1;
                else
                    g = phi<=0;
                    g = imclearborder(g,4);
                    g = imfill(g,'hole');
                    beta =-1;
                end
            end
            phi= phi0;
        end
    else
        [Converge,preArea,preLength,Error]= Convergence(phi,i,absR,preArea,preLength,teta,maxs);
        if Converge
            break
        end
    end
    error(1,i)= Error(1,i);error(2,i)= Error(2,i);
    %-Show & record evolution
    RecAC(img,phi,i,n,'AAHERBAC',rec);
    %-Next iteration
    i = i+1;
end
error(1,i)= Error(1,i);error(2,i)= Error(2,i);
toc;time = cputime-t;
%% -Final segmentation
figure(1),imagesc(img),colormap gray,hold on,...
    contour(phi0,[0 0],'g','LineWidth',2);%initial
contour(phi,[0 0],'r','LineWidth',3);%last contour
contour(phi, [0 0], 'y','LineWidth',1),hold off;
title(['AAHERBAC on ',num2str(i),' iteration ',num2str(toc),' sec']);
drawnow

figure(2),mesh(-phi),hold on,contour(phi,[0 0],'r','LineWidth',2),hold off;
title(['AAHERBAC on ',num2str(i),' iteration ',num2str(toc),' sec']);

peak = min(max(error(2,:))./(10^3),max(error(1,:)));
figure,hold on,
plot(1:i,Beta(1:i),'g','LineWidth',1);
plot(1:i,error(1,1:i),'b','LineWidth',1),...
    plot(1:i,error(2,1:i)./(10^3),'r','LineWidth',1),...
    xlabel('Iteration'),ylabel('Error');
axis([1 .5*i -1.25 peak+.25]);
legend('Beta','Error Area','Error Length(1/1000)','Location','northeast');
title(['AAHERBAC on ',num2str(i),' iteration ',num2str(toc),' sec']);
hold off

%% ===========================Additional Function=========================%
function img = im2graydouble(img)
%Converts image to grayscale double
[~,~,c] = size(img);
if(isfloat(img)) %image is a double
    if(c==3)
        img = rgb2gray(uint8(img));
    end
else %image is a int
    if(c==3)
        img = rgb2gray(img);
    end
    img = double(img);
end

function [BW, position]= maskBW(img,flag,point)
% img = 2D image
% flag = 'box','circle','freehand','poly'
% point = fixed pixel location input
% BW = binary mask
% position = contour points
%---------------------------------------%
close all;
[~, ~, c]= size (img);
if nargin<2 && isempty(img);
    errordlg('Image not selected');
    return
elseif ~exist('flag','var');
    errordlg('[BW, position]= maskBW(img,?)-->Flag is empty');
    return
elseif c == 3
    img = img(:,:,1);% red channel
end;

if nargin ==2
    figure,imagesc(img),colormap gray;
    hold on, text(.02*size(img,1),.02*size(img,2),...
        'Set contour: click left mouse & hold on','Color','yellow','FontSize',10);
    hold off;
    switch flag
        case  'box'
            himage = imhandles(gcf);
            ROI = imrect(gca);
            position = wait(ROI);
            BW = createMask(ROI, himage(end));
        case 'circle';
            himage = imhandles(gcf);
            ROI = imellipse(gca);
            position = wait(ROI);
            BW = createMask(ROI, himage(end));
        case 'freehand';
            himage = imhandles(gcf);
            ROI = imfreehand(gca);
            position = wait(ROI);
            BW = createMask(ROI, himage(end));
        case 'poly';
            himage = imhandles(gcf);
            ROI = impoly(gca);
            position = wait(ROI);
            BW = createMask(ROI,himage(end));
        case 'whole'
            [p,q] = size(img);
            r = 30;
            m = zeros(round(ceil(max(p,q)/2/(r+1))*3*(r+1)));
            siz = size(m,1);
            sx = round(siz/2);
            i = 1:round(siz/2/(r+1));
            j = 1:round(0.9*siz/2/(r+1));
            j = j-round(median(j));
            m(sx+2*j*(r+1),(2*i-1)*(r+1)) = 1;
            se = strel('disk',r);
            m = imdilate(m,se);
            BW = m(round(siz/2-p/2-6):round(siz/2-p/2-6)+p-1,...
                round(siz/2-q/2-6):round(siz/2-q/2-6)+q-1);
            BW = ~(BW==1); position = r;imagesc(BW), colormap gray;
        otherwise
            errordlg('Can not processed yet in this function','Error input');
    end
    
elseif nargin ==3
    h_im = imagesc(img,[0 255]);colormap gray;
    position = point;
    switch flag
        case  'box'
            e = imrect(gca,point);
            BW = createMask(e, h_im);
        case 'circle';
            %         e = imellipse(gca,point);
            e = imfreehand(gca,point);
            BW = createMask(e, h_im);
        case 'freehand';
            e = imfreehand(gca,point);
            BW = createMask(e, h_im);
        case 'poly';
            %         e = impoly(gca,point);
            e = imfreehand(gca,point);
            BW = createMask(e, h_im);
        case 'whole'
            [p,q] = size(img);
            r = point;
            m = zeros(round(ceil(max(p,q)/2/(r+1))*3*(r+1)));
            siz = size(m,1);
            sx = round(siz/2);
            i = 1:round(siz/2/(r+1));
            j = 1:round(0.9*siz/2/(r+1));
            j = j-round(median(j));
            m(sx+2*j*(r+1),(2*i-1)*(r+1)) = 1;
            se = strel('disk',r);
            m = imdilate(m,se);
            BW = m(round(siz/2-p/2-6):round(siz/2-p/2-6)+p-1,...
                round(siz/2-q/2-6):round(siz/2-q/2-6)+q-1);
            BW = ~(BW==1);imagesc(BW), colormap gray;
        otherwise
            errordlg('Can not processed yet in this function','Error input');
    end
end

function g = NeumannBound(f)
% Make a matric satisfy Neumann boundary condition
[nrow,ncol] = size(f);
g = f;
g([1 nrow],[1 ncol]) = g([3 nrow-2],[3 ncol-2]);
g([1 nrow],2:end-1) = g([3 nrow-2],2:end-1);
g(2:end-1,[1 ncol]) = g(2:end-1,[3 ncol-2]);

function [Kappa,absR] = Curvature(phi,g)
[nx,ny]=gradient(phi);
absR = sqrt(nx.^2+ny.^2);
absR = absR +(absR==0)*eps;
if nargin<2
    [nxx1,~]=gradient(nx./absR);
    [~,nyy1]=gradient(ny./absR);
    Kappa=nxx1+nyy1;
elseif nargin == 2
    [nxx1,~]=gradient(g.*nx./absR);
    [~,nyy1]=gradient(g.*ny./absR);
    Kappa=nxx1+nyy1;
else
    errordlg('I do not understand what you want!','Error Curvature');
end

function [c1,c2] = FittingAverage(varargin)
% [c1,c2] = FittingAverage(img)
%------------------------------------------------------------------------%
narginchk(2,5);
img = varargin{1};
validateattributes(img,{'uint8','uint16','uint32','int8','int16',...
    'int32','single','double'},{'real','nonsparse'}, mfilename,'img',1);

phi = varargin{2};
validateattributes(phi,{'single','double'},{'real','nonsparse'},...
    mfilename,'phi',2);

Hphi = Heaviside(phi);
c1 = sum(sum(img.*Hphi))/(sum(sum(Hphi)));
c2 = sum(sum(img.*(1-Hphi)))/(sum(sum((1-Hphi))));

function [Converge,Area,Length,Error]= Convergence(Phi,i,absR,preArea,preLength,Teta,Max)
%-Check Convergency
Area=sum(sum(Phi<0));%counting inner area/pixel
Error(1,i)=abs(Area-preArea);
dPhi = Dirac(Phi);
Length = sum(sum(absR.*dPhi));%counting contour length/pixel
Error(2,i)=abs(Length-preLength);
if (Error(1,i)<= Teta && Error(2,i)<= Teta) || i==Max
    Converge = true;
else
    Converge = false;
end

function H = Heaviside(varargin)
narginchk(1,3);
phi=varargin{1};
if length(varargin)== 1
    H = (1/pi)*atan(phi)+0.5;
elseif length(varargin)== 2
    epsilon= varargin{2};
    H = 0.5*(1+(2/pi)*atan(phi./epsilon));
elseif length(varargin)== 3
    epsilon= varargin{2};type= varargin{3};
    switch type
        case 1
            H=0.5*(1+(phi./epsilon)+(1/pi)*sin(pi.*phi./epsilon));
            b = (phi<=epsilon) & (phi>=-epsilon);
            c = (phi>epsilon);
            H = b.*H+c;%Heaviside functional 1
        case 2
            H = 0.5*(1+(2/pi)*atan(phi./epsilon));%Heaviside functional 2
        case 3
            H = 0.5*(1+phi);%Heaviside functional 3
        case 4
            H = (phi<=0);%Heaviside functional 4
        otherwise
            errordlg('I do not understand what you want!','Error Input');
    end
else
    errordlg('I do not understand what you want!','Error Input');
end

function D = Dirac(varargin)
narginchk(1,3);
phi=varargin{1};
if length(varargin)== 1
    D = (1/pi)./(1+phi.^2);
elseif length(varargin)== 2
    epsilon= varargin{2};
    D = (epsilon/pi)./(epsilon^2+phi.^2);
elseif length(varargin)== 3
    epsilon= varargin{2};type= varargin{3};
    switch type
        case 1
            D=(1/2/epsilon)*(1+cos(pi.*phi./epsilon));
            b = (phi<=epsilon) & (phi>=-epsilon);
            D = D.*b; % Dirac functional 1
        case 2
            D = (epsilon/pi)./(epsilon^2+phi.^2);% Dirac functional 2
        otherwise
            errordlg('I do not understand what you want!','Error Input');
    end
else
    errordlg('I do not understand what you want!','Error Input');
end

function [] = RecAC(img,phi,iteration,sampling,FileName,Record)
% Record = 'gif';'avi'
% ----------------------------------------------------------------------- %
if iteration==1 || mod(iteration,sampling)==0
    %-Display Evolution
    figure(1),...
        imagesc(img),colormap gray,hold on,...
        contour(phi,[0 0],'r','LineWidth',3);
    contour(phi, [0 0], 'y','LineWidth',1),hold off;
    title([FileName,' on ',num2str(iteration),' iteration']);drawnow
    
    figure(2),mesh(-phi),hold on,...
        contour(phi,[0 0],'r','LineWidth',3);
    contour(phi, [0 0], 'y','LineWidth',1),hold off;
    title([FileName,' on ',num2str(iteration),' iteration']);drawnow
    
    %-Record Evolution .gif file
    if ischar(Record) && strcmp(Record,'gif')
        frame1 = getframe(figure(1));
        im1 = frame2im(frame1);
        [imind1,cm1] = rgb2ind(im1,256);
        
        frame2 = getframe(figure(2));
        im2 = frame2im(frame2);
        [imind2,cm2] = rgb2ind(im2,256);
        axis tight manual;%ensures that getframe() returns a consistent size
        
        if iteration==1 || iteration == sampling
            imwrite(imind1,cm1,[FileName,'a.gif'],'gif', 'Loopcount',inf);
            imwrite(imind2,cm2,[FileName,'b.gif'],'gif', 'Loopcount',inf);
        else
            imwrite(imind1,cm1,[FileName,'a.gif'],'gif','WriteMode','append');
            imwrite(imind2,cm2,[FileName,'b.gif'],'gif','WriteMode','append');
        end
        
        %-Record Evolution .avi file
    elseif ischar(Record) && strcmp(Record,'avi')
        vidObj_a = VideoWriter([FileName,'a.avi']);
        open(vidObj_a);
        writeVideo(vidObj_a,getframe(figure(1)));
        vidObj_b = VideoWriter([FileName,'b.avi']);
        open(vidObj_b);
        writeVideo(vidObj_b,getframe(figure(2)));
    end
    pause(.01);
end

function [phi0,Obj,initial,data] = ObDetection(img,phi,ishape)
%% -Morphological operation
g = phi<=0;
% se= strel('disk',5);%Structuring element
% Obj = imopen(g,se);
Obj = imclearborder(g,4);
Obj = imfill(Obj,'hole');
%% -Region properties
% properties = regionprops(Obj,'all');
properties = regionprops(Obj,'Area','Centroid','BoundingBox','ConvexHull',...
    'MajorAxisLength','MinorAxisLength');
sorting    = sort([properties.MinorAxisLength],'descend');
index      = [properties.MinorAxisLength]/sorting(1)>=.65;
data = properties(index);
%% -Object localization
BW={1};%initialization
masking = false(size(g));%initial masking
num_Obj =length(data);
h_im = imagesc(img);colormap gray;
hold on,
for i=1:num_Obj
    ct = data(i).Centroid;
    bb = data(i).BoundingBox;
    dMax  = data(i).MajorAxisLength;%diameter maximal
    %-masking
    switch ishape
        case 'obox'
            x = bb(1)-.75*(ct(1)-bb(1));
            y = bb(2)-.75*(ct(2)-bb(2));
            P = bb(3)+1.6*(bb(1)-x);
            L = bb(4)+1.6*(bb(2)-y);
            post = imrect(gca,[x,y,P,L]);
            mask = createMask(post,h_im);
            delete(post);
        case 'ibox'
            x = bb(1)+.5*(ct(1)-bb(1));
            y = bb(2)+.5*(ct(2)-bb(2));
            P = bb(3)-1.25*(bb(1)-x);
            L = bb(4)-1.25*(bb(2)-y);
            post = imrect(gca,[x,y,P,L]);
            mask = createMask(post,h_im);
            delete(post);
        case 'ocircle'
            [Height,Wide] = size(g);
            [xx,yy] = meshgrid(1:Wide,1:Height);
            X = ct(1);% center coordinat X
            Y = ct(2);% center coordinat Y
            r = floor(.56*dMax);% radius R
            sdf = (sqrt(((xx - X).^2 + (yy - Y).^2 )) - r);% circle equation
            mask = sdf<=0;
        case'icircle'
            [Height,Wide] = size(g);
            [xx,yy] = meshgrid(1:Wide,1:Height);
            X = ct(1);% center coordinat X
            Y = ct(2);% center coordinat Y
            r = floor(.06*dMax);% radius R
            sdf = (sqrt(((xx - X).^2 + (yy - Y).^2 )) - r);% circle equation
            mask = sdf<=0;
    end
    masking = or(masking,mask);
    %-plot centroid
    plot(ct(1),ct(2), '-m+')
    a=text(ct(1)+15,ct(2), strcat('X: ', num2str(round(ct(1))),'    Y: ', num2str(round(ct(2)))));
    set(a, 'FontName', 'Arial', 'FontWeight', 'bold', 'FontSize', 10, 'Color', 'yellow');
    BW{i}=and(g,mask);
end
phi0 = -2.*masking+1;
initial0 = contour(masking,'r','LineWidth',2)';initial0 = floor(initial0);
A = initial0(:,1)>1 & initial0(:,1)< size(phi,1);
B = initial0(:,2)>1 & initial0(:,2)< size(phi,2);
idx = A & B;
a = initial0(:,1);b = initial0(:,2);
initial =[a(idx) b(idx)];
% figure(3),imshow(BW)
hold off
