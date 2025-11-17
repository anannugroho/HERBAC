function [phi,BnW,g,Beta,initial,time,error] = AACMR(varargin)
%% Autoadaptive Active Contour with Morphological Regularization %%
% by    : Anan Nugroho
% date  : July, 8-2020
% email : anannugroho@mail.unnes.ac.id
%-------------------------------------------------------------------------%
% syntax = [phi,g,Beta,initial,time,error] = AACMR(img);
% img    = Input image
% phi    = Final level set
% g      = Binary zero level set
% Beta   = Tracking of evolution mode
% initial= intial local area for edge-based GAC
% time   = Computation costRecAC(img,phi,i,2,'AACMR',rec);
% error  = Convergence error evolution
%-------------------------------------------------------------------------%
%-default parameter
maxs    = 500;       %limit evolution
dt      = 5;         %time step   
teta    = 0.5;       %for convergence parameter
beta    = 0;         %balloon force ->(+)shrinking & (-)expanding
% n       = 10;        %evolution sampling display
% win     = 5;         %window size of binary gaussian filter
% sigma   = 3;         %standar deviation of binary gaussian filter
% flag    = 'icircle'; %initial shape('ibox' =inner box,'obox' =outner box,'icircle','ocircle')
rec     = 'show';    %record evolution('show','gif','avi')

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
[Height,Wide] = size(img);
[xx,yy] = meshgrid(1:Wide,1:Height);
X = floor(Wide/2);%center coordinat
Y = floor(Height/2);%center coordinat
R = floor(min(.2*Wide,.2*Height));%radius length
phi0 = (sqrt(((xx - X).^2 + (yy - Y).^2 )) - R);%circle equation
phi0 = sign(phi0).*2;
%% Define Morphological Parameters
SEgMorfStep = strel('disk',1);
% Define MedianDisc Parameters
% Nb=[0 1 0; 1 1 1;0 1 0];
% Nb=[1 1 1; 1 1 1;1 1 1];
% med=5;
%% -Initial alocation
phi = phi0;
g = zeros(size(img));
error = zeros(2,1);Beta = zeros(1,1);
preLength = 0;preArea=0;i=1;
t=cputime;tic
%% -Level set evolution
while i>=0
    %-Save beta during evolution
    Beta(i)= beta;
    %Neumann boundary condition
    phi = NeumannBound(phi);    
    [div,absR] = Curvature(phi,g);             
    [c1,c2]=FittingAverage(img,phi);
    %% -Autoadaptive Active Contour Hybrid Model 
%     AACMR = (1-abs(beta)).*(img-(c1+c2)/2) + beta*g.*absR;
    AACMR = div.*absR + (1-abs(beta)).*(img-(c1+c2)/2) + beta*g.*absR;
%    phi = phi + dt*AACMR;
%    %% -Median Filtering Regularization
   phi = medfilt2((phi + dt*AACMR),[3 3]);
%     %% -Gaussian Regularization    
%     phi = (phi>0)-(phi<0);%making phi [1,-1]
%     kernel=fspecial('gaussian',win,sigma);
%     phi=conv2(phi,kernel,'same');
    %% -Morphological Regularization
    phi = phi>0;
    phi = imdilate(imerode(phi,SEgMorfStep),SEgMorfStep);
    phi = imerode(imdilate(phi,SEgMorfStep),SEgMorfStep);
    phi = double(2.*phi-1);
    %% -Check Convergency
    if beta==0  
        RecAC(img,phi,i,2,'AACMR',rec);
       [Converge,preArea,preLength,Error]= Convergence(phi,i,absR,preArea,preLength,teta,maxs);       
       if Converge
        BnW = phi<=0;   
        [phiShrink,gShrink,initialShrink] = ObDetection(img,phi,'ocircle');% shrinking
        [phiExpand,gExpand,initialExpand] = ObDetection(img,phi,'icircle');% expanding
        initial = {initialShrink,initialExpand};
        if mod(i,2)==0 %iterasi genap            
            g = 1-gShrink;
            beta = 1;
            phi= phiShrink;
        else %iterasi ganjil            
            g = gExpand;
            beta =-1;
            phi= phiExpand;
        end        
       end       
    elseif beta==1 % shrinking     
        phiShrink = phi;
        RecAC(img,phiShrink,i,10,'AACMR',rec);
        % switching
        g = gExpand;
        beta =-1;
        phi = phiExpand;
        [Converge,preArea,preLength,Error]= Convergence(phi,i,absR,preArea,preLength,teta,maxs);
        if Converge
            break
        end
    elseif beta==-1 % expanding    
        phiExpand = phi;
        RecAC(img,phiExpand,i,11,'AACMR',rec);
        % switching
        g = 1-gShrink;
        beta =1;
        phi = phiShrink;
        [Converge,preArea,preLength,Error]= Convergence(phi,i,absR,preArea,preLength,teta,maxs);
        if Converge
            break
        end
    end
    error(1,i)= Error(1,i);error(2,i)= Error(2,i);
    %-Show & record evolution
%     RecAC(img,phi,i,n,'AACMR',rec);
    %-Next iteration
    i = i+1;
end
    error(1,i)= Error(1,i);error(2,i)= Error(2,i);
    toc;time = cputime-t;
%% -Final segmentation
figure(1),imagesc(img),colormap gray,hold on,...
contour(phi0,[0 0],'g','LineWidth',2);%initial
contour(phiExpand,[0 0],'r','LineWidth',3);%last contour
% contour(phiExpand,[0 0], 'y','LineWidth',1);
% contour(phiShrink,[0 0],'r','LineWidth',3);%last contour
contour(phiShrink,[0 0], 'y','LineWidth',1),hold off;
title(['AACMR on ',num2str(i),' iteration ',num2str(toc),' sec']);
drawnow

figure(2),mesh(-phi),hold on,contour(phi,[0 0],'r','LineWidth',2),hold off;
title(['AACMR on ',num2str(i),' iteration ',num2str(toc),' sec']);

peak = min(max(error(2,:))./(10^3),max(error(1,:)));
figure,hold on,
plot(1:i,Beta(1:i),'g','LineWidth',1); 
plot(1:i,error(1,1:i),'b','LineWidth',1),...
plot(1:i,error(2,1:i)./(10^3),'r','LineWidth',1),...
xlabel('Iteration'),ylabel('Error');
axis([1 .5*i -1.25 peak+.25]);
legend('Beta','Error Area','Error Length(1/1000)','Location','northeast');
title(['AACMR on ',num2str(i),' iteration ',num2str(toc),' sec']);
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
     contour(phi,[0 0], 'y','LineWidth',1),hold off;
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

function [phi0,Obj,initial,data] = ObDetection(img,phi,flag)
%% -Morphological operation
g = phi<=0;
se= strel('disk',5);%Structuring element    
Obj = imopen(g,se);
Obj = imclearborder(Obj,4);
Obj = imfill(Obj,'hole');
%% -Region properties
% properties = regionprops(Obj,'all');
properties = regionprops(Obj,'Area','Centroid','BoundingBox','ConvexHull',...
             'MajorAxisLength','MinorAxisLength');
sorting    = sort([properties.MinorAxisLength],'descend');
index      = [properties.MinorAxisLength]/sorting(1)>=.7;
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
    switch flag
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
       r = floor(.45*dMax);% radius R
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
