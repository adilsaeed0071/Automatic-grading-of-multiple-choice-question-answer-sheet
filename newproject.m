clc
clear all
close all
% system('python image.py');
A = imread('scan18.jpg');
 B=imresize(A,0.5);
 A=B;
Igrey = rgb2gray(A);
K = imadjust(Igrey,[0.3 0.9],[]);
A1= im2bw(K);
A5 = imcomplement(A1);
Rmin = 20;
Rmax = 60;
[centers, radii] = imfindcircles(A5,[Rmin Rmax],'Sensitivity',0.8,'EdgeThreshold',0.1);
%   circles1= viscircles(centers, radii);
%  circles1= viscircles(centers, radii);
  imshow(A);
[ymin,ind]=min(centers(:,2));

yminx=centers(ind,1);
%  circles1= viscircles(centers(ind,:), radii);
[xmin,ind2]=min(centers(:,1));
%  circles2= viscircles(centers(ind2,:), radii);
xminy=centers(ind2,2);
[ymax,ind3]=max(centers(:,2));
%  circles3= viscircles(centers(ind3,:), radii);
ymaxx=centers(ind3,1);
[xmax,ind4]=max(centers(:,1));
xmaxy=centers(ind4,2);
%  circles4= viscircles(centers(ind4,:), radii);
centerx= centers(:,1);
centery= centers(:,2);
[xyminsum,xymin_index]= min(centers(:,1)+centers(:,2));
xymin=centers(xymin_index,:);
xyminx= xymin((1),1);
xyminy=xymin((1),2);
centerx= centers(:,1);
findmin=xmin+min(radii)
f=find (centerx<findmin);
g=centers(f,:);
h=g(:,2);
[b,c]=max(h);
pointx=g(c);
pointy=b;
changex= xyminx-pointx;
changey=pointy-xyminy;
thetha= changex/changey;
anglered=atan(thetha);
anglezero=radtodeg(anglered)
if (anglezero < -1 || anglezero > 1 )
    
if (xmaxy>xminy)
   
x1=xmin;
y1=xminy;
x2=ymaxx;
y2=ymax;
xdis= x1-x2;
ydis= y2-y1;
thetha= xdis/ydis;
anglered=atan(thetha);
angle=radtodeg(anglered)
   
end
 if (xmaxy<xminy)
    x1=yminx;
y1=ymin;
x2=xmin;
y2=xminy;
xdis= x1-x2;
ydis= y2-y1;
thetha= xdis/ydis;
anglered=atan(thetha);
angle=radtodeg(anglered);
f=45566
 end
Irot = imrotate(A,angle);
Mrot = ~imrotate(true(size(A)),angle);
Irot(Mrot&~imclearborder(Mrot)) = 255;
A=Irot;
end

A1= im2bw(A);
A2 = imcomplement(A1);
Rmin = 20;
Rmax = 65;
[centers1, radii] = imfindcircles(A2,[Rmin Rmax],'Sensitivity',0.8,'EdgeThreshold',0.3);
%   circles= viscircles(centers1, radii);

%% White mask
 
% if centers1>10
 [M,N]=size (centers1);
 %160 2 rows
 %> greater 160 and <240 4 rows
 
 if M < 160
 r1=M/2;
 centerx=centers1(1:M,1);
 centerx= sort(centerx); 
 recta= centerx(1:r1);
 rectb= centerx(r1+1:M);
 centery=centers1(1:M,2);
  extra= max(radii);
 
  
 mina= min(recta) - extra;
 minb= min(rectb) - extra;
 miny= min(centery) - 3*extra ;
 maxa= max(recta)+ extra;
  maxb= max(rectb)+ extra;
 maxy= max(centery)+ extra;
 widtha= maxa-mina;
 widthb= maxb-minb;
 length= maxy-miny;
rect1= [mina, miny, widtha,length];
rect2= [minb, miny, widthb,length];
  RGB1 = insertShape(A,'FilledRectangle',[mina, miny, widtha,length],'Color','white','Opacity',1);
 
  RGB2 = insertShape(RGB1,'FilledRectangle',[minb, miny, widthb,length],'Color','white','Opacity',1);
  %figure('name','white rectangles');

%    imshow(RGB2);
 end
 if M > 161
r1=M/4;
r2= 2*r1;
r3=3*r1;

 centerx=centers1(1:M,1);
 centerx= sort(centerx); 
 recta= centerx(1:r1);
 rectb= centerx(r1+1:r2);
 rectc= centerx(r2+1:r3);
 rectd= centerx(r3+1:M);
 centery=centers1(1:M,2);
  extra= max(radii);
 
  zerox=0;
  zeroy=0;
 mina= min(recta) - extra;
 minb= min(rectb) - extra;
 minc= min(rectc) - extra;
 mind= min(rectd) - extra;
 miny= min(centery) - 3*extra ;
 maxa= max(recta)+ extra;
  maxb= max(rectb)+ extra;
  maxc= max(rectc)+ extra;
  maxd= max(rectd)+ extra;
 maxy= max(centery)+ extra;
 widtha= maxa-mina;
 widthb= maxb-minb;
 widthc= maxc-minc;
 widthd= maxd-mind;
 length= maxy-miny;
rect1= [mina, miny, widtha,length];
rect2= [minb, miny, widthb,length];
rect3= [minc, miny, widthc,length];
rect4= [mind, miny, widthd,length];
 
 RGB1 = insertShape(A,'FilledRectangle',[zerox ,zeroy, maxd+extra,miny],'Color','white','Opacity',1);
  RGB2 = insertShape(RGB1,'FilledRectangle',[minb, miny, widthb,length],'Color','white','Opacity',1);
 
 RGB3 = insertShape(RGB2,'FilledRectangle',[minc, miny, widthc,length],'Color','white','Opacity',1);
 
  RGB4 = insertShape(RGB3,'FilledRectangle',[mind, miny, widthd,length],'Color','white','Opacity',1);
  RGB5 = insertShape(RGB4,'FilledRectangle',[mina, miny, widtha,length],'Color','white','Opacity',1);
   
%   figure('name','white rectangles 4');
RBG5=imcomplement(RGB5);
  
     
 end   
%  rectangle('Position', [minx, miny, width,length]);
% imshow(RGB1);
 A3 = imcomplement(RGB1);
%   figure('name','light and dark');
  marker = imerode(A3, strel('disk',3,0));
Iclean = imreconstruct(marker, A3);
%% Shaded and unshaded circles

% A3 = im2bw(Iclean,0.3);
% % close all
   imshow(Iclean);
  A3=Iclean;
Rmin = 13;
Rmax = 65;
 [centersBright, radiiBright] = imfindcircles(A3,[Rmin Rmax],'ObjectPolarity','bright','Sensitivity',0.85,'EdgeThreshold',0.2);
  [centersDark, radiiDark] = imfindcircles(A3,[Rmin Rmax],'ObjectPolarity','dark','Sensitivity',0.8,'EdgeThreshold',0.6);
 %Acommon = intersect(centersDark,centersDark);
Bright=round(centersBright);

Dark=round(centersDark);

%  Bright= setxor(Bright,common,'rows');
% % com= viscircles(common, rad);
%Arr3 = setxor(centersDark,Acommon);
   bright= viscircles(centersBright, radiiBright);
  %dark=viscircles(centersDark, radiiDark,'LineStyle','--');
 


%% OCR implementation
 
I = RGB5;
greyimg = rgb2gray(I);

%   figure;
%   imshow(I)
% Run OCR on the image
results = ocr(I);

a= results.Text;
 BW = im2bw(I,0.1);
 
%   figure('Name','Binary'); 
%    imshowpair(I,BW,'montage');
% Remove keypad background.
%   Icorrected = imtophat(I,strel('disk',20));
% 
%  BW1 = im2bw(Icorrected,0.2);
% 
%   figure ('Name','Image_corrected'); 
%  imshowpair(Icorrected,BW1,'montage');
% % Perform morphological reconstruction and show binarized image.
 marker = imerode(greyimg, strel('line',8,70));
Iclean = imreconstruct(marker, greyimg);

% 
% BW2 = im2bw(Iclean,0.5);
BW2= Iclean;
 
%   figure ('Name','imrode'); 
%    imshowpair(Iclean,I,'montage');
results = ocr(BW2,'TextLayout','Block');
% 
b=results.Text;
%  The regular expression, '\d', matches the location of any digit in the
%  recognized text and ignores all non-digit characters.
 regularExpr = '\d';
% 
% % Get bounding boxes around text that matches the regular expression
 bboxes = locateText(results,regularExpr,'UseRegexp',true);
% 
 digits = regexp(results.Text,regularExpr,'match');
 cc=digits;
c=results.Text;
% % draw boxes around the digits
 Idigits = insertObjectAnnotation(I,'rectangle',bboxes,digits);
% c

%    figure('Name','drawboxes'); 
%   imshow(Idigits);
% % Use the 'CharacterSet' parameter to constrain OCR
 results = ocr(BW2, 'CharacterSet','1234567890','TextLayout','block');
% 
d=results.Text;
% % Sort the character confidences.
 [sortedConf, sortedIndex] = sort(results.CharacterConfidences, 'descend');
% 
% Keep indices associated with non-NaN confidences values.
 indexesNaNsRemoved = sortedIndex( ~isnan(sortedConf) );
% 
% % Get the top ten indexes.
[m,n]= size(bboxes);
 topTenIndexes = indexesNaNsRemoved(m);
% 
% % Select the top ten results.
 digits = num2cell(results.Text(topTenIndexes));
 bboxes = results.CharacterBoundingBoxes(topTenIndexes, :);
e=results.Words;

 Idigits = insertObjectAnnotation(I,'rectangle',bboxes,digits);
% 
%    figure('Name','chracter set'); 
%   imshow(Idigits);
Iocr = insertObjectAnnotation(I, 'rectangle',results.WordBoundingBoxes,results.WordConfidences);
%[x y width height],wordbounding box    
%   figure ('name','confidence'); 
%     imshow(Iocr);
 % Find characters with high confidence.
highConfidenceIdx = results.CharacterConfidences > 0.5;

% Get the bounding box locations of the high confidence characters.
highConfBBoxes = results.CharacterBoundingBoxes(highConfidenceIdx, :);

% Get confidence values.
highConfVal = results.CharacterConfidences(highConfidenceIdx);

% Annotate image with character confidences.
str      = sprintf('confidence = %f', highConfVal);
Ilowconf = insertObjectAnnotation(Iocr,'rectangle',highConfBBoxes,str);
% 
%  figure;
%  imshow(Ilowconf);

[j,k]= size(results.WordConfidences);
i=0;
boxes=results.WordBoundingBoxes;
for  v=1:j
   
     if  results.WordConfidences(v) < 0.5
   
      disp(v)
    (e(v))
      e(v+i)= [];
      boxes(v+i,:)=[];
      
       i=i-1;
    
     end
    
end
    d= str2double(e)
    j=1
   for i=1:4:60
      
    d(i)
    i
    disp(j)
   if d(i) ~= j
    d(i)=j
   end
  disp ( d(i+1))
   j+15
   if d(i +1) ~= j+15
       d(i + 1) = j+15
   end
   disp ( d(i+2))
   j+30
   if d(i+ 2) ~= j+30
       d(i + 2) = j+ 30
   end
   disp ( d(i+3))
   j+45
     if d(i+ 3) ~= j+45
       d(i + 3) = j+ 45
   end
    j=j+1
end

  

[j,k]= size(d);

word= d
word(:,2)=boxes(:,1);
word(:,3)=boxes(:,2);
word=sortrows(word,1)
%% finding options
centerx = centers1(:,1);
centery = centers1(:,2);
ymin1=min(centery);
findminy = ymin1 +max(radii);
t=find (centery<findminy);
options=centerx(t);

options = sort(options);
[totalrows,num]=size(options);
if (totalrows > 15 ) && (totalrows < 20)
    
A1min=options(1)-max(radii);
A1max=options(1)+max(radii);

B1min=options(2)-max(radii);
B1max=options(2)+max(radii);

C1min=options(3)-max(radii);
C1max=options(3)+max(radii);

D1min=options(4)-max(radii);
D1max=options(4)+max(radii);

A2min=options(5)-max(radii);
A2max=options(5)+max(radii);

B2min=options(6)-max(radii);
B2max=options(6)+max(radii);

C2min=options(7)-max(radii);
C2max=options(7)+max(radii);

D2min=options(8)-max(radii);
D2max=options(8)+max(radii);

A3min=options(9)-max(radii);
A3max=options(9)+max(radii);

B3min=options(10)-max(radii);
B3max=options(10)+max(radii);

C3min=options(11)-max(radii);
C3max=options(11)+max(radii);

D3min=options(12)-max(radii);
D3max=options(12)+max(radii);

A4min=options(13)-max(radii);
A4max=options(13)+max(radii);

B4min=options(14)-max(radii);
B4max=options(14)+max(radii);

C4min=options(15)-max(radii);
C4max=options(15)+max(radii);

D4min=options(16)-max(radii);
D4max=options(16)+max(radii);

end
if totalrows< 15 
    
A1min=options(1)-max(radii);
A1max=options(1)+max(radii);

B1min=options(2)-max(radii);
B1max=options(2)+max(radii);

C1min=options(3)-max(radii);
C1max=options(3)+max(radii);

D1min=options(4)-max(radii);
D1max=options(4)+max(radii);

A2min=options(5)-max(radii);
A2max=options(5)+max(radii);

B2min=options(6)-max(radii);
B2max=options(6)+max(radii);

C2min=options(7)-max(radii);
C2max=options(7)+max(radii);

D2min=options(8)-max(radii);
D2max=options(8)+max(radii);
      
    
end

%% 
T = readtable('answer2.xls');
S = vartype('string');
answerkey=table2array(T);
 tf= isempty(centersBright)
 if (tf == 1)
    disp ('no circle detected')
    return
 end
F=find (centersBright(:,1)< D1max);
firstcolumn = centersBright(F,:);
 firstcolumn= sortrows(firstcolumn,2)
G=find (centersBright(:,1)< D2max & centersBright(:,1)> D1max);
secondcolumn = centersBright(G,:);
 secondcolumn= sortrows(secondcolumn,2);
H=find (centersBright(:,1)< D3max & centersBright(:,1)> D2max);
thirdcolumn = centersBright(H,:);
 thirdcolumn= sortrows(thirdcolumn,2);
 J=find (centersBright(:,1)< D4max & centersBright(:,1)> D3max);
fourthcolumn = centersBright(J,:);
 fourthcolumn= sortrows(fourthcolumn,2);

    
 %% compare location of digits and marked circles
 [j, k] = size(firstcolumn); %generally j=15
 %if j= 16 
 for i= 1:15
  
  if (i>j)
  firstcolumn(i,:)= [0, 0];
 
   end
 if (firstcolumn(i,2)- word(i,3) > 35) % comparing with location
     
b = [0,0]; 
k = i-1; %row position, can be 0,1,2 or 3 in this case
firstcolumn = [firstcolumn(1:k,:); b; firstcolumn(k+1:end,:)]
 
 end
 end
  [j, k] = size(secondcolumn);
  for i= 1:15
 
  if (i>j)
   secondcolumn(i,:)= [0, 0];
 
     
   end
 if (secondcolumn(i,2)- word(i+15,3) > 35)
     

b = [0,0]; 
k = i-1; %row position, can be 0,1,2 or 3 in this case
secondcolumn = [secondcolumn(1:k,:); b; secondcolumn(k+1:end,:)]
 
 end
  end
   [j, k] = size(thirdcolumn);
   for i= 1:15
%    disp (thirdcolumn(i,2) )
%    disp (word(i+30,3))
%  disp(i)
%  disp (thirdcolumn(i,2)- word(i+30,3))

  if (i>j)
   thirdcolumn(i,:)= [0, 0];  
   end
 
 if (thirdcolumn(i,2)- word(i+30,3) > 40)

b = [0,0]; 
k = i-1; %row position, can be 0,1,2 or 3 in this case
thirdcolumn = [thirdcolumn(1:k,:); b; thirdcolumn(k+1:end,:)]
 end
   end
    [j, k] = size(fourthcolumn);
    for i= 1:15
 
  if (i>j)
   fourthcolumn(i,:)= [0, 0];
 
   end
 if (fourthcolumn(i,2)- word(i+45,3) > 35)
     

b = [0,0]; 
k = i-1; %row position, can be 0,1,2 or 3 in this case
fourthcolumn = [fourthcolumn(1:k,:); b; fourthcolumn(k+1:end,:)]
 
 end
   end
  
 %% Sorting them in terms of options A,B,C,D
 
for x=1:15
   if (firstcolumn(x) == 0)
        answer1(x)=0;
    end
 if (firstcolumn(x) >A1min && firstcolumn(x) <A1max)
     answer1(x)='A';
 end
 if (firstcolumn(x) >B1min && firstcolumn(x) <B1max)
     answer1(x)='B';
 end
   if (firstcolumn(x) >C1min && firstcolumn(x) <C1max)
     answer1(x)='C';
 end
 if (firstcolumn(x) >D1min && firstcolumn(x) <D1max)
     answer1(x)='D';
 end      
     if (firstcolumn(x) == [])
         answer1(x)= 'no';
         disp(firstcolumn(x))
 end
end
for x=1:15
     if (secondcolumn(x) == 0)
        answer2(x)=0;
    end
 if (secondcolumn(x) >A2min && secondcolumn(x) <A2max)
     answer2(x)='A';
 end
 if (secondcolumn(x) >B2min && secondcolumn(x) <B2max)
     answer2(x)='B';
 end
   if (secondcolumn(x) >C2min && secondcolumn(x) <C2max)
     answer2(x)='C';
 end
 if (secondcolumn(x) >D2min && secondcolumn(x) <D2max)
     answer2(x)='D';
 end      
    
end
for x=1:15
    if (thirdcolumn(x) == 0)
        answer3(x)=0;
    end
 if (thirdcolumn(x) >A3min && thirdcolumn(x) <A3max)
     answer3(x)='A';
 end
 if (thirdcolumn(x) >B3min && thirdcolumn(x) <B3max)
     answer3(x)='B';
 end
   if (thirdcolumn(x) >C3min && thirdcolumn(x) <C3max)
     answer3(x)='C';
 end
 if (thirdcolumn(x) >D3min && thirdcolumn(x) <D3max)
     answer3(x)='D';
 end      


end

for x=1:15
     if (fourthcolumn(x) == 0)
        answer4(x)=0;
    end
 if (fourthcolumn(x) >A4min && fourthcolumn(x) <A4max)
     answer4(x)='A';
 end
 if (fourthcolumn(x) >B4min && fourthcolumn(x) <B4max)
     answer4(x)='B';
 end
   if (fourthcolumn(x) >C4min && fourthcolumn(x) <C4max)
     answer4(x)='C';
 end
 if (fourthcolumn(x) >D4min && fourthcolumn(x) <D4max)
     answer4(x)='D';
 end      
  
end
answerkey=cell2mat(answerkey)
[finalanswer]=[answer1,answer2,answer3,answer4]
totalcorrectanswers=0;
totalwronganswers=0;
for x= 1:60
    x
    answerkey(x)
    finalanswer(x)
   if (answerkey(x) == finalanswer(x)) 
    disp(x)
       disp ('correct')
    totalcorrectanswers=totalcorrectanswers+1;
    
   else 
       totalwronganswers=totalwronganswers+1   ;
       disp(x)
        disp ('wrong')
   end
 
       
       

end
totalwronganswers
totalcorrectanswers

%% Cross detection
% I2= imread('crossmcq.jpg');
% 
% BW = rgb2gray(I2);
% BW = edge(BW,'canny');
% [H,T,R] = hough(BW);
% imshow(H,[],'XData',T,'YData',R,...
%             'InitialMagnification','fit');
% xlabel('\theta'), ylabel('\rho');
% axis on, axis normal, hold on;
% P  = houghpeaks(H,5,'threshold',ceil(0.3*max(H(:))));
% x = T(P(:,2)); y = R(P(:,1));
% plot(x,y,'s','color','white');
% lines = houghlines(BW,T,R,P,'FillGap',10,'MinLength',20);
% figure, imshow(I2), hold on
% max_len = 0;
% for k = 1:length(lines)
%    xy = [lines(k).point1; lines(k).point2];
%    plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');
% 
%    % Plot beginnings and ends of lines
%    plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
%    plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');
% 
%    % Determine the endpoints of the longest line segment
%    len = norm(lines(k).point1 - lines(k).point2);
%    if ( len > max_len)
%       max_len = len;
%       xy_long = xy;
%    end
% end
