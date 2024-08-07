close all
clear all

% Load data
f=load('freq_.dat');
s=load('spikes_.dat');
f(s==0)=-1; % set frequency to -1 if there are no spikes
f(s==1)=0; % set frequency to 0 if there is 1 spike



% Set up grid for independent variables
x1 = 0:10:200; % basal input
y1 = 0:10:300; % tuft input
[x,y]=meshgrid(x1,y1);

% plot data to fit
figure
subplot(1,4,1); surf(x,y,f); caxis([0 180]); view([0 90])


% Multiplication model
Ca = 179.1;
Cb = 40.57;
Ta = 10.01;
Tb = 12.32;
a = -214.3;
b = 378.8;
fmult = a+b./(1+exp(-((x+Ta)./Ca.*(y+Tb)./Cb )));
subplot(1,4,2); surf(x,y,fmult); caxis([0 180]); view([0 90])

% Addition Model
Ca = 21.23;
Cb = 34.97;
Ta = 40.41;
Tb = 199.9;
a = -4.208;
b = 161.8;
fadd = a+b./(1+exp(-((x-Ta)./Ca+(y-Tb)./Cb )));
subplot(1,4,3); surf(x,y,fadd); caxis([0 180]); view([0 90])

% Sigmoid Model
a1 = 87.01;
a2 = 68.24;
a3 = 71.71;
a4 = 10.5;
b1 = 28.5;
b2 = 164.7;
b3 = 64.97;
b4 = -12.63;
M = a1 + a2./(1+exp(-(x-a3)./a4));
T = b1 + b2./(1+exp(-(x-b3)./b4));
fsigmoid = M./(1+exp(-(y-T)));
subplot(1,4,4); surf(x,y,fsigmoid); caxis([0 180]); view([0 90])
