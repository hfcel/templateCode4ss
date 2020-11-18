% load the data into the var "rawData",then run the script

%% parameters
sf=30000;
nClus=3;
nPc=2;
thFac=4;
winSize=[20,39];
%% load data
% fid=fopen(filename,'r');
% rawData=fread(fid,[1,nSamp],'int16');
% % plot(1:30000,rawData(1,1:30000));
% fclose(fid);
%% detection
%preprocess
rawData=rawData-mean(rawData);
figure
plot(1:30000,rawData(1,1:30000));
figure
[b,a]=butter(3,300/sf/2,'high');
data=filter(b,a,rawData);
plot(1:30000,data(1,1:30000));

%detection
th=thFac*std(data);
figure
hold on;
plot(1:6e4,data(1,1:6e4));
plot(1:6e4,linspace(th,th,6e4));
pos=find(data>th);
pos=pos(pos>winSize(1));
pos=pos(diff(pos)>1);

spks=[];
% ctr2=20;
% figure
% hold on
for i=pos
    tmpspk=data(1,i:i+60);
    [~,ctr]=max(tmpspk);
    ctr=ctr+i-1;
    tmpspk=data(1,ctr-winSize(1):ctr+winSize(2));
    spks=[spks;tmpspk];
    % plot(1:60,tmpspk);
end
idx=randperm(size(spks,1),100);
tmpspks=spks(idx,:);
figure
plot(1:60,tmpspks);
%% preprocess
%center
spks=spks';
spks=spks-mean(spks,2);
%normalization
spksStd=std(spks,0,2);
spks=spks./spksStd;
% figure(2)
% scatter(spks(1,:),spks(2,:),'.');
% xlabel('x');
% ylabel('y');
%% dimension reduction
%pca
[wpca,spkpca,eigvec]=pca(spks');


wpca=wpca(:,1:nPc);
Y=wpca'*spks;
% figure(3)
% scatter(Y(1,:),Y(2,:),'.');
% xlabel('pc1');
% ylabel('pc2');
% zlabel('pc3');
%% cluster

[H,U]=kmeans(Y',nClus);
figure;
hold on;
for i=1:nClus
    scatter(Y(1,H==i),Y(2,H==i),'.');
    
end
xlabel('pc1');
ylabel('pc2');
hold off;