%Rahaf Aljundi 2016
%perform the online nearest neibouring clustering
%each time after having a new track it restimates the centroids
%if any point in the new track has a distance bigger that threshold it will
%be assigned to a new cluster
%only clusters with a size higher  than a given value, their centriods
%are calculated and returened to be used (actor profile)
%clusters: cell array of inds
%mini_clusters: sub clusters within "clusters"
%track:vector of inds
%centroids: inds of the centroids in the global_dist_matrix
%global_dist_matrix: the distance between all the faces
%takes into account the initial state of not having a cluster with a
%density bigger > th
%opts: contain the optinial paramters if you wanna play with the values
function [clusters,mini_clusters,centroids]=extract_actors_profile(global_dist_matrix,clusters,mini_clusters,track,opts)
%hyper parameters
%============================================
theta=1;%theshold for entering a big cluster
density=100;%size for a cluster to be considered
theta_out=1.2;%sufficent distance to make a new cluster that is far from other clusters
mini_theta=0.5;%distance for subclusters
%============================================
if(isfield(opts,'theta'))
    theta=opts.theta;
end
if(isfield(opts,'density'))
    density=opts.density;
end
if(isfield(opts,'theta_out'))
    theta_out=opts.theta_out;
end
if(isfield(opts,'mini_theta'))
    mini_theta=opts.mini_theta;
end
centroids=[];
%loop around the points in the new track
for i=1:numel(track)
    %check its distance to the clusters
    min_dis=Inf;
    min_cl=-1;
    clear this_mini_clusters
    for cl=1:numel(clusters)
        cluster_inds=clusters{cl};
        %the average distance to the points in the cluster
        %check if minimum is better
        cl_dis=mean(global_dist_matrix(track(i),cluster_inds));
        if (cl_dis<min_dis)
            min_dis=cl_dis;
            min_cl=cl;
            
        end
    end
    if min_dis<theta
        %add the new point to the cluster
        clusters{min_cl}=[clusters{min_cl};track(i)];
        
        %look for the mini cluster
        this_mini_clusters=mini_clusters{min_cl};
        %do the same loop as beofre to look for the matching mini cluster
        %-------
        min_dis=Inf;
        this_min_cl=-1;
        for mini_cl=1:numel(this_mini_clusters)
            cluster_inds=this_mini_clusters{mini_cl};
            %the average distance to the points in the cluster
            %check if minimum is better
            cl_dis=mean(global_dist_matrix(track(i),cluster_inds));
            if (cl_dis<min_dis)
                min_dis=cl_dis;
                this_min_cl=mini_cl;
                
            end
        end
        % check if it could be attached to a mini cluster or make a new
        % mini cluster for its own
        if min_dis<mini_theta
            %add the new point to the cluster
            this_mini_clusters{this_min_cl}=[this_mini_clusters{this_min_cl};track(i)];
        else
            %make a new mini cluster
            this_mini_clusters{end+1}=track(i);
        end
        %reassign the mini clusters to their parent clusters
        mini_clusters{min_cl}=this_mini_clusters;
        %-------
        %mini cluster operation ends here
        
    else
        if( min_cl==-1 || min_dis<theta_out)
        %make a new cluster
        clusters{end+1}=track(i);
        this_mini_cluster{1}=track(i);
        mini_clusters{end+1}= this_mini_cluster;
        end
    end
    
end
%get the centroids of the clusters
biggest_cluster_size=0;
biggest_cluster_indx=-1;
for cl=1:numel(clusters)
    %if the cluster is dense enough
    if numel(clusters{cl})>density
        %get the centroids
        %of each mini cluster
        this_mini_clusters=mini_clusters{cl};
        %for each mini cluster
        for mini_cl=1:numel(this_mini_clusters)
        cluster_inds=this_mini_clusters{mini_cl};
        dis=global_dist_matrix(cluster_inds,cluster_inds);
        mean_dis=mean(dis,2);
        
        [x,min_indx]=min(mean_dis);
        centroids(1,end+1)=cluster_inds(min_indx);
        end
        
    end
    if(numel(clusters{cl})>biggest_cluster_size)
        biggest_cluster_size=numel(clusters{cl});
        biggest_cluster_indx=cl;
    end
end
%in case of all small clusters
if (numel(centroids)==0)
    for cl=1:numel(clusters)
    
        %get the centroids of all mini clusters
        this_mini_clusters=mini_clusters{cl};
        %for each mini cluster
        for mini_cl=1:numel(this_mini_clusters)
        cluster_inds=this_mini_clusters{mini_cl};
        dis=global_dist_matrix(cluster_inds,cluster_inds);
        mean_dis=mean(dis,2);
        
        [x,min_indx]=min(mean_dis);
        centroids(1,end+1)=cluster_inds(min_indx);
        end
    end
end