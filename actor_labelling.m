%Rahaf Aljundi 2016
%do the hungarian for an episode
%add the clustering concept
%selecting the centriods as actor templates
%using the nearest neighbours clustering to find the centroids
%multple representitive points of mini clusters
%opts: a structure with the input variables:
    %actor_names: the names of the main actors
    %bb_dir:directory where the tracks (feature vectors of faces in the
    %tracks) are
    %output_dir:the directory for the output
    %imdb_dir: directory where the feature of the imdb faces are
    
function actor_labelling(opts)

%% Self-labelling using the Hungarian algorithm
%  Same as graph_matching3, except this version uses the weighted distance
%  between the faces in the track and the faces in imdb
%  The Hungarian algorithm to assign one of 6 actor labels
%  to a track in an optimal way, with a threshold so that
%  not every actor needs to be present in a shot
root_path='/esat/jade/raljundi/ACCV2016/actors_verification/BBT/';
addpath(genpath(('/users/visics/pchakrav/Documents/MATLAB/utils/')));
addpath('/users/visics/pchakrav/Documents/MATLAB/utils/');
if(~isfield(opts,'actor_names'))
    opts.actor_names  = {'Aaron Paul','Anna Gunn','Betsy Brandt','Bryan Cranston','Dean Norris','Mitte'};
end

if(~isfield(opts,'bb_dir'))
    opts.bb_dir= '/esat/jade/raljundi/ACCV2016/actors_verification/tracks_features';
end
if(~isfield(opts,'imdb_dir'))
    opts.imdb_dir   = '/esat/jade/raljundi/ACCV2016/actors_verification/imdbFaceFeatures';
end
actors_size=numel(opts.actor_names );
if(~isfield(opts,'output_dir'))
    opts.output_dir = fullfile('/esat/jade/raljundi/ACCV2016/actors_verification/NN_avg_BBIMDB',strcat('BB',shot_num_str));
end
mkdir(opts.output_dir);
delete(fullfile(opts.output_dir,'*.mat'));
%where all images are for visualization purpose 
if(isfield(opts,'season_name'))
    opts.pooled_images_path='/esat/jade/raljundi/ACCV2016/actors_verification/BBT/seasons_images_tracks/';
    
    opts.pooled_images_path=fullfile(opts.pooled_images_path,opts.season_name);
end

if(~isfield(opts,'visualize'))
    opts.visualize=0;
end
%centroids_dir:directory where the centriods(actor profile elements) are saved
if (isfield(opts,'centroids_dir'))
    mkdir(opts.centroids_dir);
    delete(fullfile(opts.centroids_dir,'*.png'));
    
end
if (isfield(opts,'visualization_dir'))
    mkdir(opts.visualization_dir);
    delete(fullfile(opts.visualization_dir,'*.png'));
    
end
if (~isfield(opts,'bb'))
    opts.bb=0;
    
end
if(~isfield(opts,'actor_labels'))
    for i=1:actors_size
        opts.actor_labels{i}=i;
    end
    
    
end
%actors_dist_idx;

%% Load and normalize the actors faces features in the IMDB dataset
%contains all the faces of tracks and actors
all_faces_matrix=[];
%contains all the faces of tracks
tracks_faces_matrix=[];



size_actor_faces=0;
for i=1:numel(opts.actor_names)
    %the initial clusters
    clusters{i}=[];
    mini_clusters{i}=[];
    centroids{i}=[];
    actor_name = opts.actor_names{i};
    load(fullfile(opts.imdb_dir,actor_name));
    %4096 num of features
    face_feat_vecs_imdb_this_actor = feat_vecs_normalized(1:4096,:)';
    %L2-normalize features for face in IMDB
    actor_counter=size_actor_faces;
    for face_idx=1:size(face_feat_vecs_imdb_this_actor,1)
        this_face= l2_norm(face_feat_vecs_imdb_this_actor(face_idx,:));
        face_feat_vecs_imdb_this_actor_normalized(face_idx,:) =this_face;
        size_actor_faces=size_actor_faces+1;
        all_faces_matrix=[all_faces_matrix;this_face];
    end
    face_feat_vecs_imdb{i}=  face_feat_vecs_imdb_this_actor_normalized;
    %save the corresponding indexes of the actor faces in the all tracks
    %matrixes
    actors_dist_idx{i}=[actor_counter+1:size_actor_faces];
    clear face_feat_vecs_imdb_this_actor_normalized;
end

%initialize
size_all_faces_matrix=size(all_faces_matrix,1);
size_tracks_faces_matrix=0;


imbd_actor_inds=actors_dist_idx;
num_actors = size(opts.actor_names,2);
for act=1:num_actors
    old_centor_glb_inds{act}=[];
end

%get the right indexing among all shots

idx_start=0;
tracks_files=dir(opts.bb_dir);
tracks_files(1:2,:)=[];
all_images=[];


for track_idx = 1:numel(tracks_files)
    %-get the images as well for visualization purpose
    %uncomment it if needed

%         load(fullfile(opts.pooled_images_path,strcat(num2str(track_idx-1),'.mat')));
%         for(im_i=1:numel(track_image_split))
%             all_images{end+1}=track_image_split{im_i};

%        end
    load(fullfile(opts.bb_dir,strcat(num2str(track_idx-1),'.mat')));
    face_feat_vecs_bb_this = feat_vecs_normalized(1:4096,:)';
    %the current size of the all tracks faces matrix
    
    %L2-normalize features for each face in track
    for face_idx = 1:size(face_feat_vecs_bb_this,1)
        this_face=l2_norm(face_feat_vecs_bb_this(face_idx,:));
        face_feat_vecs_bb_this_normalized(face_idx,:) = this_face;
        
    end
    
    %store the indexes of each track faces for both matrixes
    new_size_tracks_faces_matrix=size(face_feat_vecs_bb_this_normalized,1)+size_tracks_faces_matrix;
    
    tracks_dist_idx{track_idx+idx_start}=[size_tracks_faces_matrix+1:new_size_tracks_faces_matrix];
    all_faces_tracks_dist_idx{track_idx+idx_start}= [size_tracks_faces_matrix+size_all_faces_matrix+1:new_size_tracks_faces_matrix+size_all_faces_matrix];
    size_tracks_faces_matrix=new_size_tracks_faces_matrix;
    tracks_feat_op{track_idx+idx_start}=face_feat_vecs_bb_this_normalized';
    tracks_feat{track_idx+idx_start}=face_feat_vecs_bb_this_normalized;
    
    clear face_feat_vecs_bb_this_normalized;
    
end
%% compute the distance matrix
tracks_faces_matrix=cell2mat(tracks_feat_op);
clear tracks_feat_op;
tracks_faces_matrix=tracks_faces_matrix';
all_faces_matrix=[all_faces_matrix;tracks_faces_matrix];

if~exist(strcat(root_path,opts.gobal_dist_matx,'.mat'),'file')
    tic
    gobal_dist_matx  = pdist2(tracks_faces_matrix,all_faces_matrix);
    save(strcat(root_path,opts.gobal_dist_matx),'gobal_dist_matx','-v7.3');
    toc
else
    load(strcat(root_path,opts.gobal_dist_matx),'gobal_dist_matx');
    
end
clear all_faces_matrix tracks_faces_matrix
num_tracks_orig = numel(tracks_feat);
num_tracks = num_tracks_orig;




%% Self-labelling with Hungarian Assignment



num_tracks_orig = numel(tracks_feat);
num_tracks = num_tracks_orig;
track_indices_prev_curr_iters=[1:num_tracks];

MAX_DIST_THRESH = -0.1;%1.2;
num_iterations = 1;num_tracks_added = 1;
num_of_remaining_tracks=num_tracks;
step=0;
while num_tracks > 0 && num_tracks_added > 0
    step=step+1;
   
    
    %% Assemble Hungarian matrix
    hungarian_matrix = ones(num_actors,num_tracks)*Inf;
  
    
    
    
    for track_indx=1:num_tracks
        face_feat_vecs_bb_this_normalized=tracks_feat{track_indx};
        if (size(face_feat_vecs_bb_this_normalized)>0)
            %loop around the actors and pick the one with the minimum distance
            
            all_actor_dis=0;
            for actor_idx = 1:num_actors
                %face_feat_vecs_imdb_this_actor_normalized=face_feat_vecs_imdb{actor_idx};
                
                
                %get distance indexes
                
                this_track_inds=tracks_dist_idx{track_indx};
               
                
                %instead of computing the distance with all the faces
                %that belong to the actor, do it just for the centriods
                act_centroids=centroids{actor_idx};
                shifted_act_centroids=act_centroids+size_actor_faces;
                imdb_this_actor=actors_dist_idx{actor_idx};
                this_actor_inds=[imdb_this_actor,shifted_act_centroids];
                
                %here add to this_track_inds the starter of imdb faces
                
                
                D=gobal_dist_matx(this_track_inds,this_actor_inds);
                best_match_valD = mean(D');
                best_match_val_m{actor_idx} = mean(best_match_valD);
                
                all_actor_dis=all_actor_dis+ best_match_val_m{actor_idx} ;
            end
            %compute the weighted distance
            for  actor_idx = 1:numel(opts.actor_names)
                hungarian_matrix(actor_idx, track_indx) = best_match_val_m{actor_idx}-(1/num_actors)*(all_actor_dis);
            end
            
          
        end
    end
    
    
    [assignment,cost] = munkres(hungarian_matrix);
    
    
    %% Get the top 6 tracks, assigned to their actor labels (above threshold)
    actor_track_matches = zeros(numel(opts.actor_names),1);
    for actor_idx=1:num_actors
        track_idx = assignment(actor_idx);%find(assignment==actor_idx);
        if track_idx~= 0 %hungarian sometimes returns 0 - does not assign a track to an actor
            distance = hungarian_matrix(actor_idx,track_idx);
            if distance <= MAX_DIST_THRESH
                actor_track_matches(actor_idx) = track_idx;
            end
        end
    end
    
    %% Self-labelling: add new faces from tracks to their assigned imdb clusters (actors)
    num_tracks_added = 0;
    for actor_idx=1:num_actors
        best_match_track_idx = actor_track_matches(actor_idx);
        if best_match_track_idx ~= 0
            fprintf('track #%d added for actor %d\n',best_match_track_idx,actor_idx);
            %move the indices to the actor
            %-----
            current_act_dist_idx=actors_dist_idx{actor_idx};
            this_trck_dist_idx=tracks_dist_idx{best_match_track_idx};
            %--assign the track to corresponding cluster
         
            act_clusters=clusters{actor_idx};
            act_mini_clusters=mini_clusters{actor_idx};
            [act_clusters,act_mini_clusters,act_centroids]=extract_actors_profile(gobal_dist_matx(:,size_actor_faces+1:end),act_clusters,act_mini_clusters,this_trck_dist_idx,opts);
            clusters{actor_idx}=act_clusters;
            centroids{actor_idx}=act_centroids;
            disp(strcat('this_act_idx',num2str(actor_idx),' number of cent',num2str(numel(act_centroids))));
            mini_clusters{actor_idx}=act_mini_clusters;
            %save the centroids
            %if it is asked to
            if (isfield(opts,'centroids_dir'))
                
                for(im_i=1:numel(act_centroids))
                    
                    image = imread( all_images{act_centroids(im_i)});
                    imwrite(image,fullfile(opts.centroids_dir,strcat(num2str(step),'_',num2str(actor_idx),'_',num2str(im_i),'.png')));
                end
                
            end
            
            tracks_feat{best_match_track_idx} = [];% Making the feature vectors for this track empty
            
            best_actor_idx   = opts.actor_labels{actor_idx};
           
            best_distance    = hungarian_matrix(actor_idx, best_match_track_idx);
            best_match_actor = opts.actor_names{actor_idx};
            this_trck_dist_idx=tracks_dist_idx{best_match_track_idx};
            save_distances_filename = fullfile(opts.output_dir,strcat(num2str(best_match_track_idx-1),'.mat'));
            save(save_distances_filename,'this_trck_dist_idx','best_match_actor','best_distance','best_actor_idx');
            if(opts.visualize)
                load(fullfile(opts.pooled_images_path,strcat(num2str(best_match_track_idx-1))));
                %            VISUALIZE TRACK
                for(im_i=1:numel(track_image_split))
                    image = imread( track_image_split{im_i});
                    figure(1),imshow(image);
                    text(1, 1, num2str(best_actor_idx), ...
                        'BackgroundColor',[.7 .9 .7], 'FontSize', 7, ...
                        'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom');
                    text(1, 50, num2str(best_distance), ...
                        'BackgroundColor',[.7 .9 .7], 'FontSize', 7, ...
                        'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom');
                    pause(0.01);
                end
            end
            if (isfield(opts,'visualization_dir'))
                %track_dist_idx
                %load(fullfile(opts.pooled_images_path,strcat(num2str(best_match_track_idx-1))));
                this_trck_dist_idx=tracks_dist_idx{best_match_track_idx};
                for(im_i=1:numel(this_trck_dist_idx))
                    image = imread( all_images{this_trck_dist_idx(im_i)});
                    
                    imwrite(image,fullfile(opts.visualization_dir,strcat(num2str(step),'_',num2str(actor_idx),'_',num2str(im_i),'_',num2str(best_distance),'.png')));
                end
                
            end
            
            num_tracks_added = num_tracks_added + 1;
        end
    end
    
    
    num_iterations = num_iterations + 1;
 
    num_of_remaining_tracks=num_of_remaining_tracks-num_tracks_added;
    fprintf('%d tracks added \n',num_tracks_added);
    
    fprintf('%d number of remaining tracks \n',num_of_remaining_tracks);
   
    
end
for track_indx=1:numel(tracks_feat)
    if(~isempty(tracks_feat{track_indx}))
        if (isfield(opts,'visualization_dir'))
            
            %load(fullfile(opts.pooled_images_path,strcat(num2str(track_indx-1))));
            this_trck_dist_idx=tracks_dist_idx{track_indx};
            
            for(im_i=1:numel(this_trck_dist_idx))
                image = imread( all_images{this_trck_dist_idx(im_i)});
                
                
                imwrite(image,fullfile(opts.visualization_dir,strcat(num2str(step),'_',num2str(-1),'_',num2str(track_indx-1),'_',num2str(im_i),'.png')));
            end
        end
    end
end
