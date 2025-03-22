import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#from torch.distributions.multivariate_normal import MultivariateNormal
from pdb import set_trace as breakpoint

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output

def cosine_similarity_matrix(x, y):
    x = l2_norm(x)
    y = l2_norm(y)
    cos_sim = torch.matmul(x,y.T)#F.linear(x, y)
    return cos_sim

class Momentum_Update(nn.Module):
    """Log ratio loss function. """
    def __init__(self, momentum):
        super(Momentum_Update, self).__init__()
        self.momentum = momentum

    @torch.no_grad()
    def forward(self, model_student, model_teacher):
        """
        Momentum update of the key encoder
        """
        m = self.momentum

        state_dict_s = model_student.state_dict()
        state_dict_t = model_teacher.state_dict()
        for (k_s, v_s), (k_t, v_t) in zip(state_dict_s.items(), state_dict_t.items()):
            if 'num_batches_tracked' in k_s:
                v_t.copy_(v_s)
            else:
                v_t.copy_(v_t * m + (1. - m) * v_s)
        

        #cov_inv = torch.inverse(cov_inv)

class neighbor_proj_loss(nn.Module):
    def __init__(self,args, sigma, delta, view, disable_mu, topk):
        super(neighbor_proj_loss, self).__init__()
        self.sigma = sigma
        self.delta = delta
        self.view = view
        self.disable_mu = disable_mu
        self.topk = topk
        self.args = args
        proxy_dim_f = args.embedding_size#args.bg_embedding_size
        proxy_dim_g = args.bg_embedding_size
        self.proxies_f = torch.nn.Parameter(torch.randn(args.num_proxies, proxy_dim_f).cuda())
        self.proxies_g = torch.nn.Parameter(torch.randn(args.num_proxies, proxy_dim_g).cuda())
        self.proxies_g_momentum = torch.randn(args.num_proxies, proxy_dim_g).cuda()
        nn.init.kaiming_normal_(self.proxies_f, mode='fan_out')
        nn.init.kaiming_normal_(self.proxies_g, mode='fan_out')
        nn.init.kaiming_normal_(self.proxies_g_momentum, mode='fan_out')

        self.proxy_planes_f = torch.nn.Parameter(torch.randn(args.num_proxies,args.num_dims, proxy_dim_f).cuda())
        self.proxy_planes_g = torch.nn.Parameter(torch.randn(args.num_proxies,args.num_dims, proxy_dim_g).cuda())
        self.proxy_planes_g_momentum = torch.randn(args.num_proxies,args.num_dims, proxy_dim_g).cuda()

        self.proxy_planes_g_momentum.requires_grad = False
        self.proxies_g_momentum.requires_grad = False
        nn.init.kaiming_normal_(self.proxy_planes_f, mode='fan_out')
        nn.init.kaiming_normal_(self.proxy_planes_g, mode='fan_out')
        nn.init.kaiming_normal_(self.proxy_planes_g_momentum, mode='fan_out')

    def get_proxy_point_proj_similarity(self, s_emb_orig, proxies, proxy_planes, epoch):
        s_emb = s_emb_orig.detach().clone()
        proxy_copy = proxies.detach().clone()
        proxy_planes_copy = proxy_planes.detach().clone()
        with torch.no_grad():
            # if(self.args.proxy_norm):
            #     proxy_copy = F.normalize(proxy_copy)
            fraction_similarity = torch.zeros(self.args.num_proxies, s_emb.shape[0], dtype = torch.float).cuda()
            residues = torch.zeros(self.args.num_proxies, s_emb.shape[0], dtype = torch.float).cuda()

            for num1 in range(self.args.num_proxies):
                u,s, vh = torch.linalg.svd(proxy_planes_copy[num1,:,:], full_matrices= False)
                # Get proper rank of proxy plane

                s_sum = torch.cumsum(s, 0) / s.sum()
                s_sum[s_sum < 0.99] = 2
                rank_points = torch.argmin(s_sum)
                vh = vh[:rank_points+1,:]
                # check if v are normalized , they are
                all_points_centered = s_emb - proxy_copy[num1,:]
                temp1 = self.get_residue(all_points_centered, vh)
                residues[num1,: ] = temp1 #/ temp1.max()
                #all_points = torch.matmul(s_emb - proxy_copy[num1,:] ,vh.T)
                temp_fraction = torch.div( self.get_projected_dist(all_points_centered, vh) , torch.linalg.vector_norm(all_points_centered, dim=1) )
                fraction_similarity[num1,:] = temp_fraction.unsqueeze(0)
        return fraction_similarity, residues

    def get_point_proxy_transpose_sim(self, t_emb, point_planes, proxies, epoch):
        #s_emb = s_emb_orig.detach().clone()
        proxy_copy = proxies.detach().clone()
        with torch.no_grad():
            # if(self.args.proxy_norm):
            #     proxy_copy = F.normalize(proxy_copy)
            fraction_similarity = torch.zeros(t_emb.shape[0], self.args.num_proxies, dtype = torch.float).cuda()
            residues = torch.zeros(t_emb.shape[0], self.args.num_proxies, dtype = torch.float).cuda()
    
            for num1 in range(t_emb.shape[0]):
                current_point_plane = point_planes[num1,:,:]
                all_points_centered = proxy_copy - t_emb[num1, :]
                temp1 = self.get_residue(all_points_centered, current_point_plane)
                residues[num1,: ] = temp1 #/ temp1.max()
                #all_points = torch.matmul(proxy_copy - t_emb[num1, :] , current_point_plane.T)
                temp_fraction = torch.div( self.get_projected_dist(all_points_centered, current_point_plane) , torch.linalg.vector_norm(all_points_centered , dim=1) )
                fraction_similarity[num1,:] = temp_fraction.unsqueeze(0)
        return fraction_similarity, residues

    def get_proxy_plane_loss(self, proxy_point_sim, point_planes,  proxy_planes, epoch):
        
        proxy_planes_normalized = F.normalize(proxy_planes, dim=2)
        proxy_planes_normalized = proxy_planes_normalized.reshape(proxy_planes_normalized.shape[0]* proxy_planes_normalized.shape[1], -1)
        # need (num_proxy * args.num_dim)  * num points in batch type sim
        cos_plane_distances = self.get_projected_dist_batch( proxy_planes_normalized.repeat(point_planes.shape[0],1) \
                                                             .reshape(point_planes.shape[0], proxy_planes_normalized.shape[0],-1), point_planes ) \
                                                                .reshape(point_planes.shape[0], proxy_planes_normalized.shape[0])
        cos_plane_distances = cos_plane_distances.T
        
        with torch.no_grad():
            proxy_point_sim = proxy_point_sim.repeat_interleave(proxy_planes.shape[1], dim=0)
        error = (proxy_point_sim - cos_plane_distances).pow(2)
        projection_loss = error.mean()
        #projection_loss1 = torch.div(projection_loss.sum(dim=1), sim_sum)
        return projection_loss#-1 * projection_loss1.mean()
    def get_proxy_plane_loss_old(self, s_emb, proxy_point_sim, point_planes, proxies_normalized, proxy_planes, epoch):
        # find hyperplanes of all points
        # find neighbors of all proxies
        # find projections of proxy planes on hyperplanes of neighbors
        # if(self.args.proxy_norm):
        #     proxies_normalized = F.normalize(proxies)
        # else:
        #     proxies_normalized = proxies
        with torch.no_grad():
            S_dist = torch.cdist(proxies_normalized, s_emb)
            topk_index = torch.topk(S_dist, self.args.num_local, largest=False, dim=1)[1]
            sim_sum = torch.zeros(self.args.num_proxies, dtype = torch.float).cuda()
            projection_loss = torch.zeros(self.args.num_proxies, self.args.num_local, dtype = torch.float).cuda()
        proxy_planes_normalized = F.normalize(proxy_planes, dim=2)
        for num1 in range(self.args.num_proxies):
            for num2 in range(self.args.num_local):
                neighbor_ind = topk_index[num1,num2]
                current_point_plane = point_planes[neighbor_ind,:,:]
                proj_coord = torch.matmul(proxy_planes_normalized[num1,:,:] ,current_point_plane.T)
                temp_proj = torch.linalg.vector_norm(proj_coord, dim=1).sum() / self.args.num_dims
                projection_loss[num1,num2] = proxy_point_sim[num1,neighbor_ind] * temp_proj
                sim_sum[num1] += proxy_point_sim[num1, neighbor_ind]

        projection_loss1 = torch.div(projection_loss.sum(dim=1), sim_sum)
        return -1 * projection_loss1.mean()

    def get_mse_loss_proxy(self, S_dist,similarity):
        with torch.no_grad():
            W = similarity
            #N = W.shape[0]
            #identity_matrix = torch.eye(N).cuda(non_blocking=True)

        error = (self.delta*(1-W) - S_dist).pow(2)
        error = error #* (1 - identity_matrix)
        loss = error.mean() 
        return loss

    def proxy_loss(self, s_emb, t_emb, point_planes_teacher,proxies_f , proxies_g, proxy_planes_g, proxy_planes_f, epoch):
        
        # 3 components proxy plane <--> point, point plane <--> proxy, proxy plane <--> point plane
        if self.disable_mu:
            s_emb = F.normalize(s_emb)
            # s_g = F.normalize(s_g)
        t_emb = F.normalize(t_emb)

        if(self.args.proxy_norm):
            proxies_normalized_f = F.normalize(proxies_f)
            proxies_normalized_g = F.normalize(proxies_g)
        else:
            proxies_normalized_f = proxies_f
            proxies_normalized_g = proxies_g
        


        N = len(s_emb)
        S_dist_f = torch.cdist(proxies_normalized_f, s_emb)

        # 1) proxy plane <--> point
        fractional_similarity, residues = self.get_proxy_point_proj_similarity(t_emb, proxies_normalized_f, proxy_planes_f, epoch)

        # 2) point plane <--> proxy
        transpose_similarity, transpose_residues =  self.get_point_proxy_transpose_sim(s_emb, point_planes_teacher, proxies_normalized_f, epoch)
       
        normalized_residues = residues / 1.42
        normalized_transpose_residues = transpose_residues / 1.42

        with torch.no_grad():

            # need  num_proxy * batch_size sim mat here, based on proxy planes
            projected_dist = self.get_projected_dist_batch( ( t_emb.repeat(proxies_normalized_f.shape[0],1) - proxies_normalized_f.repeat_interleave(t_emb.shape[0], dim = 0)).reshape(proxies_normalized_f.shape[0], t_emb.shape[0],-1), proxy_planes_f ).reshape(proxies_normalized_f.shape[0],t_emb.shape[0]) #point_planes.repeat_interleave(t_emb.shape[0], dim = 0)
            projected_dist = projected_dist.clone().detach()
            
            # need  batch_size * num_proxy sim mat here, based on point planes
            transpose_projected_dist = self.get_projected_dist_batch( ( proxies_normalized_f.repeat(t_emb.shape[0],1) - t_emb.repeat_interleave(proxies_normalized_f.shape[0], dim = 0)).reshape(t_emb.shape[0],proxies_normalized_f.shape[0],-1), point_planes_teacher ).reshape(t_emb.shape[0] ,proxies_normalized_f.shape[0]) #point_planes.repeat_interleave(t_emb.shape[0], dim = 0)
            transpose_projected_dist = transpose_projected_dist.clone().detach()

        
        projected_similarity = 1/ torch.pow(1+ projected_dist,self.args.projected_power )
        transpose_projected_similarity = 1/ torch.pow(1+ transpose_projected_dist,self.args.projected_power )
        
        residue_similarity = 1/ torch.pow(1+ normalized_residues,self.args.residue_power )
        transpose_residue_similarity = 1/ torch.pow(1+ normalized_transpose_residues,self.args.residue_power )


        
        if(self.args.use_gaussian_sim):
            similarity = torch.exp(-projected_dist.pow(2)*self.args.projected_power - normalized_residues.pow(2)*self.args.residue_power)
            transpose_similarity = torch.exp(-transpose_projected_dist.pow(2)*self.args.projected_power - normalized_transpose_residues.pow(2)*self.args.residue_power) 
        else:
            if(self.args.use_projected):
                if(self.args.use_additive):
                    similarity = (residue_similarity + projected_similarity) / 2
                else:
                    similarity = residue_similarity * projected_similarity
            else:
                similarity = residue_similarity
            if(self.args.use_projected):
                transpose_similarity = transpose_residue_similarity * transpose_projected_similarity
            else:
                transpose_similarity = transpose_residue_similarity


        similarity = similarity + transpose_similarity.T

        loss = self.get_mse_loss_proxy(S_dist_f, similarity)

        # 3) proxy plane <--> point plane
        proxy_plane_loss = self.get_proxy_plane_loss( similarity, point_planes_teacher, proxy_planes_f, epoch)

        return loss + proxy_plane_loss 


    def get_dist_sim(self, points1, points2):
        with torch.no_grad():
            T_dist = torch.cdist(points1, points2)
            sim_dist = torch.exp(-T_dist.pow(2) / self.sigma)
        return sim_dist
    def get_residue(self, points, plane):
        points_projected =  torch.matmul(points,plane.T)
        residue = torch.linalg.vector_norm(points - torch.matmul(points_projected, plane), dim =1)
        return residue

    def get_projected_dist_batch(self, points, planes):
        points_projected =  torch.bmm(points,torch.transpose(planes, 1,2))
        projected_dist = torch.linalg.vector_norm(points_projected, dim =2)
        return projected_dist

    def get_projected_dist(self, points, plane):
        points_projected =  torch.matmul(points,plane.T)
        projected_dist = torch.linalg.vector_norm(points_projected, dim =1)
        return projected_dist

    def get_proj_similarity_planes(self,s_emb):
        num_examples = s_emb.shape[0]
        with torch.no_grad():
            point_planes = torch.zeros(num_examples, self.args.num_dims,s_emb.shape[1], dtype = torch.float).cuda()
            plane_centers = torch.zeros(num_examples, s_emb.shape[1], dtype = torch.float).cuda()
            S_dist = torch.cdist(s_emb, s_emb)
            topk_index = torch.topk(S_dist, self.args.num_neighbors, largest=False, dim=1)[1]
            fraction_similarity = torch.zeros(num_examples, num_examples, dtype = torch.float).cuda()
            residues = torch.zeros(num_examples, num_examples, dtype = torch.float).cuda()


            for num1 in range(num_examples):
                examples_to_select = list(range(self.args.num_dims -1))
                for num_samp in range(self.args.num_dims-1, self.args.num_neighbors):
                    current_points = s_emb[topk_index[num1,examples_to_select+[num_samp]],:] #[num1,:],:]
                    origin =  current_points.mean(dim=0)#current_points[0,:]#.mean(dim = 0)
                    current_points_centered = current_points - origin
                    _,s, vh = torch.linalg.svd(current_points_centered, full_matrices= False)
                    vh = vh[:self.args.num_dims,:]
                    orig_residue = self.get_residue(current_points_centered[0,:].unsqueeze(0), vh)
                    if(orig_residue > 0.1):
                        continue
                    else:
                        examples_to_select = examples_to_select + [num_samp]
                #breakpoint()
                current_points = s_emb[topk_index[num1,examples_to_select],:] #[num1,:],:]
                origin =  current_points[0,:]#current_points.mean(dim=0)#current_points[0,:]#.mean(dim = 0)
                plane_centers[num1, :] = origin
                current_points_centered = current_points - origin
                _,s, vh = torch.linalg.svd(current_points_centered, full_matrices= False)
                vh = vh[:self.args.num_dims,:]

                point_planes[num1,:,:] = vh
                all_points_centered = s_emb - origin
                temp1 = self.get_residue(all_points_centered, vh)
                residues[num1,: ] = temp1 #/ temp1.max()
                temp_fraction = torch.div( self.get_projected_dist(all_points_centered, vh) , torch.linalg.vector_norm(all_points_centered, dim=1) )
                temp_fraction[num1] = 1
                fraction_similarity[num1,:] = temp_fraction.unsqueeze(0)
        return fraction_similarity, point_planes, plane_centers, residues

    def get_point_neighbor_loss(self, s_emb, t_emb, epoch):
        #if self.disable_mu:
        s_emb = F.normalize(s_emb)
        t_emb = F.normalize(t_emb)

        N = len(s_emb)
        S_dist = torch.cdist(s_emb, s_emb)

        fractional_similarity, point_planes, plane_centers, normalized_residues = self.get_proj_similarity_planes(t_emb)
        normalized_residues = normalized_residues /1.42
        with torch.no_grad():
            l2_dist_similarity = self.get_dist_sim(t_emb,t_emb).clone().detach()
            # m is num of planes, n is num of points in t_emb, 3 is args.num_dim
        # points are m * n * d, second dim has all points in each indice of first
        # planes are m * 3/num_dims * d, result is m*n with distance between first plane and all poitns in first row
            projected_dist = self.get_projected_dist_batch( ( t_emb.repeat(t_emb.shape[0],1) \
                                                             - plane_centers.repeat_interleave(t_emb.shape[0], dim = 0)\
                                                                ).reshape(t_emb.shape[0], t_emb.shape[0],-1), point_planes \
                                                            ).reshape(t_emb.shape[0], t_emb.shape[0]) #point_planes.repeat_interleave(t_emb.shape[0], dim = 0)
            projected_dist = projected_dist.clone().detach()

            if(self.args.use_gaussian_sim):
                similarity = torch.exp(-projected_dist.pow(2)*self.args.projected_power - normalized_residues.pow(2)*self.args.residue_power)
            else:
                if(self.args.use_projected):
                    dist_similarity = 1/ torch.pow(1+ projected_dist,self.args.projected_power )
                else:
                    dist_similarity = l2_dist_similarity

                residue_similarity = 1/ torch.pow(1+ normalized_residues,self.args.residue_power )
                
                if(self.args.use_projected):
                    if(self.args.use_additive):
                        similarity = (residue_similarity + dist_similarity) / 2
                    else:
                        similarity = residue_similarity * dist_similarity
                else:
                    similarity = residue_similarity
        #breakpoint()
        #similarity = similarity + similarity.T
        #breakpoint()
        # get gt sim
        # temp1 = y.unsqueeze(1).repeat(1,y.shape[0])
        # temp2 = y.unsqueeze(0).repeat(y.shape[0],1)
        # gt_sim = temp1 == temp2 #y.unsqueeze(1).repeat(t.shape[0]) == 
        # gt_sim = gt_sim.long()
        # gt_sim1 = gt_sim.reshape(-1)
        # similarity1= similarity.reshape(-1)
        #c1 = torch.corrcoef(torch.stack([similarity1,gt_sim1]))
        #print(c1)
        #breakpoint()
        loss = self.get_mse_loss(S_dist,similarity)
        return loss, point_planes

    def get_mse_loss(self, S_dist,similarity):
        with torch.no_grad():
            W = similarity
            N = W.shape[0]
            identity_matrix = torch.eye(N).cuda(non_blocking=True)

        error = (self.delta*(1-W) - S_dist).pow(2)
        error = error * (1 - identity_matrix)
        loss = error.sum() / (N * (N-1))#(len(s_emb) * (len(s_emb)-1))
        return loss


    def forward(self, s_f, t_f, epoch):

        if self.args.num_proxies == 0:
            dummy_loss = torch.tensor(0.0, device=s_f.device, requires_grad=True)
            return dict(RC=dummy_loss, proxy=dummy_loss, loss=dummy_loss)
        
        loss_RC_f, point_planes_teacher = self.get_point_neighbor_loss(s_f, t_f, epoch)

        loss_RC = (loss_RC_f)# + loss_RC_g)/2
        if(self.args.no_proxy):
            loss_proxy = torch.zeros_like(loss_RC)
        else:
            loss_proxy = self.proxy_loss(s_f,t_f,point_planes_teacher, self.proxies_f, self.proxies_g, self.proxy_planes_g,self.proxy_planes_f, epoch)
        if(self.args.only_proxy == True):
            loss = loss_proxy #+ loss_proxy2
        else:
            loss = loss_RC + loss_proxy #+ loss_proxy2#+ loss_KL

        
        loss = loss #+ 10*std_loss
        total_loss = dict(RC=loss_RC, proxy = loss_proxy, loss=loss)
        
        return total_loss


    def get_planes(self,s_emb):
        num_examples = s_emb.shape[0]
        point_planes = torch.zeros(num_examples, self.args.num_dims,s_emb.shape[1], dtype = torch.float).cuda()
        with torch.no_grad():
            S_dist = torch.cdist(s_emb, s_emb)
            topk_index = torch.topk(S_dist, self.args.num_neighbors, largest=False, dim=1)[1]
            for num1 in range(num_examples):
                examples_to_select = list(range(self.args.num_dims -1))
                for num_samp in range(self.args.num_dims-1, self.args.num_neighbors):
                    current_points = s_emb[topk_index[num1,examples_to_select+[num_samp]],:] #[num1,:],:]
                    origin =  current_points.mean(dim=0)
                    current_points_centered = current_points - origin
                    _,s, vh = torch.linalg.svd(current_points_centered, full_matrices= False)
                    vh = vh[:self.args.num_dims,:]
                    orig_residue = self.get_residue(current_points_centered[0,:].unsqueeze(0), vh)
                    if(orig_residue > 0.1):
                        continue
                    else:
                        examples_to_select = examples_to_select + [num_samp]

                current_points = s_emb[topk_index[num1,examples_to_select],:]
                origin =  current_points.mean(dim=0)
                current_points_centered = current_points - origin
                _,s, vh = torch.linalg.svd(current_points_centered, full_matrices= False)
                vh = vh[:self.args.num_dims,:]
                point_planes[num1,:,:] = vh

        return point_planes