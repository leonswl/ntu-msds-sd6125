import math
import numpy as np

class u_k_means():
    def __init__(self, thres, beta, gamma, rate, t_max):
        self.thres = thres
        self.beta = beta
        self.gamma = gamma
        self.rate = rate
        self.t_max = t_max

    def fit(self, points, y):
        """

        """
        # initializations
        cluster_n = points.shape[0]
        clust_cen = points + 0.0001
        alpha = np.ones((1, cluster_n)) * 1 / cluster_n
        err = 10

        c_history = []

        while cluster_n > 1 and err >= self.thres:
            self.rate = self.rate + 1

            # Step 2: Compute membership
            u = np.zeros((points.shape[0], cluster_n))
            D7 = np.zeros((points.shape[0],0))
            for k in range(cluster_n):
                D1 = np.subtract(points, clust_cen[k, :])
                D2 = np.power(D1, 2)
                D3 = np.sum(D2, 1)
                D4 = D3
                D5 = self.gamma * math.log(alpha[0, k])
                D6 = np.subtract(D4, D5)
                D7 = np.concatenate((D7, D6[:, np.newaxis]),axis=1)

            if self.rate == 1:
                D8 = D7
                np.fill_diagonal(D7, np.nan)
                val = np.min(D7, axis=1)
                idx = np.argmax(D7, axis=1)
                np.fill_diagonal(D7, np.diag(D8))
            else:
                val = np.min(D7, axis=1)
                idx = np.argmax(D7, axis=1)

            for i in range(points.shape[0]):
                u[i, idx[i]] = 1
            
            print(u)
            # Step 3: Compute gamma
            gamma = math.exp(-cluster_n / 450)

            # Step 4: Update alpha
            new_alpha = np.zeros((1, cluster_n))
            for k in range(cluster_n):
                temp1 = np.power(u[:, k], self.beta)
                temp2 = np.sum(temp1)
                temp3 = temp2 / points.shape[0]
                new_alpha[0, k] = temp3
            
            # Step 5: Update beta
            eta = 1 / points.shape[0]
            temp9 = 0
            for k in range(cluster_n):
                temp8 = math.exp(-eta * points.shape[0] * abs(new_alpha[0, k] - alpha[0, k]))
                temp9 = temp9 + temp8
            temp9 = temp9 / cluster_n
            temp10 = 1 - max(np.sum(u, 1) / points.shape[0])
            temp11 = sum(alpha[0, :] * np.log(alpha[0, :]))
            temp12 = temp10 / (-max(alpha[0, :]) * temp11)
            
            new_beta = min(temp9, temp12)
            
            # Step 6: Update number of clusters
            index = np.where(new_alpha <= 1 / points.shape[0])
            
            # ADJUST ALPHA
            adj_alpha = new_alpha
            adj_alpha = np.delete(adj_alpha, index)
            adj_alpha = adj_alpha / sum(adj_alpha)
            new_alpha = adj_alpha
            if new_alpha.shape[0] == 1:
                new_alpha = alpha
                break
            
            # Update NUMBER OF CLUSTER
            new_cluster_n = new_alpha.shape[0]
            
            # ADJUST MEMBERSHIP(U)
            adj_u = u
            adj_u = np.delete(adj_u, index, 1)
            adj_u = adj_u / np.sum(adj_u, 1)[:, None]
            adj_u[np.isnan(adj_u)] = 0
            new_u = adj_u
            
            if self.rate >= 60 and new_cluster_n - cluster_n == 0:
                new_beta = 0
                
            # Update Cluster Centers
            new_clust_cen = []
            for k in range(new_cluster_n):
                temp4 = np.zeros((1, points.shape[1]))
                temp5 = 0
                for i in range(points.shape[0]):
                    temp4 = temp4 + new_u[i, k] * points[i, :]
                    temp5 = temp5 + new_u[i, k]
                new_clust_cen.append(temp4 / temp5)
                # np.concatenate((new_clust_cen, temp4 / temp5), axis=0)
            new_clust_cen = np.array(new_clust_cen)
            print(new_clust_cen) 
            print(new_clust_cen.shape)    
            new_clust_cen[np.isnan(new_clust_cen)] = sum(np.mean(points))
            
            # STEP 8: Convergence criteria
            error = []
            for k in range(new_cluster_n):
                error = np.concatenate((error, np.linalg.norm(new_clust_cen[k, :] - clust_cen[k, :])))
                
            err = max(error)
            
            clust_cen = new_clust_cen
            cluster_n = new_cluster_n
            alpha = new_alpha
            beta = new_beta
            u = new_u
            c_history.append(cluster_n)
            c_history = np.array(c_history)
            # c_history = np.concatenate((c_history, cluster_n))
            
        # Step 9: Cluster labeling
        clust = []
        for i in range(points.shape[0]):
            clust.append(val[i])
            # np.concatenate((clust, idx[i]))
            
        return clust, c_history