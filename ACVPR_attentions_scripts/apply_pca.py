import torchvision.transforms as ttf
from src.factory import *
from scipy.spatial.distance import cdist
from tqdm import tqdm 
import sys
import torch
import os
import numpy as np
import argparse
from src.validate import validate
from extract_predictions import extract_msls_top_k, predict_poses, predict_poses_cmu, eval_pitts,extract_top_k, extract_top_k_tokyotm
""""
msls_cities = {
        'train': ["trondheim", "london", "boston", "melbourne", "amsterdam", "helsinki",
                  "tokyo", "toronto", "saopaulo", "moscow", "zurich", "paris", "bangkok",
                  "budapest", "austin", "berlin", "ottawa", "phoenix", "goa", "amman", "nairobi", "manila"],
        'val': ["cph", "sf"],
        'test': ["miami", "athens", "buenosaires", "stockholm", "bengaluru", "kampala"]
    }
"""
msls_cities = {
    'train': ["amman", "london", "zurich", "boston", "manila"],
    'val': ["sf", "cph"],
    'test': ["miami"]
}

#Whitening code by Filip Radenovic https://github.com/filipradenovic/cnnimageretrieval-pytorch/blob/master/cirtorch/utils/whiten.py
def cholesky(S):
    # Cholesky decomposition
    # with adding a small value on the diagonal
    # until matrix is positive definite
    alpha = 0
    while 1:
        try:
            L = np.linalg.cholesky(S + alpha*np.eye(*S.shape))
            return L
        except:
            if alpha == 0:
                alpha = 1e-10
            else:
                alpha *= 10
            print(">>>> {}::cholesky: Matrix is not positive definite, adding {:.0e} on the diagonal"
                .format(os.path.basename(__file__), alpha))


def whitenapply(X, m, P, dimensions=None):
    
    if not dimensions:
        dimensions = P.shape[0]

    X = np.dot(P[:dimensions, :], X-m)
    X = X / (np.linalg.norm(X, ord=2, axis=0, keepdims=True) + 1e-6)

    return X.copy(order='C')


def pcawhitenlearn(X):

    N = X.shape[1]

    # Learning PCA w/o annotations
    m = X.mean(axis=1, keepdims=True)
    Xc = X - m
    Xcov = np.dot(Xc, Xc.T)
    Xcov = (Xcov + Xcov.T) / (2*N)
    eigval, eigvec = np.linalg.eig(Xcov)
    order = eigval.argsort()[::-1]
    eigval = eigval[order]
    eigvec = eigvec[:, order]

    P = np.dot(np.linalg.inv(np.sqrt(np.diag(eigval))), eigvec.T)
    if P.dtype == "complex128":
        P = np.real(P)
        print("Warning: complex numbers in eigenvec and eigenvals")
    
    return m, P

def mapquery_pcawhitenlearn(params):
    features_dir ="results/" + params.dataset+"/"+params.subset+"/"
    db=np.load(params.map_feats_file).astype(np.float16).T
    return pcawhitenlearn(db)

def mapquery_whitenapply(dataset,name, root_dir, subset, map_feats_file, query_feats_file, m, P,m_idx_file="",q_idx_file="", m_raw_file="", result_file="", dimensions=[2048, 1024, 512, 256, 128, 64, 32]):
    features_dir ="results/" + dataset+"/"+subset+"/"
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)
    db = np.load(map_feats_file).astype(np.float16).T
    q = np.load(query_feats_file).astype(np.float16).T
    for d in tqdm(dimensions, desc="Applying PCA whitening..."):
        
        q_whiten_file=query_feats_file.replace(".npy", "_whiten_"+str(d)+".npy")
        if not os.path.exists(q_whiten_file):
            print("Getting query features...")
            q_whiten=whitenapply(q, m, P, dimensions=d).T
            q_whiten=q_whiten.copy(order='C')
            np.save(q_whiten_file, q_whiten)
        else:
            print("Loading query features...")
            q_whiten = np.load(q_whiten_file)
        db_whiten_file=map_feats_file.replace(".npy", "_whiten_"+str(d)+".npy")
        if not os.path.exists(db_whiten_file):
            print("Getting map features...")
            db_whiten=whitenapply(db, m, P, dimensions=d).T
            db_whiten=db_whiten.copy(order='C')
            np.save(db_whiten_file, db_whiten)
        else:
            print("Loading map features...")
            db_whiten = np.load(db_whiten_file)
        if dataset.lower() == "robotcarseasons":
            predict_poses(root_dir, db_whiten_file,q_whiten_file)
        elif dataset.lower() == "extendedcmu" or dataset.lower() == "cmu":
            predict_poses_cmu(root_dir, db_whiten_file,q_whiten_file)
        elif "pitts" in dataset.lower() or dataset.lower() == "tokyo247":
            result_file=db_whiten_file.replace("_mapfeats", "").replace(".npy", "_predictions.npy")
            extract_top_k(db_whiten_file,q_whiten_file, result_file, 30)
            eval_pitts(root_dir, dataset, result_file)
        elif dataset.lower() == "tokyotm":
            result_file=db_whiten_file.replace("_mapfeats", "").replace(".npy", "_predictions.npy")
            m_idx_file=root_dir+"val_db.json"
            q_idx_file=root_dir+"val_q.json"
            extract_top_k_tokyotm(db_whiten_file,q_whiten_file, m_idx_file, q_idx_file, result_file, 50)
            eval_pitts(root_dir, dataset, result_file)
        elif dataset.lower() =="msls":
            result_file=features_dir+name+"_retrieved_whiten_"+str(d)+".csv"
            extract_msls_top_k(db_whiten_file,q_whiten_file,m_idx_file, q_idx_file, result_file, 50, m_raw_file)

def mapquery_whitenapply_from_file(dataset, name, root_dir, subset, dim, map_feats_file, query_feats_file, checkpoint_file,m_idx_file="",q_idx_file="", m_raw_file=""):
    checkpoint = torch.load(checkpoint_file+str(dim)+".pth")
    assert dim == checkpoint["num_pcs"]
    pca_conv = torch.nn.Conv2d(32768, dim, kernel_size=(1, 1), stride=1, padding=0)
    pca_conv.weight=torch.nn.Parameter(checkpoint["state_dict"]["WPCA.0.weight"].to(torch.float32))
    pca_conv.bias=torch.nn.Parameter(checkpoint["state_dict"]["WPCA.0.bias"].to(torch.float32))
    features_dir ="results/" + dataset+"/"+subset+"/"
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)
    with torch.no_grad():
        b_size = 1024
        q_whiten_file =query_feats_file.replace(".npy", "_whiten_"+str(dim)+".npy")
        if not os.path.exists(q_whiten_file):
            print("Getting query features...")
            aux=0
            q = torch.Tensor(np.load(query_feats_file))
            w_q_feats = torch.zeros((q.shape[0], dim))
            while aux < q.shape[0]:
                print(aux)
                w_q_feats[aux: aux+b_size,:] = torch.nn.functional.normalize(pca_conv(q[aux:aux+b_size,:].unsqueeze(-1).unsqueeze(-1)).squeeze(), p=2, dim =-1)
                aux = aux + b_size
            #w_q_feats = torch.nn.functional.normalize(pca_conv(q.unsqueeze(-1).unsqueeze(-1)).squeeze(), p=2, dim =-1)
            np.save(q_whiten_file, w_q_feats.detach().numpy())
        else:
            print(q_whiten_file, "already exists. Skipping...")

        db_whiten_file =map_feats_file.replace(".npy", "_whiten_"+str(dim)+".npy")
        if not os.path.exists(db_whiten_file):
            
            print("Getting map features...")
            db = torch.Tensor(np.load(map_feats_file))
            aux=0
            w_m_feats = torch.zeros((db.shape[0], dim))
            while aux < db.shape[0]:
                print(aux)
                w_m_feats[aux:aux+b_size,:] = torch.nn.functional.normalize(pca_conv(db[aux: aux+b_size,:].unsqueeze(-1).unsqueeze(-1)).squeeze(), p=2, dim =-1)
                aux = aux + b_size
            #w_m_feats = torch.nn.functional.normalize(pca_conv(db.unsqueeze(-1).unsqueeze(-1)).squeeze(), p=2, dim =-1)
            np.save(db_whiten_file, w_m_feats.detach().numpy())
        else:
            print(db_whiten_file, "already exists. Skipping...")
    if dataset.lower() == "robotcarseasons":
        predict_poses(root_dir, db_whiten_file,q_whiten_file)
    elif dataset.lower() == "extendedcmu" or dataset.lower() == "cmu":
        predict_poses_cmu(root_dir, db_whiten_file,q_whiten_file)
    elif "pitts" in dataset.lower() or dataset.lower() == "tokyo247":
        result_file=db_whiten_file.replace("_mapfeats", "").replace(".npy", "_predictions.npy")
        extract_top_k(db_whiten_file,q_whiten_file, result_file, 30)
        eval_pitts(root_dir, dataset, result_file)
    elif dataset.lower() == "tokyotm":
        result_file=db_whiten_file.replace("_mapfeats", "").replace(".npy", "_predictions.npy")
        m_idx_file=root_dir+"val_db.json"
        q_idx_file=root_dir+"val_q.json"
        extract_top_k_tokyotm(db_whiten_file,q_whiten_file, m_idx_file, q_idx_file, result_file, 50)
        eval_pitts(root_dir, dataset, result_file)
    elif dataset.lower() =="msls":
        result_file=features_dir+name+"_retrieved_whiten_"+str(dim)+".csv"
        extract_msls_top_k(db_whiten_file,q_whiten_file,m_idx_file, q_idx_file, result_file, 50, m_raw_file)



def msls_pcawhitenlearn(params):
    features_dir ="results/" + params.dataset+"/"+params.subset+"/"
    cities = msls_cities[params.subset]
    db=[]
    for city in cities:
        db_file=features_dir+params.name+"_"+city+"_mapfeats.npy"
        db.append(np.load(db_file).T)
    db=np.hstack(db)
    return pcawhitenlearn(db)

def msls_whitenapply_from_file(params):

    cities = msls_cities[params.subset]
    features_dir ="results/" + params.dataset+"/"+params.subset+"/"
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)
    d = params.dim
    result_file=features_dir+params.name+"_retrieved_whiten_"+str(d)+".csv"
    f=open(result_file, "w+")
    f.close()
    for c in tqdm(cities):
        db_file=features_dir+params.name+"_"+c+"_mapfeats.npy"
        q_file=features_dir+params.name+"_"+c+"_queryfeats.npy"
        ds_folder=params.subset if params.subset=="test" else "train_val"
        q_idx_file=params.root_dir+ds_folder+"/"+c+"/query.json"
        m_idx_file=params.root_dir+ds_folder+"/"+c+"/database.json"
        m_raw_file=params.root_dir+ds_folder+"/"+c+"/database/raw.csv"
        q_raw_file=params.root_dir+ds_folder+"/"+c+"/query/raw.csv"

        mapquery_whitenapply_from_file(params.dataset, params.name, params.root_dir, params.subset, params.dim, db_file, q_file, params.checkpoint,m_idx_file=m_idx_file,q_idx_file=q_idx_file, m_raw_file=m_raw_file)
    if params.subset =="val":
        result_file=features_dir+params.name+"_retrieved_whiten_"+str(params.dim)+".csv"
        validate(result_file, params.root_dir, result_file.replace("retrieved", "result").replace(".csv", ".txt"))


def msls_whitenapply(params, m, P, dimensions=[2048, 1024, 512, 256, 128, 64, 32]):

    cities = msls_cities[params.subset]
    features_dir ="results/" + params.dataset+"/"+params.subset+"/"
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)
    for d in dimensions:
        result_file=features_dir+params.name+"_retrieved_whiten_"+str(d)+".csv"
        f=open(result_file, "w+")
        f.close()
    for c in cities:
        db_file=features_dir+params.name+"_"+c+"_mapfeats.npy"
        q_file=features_dir+params.name+"_"+c+"_queryfeats.npy"
        ds_folder=params.subset if params.subset=="test" else "train_val"
        q_idx_file=params.root_dir+ds_folder+"/"+c+"/query.json"
        m_idx_file=params.root_dir+ds_folder+"/"+c+"/database.json"
        m_raw_file=params.root_dir+ds_folder+"/"+c+"/database/raw.csv"
        q_raw_file=params.root_dir+ds_folder+"/"+c+"/query/raw.csv"

        mapquery_whitenapply(params.dataset, params.name, params.root_dir, params.subset, db_file, q_file, m, P,m_idx_file=m_idx_file,q_idx_file=q_idx_file, m_raw_file=m_raw_file, dimensions=dimensions)
   
    if params.subset =="val":
        for d in tqdm(dimensions):
            result_file=features_dir+params.name+"_retrieved_whiten_"+str(d)+".csv"
            validate(result_file, params.root_dir, result_file.replace("retrieved", "result").replace(".csv", ".txt"))






if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', required=True, default='MSLS', help='Name of the dataset [MSLS|7Scenes|TB_Places]')

    parser.add_argument('--root_dir', required=True, help='Root directory of the dataset')
    parser.add_argument('--subset', required=False, default='val', help='For MSLS. Subset to test')
    parser.add_argument('--checkpoint', required=False, default=None, help='Checkpoint containing the PCs')

    parser.add_argument('--query_feats_file', type=str, required=False, help='Query features file, .npy')
    parser.add_argument('--map_feats_file', type=str, required=False, help='Map features file, .npy')
    parser.add_argument('--name', type=str, required=True, help='Name of the experiment')
    parser.add_argument('--dim', type=int, required=False, help='dimension size')

    params = parser.parse_args()
    if "vgg" in params.name:
        dimensions = [512, 256, 128, 64, 32]
    elif "vlad" in params.name.lower():
        dimensions = [32768,16384,8192,4096,2048,1024, 512, 256, 128, 64, 32]
    else:
        dimensions=[2048, 1024, 512, 256, 128, 64, 32]

    if params.dim is not None:
        dimensions = [params.dim]
    
    if params.dataset== "MSLS":
        if params.checkpoint is None:
            m, P =msls_pcawhitenlearn(params)
            msls_whitenapply(params, m, P, dimensions = dimensions)
        else:
            msls_whitenapply_from_file(params)
            
    else:
        if params.checkpoint is None:

            m, P =mapquery_pcawhitenlearn(params)

            mapquery_whitenapply(params.dataset, params.name, params.root_dir, params.subset, params.map_feats_file, params.query_feats_file, m, P, dimensions = dimensions)
        else:
            mapquery_whitenapply_from_file(params.dataset, params.name, params.root_dir, params.subset, params.dim, params.map_feats_file, params.query_feats_file,params.checkpoint)


    

