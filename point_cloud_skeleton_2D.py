# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 07:40:37 2022

@author: SATuszynskiJ
"""
import numpy as np
import numpy.random as nr
import matplotlib.pyplot as plt
import laspy
import scipy.ndimage
import skimage.morphology 
#import trimesh
from shapely.geometry import MultiLineString
from   matplotlib.patches import Polygon
import pandas as pd

import trace_skeleton

# ============================================= ================================
def points_on_plane(points, plane_point, plane_norm, dist, radius):
    pts = points-plane_point
    dis = np.linalg.norm(pts[:,:2], axis=1)
    pts = pts[dis<radius,:]
    # Equation of the plane a*x+b*y+c*z+d=0 where [a,b,c] = plane_norm
    
    nrm = plane_norm/np.linalg.norm(plane_norm) # normal unit vector 
    dis = np.abs(np.dot(pts, nrm))
    pts = pts[dis<dist,:]
    return pts

# =============================================================================
def triangle_point_cloud():
    n = 500
    sigma = 1
    straigh_edge = lambda p, q: p +  np.outer(nr.random((n,1)), q-p) 
    d = 20
    a = np.array([-d, 0])
    b = np.array([ d, 0])
    c = np.array([0, 2*d]) 
    p1 = straigh_edge(a, b)
    p2 = straigh_edge(b, c)
    p3 = straigh_edge(c, a)
    points = np.vstack((p1, p2, p3))  +  sigma * nr.normal(size=(3*n,2))
    return points 

# =============================================================================
def envelope(points):
    x = points[:,0]
    y = points[:,1]
    minx = min(x)
    miny = min(y)
    return np.array([minx, miny, max(x)-minx, max(y)-miny])

# =============================================================================
def plt_show():
    plt.axis('equal')
    plt.axis('off')
    plt.show()
    
# =============================================================================
def plt_savefig(fname):
    plt.axis('equal')
    #plt.axis('off')
    plt.savefig(fname, dpi=300)
    
# =============================================================================
def npoints(mls):
    ctr = 0
    for poly in mls.geoms:
        ctr += len(poly.xy[0])
    return ctr

# =============================================================================
def mls_to_mesh(mls):
    points = np.zeros((0,2))
    edges  = np.zeros((0,2))
    for poly in mls.geoms:
        n = points.shape[0]
        m = len(poly.xy[0])
        points = np.vstack((points, np.array(poly.xy).T))
        e = np.arange(n, n+m-1)[...,None]
        edges = np.vstack((edges, np.hstack((e, e+1)) ))
    return points, edges.astype(int)

# =============================================================================
def plot_mls(mls):
    for poly in mls.geoms:
        plt.plot(*poly.xy)
        plt.plot(*poly.xy, '.k')
        
# =============================================================================
def plot_mesh(points, edges):
    for edge in edges:
        a = points[edge[0],:] 
        b = points[edge[1],:] 
        plt.plot([a[0],b[0]], [a[1],b[1]], zorder=2)
        plt.plot([a[0],b[0]], [a[1],b[1]], '.k', zorder=3)

# =============================================================================
def main():
    source = 1
    if source == 1:
        # syntax for laspy 1.7.0
        las = laspy.file.File('input.las')
        points = np.vstack((las.x, las.y, las.z)).T
        # origin = [min(las.x), min(las.y), min(las.z)]
        # points -= origin
        las = None
    else:
        points = triangle_point_cloud()
        
    plt.clf()
    plt.plot(points[:,0], points[:,1], '.', zorder=1)
    plt_savefig('plots2/nadir0.png')
    
    # -------------------------------------------------------------------------    
    # normalize the data: remove the mean and aligh with major/minor axis    
    mean_p  = points.mean(axis=0)    
    points0 = points - mean_p
    
    # aligh with major/minor axis 
    xy = points[:,:2]
    cov = np.cov(xy, rowvar=False)
    ecals, evecs = np.linalg.eig(cov)
    rot_mat = np.eye(3)
    rot_mat[:2,0] =  evecs[:,1]
    rot_mat[:2,1] = -evecs[:,0]
    points0 = points0 @ rot_mat
    
    # plot the results as point cloud
    plt.clf()
    plt.plot(points0[:,0], points0[:,1], '.', zorder=1)
    plt_savefig('plots2/nadir1.png')
        
    # -------------------------------------------------------------------------    
    # move from point cloud to a boolean image "msk" where each point is one 
    # or more pixels
    n = 1000                    # image will have n^2 pixels
    box = envelope(points0)      # bounding box of the point cloud
    r  = np.sqrt(box[2]/box[3]) # aspect ratio
    ny = int(n/r)               # number of rows
    nx = int(n*r)               # number of columns
    x = points0[:,0]
    y = points0[:,1]
    # xvec = np.linspace(min(x), max(x), nx)
    # yvec = np.linspace(min(x), max(x), ny)
    col = np.round(nx*(x - box[0])/box[2]).astype(int).clip(0,nx-1)
    row = np.round(ny*(y - box[1])/box[3]).astype(int).clip(0,ny-1)
    msk = np.full((ny,nx), False, dtype=bool)
    msk[row,col] = True
    
    # -------------------------------------------------------------------------    
    # make each point a circle and remove details
    # prepare circular filter with radius "m"
    m = 4 # radius of the filter
    v = np.linspace(-m, m, 2*m+1)
    xm, ym = np.meshgrid(v, v)
    flt = ((xm**2 + ym**2) <= m**2) # create circular filter
    msk = scipy.ndimage.convolve(msk, flt)
    
    # remove small features
    size = msk.sum() # number of "true" pixels in the "msk" image
    small = int(0.01*size) # definition of "small"
    msk = skimage.morphology.remove_small_holes  (msk, small)
    msk = skimage.morphology.remove_small_objects(msk, small)
    # flr = skimage.morphology.disk(2*m)
    # msk = skimage.morphology.opening(msk, flr)

    # plot the boolean image
    plt.clf()
    plt.imshow(msk)
    plt.gca().invert_yaxis()
    plt_savefig('plots2/nadir2.png')

    # -------------------------------------------------------------------------    
    # skeletonize
    skeleton = skimage.morphology.skeletonize(msk)

    # plot the boolean image
    plt.clf()
    plt.imshow(skeleton)
    plt.gca().invert_yaxis()
    plt_savefig('plots2/nadir3.png')

    # -------------------------------------------------------------------------    
    # convert skeleton image to polygons
    csize, maxIter = 100, 999
    polys = trace_skeleton.traceSkeleton(msk, 0, 0, nx, ny, csize, maxIter, None)
    mls = MultiLineString(polys)
    print(npoints(mls))
    
    # plot the polygons
    plt.clf()
    plot_mls(mls)
    #plt.gca().invert_yaxis()
    plt_savefig('plots2/nadir4.png')
    
    # simplify polygons
    mls = mls.simplify(tolerance=2, preserve_topology=False)
    print(npoints(mls))
    
    # plot the polygons
    plt.clf()
    plot_mls(mls)
    #plt.gca().invert_yaxis()
    plt_savefig('plots2/nadir5.png')

    # -------------------------------------------------------------------------    
    # convert to a mesh representation and undo all the point transformations
    points1, edges =  mls_to_mesh(mls)
    z = np.zeros((points1.shape[0], 1))
    points1 = np.hstack((points1, z))
    p = np.array([box[0],    box[1],    0])
    s = np.array([box[2]/nx, box[3]/ny, 0])
    points1 = points1 * s + p
    points1 = points1 @ np.linalg.inv(rot_mat) + mean_p

    # plot the polygons
    plt.clf()
    plt.plot(points[:,0], points[:,1], '.', zorder=1)
    plot_mesh(points1, edges)
    plt_savefig('plots2/nadir6.png')

    # -------------------------------------------------------------------------    
    # Calculate elevation for the mesh points
    points1[:,2] = 0
    for i_point, point in enumerate(points1):
        dist = 5
        pts = points-points1[i_point,:]
        dis = np.linalg.norm(pts[:,:2], axis=1)
        for iter in range(5):
            z = pts[dis<dist,2]
            if len(z)<20:
                dist *= 2
            else:
                points1[i_point, 2] = np.median(z) 
                break

    # ------------------------------------------------------------------------- 
    # save results
    csv_path = 'skeleton.csv'
    with open(csv_path, 'w') as fid:
        fid.write('Classification,U//CUI\n')
        fid.write('Segment_Number,St_Easting,St_Northing,St_Elev,End_Easting,End_Northing,End_Elev\n')
        for iedge, edge in enumerate(edges):
            a = points1[edge[0],:] 
            b = points1[edge[1],:] 
            nums = [iedge, a[0], a[1], a[2], b[0], b[1], b[2] ]
            fid.write(','.join(map(str, nums))+'\n')
        
    writer = pd.ExcelWriter('skeleton.xlsx', engine='xlsxwriter')
    pd.DataFrame(edges).to_excel(writer, sheet_name='edges')
    pd.DataFrame(points1).to_excel(writer, sheet_name='points')
    writer.save()

    a = 1


# =============================================================================
if __name__ == "__main__":
    main()
