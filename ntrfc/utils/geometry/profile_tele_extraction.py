import numpy as np
from matplotlib import path
import pyvista as pv
from scipy.spatial import Delaunay
from scipy import interpolate

from ntrfc.utils.geometry.alphashape import calc_concavehull
from ntrfc.utils.math.vectorcalc import findNearest, vecDir
from ntrfc.utils.pyvista_utils.line import lines_from_points, polyline_from_points,refine_spline


# read the base-data
def read_data(sortedPoly,alpha):

    # set data to m from mm
    X = sortedPoly.points[:,0]
    Y = sortedPoly.points[:,1]
    Z = sortedPoly.points[:,2]
    # find smallest values for shifting to touch x and y axis
    move_X = min(X)
    move_Y = min(Y)
    # shift
    X = X - min(X)
    Y = Y - min(Y)
    idx = np.argsort(X)
    X = np.sort(X)
    Y = Y[idx]
    Z = Z[idx]
    # find geometry with boundary function. Given parameter is
    # sensibility for ignoring konvex outliners.
    ord_x,ia=np.unique(X,return_index=True)
    Y=Y[ia]
    Z=Z[ia]
    X=ord_x

    X,Y=calc_concavehull(X,Y,alpha)

    return X,Y,Z,move_X,move_Y

def split_up(ordering):
    """
    returns coordinates from the pseudo pressure and suction side. Argument
    are coordinates from profile
    """
    k_SS = 0;
    k_DS = 0;
    n = 0;
    ds = np.zeros([1,3])
    ss = np.zeros([1,3])
    # checking if the next point is above or under the smallest x-Element to
    # determine which direction the algorithm should run. Either clockwise or
    # counterclockwise
    if ordering[n,1] > ordering[n+1,1]:
        i=0
        #going through the points until x-value changes direction
        #everything else belongs to the other side
        while (ordering[i,0] < ordering[i+1,0]) or (ordering[i+1,0] < ordering[i+2,0]) or i < 10:
            ds[0,:]=ordering[0,:]
            ds=np.append(ds,ordering[[i+1],:],axis=0)
            k_DS=k_DS+1
            i=i+1
        idx=np.arange(i,len(ordering))
        # everything else add to suction side
        ss=np.vstack((ordering[idx,:],ordering[[0],:]))

    else:
        i=0

        while (ordering[i,0] < ordering[i+1,0]) or (ordering[i+1,0] < ordering[i+2],0):
            ss[0,:]=ordering[0,:]
            ss=np.append(ss,ordering[[i+1],:],axis=0)
            k_SS=k_SS+1
            i=i+1
        idx=np.arange(i,len(ordering))
        # everything else add to suction side
        ds=np.vstack((ordering[idx,:],ordering[[0],:]))

    return ss,ds



# x-Value of Interpolation via Chebychev-Nodes
def x_function_chebychev(x_min,x_max,n):
    """
    returns values from chebychev polynom and requests axial length of
    profile and number of points
    """
    # lower border
    b=x_min
    # upper border
    a=x_max
    # how many points of interpolation?
    k=np.array(range(0,n))
    n=len(k)
    # Chebychev-Nodes for an Interval [a;b] with n Elements
    x = 0.5*(a+b)+0.5*(b-a)*np.cos((2*k-1)/(2*n)*np.pi);
    return x

# Circumcenter function
def circumcenter(xvals,yvals):
    """
    calculates the circumcenter for three 2D points
    """
    x1, x2, x3, y1, y2, y3 = xvals[ 0 ], xvals[1], xvals[2], yvals[0],yvals[1], yvals[2]
    A = 0.5*((x2-x1)*(y3-y2)-(y2-y1)*(x3-x2))
    if A==0:
        print('Failed: Points are either colinear or not distinct')
        return 0
    xnum=((y3-y1)*(y2-y1)*(y3-y2))-((x2**2-x1**2)*(y3-y2)) + ((x3**2- x2**2)*(y2-y1))
    x=xnum/(-4*A)
    y=(-1*(x2-x1)/(y2-y1))*(x-0.5*(x1+x2))+0.5*(y1+y2)
    return x,y

def knn_search(x, D, K):
    #https://glowingpython.blogspot.com/2012/04/k-nearest-neighbor-search.html
    """ find K nearest neighbours of data among D """
    ndata = D.shape[0]
    K = K if K < ndata else ndata
    dist=np.zeros([len(D),30])
    #idx=np.zeros([len(D),30])
    for i in range(len(D)):
        # euclidean distances from the other points
        sqd = np.sqrt(((D[i,:] - x[:ndata,:])**2).sum(axis=1))
        psort= np.sort(sqd)
        dist[i,:]=psort[0:30]

    # return the indexes of K nearest neighbours
    #return idx,dist
    return dist


# Delauney triangulation
def delaunay_calc(SS_y,DS_y,x_interp,para_filt):
    """
    returns the middle points of the delaunay calc which is the camberline of the profile
    requests X and Y coordinates for pressure and suction side as well as
    filter parameter
    """

    # create data and sort for triangulation
    idx=np.argsort(np.hstack((x_interp,np.flipud(x_interp))))
    data_x=np.sort(np.hstack((x_interp,np.flipud(x_interp))))

    # create a closed figure
    data_y=np.hstack((DS_y,np.flipud(SS_y)))
    data_y=data_y[idx]

    # Delaunay triangulation
    points=np.vstack((data_x,data_y))
    tri = Delaunay(points.T)

    # center of circle around triangle
    tri1=tri.points[tri.vertices]
    tri_mid=np.zeros([len(tri1),2])
    for i in range(len(tri1)):
        tri_mid1=circumcenter(tri1[i,:,0],tri1[i,:,1])
        tri_mid[i,]=np.array(tri_mid1)

    # filter everything outside of the geometry with inpolygon. Requests
    # points for filter and geometry as polygon
    p=path.Path(np.vstack([np.hstack([x_interp,np.flipud(x_interp)]),np.hstack([DS_y,np.flipud(SS_y)])]).T)
    idx_in=p.contains_points(np.vstack([tri_mid[:,0],tri_mid[:,1]]).T)

    # setting data
    # distribute on X and Y
    tri_X=tri_mid[:,0]
    tri_Y=tri_mid[:,1]
    # use filter indices
    tri_X=tri_X[idx_in,]
    tri_Y=tri_Y[idx_in,]
    # build a single matrix
    tri=tri_mid[idx_in,:]
    ind = np.argsort(tri[:,0])
    #sort matrix
    tri=np.vstack([tri[ind,0],tri[ind,1]]).T
    # filter every potential outlier with knnsearch. Calculates distance to
    # next points. How many can be decided by the last parameter.
    # Increasing points results in more sensitive filtering
    d=knn_search(tri,tri,30)

    # para_filt[5] decides if 3 section filter are used or one single
    # filter for whole camberline
    if para_filt[0,4]==1:

        # first 20% of camberline
        #while i < round(len(tri)*0.2):
            # distance to next neighbour based on knnsearch is used as
            # filter criteria. Each value above the mean is deleted.
        idx_1_sec = np.argwhere(abs(d[:,1])< abs(np.mean(d[:,1])))
            # create index for points that will be deleted
        idx_1_sec=idx_1_sec[(idx_1_sec <= round(len(tri)*0.2))]
        #i = i + 1;

        # between 20 and 46% camberline. Second value should be shortly
        # after max curvature to avoid filtering too many points from
        # straighter section.
        idx_a= np.argwhere(abs(d[:,1])<1*abs(np.mean(d[:,1])))
        idx_b= np.argwhere(abs(d[:,28])<1*abs(np.mean(d[:,28])))
        # find unique filter parameter
        idx_2_sec = np.unique(np.vstack([idx_a,idx_b]));
        # create index for points that will be deleted
        idx_2_sec=idx_2_sec[(idx_2_sec<round(len(tri)*0.46))]
        idx_2_sec=idx_2_sec[(idx_2_sec>round(len(tri)*0.2))]

        # last 54% of camber
        idx_3_sec= np.argwhere(abs(d[:,1]) < abs(np.mean(d[:,1])))
        idx_3_sec=idx_3_sec[(idx_3_sec>round(len(tri)*0.46))]
        idx=np.hstack([idx_1_sec,idx_2_sec,idx_3_sec])

    else:
        idx=np.nonzero(abs(d[:,1])<abs(np.mean(d[:,1]))+para_filt[0,0])
        idx=idx[0]

    tri=tri[idx,:]
    return tri

# Filter the camberline and cut the front/end
def filter_camb(tri_y,camb,para_filt):
    """
    returns filtered camberline and requests camberline, filter parameter
    only deletes points near leading and trailing edge!
    """

    # gradient and curvature of the camberline in y-direction
    dc=np.gradient(tri_y)
    dcdc=np.gradient(dc)

    # selecting sections and building the mean curvature. Section 1 and last
    # will be used. Everything in between should already be filtered
    # beforehand.
    l=len(dcdc)/8
    m=np.mean(dcdc)

    # filtering of section 1 for leading edge
    dcdc_s1=dcdc[range(0,int(l))]
    # setting the range for the filter based on given sensibility
    dif=para_filt[0,1]
    # searching every element out of range mean + filter range
    idx = np.nonzero(abs(dcdc_s1)>m+dif);
    idx=idx[0]
    # setting new camberline
    if idx.size==0:
        idx=1

    cambb=camb[range(int(np.max(idx)*para_filt[0,3]),len(camb)),:]

    # last section for trailing edge
    dcdc_s4=dcdc[range(int(l*7),len(dcdc))]
    # algorithm equivalent to section 1
    dif1=para_filt[0,2]
    idx=np.argsort(abs(dcdc_s4)>m+dif1)
    cambb=cambb[1:len(cambb)-(len(dcdc_s4)-np.min(idx)),:]

    return cambb





def extract_vk_hk(sortedPoly, verbose=False):


    para_int_points = np.array([3000, 3000])
    para_filt = np.array([[1 * 10 ** -5, 3 * 10 ** -5, 8 * 10 ** -5, 1.1, 1]])


    # read data from cuts
    X, Y, Z= sortedPoly.points[::,0],sortedPoly.points[::,1],sortedPoly.points[::,2]


    # setting coordinates in right order and a single matrix
    coord = np.stack((X, Y, Z), axis=1)
    idx = np.argmin(X)
    coord = np.vstack((coord[idx:len(X), :], coord[0:idx, :]))

    # delete non-unique values to allow interpolation
    ordering = coord

    # splitting of the geometry in pseudo SS and DS (doesn't fulfill the real
    # definition, because it starts at smallest x-value not leading edge resp. trailing edge)
    ss, ds = split_up(ordering)

    # interpolate of geometry via chebychev polynom which improves the geometry
    # in comparison with equidistant points
    x_interp = x_function_chebychev(0, max(X), para_int_points[0])

    # interpolation of geometry with chebychev supporting points with piecewise
    # cubic hermite interpolation polynom

    # interpolation of suction side
    u, idx = np.unique(ss[:, 0], return_index=True)
    ss = ss[idx, :]

    SS_y = interpolate.pchip_interpolate(ss[:, 0], ss[:, 1], x_interp)
    DS_y = interpolate.pchip_interpolate(ds[:, 0], ds[:, 1], x_interp)


    tri = delaunay_calc(SS_y, DS_y, x_interp, para_filt)

    # sort and filter camberline derived from delaunay algorithm
    p_tri = np.sort(tri[:, 0])
    k = np.argsort(tri[:, 0])
    tri_y = tri[:, 1]
    camb = filter_camb(tri_y[k], np.vstack([p_tri, tri_y[k]]).T, para_filt)
    cama_evenx,cama_eveny = refine_spline(camb[::,0],camb[::,1],100)
    camberpoints = np.stack([cama_evenx,cama_eveny,np.zeros(len(cama_eveny))]).T
    camberline = lines_from_points(camberpoints)
    LE_camber = camberline.extract_cells(0)
    LE_dir = vecDir(LE_camber.points[-1]-LE_camber.points[0])
    TE_camber = camberline.extract_cells(camberline.number_of_cells-1)
    TE_dir = vecDir(TE_camber.points[0]-TE_camber.points[-1])

    profilepoly = polyline_from_points(np.vstack([np.stack([X,Y,Z]).T, np.stack([X[0],Y[0],Z[0]]).T]))


    camber_le_extension = pv.Line(LE_camber.points[0]-LE_dir*10,camberpoints[0],resolution = 10000)
    camber_te_extension = pv.Line(camberpoints[-1],TE_camber.points[0]-TE_dir*10,resolution = 10000)
    camberline_extended = lines_from_points(np.vstack([camber_le_extension.points,
                                                       camberpoints[1:-2],
                                                       camber_te_extension.points]))

    helpersurface = profilepoly.copy().extrude([0,0,-1],inplace=True)
    helpersurface= helpersurface.translate([0,0,.5],inplace=True)

    camberline_computed=camberline_extended.clip_surface(helpersurface,invert=False)


    le_ind = findNearest(np.stack([X, Y, Z]).T,camberline_computed.points[0])
    te_ind = findNearest(np.stack([X, Y, Z]).T,camberline_computed.points[-1])

    if verbose:
        p = pv.Plotter()
        p.add_mesh(camberpoints)
        p.add_mesh(profilepoly)
        p.add_mesh(camberline_computed)
        p.add_mesh(profilepoly.points[le_ind],color="red",point_size=20)
        p.add_mesh(profilepoly.points[te_ind],color="green",point_size=20)
        p.show()

    return le_ind, te_ind











