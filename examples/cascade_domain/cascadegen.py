from ntrfc.utils.geometry.airfoil_generators.naca_airfoil_creator import naca
from ntrfc.utils.geometry.domaingen_cascade import cascade_2d_domain,cascade_3d_domain
import pyvista as pv
from py2gmsh import (Mesh, Entity, Field)

profilepoints2d = naca("6504", 260, finite_te=False, half_cosine_spacing=True)

sortedPoly,psPoly, ssPoly, per_y_upper, per_y_lower, inletPoly, outletPoly = cascade_2d_domain(profilepoints2d,-0.9,0.2,0.4,"m",0.001,1, verbose=False)

# p=pv.Plotter()
# p.add_mesh(sortedPoly)
# p.add_mesh(per_y_upper)
# p.add_mesh(per_y_lower)
# p.add_mesh(inletPoly)
# p.add_mesh(outletPoly)
# p.show()
#
sortedPoly_lowz,sortedPoly_high,per_y_upper_lowz,per_y_upper_highz,per_y_lower_lowz,per_y_lower_highz,inletPoly_lowz ,inletPoly_highz,outletPoly_lowz,outletPoly_highz= cascade_3d_domain(sortedPoly,psPoly, ssPoly,  per_y_upper, per_y_lower, inletPoly, outletPoly, 0.04, 0.5, verbose=False)




def pyvista2gmsh_3d(sortedPoly_lowz,sortedPoly_high,per_y_upper_lowz,per_y_upper_highz, per_y_lower_highz, per_y_lower_lowz,filename):
    my_mesh = Mesh()


    def curveloop_gen(pointlist,mesh):
        sortedpoints_ps_to_ss_lower = pointlist
        for pt in sortedpoints_ps_to_ss_lower:
            gmshpoints.append(Entity.Point(pt, mesh=mesh))

        numpts = len(gmshpoints)
        for idx in range(numpts):
            p1 = gmshpoints[idx]
            p2 = gmshpoints[(idx+1) % numpts]
            gmshcurves.append(Entity.Curve([p1, p2], mesh=mesh))
        curveloop = Entity.CurveLoop(gmshcurves, mesh=mesh)
        return curveloop

    def curves_gen(mesh, pointlist):
        sortedpoints_ps_to_ss_lower = pointlist
        for pt in sortedpoints_ps_to_ss_lower:
            gmshpoints.append(Entity.Point(pt, mesh=mesh))
        numpts = len(gmshpoints)
        for idx in range(numpts-1):
            p1 = gmshpoints[idx]
            p2 = gmshpoints[(idx + 1) % numpts]
            gmshcurves.append(Entity.Curve([p1, p2], mesh=mesh))
        return gmshcurves



    lower_blade_points = [Entity.Point(pt,mesh=my_mesh) for pt in sortedPoly_lowz.points]
    lower_blade_curves = [Entity.Curve([lower_blade_points[idx],lower_blade_points[(idx + 1) % len(lower_blade_points)]],mesh=my_mesh) for idx in range(len(lower_blade_points))]
    lower_blade_curveloop = Entity.CurveLoop(lower_blade_curves,mesh=my_mesh)

    upper_blade_points = [Entity.Point(pt,mesh=my_mesh) for pt in sortedPoly_high.points]
    upper_blade_curves = [Entity.Curve([upper_blade_points[idx],upper_blade_points[(idx + 1) % len(upper_blade_points)]],mesh=my_mesh) for idx in range(len(upper_blade_points))]
    upper_blade_curveloop = Entity.CurveLoop(upper_blade_curves,mesh=my_mesh)


    # todo: check if ymax and ymin are named right or if its points&curves have to be renamned (are the points named right?)
    lower_inlet_points = [Entity.Point(pt,mesh=my_mesh) for pt in [per_y_lower_lowz.points[0], per_y_upper_lowz.points[0]]]
    lower_inlet_curves = [Entity.Curve([lower_inlet_points[0],lower_inlet_points[1]],mesh=my_mesh)]

    lower_outlet_points = [Entity.Point(pt,mesh=my_mesh) for pt in [per_y_upper_lowz.points[-1], per_y_lower_lowz.points[-1]]]
    lower_outlet_curves = [Entity.Curve([lower_outlet_points[0],lower_outlet_points[1]],mesh=my_mesh)]

    lower_ypermin_points = [lower_inlet_points[-1]] + [Entity.Point(pt,mesh=my_mesh) for pt in per_y_upper_lowz.points[1:-1]] +  [lower_outlet_points[0]]
    lower_ypermin_curves = [Entity.Curve([lower_ypermin_points[idx],lower_ypermin_points[(idx + 1) % len(lower_ypermin_points)]],mesh=my_mesh) for idx in range(len(lower_ypermin_points)-1)]

    lower_ypermax_points = [lower_outlet_points[-1]] + [Entity.Point(pt,mesh=my_mesh) for pt in per_y_lower_lowz.points[::-1][1:-1]] +  [lower_inlet_points[0]]
    lower_ypermax_curves = [Entity.Curve([lower_ypermax_points[idx],lower_ypermax_points[(idx + 1) % len(lower_ypermax_points)]],mesh=my_mesh) for idx in range(len(lower_ypermax_points)-1)]

    lower_domain_curveloop = Entity.CurveLoop([*lower_inlet_curves, *lower_ypermin_curves ,
                                           *lower_outlet_curves,*lower_ypermax_curves], mesh=my_mesh)

    lower_domain_face = Entity.PlaneSurface([lower_domain_curveloop,lower_blade_curveloop], mesh=my_mesh)



    upper_inlet_points = [Entity.Point(pt,mesh=my_mesh) for pt in [per_y_lower_highz.points[0], per_y_upper_highz.points[0]]]
    upper_inlet_curves = [Entity.Curve([upper_inlet_points[0],upper_inlet_points[1]],mesh=my_mesh)]

    upper_outlet_points = [Entity.Point(pt,mesh=my_mesh) for pt in [per_y_upper_highz.points[-1], per_y_lower_highz.points[-1]]]
    upper_outlet_curves = [Entity.Curve([upper_outlet_points[0],upper_outlet_points[1]],mesh=my_mesh)]

    upper_ypermin_points = [upper_inlet_points[-1]] + [Entity.Point(pt,mesh=my_mesh) for pt in per_y_upper_highz.points[1:-1]] +  [upper_outlet_points[0]]
    upper_ypermin_curves = [Entity.Curve([upper_ypermin_points[idx],upper_ypermin_points[(idx + 1) % len(upper_ypermin_points)]],mesh=my_mesh) for idx in range(len(upper_ypermin_points)-1)]

    upper_ypermax_points = [upper_outlet_points[-1]] + [Entity.Point(pt,mesh=my_mesh) for pt in per_y_lower_highz.points[::-1][1:-1]] +  [upper_inlet_points[0]]
    upper_ypermax_curves = [Entity.Curve([upper_ypermax_points[idx],upper_ypermax_points[(idx + 1) % len(upper_ypermax_points)]],mesh=my_mesh) for idx in range(len(upper_ypermax_points)-1)]

    upper_domain_curveloop = Entity.CurveLoop([*upper_inlet_curves, *upper_ypermin_curves ,
                                           *upper_outlet_curves,*upper_ypermax_curves], mesh=my_mesh)

    upper_domain_face = Entity.PlaneSurface([upper_domain_curveloop,upper_blade_curveloop], mesh=my_mesh)

    inlet_curves = [Entity.Curve([lower_inlet_points[0],upper_inlet_points[0]],mesh=my_mesh),
                    *upper_inlet_curves,
                    Entity.Curve([upper_inlet_points[-1], lower_inlet_points[-1]], mesh=my_mesh),
                    *lower_inlet_curves
                    ]

    inlet_curveloop = Entity.CurveLoop(inlet_curves,mesh=my_mesh)

    inlet_face = Entity.PlaneSurface([inlet_curveloop],mesh=my_mesh)



    # inlet_lowtohigh_lower= curves_gen(my_mesh,[per_y_lower_lowz.points[-1],per_y_lower_highz.points[-1]])
    # inlet_lowtohigh_upper = curves_gen(my_mesh,[per_y_upper_highz.points[-1],per_y_lower_highz.points[-1]])
    # outlet_hightolow_lower= curves_gen(my_mesh,[per_y_lower_highz.points[0],per_y_lower_lowz.points[0]])
    # outlet_lowtohigh_upper = curves_gen(my_mesh,[per_y_upper_highz.points[0],per_y_lower_highz.points[0]])
    #
    # ylower_curveloop =Entity.CurveLoop([*inlet_lowtohigh_lower,*per_y_lower_line_highz,*outlet_hightolow_lower,*per_y_lower_line_lowz],mesh=my_mesh)
    #
    # outlet_curve=curveloop_gen([per_y_lower_lowz.points[-1],per_y_lower_highz.points[-1], per_y_upper_highz.points[-1], per_y_upper_lowz.points[-1]], my_mesh=my_mesh)
    # inlet_curve=curveloop_gen([per_y_lower_lowz.points[0],per_y_lower_highz.points[0], per_y_upper_highz.points[0], per_y_upper_lowz.points[0]], my_mesh=my_mesh)
    # inletface = Entity.PlaneSurface([inlet_curve], mesh=my_mesh)
    # outletface = Entity.PlaneSurface([outlet_curve], mesh=my_mesh)
    #

    # zspan_high_face = Entity.PlaneSurface([upper_domain_curve,upper_blade_curve], mesh=my_mesh)

    # create fields
    f1 = Field.MathEval(mesh=my_mesh)
    grading = 1.1
    he = 0.01
    f1.F = '(abs(y-0.5)*({grading}-1)+{he})/{grading}'.format(grading=grading,
                                                              he=he)
    # create minimum field
    fmin = Field.Min(mesh=my_mesh)
    fmin.FieldsList = [f1]  # could add more fields in the list if necessary

    # set the background field as minimum field
    my_mesh.setBackgroundField(fmin)

    # set max element size
    my_mesh.Options.Mesh.CharacteristicLengthMax = 0.1

    # adding Coherence option
    my_mesh.Coherence = True
    my_mesh.writeGeo(filename)



def pyvista2gmsh_2d(sspoly, pspoly, yupperpoly, ylowerpoly,filename):
    my_mesh = Mesh()
    bladepts = []
    sortedpoints_ps_to_ss = [*pspoly.points, *sspoly.points[::-1]]

    for pt in sortedpoints_ps_to_ss:
        bladepts.append(Entity.Point(pt, mesh=my_mesh))
    bladecurves = []
    numpts_blade = len(bladepts)
    for idx in range(numpts_blade):
        p1 = bladepts[idx]
        p2 = bladepts[(idx+1)%numpts_blade]
        bladecurves.append(Entity.Curve([p1, p2], mesh=my_mesh))

    # create curveloop
    blade = Entity.CurveLoop(bladecurves, mesh=my_mesh)


    yupperpolypts = []
    yupperpolycurves = []
    for pt in yupperpoly.points:
        yupperpolypts.append(Entity.Point(pt, mesh=my_mesh))
    yupperpolypts.reverse()
    for i in range(len(yupperpolypts)-1):
        yupperpolycurves.append(Entity.Curve([yupperpolypts[i],yupperpolypts[i+1]], mesh=my_mesh))

    ylowerpts = []
    ylowerpolycurves = []
    for pt in ylowerpoly.points:
        ylowerpts.append(Entity.Point(pt, mesh=my_mesh))
    for i in range(len(ylowerpts)-1):
        ylowerpolycurves.append(Entity.Curve([ylowerpts[i],ylowerpts[i+1]], mesh=my_mesh))

    inletpts = [ylowerpts[-1],yupperpolypts[0]]
    inletcurves = []
    for i in range(len(inletpts)-1):
        inletcurves.append(Entity.Curve([inletpts[i], inletpts[i+1]], mesh=my_mesh))

    outletpts = [yupperpolypts[-1],ylowerpts[0]]
    outletcurves = []
    outletpts.reverse()
    for i in range(len(outletpts)-1):
        outletcurves.append(Entity.Curve([outletpts[i],outletpts[i+1]], mesh=my_mesh))



    # create surface
    domaincurveloop = Entity.CurveLoop([*inletcurves,*yupperpolycurves,*outletcurves,*ylowerpolycurves], mesh=my_mesh)
    domainface = Entity.PlaneSurface([domaincurveloop,blade], mesh=my_mesh)

    # create fields
    f1 = Field.MathEval(mesh=my_mesh)
    grading = 1.1
    he = 0.01
    f1.F = '(abs(y-0.5)*({grading}-1)+{he})/{grading}'.format(grading=grading,
                                                              he=he)
    # create minimum field
    fmin = Field.Min(mesh=my_mesh)
    fmin.FieldsList = [f1]  # could add more fields in the list if necessary

    # set the background field as minimum field
    my_mesh.setBackgroundField(fmin)

    # set max element size
    my_mesh.Options.Mesh.CharacteristicLengthMax = 0.1

    # adding Coherence option
    my_mesh.Coherence = True
    my_mesh.writeGeo(filename)
    return my_mesh


blade_gmshgeo = pyvista2gmsh_2d(psPoly, ssPoly, per_y_upper, per_y_lower,"domain2d.geo")
blade_gmshgeo3d = pyvista2gmsh_3d(sortedPoly_lowz,sortedPoly_high,per_y_upper_lowz,per_y_upper_highz, per_y_lower_highz, per_y_lower_lowz,"domain3d.geo")
