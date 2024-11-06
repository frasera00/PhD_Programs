import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import math
from MDAnalysis.transformations import wrap


class ContactAngle:
    def __init__(self, root, universe):
        self.folder = root
        self.universe = universe

    def circle_residuals(self,parameters, points):
        x_center, y_center, radius = parameters
        residuals = np.sqrt((points[:, 0] - x_center)**2 + (points[:, 1] - y_center)**2) - radius
        return residuals

    def fit_circle(self, v, vc_guess):
        initial_guess = np.concatenate((vc_guess, [1.0]))
        result = least_squares(self.circle_residuals, initial_guess, args=(v,))
        x_center, y_center, radius = result.x
        v_center = np.array([x_center, y_center])
        return v_center, radius

    def tangent_line_to_circle(self, vc, R, point_contact):
        x1, y1 = point_contact
        h, k = vc

        # Calculate the slope of the tangent line
        slope_tangent = -(x1 - h) / (y1 - k)

        # The equation of the tangent line: y - y1 = m(x - x1)
        def tangent_line(x):
            return slope_tangent * (x - x1) + y1

        return tangent_line, slope_tangent

    def angle_between_lines(self, slope_tangent, slope_base_line):
        angle = np.arctan(np.abs((slope_tangent - slope_base_line) / (1 + slope_tangent * slope_base_line)))
        return np.degrees(angle)


    def plot_fit(self, ax, va, vb, vc, R, baseLineF, slopeF, label=None, contour=False):
        """
        Plot data and a fitted circle.
        Inputs:
            ax : axis to plot
            va : data to fit
            vb : data to discard
            vc : center vector
            R : fit circle radius
        Output:
            v_fit: fit coords
        """

        ax.set_aspect('equal')
        theta_fit = np.linspace(-np.pi, np.pi, 180)
        v_fit =  vc + R*np.column_stack((np.cos(theta_fit), np.sin(theta_fit)))
        xvals = np.linspace(va[:,0].min()-30, va[:,0].max()+30,100)
        baseLine = np.array([(x,baseLineF(x)) for x in xvals])

        def find_intersections(vc, R, v_line):

            # Calculate the coefficients for the quadratic equation of the intersection
            a = np.dot(v_line[1] - v_line[0], v_line[1] - v_line[0])
            b = 2 * np.dot(v_line[1] - v_line[0], v_line[0] - vc)
            c = np.dot(v_line[0] - vc, v_line[0] - vc) - R**2

            discriminant = b**2 - 4*a*c

            if discriminant < 0:
                return None  # No real solutions, no intersection

            t1 = (-b + np.sqrt(discriminant)) / (2 * a)
            t2 = (-b - np.sqrt(discriminant)) / (2 * a)

            intersection_points = [tuple(v_line[0] + t1 * (v_line[1] - v_line[0])),
                                tuple(v_line[0] + t2 * (v_line[1] - v_line[0]))]

            return intersection_points

        intersection_points = find_intersections(vc, R, baseLine)
        print("Intersection points:", intersection_points)
        # plot data
        if contour:
            ax.plot(va[:,0], va[:,1], ls='none', marker='o', mec='black', mfc='none', mew=0.8, label='Isodensity', ms=3, alpha=0.7)

        ax.plot(vb[:,0], vb[:,1], ls='none', marker='x', color='gray', ms=3, label="Discarded")
        ax.plot(baseLine[:,0], baseLine[:,1], color='black', lw=2, label='Base')
        ax.plot(v_fit[:,0], v_fit[:,1], ls='dashed', c='red' , label='Fit', lw=2, alpha=1)
        ax.plot(vc[0], vc[1], marker='o', ls='none', c='red', mfc='white', mew=1.5, mec='xkcd:tomato', alpha=1)
        
        #plot tangent line to circle at intersection
        thetas = []
        i = 0
        for point,side in zip(intersection_points,["right","left"]):    
            tangent_line_func, slope_tangent = self.tangent_line_to_circle(vc, R, point)
            theta = 180 - self.angle_between_lines(slope_tangent, slopeF)
            
            if not label:
                label = ""

            xx = np.linspace(vc[0]-200,vc[0]+200,1000)
            yy = tangent_line_func(xx)
            mask = (yy > vc[1]-100) & (yy < vc[1]+100)
            xx = xx[mask]
            yy = yy[mask]
                
            ax.plot([vc[0], point[0]], [vc[1], point[1]], color='black', lw=2, ls='dashed', zorder=5)
            ax.plot(xx, yy, color='black', lw=2, ls='dashed', zorder=5)

            ax.legend(loc='upper left', handlelength=1.8, labelspacing=0.1, fontsize=11)
            tlabel = "$\\theta^{%s}_{Y}$" % side + "$ = %.2f^{\circ}$" % theta
            ax.text(0.9, 0.9-i*0.08, tlabel, color="white", horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, fontsize=12, zorder=5)
            thetas.append(theta)
            i+=1

        return thetas

    def perp_line(self,xc, yc, x_intersect, y_intersect):   
        a = (yc-y_intersect)/(xc-x_intersect)
        b = yc - a*xc
        a_perp = -1/a
        b_perp = y_intersect - a_perp*x_intersect

        return a_perp, b_perp

    def density(self, fig, ax, nframes, axd, cuts=None, selection=None, cmap="coolwarm", mols_to_plot=None, delta=0):
        from MDAnalysis.analysis.density import DensityAnalysis        

        try:
            u = self.universe
        except AttributeError:
            raise Exception("No universe found")

        if selection is not None:
            SOLV = u.select_atoms(selection)

        totalframes = len(u.trajectory)
        start = int( totalframes - nframes )
        if start < 0:
            start = totalframes
            nframes = totalframes
        print(f"{nframes} frames from {totalframes} total. Starting from frame {start}.")

        D = DensityAnalysis(SOLV, delta=1.0)
        D.run(start=start, verbose=True)
        DENS_edges = D.results.density.edges

        density = D.results.density.grid
        dens = {}
        dens[axd] = np.mean(density[:,:,:],axis=axd)
        dens[axd] = np.swapaxes(dens[axd],0,1)
        
        ts = u.trajectory[-1]
        if cuts is not None:
            extent = cuts
        else:
            ts = u.trajectory[-1]
            extent = [0,ts.dimensions[0],0,ts.dimensions[2]]
        img = ax.imshow(dens[axd], cmap=cmap, vmin=0, vmax=0.015, aspect="equal", interpolation="bicubic", alpha=1.0, origin="lower", extent=extent)
        fig.colorbar(img, ax=ax, label="Density ($\\AA^{-2}$)", shrink=0.7)

        if mols_to_plot is not None:
            for mol in mols_to_plot:
                sel = u.select_atoms(f"name {mol}")
                ax.scatter(sel.positions[:,0], sel.positions[:,2]+delta, s=0.2, c=mols_to_plot[mol], alpha=0.7, zorder=3)

        return fig, ax, dens, img

    def rotate_line(self, base_line, angle_degrees, center_point):
        """
        Rotate a line around a center point.

        Parameters:
        - base_line: The constant line in the form y = BaseLine.
        - angle_degrees: The angle in degrees by which to rotate the line.
        - center_point: The center point of rotation in the form (xc, yc).

        Returns:
        - A function that takes an x value and returns the corresponding y value for the rotated line.
        """

        # Convert angle to radians
        angle_radians = math.radians(angle_degrees)

        def rotated_line_function(x):
            # Translate the coordinates to the origin
            translated_x = x - center_point[0]
            translated_y = base_line - center_point[1]

            # Rotate the translated coordinates
            rotated_x = translated_x * math.cos(angle_radians) - translated_y * math.sin(angle_radians)
            rotated_y = translated_x * math.sin(angle_radians) + translated_y * math.cos(angle_radians)

            # Translate the rotated coordinates back to the original center point
            final_x = rotated_x + center_point[0]
            final_y = rotated_y + center_point[1]

            return final_y
        
        slope_rotated_line = math.tan(angle_radians)
        
        return rotated_line_function, slope_rotated_line

    def calc_contact_angle(self, solvent, nframes, axis=[1], cmap="coolwarm", baseLine=15, distFromBase=25, contour=False, cuts=None, fig=None, tilt=None, molsDict=None, delta=0):
        from scipy.stats import gaussian_kde
        from scipy.signal import find_peaks
        selection, selection_beads, selection_cuts = "","",""

        beads = {"water":["W","SW","TW"], "toluene":["TOLU"], "octane":["OCT"]}

        try:
            u = self.universe
            transform = wrap(u.atoms)
            u.trajectory.add_transformations(transform)
            ts = u.trajectory[-1]

        except AttributeError:
            raise Exception("No universe found")
        
        if len(beads[solvent])<2:
            selection_beads = f"resname {beads[solvent][0]}"
        else:
            selection_beads = " or ".join([f"resname {bead}" for bead in beads[solvent]])
            
        if cuts is not None:
            selection_cuts = f" and (prop z > {cuts[2]} and prop z < {cuts[3]} and prop x > {cuts[0]} and prop x < {cuts[1]})"
            extent=cuts
        else:
            extent=[0,ts.dimensions[0],0,ts.dimensions[2]]

        selection = selection_beads + selection_cuts

        if fig is None:
            fig, ax = plt.subplots(1,1)
        else:
            ax = fig[1]
            fig = fig[0]
        
        figs = []
        for axd in axis:
            fig, ax, dens, _ = self.density(fig, ax, nframes, axd, cuts=cuts, cmap=cmap, selection=selection, mols_to_plot=molsDict, delta=delta)

            data = dens[axd]
            kde = gaussian_kde(data.flatten("C"))
            x = np.linspace(data.min(),data.max(),100)
            y = [0] + list(kde(x)/nframes)
            x = [0] + list(x)

            peak_indices, peak_dict = find_peaks(y, height=0.1)
            cutDen = np.mean([x[i] for i in peak_indices])
            f,a = plt.subplots()
            a.plot(x,y)

            for p in peak_indices:
                a.scatter(x[p], y[p], c="red")

            cs = ax.contour(data, levels=[cutDen], extent=extent, colors="none", linewidths=1.5, alpha=0)
            paths = cs.collections[0].get_paths()[0]
            v = np.array(paths.vertices)
            
            if tilt is not None:      
                pc = np.mean(v, axis=0)
                baseLineF, slopeF = self.rotate_line(baseLine, tilt, pc)
            else:
                baseLineF = lambda x: baseLine

            va = v[v[:,1] > distFromBase + np.array([baseLineF(x) for x in v[:,0]])]
            vb = v[v[:,1] < distFromBase + np.array([baseLineF(x) for x in v[:,0]])]
            vc, R = self.fit_circle(va, np.mean(va, axis=0))

            theta = self.plot_fit(ax, va, vb, vc, R, baseLineF=baseLineF, slopeF=slopeF, contour=contour)
            figs.append([fig,ax])

            if cuts is not None:
                ax.set_xlim(0,cuts[1]-cuts[0])
                ax.set_ylim(bottom=0)
                
            ax.set_ylabel("z $(\\AA)$")
            ax.set_xlabel("x $(\\AA)$")

        return figs, theta
