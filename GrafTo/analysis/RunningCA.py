import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import math
from MDAnalysis.transformations import wrap

class ContactAngle:
    """
    Class to calculate the contact angle of a droplet on a surface.
    The contact angle is calculated by fitting a circle to the droplet and calculating the angle between the tangent line to the circle at the intersection with a base line and the base line itself.
    The angle is calculated for both sides of the droplet and the average is taken as the contact angle.

    Initial attributes:
        - root: The root folder where the data is stored.
        - universe: The MDAnalysis universe object containing the data.

    """

    def __init__(self, root, universe):
        self.root = root
        self.universe = universe

    def circle_residuals(self,parameters, points):
        """
        Calculate the residuals of the circle fit.
        Inputs:
            - parameters: The parameters of the circle fit in the form (x_center, y_center, radius).
            - points: The coordinates of the points to fit in the form (x, y).
        Output:
            - residuals: The residuals of the circle fit.
        """

        x_center, y_center, radius = parameters
        residuals = np.sqrt((points[:, 0] - x_center)**2 + (points[:, 1] - y_center)**2) - radius
        return residuals

    def fit_circle(self, v, vc_guess):
        """
        Fit a circle to a set of points.
        Inputs:
            - v: The coordinates of the points to fit in the form (x, y).
            - vc_guess: An initial guess for the center of the circle in the form (x_center, y_center).
        Output:
            - v_center: The center of the fitted circle in the form (x_center, y_center).
            - radius: The radius of the fitted circle.
        """ 

        initial_guess = np.concatenate((vc_guess, [1.0]))
        result = least_squares(self.circle_residuals, initial_guess, args=(v,))
        x_center, y_center, radius = result.x
        v_center = np.array([x_center, y_center])
        return v_center, radius

    def tangent_line_to_circle(self, vc, R, point_contact):
        """
        Calculate the tangent line to a circle at a given point.
        Inputs:
            - vc: The center of the circle in the form (x_center, y_center).
            - R: The radius of the circle.
            - point_contact: The coordinates of the point of contact in the form (x, y).
        Output:
            - tangent_line: A function that takes an x value and returns the corresponding y value for the tangent line.
            - slope_tangent: The slope of the tangent line
        """

        x1, y1 = point_contact
        h, k = vc

        slope_tangent = -(x1 - h) / (y1 - k)
        def tangent_line(x):
            return slope_tangent * (x - x1) + y1
        
        #print(slope_tangent)
        return tangent_line, slope_tangent

    def angle_between_lines(self, slope_tangent, slope_base_line):
        angle = np.arctan(np.abs((slope_tangent - slope_base_line) / (1 + slope_tangent * slope_base_line)))
        return np.degrees(angle)


    def plot_fit(self, ax, va, vb, vc, R, extent,baseLineF, slopeF, label=None, contour=False):
        """
        Plot the fit of a circle to a set of points and the tangent line to the circle at the intersection with a base line.
        Inputs:
            - ax: The matplotlib axis to plot on.
            - va: The coordinates of the points to fit in the form (x, y).
            - vb: The coordinates of the points to discard in the form (x, y).
            - vc: The center of the fitted circle in the form (x_center, y_center).
            - R: The radius of the fitted circle.
            - baseLineF: The function of the base line in the form y = BaseLine.
            - slopeF: The slope of the base line.
            - label: The label of the plot.
            - contour: Whether to plot the points as a contour plot.
        Output:
            - theta: The angles between the tangent lines and the base line for both sides of the droplet.
        """

        ax.set_aspect('equal')
        theta_fit = np.linspace(-np.pi, np.pi, 180)
        v_fit =  vc + R*np.column_stack((np.cos(theta_fit), np.sin(theta_fit)))
        xvals = np.linspace(va[:,0].min()-30, va[:,0].max()+30,100)
        baseLine = np.array([(x,baseLineF(x)) for x in xvals])

        def find_intersections(vc, R, v_line):
            """
            Find the intersection points of a circle and a line.
            Inputs:
                - vc: The center of the circle in the form (x_center, y_center).
                - R: The radius of the circle.
                - v_line: The coordinates of the line in the form [(x1, y1), (x2, y2)].
            Output:
                - intersection_points: The coordinates of the intersection points in the form [(x1, y1), (x2, y2)].
            """ 

            a = np.dot(v_line[1] - v_line[0], v_line[1] - v_line[0])
            b = 2 * np.dot(v_line[1] - v_line[0], v_line[0] - vc)
            c = np.dot(v_line[0] - vc, v_line[0] - vc) - R**2

            discriminant = b**2 - 4*a*c

            if discriminant < 0:
                return None  # No real solutions, no intersection

            t1 = (-b + np.sqrt(discriminant)) / (2 * a)
            t2 = (-b - np.sqrt(discriminant)) / (2 * a)

            intersection_points = [tuple(v_line[0] + t1 * (v_line[1] - v_line[0])), tuple(v_line[0] + t2 * (v_line[1] - v_line[0]))]

            return intersection_points

        intersection_points = find_intersections(vc, R, baseLine)
        #print("Intersection points:", intersection_points)

        if contour:
            ax.plot(va[:,0], va[:,1], ls='none', marker='o', mec='black', mfc='none', mew=0.8, label='Isodensity', ms=3, alpha=0.7)

        ax.plot(vb[:,0], vb[:,1], ls='none', marker='x', color='gray', ms=3, label="Discarded")
        ax.plot(baseLine[:,0], baseLine[:,1], color='black', ls="solid", lw=2, label='Base', zorder=10)
        ax.plot(v_fit[:,0], v_fit[:,1], ls='dashed', c='red' , label='Fit', lw=2, alpha=1)
        ax.plot(vc[0], vc[1], marker='o', ls='none', c='red', mfc='white', mew=1.5, mec='xkcd:tomato', alpha=1)
        
        thetas = []
        i = 0
        for point,side in zip(intersection_points,["right","left"]):    
            tangent_line_func, slope_tangent = self.tangent_line_to_circle(vc, R, point)
            
            # Debug prints for intersection points and slopes
            #print(f"Side: {side}, Intersection point: {point}, Slope of tangent: {slope_tangent}")

            angle_between = self.angle_between_lines(slope_tangent, slopeF)
            
            if slope_tangent < 0:
                if side=="right":
                    theta = angle_between
                else:
                    theta = 180 - angle_between
            else:
                if side=="right":
                    theta = 180 - angle_between
                else:
                    theta = angle_between

            if not label:
                label = ""

            xx = np.linspace(point[0] - 150, point[0] + 150, 100)
            yy = tangent_line_func(xx)
            mask = (yy > extent[2]) & (yy < extent[3])  # extent = [xmin, xmax, ymin, ymax]
            xx = xx[mask]
            yy = yy[mask]
                
            ax.plot([vc[0], point[0]], [vc[1], point[1]], color='black', lw=2, ls='dashed', zorder=5)
            ax.plot(xx, yy, color='black', lw=2, ls='dashed', zorder=5)

            ax.legend(loc='upper left', handlelength=1.8, labelspacing=0.1, fontsize=11)
            tlabel = "$\\theta^{%s}_{Y}$" % side + "$ = %.2f^{\\circ}$" % theta
            ax.text(0.9, 0.9-i*0.2, tlabel, color="white", horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, fontsize=12, zorder=5)
            thetas.append(theta)
            i+=1

        return thetas

    def perp_line(self,xc, yc, x_intersect, y_intersect):
        """
        Calculate the perpendicular line to a line passing through a point.
        Inputs:
            - xc: The x coordinate of the point.
            - yc: The y coordinate of the point.
            - x_intersect: The x coordinate of the point the line passes through.
            - y_intersect: The y coordinate of the point the line passes through.
        Output:
            - a_perp: The slope of the perpendicular line.
            - b_perp: The y-intercept of the perpendicular line.
        """

        a,b = (yc-y_intersect)/(xc-x_intersect), yc - a*xc
        a_perp = -1/a
        b_perp = y_intersect - a_perp*x_intersect

        return a_perp, b_perp

    def density(self, fig, ax, nframes, axd, cuts=None, selection=None, cmap="coolwarm", mols_to_plot=None, delta=0):
        """
        Plot the density of a selection of atoms in the universe.
        Inputs:
            - fig: The matplotlib figure to plot on.
            - ax: The matplotlib axis to plot on.
            - nframes: The number of frames to analyze.
            - axd: The axis to plot the density on (0 for x, 1 for y, 2 for z).
            - cuts: The limits of the plot in the form (xmin, xmax, ymin, ymax).
            - selection: The selection of atoms to analyze.
            - cmap: The colormap to use for the plot.
            - mols_to_plot: The molecules to plot on the density plot.
            - delta: The offset of the molecules from the density plot.
        Output:
            - fig: The matplotlib figure.
            - ax: The matplotlib axis.
            - dens: The density data.
            - img: The density plot.
        """
        from MDAnalysis.analysis.density import DensityAnalysis        

        try:
            u = self.universe
        except AttributeError:
            raise Exception("No universe found")

        if selection is not None:
            SOLV = u.select_atoms(selection, updating=True)

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
        
        coord_axes = [i for i in range(3) if i != axd]
        centers_x = 0.5 * (DENS_edges[coord_axes[0]][:-1] + DENS_edges[coord_axes[0]][1:])
        centers_y = 0.5 * (DENS_edges[coord_axes[1]][:-1] + DENS_edges[coord_axes[1]][1:])

        extent = [centers_x[0], centers_x[-1],centers_y[0], centers_y[-1],]

        img = ax.imshow(dens[axd], cmap=cmap, vmin=0, vmax=0.015, aspect="equal", interpolation="bicubic", alpha=1.0, origin="lower", extent=extent)
        fig.colorbar(img, ax=ax, label="Density ($\\AA^{-2}$)", shrink=0.7)

        if mols_to_plot is not None:
            for mol in mols_to_plot:
                sel = u.select_atoms(f"name {mol} or resname {mol}")
                if not delta:
                    delta = 0
                ax.scatter(sel.positions[:,0], sel.positions[:,2]+delta, s=0.2, c=mols_to_plot[mol], alpha=0.7, zorder=3)

        return fig, ax, dens, img, extent

    def rotate_line(self, base_line, angle_degrees, center_point):
        """
        Rotate a line around a center point.
        Inputs:
            - base_line: The y-intercept of the line.
            - angle_degrees: The angle to rotate the line by.
            - center_point: The center point to rotate the line around in the form (x_center, y_center).
        Output:
            - rotated_line_function: A function that takes an x value and returns the corresponding y value for the rotated line.
            - slope_rotated_line: The slope of the rotated line.
        """ 

        angle_radians = math.radians(angle_degrees)

        def rotated_line_function(x):
            translated_x = x - center_point[0]
            translated_y = base_line - center_point[1]

            rotated_x = translated_x * math.cos(angle_radians) - translated_y * math.sin(angle_radians)
            rotated_y = translated_x * math.sin(angle_radians) + translated_y * math.cos(angle_radians)

            final_x, final_y = rotated_x + center_point[0], rotated_y + center_point[1]

            return final_y
        
        slope_rotated_line = math.tan(angle_radians)
        
        return rotated_line_function, slope_rotated_line

    def calc_contact_angle(self, solvent_beads, nframes, averaged_axis=1, cmap="coolwarm", baseLine=15, distFromBase=25, contour=False, cuts=None, fig=None, tilt=None, mols_to_plot=None, delta=0):
        """
        Calculate the contact angle of a droplet on a surface.
        Inputs:
            - solvent_beads: The beads composing the droplet.
            - nframes: The number of frames to analyze.
            - averaged_axes: The axis to be averaged in the density calculation (0 for x, 1 for y, 2 for z). If more than one, produces a figure for each.
            - cmap: The colormap to use for the density plot.
            - baseLine: The y-intercept of the base line.
            - distFromBase: The distance from the base line to the droplet.
            - contour: Whether to plot the points as a contour plot.
            - cuts: The limits of the plot in the form (xmin, xmax, ymin, ymax).
            - tilt: The angle to tilt the base line by.
            - molsDict: The molecules to plot on the density plot.
            - delta: The offset of the molecules from the density plot.
        Output:
            - fig: The matplotlib figures.
            - ax: The matplotlib axes.
            - theta: The contact angle of the droplet.
        """

        from scipy.stats import gaussian_kde
        from scipy.signal import find_peaks
        selection, selection_beads, selection_cuts = "","",""

        beads = {"solvent": solvent_beads}

        try:
            u = self.universe
            transform = wrap(u.atoms)
            u.trajectory.add_transformations(transform)
            ts = u.trajectory[-1]

        except AttributeError:
            raise Exception("No universe found")
        
        if len(beads["solvent"])<2:
            selection_beads = f"name {beads["solvent"][0]} or resname {beads["solvent"][0]}"
        else:
            selection_beads = " or ".join([f"name {bead} or resname {bead}" for bead in beads["solvent"]])
            
        if cuts is not None:
            selection_cuts = f" and (prop z > {cuts[2]} and prop z < {cuts[3]} and prop x > {cuts[0]} and prop x < {cuts[1]})"

        selection = selection_beads + selection_cuts

        fig, ax = plt.subplots(1,1)
        
        _, _, dens, _, extent = self.density(fig, ax, nframes, averaged_axis, cuts=cuts, cmap=cmap, selection=selection, mols_to_plot=mols_to_plot, delta=delta)

        data = dens[averaged_axis]
        kde = gaussian_kde(data.flatten("C"))
        x = np.linspace(data.min(),data.max(),100)
        y = [0] + list(kde(x)/nframes)
        x = [0] + list(x)

        peak_indices, peak_dict = find_peaks(y, height=0.1)
        cutDen = np.mean([x[i] for i in peak_indices])
        f,a = plt.subplots(figsize=(5,4))
        a.plot(x,y)
        a.set_xlabel("Density")
        a.set_ylabel("Frequency")
        a.set_title("KDE of the density data")

        for p in peak_indices:
            a.scatter(x[p], y[p], c="red")

        cs = ax.contour(data, levels=[cutDen], extent=extent, colors="none", linewidths=1.5, alpha=0)
        paths = cs.collections[0].get_paths()[0]
        v = np.array(paths.vertices)
        
        slopeF = 0
        if tilt is not None:      
            pc = np.mean(v, axis=0)
            baseLineF, slopeF = self.rotate_line(baseLine, tilt, pc)
        else:
            baseLineF = lambda x: baseLine

        va = v[v[:,1] > distFromBase + np.array([baseLineF(x) for x in v[:,0]])]
        vb = v[v[:,1] < distFromBase + np.array([baseLineF(x) for x in v[:,0]])]
        vc, R = self.fit_circle(va, np.mean(va, axis=0))

        theta = self.plot_fit(ax, va, vb, vc, R, extent,baseLineF=baseLineF, slopeF=slopeF, contour=contour)

        if cuts is not None:
            ax.set_xlim(0,cuts[1]-cuts[0])
            ax.set_ylim(0,cuts[3]-cuts[2])
            
        ax.set_ylabel("z $(\\AA)$")
        ax.set_xlabel("x $(\\AA)$")

        return fig, ax, theta
