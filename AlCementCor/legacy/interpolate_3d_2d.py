import math

# from interpolate import interpolate_values
import numpy as np
from input_file import ExternalInput
from input_file import process_input_tensors


def interpolate_3d_2d():
    loaded_vars = process_input_tensors('../CementOutput.json')

    x_coords = loaded_vars['X']
    y_coords = loaded_vars['Y']
    z_coords = loaded_vars['Z']

    x_init = x_coords[:, 0]
    y_init = y_coords[:, 0]
    z_init = z_coords[:, 0]

    init_coords = np.c_[x_init, y_init, z_init]

    # for key, value in loaded_vars.items():
    #     print(f"{key}")

    group1 = loaded_vars[ExternalInput.OUTSIDE_P]
    group2 = loaded_vars[ExternalInput.INSIDE_P]

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # ax.scatter(group1[:, 0], group1[:, 1], group1[:, 2], c='b', marker='o')
    # ax.scatter(group2[:, 0], group2[:, 1], group2[:, 2], c='r', marker='o')

    # ax.set_xlabel('X-axis')
    # ax.set_ylabel('Y-axis')
    # ax.set_zlabel('Z-axis')

    unique_x1 = np.sort(np.unique(group1[:, 0]))
    unique_x2 = np.sort(np.unique(group2[:, 0]))

    z_sel = np.max(np.unique(group1[:, 2]))
    x_sel1 = unique_x1[1]
    x_sel2 = unique_x2[1]

    sel_id1 = np.arange(0, x_init.shape[0])[(init_coords[:, 0] == x_sel1) & (init_coords[:, 2] == z_sel)]
    sel_id2 = np.arange(0, x_init.shape[0])[(init_coords[:, 0] == x_sel2) & (init_coords[:, 2] == z_sel)]
    sel_pt1 = init_coords[sel_id1][0]
    sel_pt2 = init_coords[sel_id2][0]
    # group_sel = np.c_[sel_pt1, sel_pt2].T

    # ax.scatter(group_sel[:, 0], group_sel[:, 1], group_sel[:, 2], c='g', s=200)

    # displacements = loaded_vars["displacement"]
    displacements = loaded_vars["LE11"]
    init_disp = displacements[:, 0]
    sel_disp1 = init_disp[sel_id1]
    sel_disp2 = init_disp[sel_id2]

    # num_int_pts = 10
    # interpolated_points = np.linspace(sel_pt1, sel_pt2, num_int_pts)
    # inteprolated_disp = np.linspace(sel_disp1, sel_disp2, num_int_pts)
    # ax.scatter(interpolated_points[:, 0], interpolated_points[:, 1], interpolated_points[:, 2], c='g', s=10)

    # group = np.r_[group1, group2]
    # x_min, x_max = np.min(group1[:, 0]), np.max(group[:, 0])
    # y_min, y_max = np.min(group1[:, 1]), np.max(group[:, 1])

    # plt.show()

    distance = math.sqrt((sel_pt2[0] - sel_pt1[0]) ** 2 + (sel_pt2[1] - sel_pt1[1]) ** 2)

    return distance, sel_disp1, sel_disp2

    # loading = Expression("h1 + (h2-h1)/l*x[0]", val1, val2, l, degree=1)
