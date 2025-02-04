#!/usr/bin/env python
# -*- coding: utf-8 -*-
# LEAF_GRASP_NODE_2

import rospy
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from std_msgs.msg import String, Float64
from scipy import signal
import matplotlib.patches as patches

def leaf_grasp_node():
    rospy.init_node('leaf_grasp_node_2', anonymous=True)
    pub = rospy.Publisher('grasp_point_topic', String, queue_size=10)
    angle_pub = rospy.Publisher('grasp_angle_topic', Float64, queue_size=10)
    pre_grasp_pub = rospy.Publisher('pre_grasp_point_topic', String, queue_size=10)
    difference_pub = rospy.Publisher('grasp_diff_topic', String, queue_size=10)

    rospy.sleep(1.0)

    # Hardcoded camera intrinsic parameters
    fx = 1750.682853435758
    fy = 1749.705525555765
    cx = 707.8665402220275
    cy = 494.070091507807

    offset_x = -0.088
    offset_y = -0.003
    offset_z = 0.1317

    gantryX = 0.673
    gantryY = -0.150
    gantryZ = 0.270

    masked_image_path = '/home/buggspray/SDF_OUT/test2/aggrigated_masks0.png'
    depth_data_path = '/home/buggspray/SDF_OUT/test2/depth0.npy'

    masked_image = cv2.imread(masked_image_path)
    masked_image_rgb = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
    depth_data = np.load(depth_data_path)

    unique_colors = np.unique(masked_image_rgb.reshape(-1, masked_image_rgb.shape[2]), axis=0)

    leaf_centroids = []
    leaf_areas = []
    leaf_depths = []
    leaf_3d_points = []
    leaf_3d_points_ee = []
    graspable_areas = []
    contours_list = []
    min_area_threshold = 1500

    image_center_x, image_center_y = masked_image_rgb.shape[1] // 2, masked_image_rgb.shape[0] // 2
    max_distance_from_center = 50

    def compute_graspable_area(mask, kernel_size):
        convolved = signal.convolve2d(mask.astype('uint8'), np.ones((kernel_size, kernel_size)), boundary='symm', mode='same')
        graspable_area = np.where(convolved < np.amax(convolved) * 0.95, 1, 0)
        return graspable_area

    # Contour-based ellipse fitting function
    def fit_ellipse_contour_based(contour):
        if len(contour) >= 5:  # Minimum 5 points are needed for fitEllipse
            ellipse = cv2.fitEllipse(contour)
            return ellipse
        else:
            print("Not enough points to fit ellipse.")
            return None

    for color in unique_colors:
        if np.array_equal(color, [128, 0, 128]):
            continue

        color_mask = cv2.inRange(masked_image_rgb, color, color)

        leaf_area = np.sum(color_mask > 0)
        if leaf_area < min_area_threshold:
            continue

        labels = measure.label(color_mask)
        props = measure.regionprops(labels)

        _, contours, _ = cv2.findContours(color_mask.astype("uint8"), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        if contours:
            contour_max = max(contours, key=cv2.contourArea)
            MOMENT = cv2.moments(contour_max)
            if MOMENT["m00"] > 0:
                centroid_x = int(MOMENT["m10"] / MOMENT["m00"])
                centroid_y = int(MOMENT["m01"] / MOMENT["m00"])

                distance_to_center = np.sqrt((centroid_x - image_center_x) ** 2 + (centroid_y - image_center_y) ** 2)
                if distance_to_center > max_distance_from_center:
                    leaf_centroids.append((centroid_x, centroid_y))
                    leaf_areas.append(leaf_area)
                    depth_value = depth_data[centroid_y, centroid_x]
                    leaf_depths.append(depth_value)

                    z = depth_value
                    x = (centroid_x - cx) * z / fx
                    y = (centroid_y - cy) * z / fy

                    x_robot = y
                    y_robot = x

                    leaf_3d_points.append((x_robot, y_robot, z))

                    x_ee = x_robot + offset_x
                    y_ee = y_robot + offset_y
                    z_ee = z - offset_z

                    leaf_3d_points_ee.append((-x_ee, -y_ee, z_ee))

                    graspable_area = compute_graspable_area(color_mask, kernel_size=int(0.005 * fx / z))
                    graspable_areas.append(graspable_area)

                    # Save the contour for later use with optimal leaf
                    contours_list.append(contour_max)
            else:
                print("Leaf has no valid moment for centroid calculation.")
        else:
            print("Leaf has no contour.")

    leaf_areas = np.array(leaf_areas)
    leaf_depths = np.array(leaf_depths)

    print("Number of identified leaves: {}".format(len(leaf_centroids)))

    for i, (centroid, area, depth) in enumerate(zip(leaf_centroids, leaf_areas, leaf_depths), start=1):
        print("Leaf {}: Centroid = {}, Area = {} pixels, Depth = {}".format(i, centroid, area, depth))

    if leaf_centroids:
        score = leaf_areas / leaf_depths
        optimal_leaf_index = np.argmax(score)
        optimal_centroid = leaf_centroids[optimal_leaf_index]
        optimal_3d_point = leaf_3d_points[optimal_leaf_index]
        optimal_3d_point_ee = leaf_3d_points_ee[optimal_leaf_index]

        x_world = gantryX + optimal_3d_point_ee[0]
        y_world = gantryY + optimal_3d_point_ee[1]

        if optimal_3d_point_ee[2] < gantryZ:
            z_world = gantryZ + (gantryZ - optimal_3d_point_ee[2])
        else:
            z_world = gantryZ + (gantryZ - optimal_3d_point_ee[2]) + gantryZ

        print("\n3D Coordinates of Optimal Grasp Point (with respect to inhand_link): {}".format(tuple(np.float32(optimal_3d_point))))
        print("3D Coordinates of Optimal Grasp Point : X={:.6f} meters, Y={:.6f} meters, Z={:.6f} meters".format(optimal_3d_point[0], optimal_3d_point[1], optimal_3d_point[2]))
        print("Transformed 3D Coordinates with respect to ee_link (numpy float32): {}".format(np.array(optimal_3d_point_ee, dtype=np.float32)))
        print("Grasp Point with respect to world frame (base_link) : X={:.6f} meters, Y={:.6f} meters, Z={:.6f} meters".format(x_world, y_world, z_world))

        grasp_point_msg = "%.6f %.6f %.6f" % (optimal_3d_point_ee[0], optimal_3d_point_ee[1], optimal_3d_point_ee[2])
        rospy.loginfo("Publishing Grasp Point: %s" % grasp_point_msg)
        pub.publish(grasp_point_msg)

        # Only fit the ellipse to the optimal leaf's contour
        optimal_leaf_contour = contours_list[optimal_leaf_index]
        ellipse_fitted = fit_ellipse_contour_based(optimal_leaf_contour)

        if ellipse_fitted is not None:
            # Get ellipse parameters (center, axes, angle)
            ellipse_center = (int(ellipse_fitted[0][0]), int(ellipse_fitted[0][1]))
            ellipse_axes = (int(ellipse_fitted[1][0] / 2), int(ellipse_fitted[1][1] / 2))  # Semi-major and semi-minor axes
            ellipse_angle = ellipse_fitted[2]

            # Check which axis is the major axis based on the lengths
            if ellipse_fitted[1][0] > ellipse_fitted[1][1]:  # If width is greater than height, width is the major axis
                ellipse_axis_length = ellipse_fitted[1][0] / 2  # Major axis length
                ellipse_angle_major = ellipse_angle  # No need to add 90 degrees
            else:
                ellipse_axis_length = ellipse_fitted[1][1] / 2  # Major axis length
                ellipse_angle_major = ellipse_angle + 90  # Add 90 degrees if height is the major axis

            # Plot ellipse for the optimal leaf
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(masked_image_rgb)

            # Add ellipse patch for the optimal leaf
            ellipse_patch = patches.Ellipse(
                ellipse_center,
                width=ellipse_fitted[1][0],  # Full length of the major axis
                height=ellipse_fitted[1][1],  # Full length of the minor axis
                angle=ellipse_angle,
                color='cyan',
                alpha=0.5,  # Translucency
                fill=True
            )
            ax.add_patch(ellipse_patch)

            # Calculate the correct lengths for the axes
            ellipse_axis_start = (
                int(ellipse_center[0] - ellipse_axis_length * np.cos(np.radians(ellipse_angle_major))),
                int(ellipse_center[1] - ellipse_axis_length * np.sin(np.radians(ellipse_angle_major)))
            )
            ellipse_axis_end = (
                int(ellipse_center[0] + ellipse_axis_length * np.cos(np.radians(ellipse_angle_major))),
                int(ellipse_center[1] + ellipse_axis_length * np.sin(np.radians(ellipse_angle_major)))
            )

            # Plot the blue solid line along the correct major axis
            ax.plot([ellipse_axis_start[0], ellipse_axis_end[0]], [ellipse_axis_start[1], ellipse_axis_end[1]], 'b-', linewidth=2)

            # Extend the red dotted line to the boundaries
            img_height, img_width, _ = masked_image_rgb.shape

            def get_boundary_intersection(x0, y0, angle, img_width, img_height):
                points = []
                if np.cos(np.radians(angle)) != 0:
                    x_left = 0
                    y_left = y0 + (x_left - x0) * np.tan(np.radians(angle))
                    if 0 <= y_left <= img_height:
                        points.append((x_left, y_left))

                    x_right = img_width
                    y_right = y0 + (x_right - x0) * np.tan(np.radians(angle))
                    if 0 <= y_right <= img_height:
                        points.append((x_right, y_right))

                if np.sin(np.radians(angle)) != 0:
                    y_top = 0
                    x_top = x0 + (y_top - y0) / np.tan(np.radians(angle))
                    if 0 <= x_top <= img_width:
                        points.append((x_top, y_top))

                    y_bottom = img_height
                    x_bottom = x0 + (y_bottom - y0) / np.tan(np.radians(angle))
                    if 0 <= x_bottom <= img_width:
                        points.append((x_bottom, y_bottom))

                return points

            intersections = get_boundary_intersection(
                ellipse_center[0], ellipse_center[1], ellipse_angle_major, img_width, img_height
            )

            if len(intersections) == 2:
                ax.plot([intersections[0][0], intersections[1][0]], [intersections[0][1], intersections[1][1]], 'r--', linewidth=2)

            # Black dotted vertical line
            ax.plot([optimal_centroid[0], optimal_centroid[0]], [0, img_height], 'k--', linewidth=2)

            # Mark all leaf centroids with red dots
            for i, centroid in enumerate(leaf_centroids):
                if i != optimal_leaf_index:  # Mark all other leaves as red
                    ax.plot(centroid[0], centroid[1], 'ro', markersize=10)

            # Mark the optimal grasp point with a green dot
            ax.plot(optimal_centroid[0], optimal_centroid[1], 'go', markersize=12, markeredgecolor='black', markeredgewidth=2)

            # Calculate the angle between the vertical axis and the major axis
            angle_between_axes = abs(ellipse_angle_major - 90)

            # Ensure the angle is less than 90 degrees (smallest possible angle)
            if angle_between_axes > 90:
                angle_between_axes = 180 - angle_between_axes

            # Determine positive or negative angle based on leaf's side (left or right)
            if optimal_centroid[0] < image_center_x:
                angle_between_axes = abs(angle_between_axes)  # Left side (positive angle)
            else:
                angle_between_axes = -abs(angle_between_axes)  # Right side (negative angle)

            print("Small angle between vertical axis and major axis: %.2f degrees" % angle_between_axes)

            plt.title('Optimal Grasp Point on Optimal Leaf')

            # Save and display the image
            plt.axis('off')
            plt.savefig('/home/buggspray/ros/catkin_ws/src/leaf_grasp_srecharan/grasp_point_visualization/leaf_grasp_visualization.png')
            plt.show()
            plt.close()
        else:
            print("Could not fit ellipse on the optimal leaf contour.")
    else:
        print("No valid centroids found.")

if __name__ == '__main__':
    try:
        leaf_grasp_node()
    except rospy.ROSInterruptException:
        pass
