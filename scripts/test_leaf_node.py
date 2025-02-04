#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from scipy import signal
from std_msgs.msg import String, Float64  # Add Float64 for angle publishing
import matplotlib.patches as patches

def leaf_grasp_node():
    rospy.init_node('leaf_grasp_node_2', anonymous=True)
    pub = rospy.Publisher('grasp_point_topic', String, queue_size=10)
    angle_pub = rospy.Publisher('grasp_angle_topic', Float64, queue_size=10)  # Publisher for angle
    pre_grasp_pub = rospy.Publisher('pre_grasp_point_topic', String, queue_size=10)  # Publisher for pre-grasping point
    difference_pub = rospy.Publisher('grasp_diff_topic', String, queue_size=10)  # Publisher for grasp point difference

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

    masked_image_path = '/home/buggspray/SDF_OUT/test3/aggrigated_masks0.png'
    depth_data_path = '/home/buggspray/SDF_OUT/test3/depth0.npy'

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
    min_area_threshold = 1500

    image_center_x, image_center_y = masked_image_rgb.shape[1] // 2, masked_image_rgb.shape[0] // 2
    max_distance_from_center = 50

    def compute_graspable_area(mask, kernel_size):
        convolved = signal.convolve2d(mask.astype('uint8'), np.ones((kernel_size, kernel_size)), boundary='symm', mode='same')
        graspable_area = np.where(convolved < np.amax(convolved) * 0.95, 1, 0)
        return graspable_area

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
                else:
                    print("Skipping centroid near the center at ({}, {})".format(centroid_x, centroid_y))
            else:
                print("Leaf has no valid moment for centroid calculation")
        else:
            print("Leaf has no contour")
            color_mask = np.where(color_mask == color, 0, color_mask)

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

        contour_2d = contour_max.reshape(-1, 2).astype(np.float32)

        mean, eigenvectors = cv2.PCACompute(contour_2d, mean=None)
        eigenvalues = np.array([cv2.norm(np.dot(contour_2d, eigenvectors[0])), cv2.norm(np.dot(contour_2d, eigenvectors[1]))])

        major_axis_vector = eigenvectors[0] * np.sqrt(eigenvalues[0])
        minor_axis_vector = eigenvectors[1] * np.sqrt(eigenvalues[1])

        # Ensure the major axis aligns with the centerline of the leaf and the minor axis with the width
        if np.linalg.norm(minor_axis_vector) > np.linalg.norm(major_axis_vector):
            major_axis_vector, minor_axis_vector = minor_axis_vector, major_axis_vector

        major_axis_start = (int(optimal_centroid[0] - major_axis_vector[0]), int(optimal_centroid[1] - major_axis_vector[1]))
        major_axis_end = (int(optimal_centroid[0] + major_axis_vector[0]), int(optimal_centroid[1] + major_axis_vector[1]))

        minor_axis_start = (int(optimal_centroid[0] - minor_axis_vector[0]), int(optimal_centroid[1] - minor_axis_vector[1]))
        minor_axis_end = (int(optimal_centroid[0] + minor_axis_vector[0]), int(optimal_centroid[1] + minor_axis_vector[1]))

        print("\n3D Coordinates of Optimal Grasp Point (with respect to inhand_link): {}".format(tuple(np.float32(optimal_3d_point))))
        print("3D Coordinates of Optimal Grasp Point : X={:.6f} meters, Y={:.6f} meters, Z={:.6f} meters".format(optimal_3d_point[0], optimal_3d_point[1], optimal_3d_point[2]))
        print("Transformed 3D Coordinates with respect to ee_link (numpy float32): {}".format(np.array(optimal_3d_point_ee, dtype=np.float32)))
        print("Grasp Point with respect to world frame (base_link) : X={:.6f} meters, Y={:.6f} meters, Z={:.6f} meters".format(x_world, y_world, z_world))

        grasp_point_msg = "{:.6f} {:.6f} {:.6f}".format(optimal_3d_point_ee[0], optimal_3d_point_ee[1], optimal_3d_point_ee[2])
        rospy.loginfo("Publishing Grasp Point: {}".format(grasp_point_msg))
        pub.publish(grasp_point_msg)

        # Calculate the center and angle of rotation for the ellipse
        ellipse_center = (optimal_centroid[0], optimal_centroid[1])
        ellipse_angle = np.arctan2(major_axis_vector[1], major_axis_vector[0]) * 180.0 / np.pi

        # Calculate the angle between the major axis and the vertical line
        angle_with_vertical = 90 - ellipse_angle if ellipse_angle < 90 else 270 - ellipse_angle
        print("Angle between Major Axis and Vertical Line: {:.2f} degrees".format(angle_with_vertical))

        # Publish the angle in radians
        angle_in_radians = np.radians(angle_with_vertical)
        angle_pub.publish(angle_in_radians)
        rospy.loginfo("Publishing Grasp Angle: {:.6f} radians".format(angle_in_radians))

        # Calculate the pre-grasping point, taking space into account
        pre_grasp_distance = 0.05  # Increase the distance from the grasping point in meters
        img_height, img_width, _ = masked_image_rgb.shape

        if abs(optimal_centroid[0] - image_center_x) < abs(optimal_centroid[1] - image_center_y):
            # Move along the horizontal axis if closer to the vertical center line and check boundaries
            if optimal_centroid[0] - int(pre_grasp_distance * fx / z) > 0:
                pre_grasp_3d_point_ee = (optimal_3d_point_ee[0] - pre_grasp_distance, optimal_3d_point_ee[1], optimal_3d_point_ee[2])
                pre_grasp_visual_point = (optimal_centroid[0] - int(pre_grasp_distance * fx / z), optimal_centroid[1])
            else:
                pre_grasp_3d_point_ee = (optimal_3d_point_ee[0] + pre_grasp_distance, optimal_3d_point_ee[1], optimal_3d_point_ee[2])
                pre_grasp_visual_point = (optimal_centroid[0] + int(pre_grasp_distance * fx / z), optimal_centroid[1])
        else:
            # Move along the vertical axis if closer to the horizontal center line and check boundaries
            if optimal_centroid[1] - int(pre_grasp_distance * fy / z) > 0:
                pre_grasp_3d_point_ee = (optimal_3d_point_ee[0], optimal_3d_point_ee[1] - pre_grasp_distance, optimal_3d_point_ee[2])
                pre_grasp_visual_point = (optimal_centroid[0], optimal_centroid[1] - int(pre_grasp_distance * fy / z))
            else:
                pre_grasp_3d_point_ee = (optimal_3d_point_ee[0], optimal_3d_point_ee[1] + pre_grasp_distance, optimal_3d_point_ee[2])
                pre_grasp_visual_point = (optimal_centroid[0], optimal_centroid[1] + int(pre_grasp_distance * fy / z))

        pre_grasp_point_msg = "{:.6f} {:.6f} {:.6f}".format(pre_grasp_3d_point_ee[0], pre_grasp_3d_point_ee[1], pre_grasp_3d_point_ee[2])
        rospy.loginfo("Publishing Pre-Grasp Point: {}".format(pre_grasp_point_msg))
        pre_grasp_pub.publish(pre_grasp_point_msg)

        # Print and Publish the difference between the grasping point and pre-grasping point
        x_diff = max(0, optimal_3d_point_ee[0] - pre_grasp_3d_point_ee[0])
        y_diff = max(0, optimal_3d_point_ee[1] - pre_grasp_3d_point_ee[1])
        z_diff = max(0, optimal_3d_point_ee[2] - pre_grasp_3d_point_ee[2])

        print("Difference between Grasping Point and Pre-Grasping Point: X={:.6f}, Y={:.6f}, Z={:.6f}".format(x_diff, y_diff, z_diff))
        diff_msg = "{:.6f} {:.6f} {:.6f}".format(x_diff, y_diff, z_diff)
        rospy.loginfo("Publishing Grasp Point Difference: {}".format(diff_msg))
        difference_pub.publish(diff_msg)

        # Add the ellipse to the plot
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(masked_image_rgb)

        # Draw translucent ellipse with the same size as the major and minor axes
        ellipse = patches.Ellipse(
            ellipse_center,
            width=np.linalg.norm(major_axis_vector) * 2,  # Major axis length
            height=np.linalg.norm(minor_axis_vector) * 2,  # Minor axis length
            angle=ellipse_angle,
            color='cyan',
            alpha=0.5,  # Translucency
            fill=True
        )
        ax.add_patch(ellipse)

        # Draw the major and minor axes
        ax.plot([major_axis_start[0], major_axis_end[0]], [major_axis_start[1], major_axis_end[1]], 'r-', linewidth=2)
        ax.plot([minor_axis_start[0], minor_axis_end[0]], [minor_axis_start[1], minor_axis_end[1]], 'b-', linewidth=2)

        # Draw the horizontal and vertical lines through the grasping point
        # Horizontal line
        ax.plot([0, img_width], [optimal_centroid[1], optimal_centroid[1]], 'g--', linewidth=1)

        # Vertical line
        ax.plot([optimal_centroid[0], optimal_centroid[0]], [0, img_height], 'g--', linewidth=1)

        # Draw the pre-grasping point as a square
        square_size = 15  # Size of the square marker
        ax.add_patch(patches.Rectangle(
            (pre_grasp_visual_point[0] - square_size / 2, pre_grasp_visual_point[1] - square_size / 2),
            square_size,
            square_size,
            linewidth=2,
            edgecolor='yellow',
            facecolor='yellow',
            fill=True
        ))

        # Optionally, draw the angle symbol if possible
        # This is a simplified representation of the angle near the intersection point
        ax.text(optimal_centroid[0] + 10, optimal_centroid[1] - 10, '{:.1f}\u00B0'.format(angle_with_vertical), color='white', fontsize=12, fontweight='bold')

        # Mark the centroids
        for i, centroid in enumerate(leaf_centroids):
            if i == optimal_leaf_index:
                ax.plot(centroid[0], centroid[1], 'go', markersize=12)
            else:
                ax.plot(centroid[0], centroid[1], 'ro')

        plt.title('Optimal Grasp Point on Leaves')
        plt.axis('off')
        plt.savefig('/home/buggspray/ros/catkin_ws/src/leaf_grasp_srecharan/test_visualization/test_visualization.png')
        plt.close()

    else:
        print("No valid centroids found.")

if __name__ == '__main__':
    try:
        leaf_grasp_node()
    except rospy.ROSInterruptException:
        pass

