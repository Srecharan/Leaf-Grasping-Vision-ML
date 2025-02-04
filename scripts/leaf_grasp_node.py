#!/usr/bin/env python

import rospy
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from scipy import signal
import rosbag
from std_msgs.msg import String

def leaf_grasp_node():
    rospy.init_node('leaf_grasp_node', anonymous=True)
    pub = rospy.Publisher('grasp_point_topic', String, queue_size=10)

    # Initialize gantry position variables
    global gantryX, gantryY, gantryZ
    gantryX, gantryY, gantryZ = None, None, None

    # Callback to update gantry positions
    def gantry_position_callback(msg):
        global gantryX, gantryY, gantryZ
        positions = msg.data.split()
        gantryX = float(positions[0])
        gantryY = float(positions[1])
        gantryZ = float(positions[2])
        rospy.loginfo("Received gantry position: X={}, Y={}, Z={}".format(gantryX, gantryY, gantryZ))

    # Subscribe to the topic to receive gantry positions
    rospy.Subscriber('/gantry_position_topic', String, gantry_position_callback)

    # Ensure the publisher is ready
    rospy.sleep(1.0)

    # Load the ROSbag
    bag = rosbag.Bag('/home/buggspray/gantry_robot_data.bag')

    # Extract the last camera information message from the ROSbag
    camera_info_topic = '/theia/left/camera_info'
    camera_info_msg = None
    for topic, msg, t in bag.read_messages(topics=[camera_info_topic]):
        camera_info_msg = msg

    if camera_info_msg is None:
        raise ValueError("No camera info found in the ROSbag.")

    # Camera intrinsic parameters (from your camera info file)
    fx = camera_info_msg.K[0]  # Focal length in x
    fy = camera_info_msg.K[4]  # Focal length in y
    cx = camera_info_msg.K[2]  # Principal point x
    cy = camera_info_msg.K[5]  # Principal point y

    # Offsets from inhand_link to ee_link (in meters)
    offset_x = -0.088
    offset_y = -0.003
    offset_z = 0.119

    # Wait until the gantry positions are received
    while gantryX is None or gantryY is None or gantryZ is None:
        rospy.loginfo("Waiting for gantry positions...")
        rospy.sleep(0.1)

    # Load the masked image and depth data for the last image
    masked_image_path = '/home/buggspray/SDF_OUT/masked_images/aggrigated_masks8.png'
    depth_data_path = '/home/buggspray/SDF_OUT/npy/depth8.npy'

    masked_image = cv2.imread(masked_image_path)
    masked_image_rgb = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
    depth_data = np.load(depth_data_path)

    # Identify unique colors in the image
    unique_colors = np.unique(masked_image_rgb.reshape(-1, masked_image_rgb.shape[2]), axis=0)

    leaf_centroids = []
    leaf_areas = []
    leaf_depths = []
    leaf_3d_points = []
    leaf_3d_points_ee = []
    graspable_areas = []
    min_area_threshold = 1500

    # Image center coordinates for filtering
    image_center_x, image_center_y = masked_image_rgb.shape[1] // 2, masked_image_rgb.shape[0] // 2
    max_distance_from_center = 50

    def compute_graspable_area(mask, kernel_size):
        convolved = signal.convolve2d(mask.astype('uint8'), np.ones((kernel_size, kernel_size)), boundary='symm', mode='same')
        graspable_area = np.where(convolved < np.amax(convolved) * 0.95, 1, 0)
        return graspable_area

    for color in unique_colors:
        if np.array_equal(color, [128, 0, 128]):  # Skip the background (purple)
            continue

        # Create a binary mask for the current color
        color_mask = cv2.inRange(masked_image_rgb, color, color)

        # Calculate area and filter out small regions
        leaf_area = np.sum(color_mask > 0)
        if leaf_area < min_area_threshold:
            continue

        # Label the regions in the mask
        labels = measure.label(color_mask)
        props = measure.regionprops(labels)

        # Find contours
        _, contours, _ = cv2.findContours(color_mask.astype("uint8"), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        if contours:
            contour_max = max(contours, key=cv2.contourArea)
            MOMENT = cv2.moments(contour_max)
            if MOMENT["m00"] > 0:
                centroid_x = int(MOMENT["m10"] / MOMENT["m00"])
                centroid_y = int(MOMENT["m01"] / MOMENT["m00"])

                # Check if the centroid is too close to the image center
                distance_to_center = np.sqrt((centroid_x - image_center_x) ** 2 + (centroid_y - image_center_y) ** 2)
                if distance_to_center > max_distance_from_center:
                    leaf_centroids.append((centroid_x, centroid_y))
                    leaf_areas.append(leaf_area)
                    depth_value = depth_data[centroid_y, centroid_x]
                    leaf_depths.append(depth_value)

                    # Convert the 2D centroid to a 3D point in the camera's coordinate system
                    z = depth_value
                    x = (centroid_x - cx) * z / fx
                    y = (centroid_y - cy) * z / fy

                    # Swap the X and Y axes to match the robot's coordinate system
                    x_robot = y
                    y_robot = x

                    # Store the 3D point with respect to inhand_link
                    leaf_3d_points.append((x_robot, y_robot, z))

                    # Apply the offset to convert to ee_link frame
                    x_ee = x_robot + offset_x
                    y_ee = y_robot + offset_y
                    z_ee = z - offset_z  # z_ee = z + offset_z

                    # Store the 3D point with respect to ee_link
                    leaf_3d_points_ee.append((-x_ee, -y_ee, z_ee))

                    # Compute graspable area
                    graspable_area = compute_graspable_area(color_mask, kernel_size=int(0.005 * fx / z))
                    graspable_areas.append(graspable_area)
                else:
                    rospy.loginfo("Skipping centroid near the center at ({}, {})".format(centroid_x, centroid_y))
            else:
                rospy.loginfo("Leaf has no valid moment for centroid calculation")
        else:
            rospy.loginfo("Leaf has no contour")
            color_mask = np.where(color_mask == color, 0, color_mask)

    leaf_areas = np.array(leaf_areas)
    leaf_depths = np.array(leaf_depths)

    rospy.loginfo("Number of identified leaves: {}".format(len(leaf_centroids)))

    for i, (centroid, area, depth) in enumerate(zip(leaf_centroids, leaf_areas, leaf_depths), start=1):
        rospy.loginfo("Leaf {}: Centroid = {}, Area = {} pixels, Depth = {}".format(i, centroid, area, depth))

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

        rospy.loginfo("\n3D Coordinates of Optimal Grasp Point (with respect to inhand_link): {}".format(tuple(np.float32(optimal_3d_point))))
        rospy.loginfo("3D Coordinates of Optimal Grasp Point : X={:.6f} meters, Y={:.6f} meters, Z={:.6f} meters".format(optimal_3d_point[0], optimal_3d_point[1], optimal_3d_point[2]))
        rospy.loginfo("Transformed 3D Coordinates with respect to ee_link (numpy float32): {}".format(np.array(optimal_3d_point_ee, dtype=np.float32)))
        rospy.loginfo("Transformed 3D Coordinates with respect to ee_link : X={:.6f} meters, Y={:.6f} meters, Z={:.6f} meters".format(optimal_3d_point_ee[0], optimal_3d_point_ee[1], optimal_3d_point_ee[2]))
        rospy.loginfo("Grasp Point with respect to world frame (base_link) : X={:.6f} meters, Y={:.6f} meters, Z={:.6f} meters".format(x_world, y_world, z_world))

        plt.figure(figsize=(10, 10))
        plt.imshow(masked_image_rgb)
        for i, centroid in enumerate(leaf_centroids):
            if i == optimal_leaf_index:
                plt.plot(centroid[0], centroid[1], 'go', markersize=12)
            else:
                plt.plot(centroid[0], centroid[1], 'ro')
        plt.title('Optimal Grasp Point on Leaves')
        plt.axis('off')
        plt.savefig('/home/buggspray/ros/catkin_ws/src/leaf_grasp_srecharan/grasp_point_visualization/leaf_grasp_visualization.png')
        plt.close()

        # Ensure the grasp point is being published correctly (values in ee_link frame)
        grasp_point_msg = "{:.6f} {:.6f} {:.6f}".format(optimal_3d_point_ee[0], optimal_3d_point_ee[1], optimal_3d_point_ee[2])
        rospy.loginfo("Publishing Grasp Point: {}".format(grasp_point_msg))

        # Attempt to publish several times for reliability
        for _ in range(10):
            pub.publish(grasp_point_msg)
            rospy.sleep(0.5)
    else:
        rospy.loginfo("No valid centroids found.")

    bag.close()

if __name__ == '__main__':
    try:
        leaf_grasp_node()
    except rospy.ROSInterruptException:
        pass
