import numpy as np
import math
import cv2

# Define the indices of the joints in SMPL that correspond to OpenPose joints
# Note: These indices and mappings are hypothetical and should be adjusted based on actual joint correspondences
smpl_to_openpose = {
    0: 8,   # MidHip
    1: 12,  # RHip
    2: 9,   # RKnee
    3: 10,  # RAnkle
    4: 11,  # RFoot (not typically used in OpenPose, here for completeness)
    5: 13,  # LHip
    6: 14,  # LKnee
    7: 15,  # LAnkle
    8: 16,  # LFoot (not typically used in OpenPose, here for completeness)
    9: 1,   # Spine
    10: 0,  # Neck/Nose (considered as neck in OpenPose)
    11: 0,  # HeadTop (no exact match, map to Nose)
    12: 2,  # RShoulder
    13: 3,  # RElbow
    14: 4,  # RWrist
    15: 5,  # LShoulder
    16: 6,  # LElbow
    17: 7   # LWrist
}

def convert_smpl_to_openpose(smpl_joints):
    """
    Converts SMPL joint coordinates to OpenPose format.
    
    Args:
    smpl_joints (np.array): A (24, 3) array of SMPL joint coordinates (x, y, z)
    
    Returns:
    np.array: An (18, 3) array of joint coordinates in OpenPose format
    """
    # Initialize the output array with NaNs for OpenPose format
    openpose_joints = np.full((18, 2), np.nan)
    
    # Map the joints from SMPL to OpenPose
    for op_idx, smpl_idx in smpl_to_openpose.items():
        openpose_joints[op_idx] = smpl_joints[smpl_idx]
        
    return openpose_joints


def draw_bodypose(canvas, keypoints) -> np.ndarray:
    """
    Draw keypoints and limbs representing body pose on a given canvas.

    Args:
        canvas (np.ndarray): A 3D numpy array representing the canvas (image) on which to draw the body pose.
        keypoints (List[Keypoint]): A list of Keypoint objects representing the body keypoints to be drawn.

    Returns:
        np.ndarray: A 3D numpy array representing the modified canvas with the drawn body pose.

    Note:
        The function expects the x and y coordinates of the keypoints to be normalized between 0 and 1.
    """
    H, W, C = canvas.shape
    stickwidth = 4

    limbSeq = [
        [2, 3], [2, 6], [3, 4], [4, 5], 
        [6, 7], [7, 8], [2, 9], [9, 10], 
        [10, 11], [2, 12], [12, 13], [13, 14], 
        [2, 1], [1, 15], [15, 17], [1, 16], 
        [16, 18],
    ]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    #print(keypoints)
    for (k1_index, k2_index), color in zip(limbSeq, colors):
        keypoint1 = keypoints[k1_index - 1]
        keypoint2 = keypoints[k2_index - 1]

        if keypoint1 is None or keypoint2 is None:
            continue
        if np.sum(keypoint1) <0 or np.sum(keypoint2) <0:
            continue
        #print(keypoint1, keypoint2)
        Y = np.array([keypoint1[0], keypoint2[0]])# * float(W)
        X = np.array([keypoint1[1], keypoint2[1]])# * float(H)
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, [int(float(c) * 0.6) for c in color])

    for keypoint, color in zip(keypoints, colors):
        if keypoint is None:
            continue
        if np.sum(keypoint) <0:
            continue
        x, y = keypoint[0], keypoint[1]
        x = int(x) #int(x * W)
        y = int(y) #int(y * H)
        #print(x,y)
        cv2.circle(canvas, (int(x), int(y)), 4, color, thickness=-1)
    #print(canvas.shape, np.max(canvas), '???')
    return canvas


def draw_poses(pose, H, W, draw_body=True):
    """
    Draw the detected poses on an empty canvas.

    Args:
        poses 24, 2
        H (int): The height of the canvas.
        W (int): The width of the canvas.
        draw_body (bool, optional): Whether to draw body keypoints. Defaults to True.
        draw_hand (bool, optional): Whether to draw hand keypoints. Defaults to True.
        draw_face (bool, optional): Whether to draw face keypoints. Defaults to True.

    Returns:
        numpy.ndarray: A 3D numpy array representing the canvas with the drawn poses.
    """
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    #pose = convert_smpl_to_openpose(pose) # smpl to openpose
    #print(pose.shape)
    return draw_bodypose(canvas, pose)



def paper_draw_bodypose(canvas, keypoints) -> np.ndarray:
    """
    Draw keypoints and limbs representing body pose on a given canvas.

    Args:
        canvas (np.ndarray): A 3D numpy array representing the canvas (image) on which to draw the body pose.
        keypoints (List[Keypoint]): A list of Keypoint objects representing the body keypoints to be drawn.

    Returns:
        np.ndarray: A 3D numpy array representing the modified canvas with the drawn body pose.

    Note:
        The function expects the x and y coordinates of the keypoints to be normalized between 0 and 1.
    """
    H, W, C = canvas.shape
    stickwidth = 4

    limbSeq = [
        [2, 3], [2, 6], [3, 4], [4, 5], 
        [6, 7], [7, 8], [2, 9], [9, 10], 
        [10, 11], [2, 12], [12, 13], [13, 14], 
        [2, 1], [1, 15], [15, 17], [1, 16], 
        [16, 18],
    ]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    #colors = [[224, 224, 224] * len(colors)]
    #ine_colors = [[149, 149, 149] * len(colors)]



    #print(keypoints)
    for (k1_index, k2_index), color in zip(limbSeq, colors):
        keypoint1 = keypoints[k1_index - 1]
        keypoint2 = keypoints[k2_index - 1]
        color = [149, 149, 149]
        if keypoint1 is None or keypoint2 is None:
            continue
        if np.sum(keypoint1) <0 or np.sum(keypoint2) <0:
            continue
        #print(keypoint1, keypoint2)
        Y = np.array([keypoint1[0], keypoint2[0]])# * float(W)
        X = np.array([keypoint1[1], keypoint2[1]])# * float(H)
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        #polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        #cv2.fillConvexPoly(canvas, polygon, [int(float(c) * 0.6) for c in color])
        cv2.line(canvas, (int(keypoint1[0]), int(keypoint1[1])), (int(keypoint2[0]), int(keypoint2[1])), color, 6) 

    for keypoint, color in zip(keypoints, colors):
        color = [224, 224, 224] 
        if keypoint is None:
            continue
        if np.sum(keypoint) <0:
            continue
        x, y = keypoint[0], keypoint[1]
        x = int(x) #int(x * W)
        y = int(y) #int(y * H)
        print(color)
        #print(x,y)
        cv2.circle(canvas, (int(x), int(y)), 6, color, thickness=-1)
        #cv2.circle(image, center, radius, circle_color, -1)

        cv2.circle(canvas, (int(x), int(y)), 6, (0., 0., 0.), thickness=3)
    #print(canvas.shape, np.max(canvas), '???')
    return canvas



def paper_draw_poses(pose, H, W, draw_body=True):
    """
    Draw the detected poses on an empty canvas.

    Args:
        poses 24, 2
        H (int): The height of the canvas.
        W (int): The width of the canvas.
        draw_body (bool, optional): Whether to draw body keypoints. Defaults to True.
        draw_hand (bool, optional): Whether to draw hand keypoints. Defaults to True.
        draw_face (bool, optional): Whether to draw face keypoints. Defaults to True.

    Returns:
        numpy.ndarray: A 3D numpy array representing the canvas with the drawn poses.
    """
    canvas = np.ones(shape=(H, W, 3), dtype=np.uint8) * 255

    #pose = convert_smpl_to_openpose(pose) # smpl to openpose
    #print(pose.shape)
    return paper_draw_bodypose(canvas, pose)