import cv2
import numpy as np
import math


def mapping3d_2d():
    lmk_3d = [-7.308957, 0.913869, 0.000000, -6.775290, -0.730814, -0.012799, -5.665918, -3.286078, 1.022951, -5.011779,
              -4.876396, 1.047961, -4.056931, -5.947019, 1.636229, -1.833492, -7.056977, 4.061275, 0.000000, -7.415691,
              4.070434, 1.833492, -7.056977, 4.061275, 4.056931, -5.947019, 1.636229, 5.011779, -4.876396, 1.047961,
              5.665918, -3.286078, 1.022951, 6.775290, -0.730814, -0.012799, 7.308957, 0.913869, 0.000000, 5.311432,
              5.485328, 3.987654, 4.461908, 6.189018, 5.594410, 3.550622, 6.185143, 5.712299, 2.542231, 5.862829,
              4.687939,
              1.789930, 5.393625, 4.413414, 2.693583, 5.018237, 5.072837, 3.530191, 4.981603, 4.937805, 4.490323,
              5.186498,
              4.694397, -5.311432, 5.485328, 3.987654, -4.461908, 6.189018, 5.594410, -3.550622, 6.185143, 5.712299,
              -2.542231, 5.862829, 4.687939, -1.789930, 5.393625, 4.413414, -2.693583, 5.018237, 5.072837, -3.530191,
              4.981603, 4.937805, -4.490323, 5.186498, 4.694397, 1.330353, 7.122144, 6.903745, 2.533424, 7.878085,
              7.451034,
              4.861131, 7.878672, 6.601275, 6.137002, 7.271266, 5.200823, 6.825897, 6.760612, 4.402142, -1.330353,
              7.122144,
              6.903745, -2.533424, 7.878085, 7.451034, -4.861131, 7.878672, 6.601275, -6.137002, 7.271266, 5.200823,
              -6.825897, 6.760612, 4.402142, -2.774015, -2.080775, 5.048531, -0.509714, -1.571179, 6.566167, 0.000000,
              -1.646444, 6.704956, 0.509714, -1.571179, 6.566167, 2.774015, -2.080775, 5.048531, 0.589441, -2.958597,
              6.109526, 0.000000, -3.116408, 6.097667, -0.589441, -2.958597, 6.109526, -0.981972, 4.554081, 6.301271,
              -0.973987, 1.916389, 7.654050, -2.005628, 1.409845, 6.165652, -1.930245, 0.424351, 5.914376, -0.746313,
              0.348381, 6.263227, 0.000000, 0.000000, 6.763430, 0.746313, 0.348381, 6.263227, 1.930245, 0.424351,
              5.914376,
              2.005628, 1.409845, 6.165652, 0.973987, 1.916389, 7.654050, 0.981972, 4.554081, 6.301271]

    lmk_3d = np.array(lmk_3d).reshape([-1, 3])
    mincoord = np.min(lmk_3d, axis=0)
    maxcoord = np.max(lmk_3d, axis=0)
    normalized_lmk_3d = (lmk_3d[:, :2] - mincoord[:2]) / (maxcoord[:2] - mincoord[:2]) * 112

    normalized_lmk_3d[:,:2] = 112-normalized_lmk_3d[:,:2]
    normalized_lmk_3d[:,:2] = normalized_lmk_3d
    #left eye brow left,right , right eye brow left, right , nose left, right, left eye left,right, right eye left right,
    #mouth left , right, lower lip
    key_index = [33, 29, 34, 38, 54, 50, 13, 17, 25,21, 43, 39, 45]
    key_lmk_3d = normalized_lmk_3d[key_index]

    lmk_2d = key_lmk_3d[:,:2]

    # print mlk_2d

    import processimg
    from config import config
    config = config.config

    img = np.zeros((112,112),dtype=np.uint8)
    img = cv2.merge((img,img,img))
    img = processimg.drawPoints(img,lmk_2d)
    cv2.imwrite(config.outputdir+'/3d-mapping.jpg',img)

def calculate_pitch_yaw_roll(landmarks_2D ,cam_w=112, cam_h=112,radians=False, mean_lmk = None):
    """ Return the the pitch  yaw and roll angles associated with the input image.
    @param radians When True it returns the angle in radians, otherwise in degrees.
    """
    c_x = cam_w/2
    c_y = cam_h/2
    f_x = c_x / np.tan(60/2 * np.pi / 180)
    f_y = f_x

    #Estimated camera matrix values.
    camera_matrix = np.float32([[f_x, 0.0, c_x],
                                [0.0, f_y, c_y],
                                [0.0, 0.0, 1.0]])

    camera_distortion = np.float32([0.0, 0.0, 0.0, 0.0, 0.0])

    #The dlib shape predictor returns 68 points, we are interested only in a few of those
    # TRACKED_POINTS = [17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8]
    #wflw(98 landmark) trached points
    # TRACKED_POINTS = [33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]
    #X-Y-Z with X pointing forward and Y on the left and Z up.
    #The X-Y-Z coordinates used are like the standard
    # coordinates of ROS (robotic operative system)
    #OpenCV uses the reference usually used in computer vision:
    #X points to the right, Y down, Z to the front
    # LEFT_EYEBROW_LEFT  = [6.825897, 6.760612, 4.402142]
    # LEFT_EYEBROW_RIGHT = [1.330353, 7.122144, 6.903745]
    # RIGHT_EYEBROW_LEFT = [-1.330353, 7.122144, 6.903745]
    # RIGHT_EYEBROW_RIGHT= [-6.825897, 6.760612, 4.402142]
    # LEFT_EYE_LEFT  = [5.311432, 5.485328, 3.987654]
    # LEFT_EYE_RIGHT = [1.789930, 5.393625, 4.413414]
    # RIGHT_EYE_LEFT = [-1.789930, 5.393625, 4.413414]
    # RIGHT_EYE_RIGHT= [-5.311432, 5.485328, 3.987654]
    # NOSE_LEFT  = [2.005628, 1.409845, 6.165652]
    # NOSE_RIGHT = [-2.005628, 1.409845, 6.165652]
    # MOUTH_LEFT = [2.774015, -2.080775, 5.048531]
    # MOUTH_RIGHT=[-2.774015, -2.080775, 5.048531]
    # LOWER_LIP= [0.000000, -3.116408, 6.097667]
    # CHIN     = [0.000000, -7.415691, 4.070434]
    #
    # landmarks_3D = np.float32( [LEFT_EYEBROW_LEFT,
    #                             LEFT_EYEBROW_RIGHT,
    #                             RIGHT_EYEBROW_LEFT,
    #                             RIGHT_EYEBROW_RIGHT,
    #                             LEFT_EYE_LEFT,
    #                             LEFT_EYE_RIGHT,
    #                             RIGHT_EYE_LEFT,
    #                             RIGHT_EYE_RIGHT,
    #                             NOSE_LEFT,
    #                             NOSE_RIGHT,
    #                             MOUTH_LEFT,
    #                             MOUTH_RIGHT,
    #                             LOWER_LIP,
    #                             CHIN])

    from config import config
    config = config.config
    if mean_lmk is None:
        mean_lmk = np.loadtxt(config.mean_pts,dtype=float)
    mean_lmk = mean_lmk.reshape([-1,2])
    key_mean_lmk = mean_lmk[config.mainPlaneLmkIndex]
    landmarks_3D = np.zeros((len(key_mean_lmk),3))
    landmarks_3D[:,2] = config.face3ddepth
    landmarks_3D[:,:2] = key_mean_lmk

    #Return the 2D position of our landmarks
    assert landmarks_2D is not None ,'landmarks_2D is None'
    landmarks_2D = np.asarray(landmarks_2D,dtype=np.float32).reshape(-1,2)
    # print (landmarks_2D)
    #Applying the PnP solver to find the 3D pose
    #of the head from the 2D position of the
    #landmarks.
    #retval - bool
    #rvec - Output rotation vector that, together with tvec, brings
    #points from the world coordinate system to the camera coordinate system.
    #tvec - Output translation vector. It is the position of the world origin (SELLION) in camera co-ords
    retval, rvec, tvec = cv2.solvePnP(landmarks_3D,
                                      landmarks_2D,
                                      camera_matrix,
                                      camera_distortion)

    #Get as input the rotational vector
    #Return a rotational matrix
    rmat, _ = cv2.Rodrigues(rvec)
    pose_mat = cv2.hconcat((rmat,tvec))

    #euler_angles contain (pitch, yaw, roll)
    # euler_angles = cv2.DecomposeProjectionMatrix(projMatrix=rmat, cameraMatrix=self.camera_matrix, rotMatrix, transVect, rotMatrX=None, rotMatrY=None, rotMatrZ=None)
    _, _, _, _, _, _,euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    pitch,yaw,roll =map(lambda temp:temp[0],euler_angles)
    return pitch,yaw,roll

    # head_pose = [ rmat[0,0], rmat[0,1], rmat[0,2], tvec[0],

                   # rmat[1,0], rmat[1,1], rmat[1,2], tvec[1],

                   # rmat[2,0], rmat[2,1], rmat[2,2], tvec[2],

                         # 0.0,      0.0,        0.0,    1.0 ]

    #print(head_pose) #TODO remove this line

    # return self.rotationMatrixToEulerAngles(rmat)
# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).

def rotationMatrixToEulerAngles(R) :
    #assert(isRotationMatrix(R))
    #To prevent the Gimbal Lock it is possible to use
    #a threshold of 1e-6 for discrimination
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        # math.atan2()
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])