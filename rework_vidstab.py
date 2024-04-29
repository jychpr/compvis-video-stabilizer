import numpy as np
import cv2
import click


SMOOTHING_RADIUS = 100


def movingAverage(curve, radius):
    window_size = 2 * radius + 1
    # define the filter
    filter = np.ones(window_size) / window_size
    # add padding to the boundaries
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    # apply convolution
    curve_smoothed = np.convolve(curve_pad, filter, mode='same')
    # remove padding
    curve_smoothed = curve_smoothed[radius:-radius]
    # return smoothed curve
    return curve_smoothed

def smooth(trajectory):
    smoothed_trajectory = np.copy(trajectory)
    # filter the x, y, and angle curves
    for i in range(3):
        smoothed_trajectory[:, i] = movingAverage(trajectory[:, i], radius=SMOOTHING_RADIUS)
    return smoothed_trajectory

def fixBorder(frame):
    frame_shape = frame.shape
    # scale the image 5% without moving the center
    border_correction_matrix = cv2.getRotationMatrix2D((frame_shape[1] / 2, frame_shape[0] / 2), 0, 1.05)
    frame = cv2.warpAffine(frame, border_correction_matrix, (frame_shape[1], frame_shape[0]), borderMode = cv2.BORDER_CONSTANT, borderValue = 0, flags = cv2.INTER_CUBIC)
    # frame = cv2.warpAffine(frame, border_correction_matrix, (frame_shape[1], frame_shape[0]))
    return frame

@click.command()
@click.option('--video_input', '-vi', type=str, required=True)
@click.option('--video_output', '-vo', type=str, required=True, default='output.avi')
@click.option('--video_format', '-vf', type=str, required=True, default='XVID')

def run_stab(video_input, video_output, video_format):
    print(video_input)
    print(video_output)
    print(video_format)

    # get video input
    vid_captured = cv2.VideoCapture(video_input)

    # get video information
    vid_n_frames = int(vid_captured.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_fps = vid_captured.get(cv2.CAP_PROP_FPS)
    vid_width = int(vid_captured.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(vid_captured.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("frame count in video : ", vid_n_frames)
    print("video fps rate : ", vid_fps)
    print('width : ', vid_width)
    print('height : ', vid_height)

    # get output video
    vid_fourcc = cv2.VideoWriter.fourcc(*video_format)
    vid_out = cv2.VideoWriter(video_output, vid_fourcc, vid_fps, (vid_width, vid_height))

    # read first frame
    _, previous = vid_captured.read()

    # convert frame to greyscale
    previous_grayscaled = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)

    # pre-define transformation-store array
    transform_matrix = np.zeros((vid_n_frames-1,3), np.float64)

    for i in range(vid_n_frames-2):
        # detect feature points in previous frame
        previous_points = cv2.goodFeaturesToTrack(previous_grayscaled,
                                                    maxCorners=100,
                                                    qualityLevel=0.01,
                                                    minDistance=10,
                                                    blockSize=3)
        
        # read next frame
        success, current = vid_captured.read()
        if (success == False):
            break

        # convert to grayscale
        current_grayscaled = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)

        # calculate optical flow to track feature points
        current_points, status, errors = cv2.calcOpticalFlowPyrLK(previous_grayscaled,
                                                                    current_grayscaled,
                                                                    previous_points,
                                                                    None)
        
        # sanity check
        assert previous_points.shape == current_points.shape

        # filter only valid points
        index = np.where(status == 1)[0]
        previous_points = previous_points[index]
        current_points = current_points[index]

        # find transformation matrix
        matrix = cv2.estimateAffine2D(previous_points, current_points)[0]

        # extract transformation
        dx = matrix[0, 2]
        dy = matrix[1, 2]
        da = np.arctan2(matrix[1, 0], matrix[0, 0])

        # store transformation
        transform_matrix[i] = [dx, dy, da]

        # move to next frame
        previous_grayscaled = current_grayscaled

        print("Frame : " + str(i) + "/" + str(vid_n_frames) + " - Tracked points : " + str(len(previous_points)))

    # compute trajectory using cumulative sum of transformation
    trajectory = np.cumsum(transform_matrix, axis=0)

    # smooth the trajectory
    smoothed_trajectory = smooth(trajectory)

    # calculate difference in smoothed_trajectory and trajectory
    difference = smoothed_trajectory - trajectory

    # calculate newer transformation array
    transform_matrix_smooth = transform_matrix + difference

    # reset stream to first frame
    vid_captured.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # write n_frames-1 transformed frames --> apply transformation to video
    for i in range(vid_n_frames-2):
        # read next frame
        success, frame = vid_captured.read()
        if not success:
            break

        # extract transformations from the new transformation array
        dx = transform_matrix_smooth[i, 0]
        dy = transform_matrix_smooth[i, 1]
        da = transform_matrix_smooth[i, 2]

        # reconstruct transformation matrix according to new values
        matrix = np.zeros((2, 3), np.float64)
        matrix[0, 0] = np.cos(da)
        matrix[0, 1] = -np.sin(da)
        matrix[1, 0] = np.sin(da)
        matrix[1, 1] = np.cos(da)
        matrix[0, 2] = dx
        matrix[1, 2] = dy

        # apply affine wrapping to the given frame
        frame_stabilized = cv2.warpAffine(frame, matrix, (vid_width, vid_height))

        # fix border artifacts
        frame_stabilized = fixBorder(frame_stabilized)

        # write the frame to file
        frame_output = cv2.hconcat([frame, frame_stabilized])
        
        vid_out.write(frame_stabilized)

        # if the image is too big, resize it
        # if frame_output.shape[i] > 1920:
        #     frame_output = cv2.resize(frame_output, (frame_output.shape[1] // 2, frame_output[0] //2))
        
        cv2.imshow("Before and After", frame_output)
        cv2.waitKey(10)
        
        

    vid_captured.release()
    vid_out.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    run_stab()