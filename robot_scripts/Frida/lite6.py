#! /usr/bin/env python
import os
import numpy as np
from robot import Robot
import time

class Lite6(Robot, object):
    '''
        Low-level action functionality of the Franka robot.
    '''

    def __init__(self, debug, node_name="painting"):
        from xarm.wrapper import XArmAPI

        self.debug_bool = debug
        self.robot =XArmAPI('192.168.1.162')
        self.robot.connect()
        self.robot.motion_enable(enable=True)

        # reset lite6 to its home joints
        self.reset()

    def zero_joints(self):
        goal_pos = [0.,-44.2, 3.2, 0., 47.4, 0., 0.]
        self.reset(goal_pos=goal_pos)
        #self.rob360ot.set_mode(0)
        #self.robot.set_state(0)
        #self.robot.reset()
        #self.robot.set_servo_attach()
        #self.robot.move_gohome()
        #self.robot.set_position(x=200,y=0,z=200,roll=3.14,pitch=0,yaw=3.14*1.25)
        #self.robot.set_servo_angle_j()
         

    def reset(self, goal_pos=None):
        #self.robot.reset()
        self.robot.set_mode(1)
        self.robot.set_state(0)
        
        if goal_pos is None:
            #goal_pos = [0., 9.9, 31.8, 0., 21.9, 0., 0.]#self.robot.
            goal_pos = [0., 9.9, 31.8, 0., 21.9, 0., 0.]#self.robot.
        code, curr_pos = self.robot.get_servo_angle(is_radian=False)
        try:
            if code==0:
                steps = (np.array(goal_pos) - np.array(curr_pos))/200.0
                for i in range(0,200):
                    curr_pos += steps
                    self.robot.set_servo_angle_j(curr_pos, is_radian=False, wait=True)
                    time.sleep(0.01)
                print("Moved to an initial point.")
        except Exception as e:
            print("Error: ", e)
            self.reset(goal_pos)

    def debug(self, msg):
        if self.debug_bool:
            print(msg)

    def good_morning_robot(self):
        # reset franka to its home joints
        self.reset()

    def good_night_robot(self):
        # reset franka back to home
        self.reset()

    def create_rotation_transform(pos, quat):
        from autolab_core import RigidTransform
        rot = RigidTransform.rotation_from_quaternion(quat)
        rt = RigidTransform(rotation=rot, translation=pos,
                from_frame='franka_tool', to_frame='world')
        return rt

    def euler_from_rt(_rt):
        from autolab_core import RigidTransform
        rt = RigidTransform(rotation = _rt.rotation, translation = _rt.translation)
        return rt.euler

    #def sawyer_to_franka_position(pos):
    #    # Convert from sawyer code representation of X,Y to Franka
    #    pos[0] *= -1 # The x is oposite sign from the sawyer code
    #    pos[:2] = pos[:2][::-1] # The x and y are switched compared to sawyer for which code was written
    #    return pos
    
    def go_to_cartesian_pose(self, positions, orientations, precise=False):
        """
            Move to a list of points in space
            args:
                positions (np.array(n,3)) : x,y,z coordinates in meters from robot origin
                orientations (np.array(n,4)) : x,y,z,w quaternion orientation
                precise (bool) : use precise for slow short movements. else use False, which is fast but unstable
        """
        self.robot.set_mode(0)
        self.robot.set_state(0)
        
        positions, orientations = np.array(positions), np.array(orientations)
        if len(positions.shape) == 1:
            positions = positions[None,:]
            orientations = orientations[None,:]

        if precise:
            #self.go_to_cartesian_pose_precise(positions, orientations)
            print("Precise Mode Requested but will use Stable Mode.")
            self.go_to_cartesian_pose_stable(positions, orientations, wait=True)
        else:
            print("Stable mode request.")
            self.go_to_cartesian_pose_stable(positions, orientations, wait=False)

    def rotate(self, z_degree, wait=False):
        self.robot.set_mode(1)
        self.robot.set_servo_angle(servo_id=6, angle=z_degree, is_radian=False)

    def go_to_cartesian_pose_stable(self, positions, orientations, wait=False):
        self.robot.set_mode(0)
        self.robot.set_state(0)
        for i in range(len(positions)):
            pos = positions[i]
            rt = Lite6.create_rotation_transform(pos, orientations[i])
            euler = Lite6.euler_from_rt(rt)

            # Determine speed/duration
            code = 1
            while code!=0:
                try:
                    code, curr_pose = self.robot.get_position() 
                except Exception as e:
                    print('Could not get pos', e)
            curr_pos = curr_pose[:3]
            # print(curr_pos, pos)

            dist = ((curr_pos - pos)**2).sum()**.5
            # print('distance', dist)
            duration = dist * 7 # 1cm=.1s 1m=10s
            duration = max(0.6, duration) # Don't go toooo fast
            # print('duration', duration, type(duration))
            duration = float(duration)
            
            print("MOVE TO: ", pos)
            if pos[2] < 0.05:
                print('below threshold!!', pos[2])
                continue
            try:
                self.robot.set_position(x=pos[0],y=pos[1],z=pos[2
                ],roll=euler[0],pitch=euler[1],yaw=euler[2], wait=wait) #euler[2]
                #self.robot.set_position(x=pos[0],y=pos[1],z=pos[2
                #],roll=euler[0],pitch=euler[1],yaw=-euler[2]+Lite6.rz_rad, wait=wait) #euler[2]
                
            except Exception as e:
                print('Could not goto_pose', e)
                # clear error
                self.robot.clean_error()
                self.robot.set_position(x=pos[0],y=pos[1],z=pos[2
                ],roll=euler[0],pitch=euler[1],yaw=euler[2], wait=wait) 
                

           

    def go_to_cartesian_pose_precise(self, positions, orientations, hertz=200, stiffness_factor=3.0):
        """
            This is a smooth version of this function. It can very smoothly go betwen the positions.
            However, it is unstable, and will result in oscilations sometimes.
            Recommended to be used only for fine, slow motions like the actual brush strokes.
        """
        
        def get_duration(here, there):
            dist = ((here.translation - there.translation)**2).sum()**.5
            duration = dist *  10#5 # 1cm=.1s 1m=10s
            duration = max(0.2, duration) # Don't go toooo fast
            duration = float(duration)
            return duration, dist

        def smooth_trajectory(poses, window_width=50):
            # x = np.cumsum(delta_ts)
            from scipy.interpolate import interp1d
            from scipy.ndimage import gaussian_filter1d

            for c in range(3):
                coords = np.array([p.translation[c] for p in poses])

                coords_smooth = gaussian_filter1d(coords, 31)
                print(len(poses), len(coords_smooth))
                for i in range(len(poses)-1):
                    coords_smooth[i]
                    poses[i].translation[c] = coords_smooth[i]
            return poses

        pose_trajs = []
        delta_ts = []
        
        # Loop through each position/orientation and create interpolations between the points
        p0 = self.robot.get_position()
        for i in range(len(positions)):
            #p1 = Lite6.create_rotation_transform(positions[i], orientations[i])
            
            duration, distance = get_duration(p0[:3], positions[i,:3])

            # needs to be high to avoid torque discontinuity error controller_torque_discontinuity
            STEPS = max(10, int(duration*hertz))
            # print(STEPS, distance)

            # if distance*100 > 5:
            #     print("You're using the precise movement wrong", distance*100)

            ts = np.arange(0, duration, duration/STEPS)
            # ts = np.linspace(0, duration, STEPS)
            #weights = [min_jerk_weight(t, duration) for t in ts]

            #if i == 0 or i == len(positions)-1:
            #    # Smooth for the first and last way points
            #    pose_traj = [p0.interpolate_with(p1, w) for w in weights]
            #else:
            #    # linear for middle points cuz it's fast and accurate
            #    pose_traj = p0.linear_trajectory_to(p1, STEPS)
            # pose_traj = [p0.interpolate_with(p1, w) for w in weights]
            # pose_traj = p0.linear_trajectory_to(p1, STEPS)
            
            from scipy.interpolate import BSpline, make_interp_spline
            spl = make_interp_spline(x, y)
            x_new, y_new = spl(ts).T
            pose_traj = [x_new,y_new]

            pose_trajs += pose_traj
            
            delta_ts += [duration/len(pose_traj),]*len(pose_traj)
            
            p0 = p1
            
        T = float(np.array(delta_ts).sum())

        # pose_trajs = smooth_trajectory(pose_trajs)

        
        # To ensure skill doesn't end before completing trajectory, make the buffer time much longer than needed
        print("[GoToCartesianPosePrecise] pose_trajs[1] ",pose_trajs[1])
        #self.fa.goto_pose(pose_trajs[1], duration=T, dynamic=True, 
        #    buffer_time=T+10,
        #    force_thresholds=[10,10,10,10,10,10],
        #    cartesian_impedances=(np.array(FC.DEFAULT_TRANSLATIONAL_STIFFNESSES)*stiffness_factor).tolist() + FC.#DEFAULT_ROTATIONAL_STIFFNESSES,
        #    ignore_virtual_walls=True,
        #)

        '''
        pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)
        
        try:
            init_time = rospy.Time.now().to_time()
            for i in range(2, len(pose_trajs)):
                timestamp = rospy.Time.now().to_time() - init_time
                traj_gen_proto_msg = PosePositionSensorMessage(
                    id=i, timestamp=timestamp, 
                    position=pose_trajs[i].translation, quaternion=pose_trajs[i].quaternion
                )
                fb_ctrlr_proto = CartesianImpedanceSensorMessage(
                    id=i, timestamp=timestamp,
                    # translational_stiffnesses=FC.DEFAULT_TRANSLATIONAL_STIFFNESSES[:2] + [z_stiffness_trajs[i]],
                    translational_stiffnesses=(np.array(FC.DEFAULT_TRANSLATIONAL_STIFFNESSES)*stiffness_factor).tolist(),
                    rotational_stiffnesses=FC.DEFAULT_ROTATIONAL_STIFFNESSES
                )
                
                ros_msg = make_sensor_group_msg(
                    trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                        traj_gen_proto_msg, SensorDataMessageType.POSE_POSITION),
                    feedback_controller_sensor_msg=sensor_proto2ros_msg(
                        fb_ctrlr_proto, SensorDataMessageType.CARTESIAN_IMPEDANCE)
                )

                # rospy.loginfo('Publishing: ID {}'.format(traj_gen_proto_msg.id))
                pub.publish(ros_msg)
                # rate = rospy.Rate(1 / (delta_ts[i]))
                rate = rospy.Rate(hertz)
                rate.sleep()

                # if i%100==0:
                #     print(self.fa.get_pose().translation[-1] - pose_trajs[i].translation[-1], 
                #         '\t', self.fa.get_pose().translation[-1], '\t', pose_trajs[i].translation[-1])
        except Exception as e:
            print('unable to execute skill', e)
        '''
        # Stop the skill
        #self.fa.stop_skill()

