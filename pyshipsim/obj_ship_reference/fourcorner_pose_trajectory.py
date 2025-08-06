import numpy as np

from .reference_pose_trajectory import ReferencePoseTrajectory


class FourCornerReferencePoseTrajectory(ReferencePoseTrajectory):
    def __init__(
        self,
        edge_length: float = 5.0,
        beta_angle: float = np.pi / 4,
        T_step: float = 300.0,
        T_between: float = 0.0,
        **kwargs,
    ):
        # 5 points of four corner
        ref_x_0 = [edge_length, edge_length, edge_length, 0.0, 0.0]
        ref_y_0 = [0.0, edge_length, edge_length, edge_length, 0.0]
        ref_psi = [0.0, 0.0, beta_angle, beta_angle, 0.0]
        # Generate reference trajectory
        t_seq = [0.0]
        pose_seq = [[0.0, 0.0, 0.0]]
        for i in range(5):
            t_seq.append(t_seq[-1] + T_step)
            pose_seq.append([ref_x_0[i], ref_y_0[i], ref_psi[i]])
            t_seq.append(t_seq[-1] + T_between)
            pose_seq.append([ref_x_0[i], ref_y_0[i], ref_psi[i]])
        t_seq.append(np.inf)
        pose_seq.append(pose_seq[-1])
        super().__init__(np.array(t_seq), np.array(pose_seq), **kwargs)
