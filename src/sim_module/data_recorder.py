"""Simulation data recorder owned by the simulation module."""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


class DataRecorder:
    """Record simulation state, control, and prediction traces."""

    def __init__(self, output_dir: str = "simulation_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.data = {
            "time": [],
            "theta1_real": [], "theta2_real": [], "theta3_real": [],
            "omega1_real": [], "omega2_real": [], "omega3_real": [],
            "theta1_pred": [], "theta2_pred": [], "theta3_pred": [],
            "omega1_pred": [], "omega2_pred": [], "omega3_pred": [],
            "theta1_des": [], "theta2_des": [], "theta3_des": [],
            "omega1_des": [], "omega2_des": [], "omega3_des": [],
            "u1": [], "u2": [], "u3": [], #true torque inputs from the PD+
            "ddq_mpc_1": [], "ddq_mpc_2": [], "ddq_mpc_3": [],
            "ddq_local_cbf_1": [], "ddq_local_cbf_2": [], "ddq_local_cbf_3": [],
            "p_ref_x": [], "p_ref_y": [],
            "q_ref_1": [], "q_ref_2": [], "q_ref_3": [],
            "w1": [], "w2": [], "w3": [], #disturbance
            "ee_x_real": [], "ee_y_real": [],
            "ee_x_des": [], "ee_y_des": [],
            "ee_x_pred": [], "ee_y_pred": [],
            "local_cbf_active": [], "local_cbf_slack": [], "local_cbf_feasible": [],
            "local_cbf_solvetime": [], "local_cbf_min": [],
            "mpc_cbf_active": [], "mpc_cbf_slack": [],"mpc_feasible": [], 
            "mpc_solvetime": [], "mpc_cbf_min": [],

        }
        self.step_count = 0

    def record_step(
        self,
        time,
        z_real=None,
        z_pred=None,
        u=None,
        ddq_mpc=None,
        ddq_local_cbf=None,
        z_des=None,
        p_ref=None,
        q_ref=None,
        w=None,
        ee_pos_real=None,
        ee_pos_des=None,
        ee_pos_pred=None,
        local_cbf_active = None,
        local_cbf_slack = None,
        local_cbf_feasible = None,
        local_cbf_solvetime = None, 
        local_cbf_min = None,
        mpc_cbf_active = None, 
        mpc_cbf_slack = None,
        mpc_feasible = None, 
        mpc_solvetime = None,
        mpc_cbf_min = None,
    ):
        self.data["time"].append(time)

        if z_real is not None:
            for index in range(3):
                self.data[f"theta{index + 1}_real"].append(z_real[index])
                self.data[f"omega{index + 1}_real"].append(z_real[index + 3])
        else:
            for index in range(3):
                self.data[f"theta{index + 1}_real"].append(np.nan)
                self.data[f"omega{index + 1}_real"].append(np.nan)

        if z_pred is not None:
            for index in range(3):
                self.data[f"theta{index + 1}_pred"].append(z_pred[index])
                self.data[f"omega{index + 1}_pred"].append(z_pred[index + 3])
        else:
            for index in range(3):
                self.data[f"theta{index + 1}_pred"].append(np.nan)
                self.data[f"omega{index + 1}_pred"].append(np.nan)

        if z_des is not None:
            for index in range(3):
                self.data[f"theta{index + 1}_des"].append(z_des[index])
                self.data[f"omega{index + 1}_des"].append(z_des[index + 3])
        else:
            for index in range(3):
                self.data[f"theta{index + 1}_des"].append(np.nan)
                self.data[f"omega{index + 1}_des"].append(np.nan)

        if u is not None:
            for index in range(3):
                self.data[f"u{index + 1}"].append(u[index])
        else:
            for index in range(3):
                self.data[f"u{index + 1}"].append(np.nan)

        if ddq_mpc is not None:
            for index in range(3):
                self.data[f"ddq_mpc_{index + 1}"].append(ddq_mpc[index])
        else:
            for index in range(3):
                self.data[f"ddq_mpc_{index + 1}"].append(np.nan)

        if ddq_local_cbf is not None:
            for index in range(3):
                self.data[f"ddq_local_cbf_{index + 1}"].append(ddq_local_cbf[index])
        else:
            for index in range(3):
                self.data[f"ddq_local_cbf_{index + 1}"].append(np.nan)

        if p_ref is not None:
            self.data["p_ref_x"].append(p_ref[0])
            self.data["p_ref_y"].append(p_ref[1])
        else:
            self.data["p_ref_x"].append(np.nan)
            self.data["p_ref_y"].append(np.nan)

        if q_ref is not None:
            for index in range(3):
                self.data[f"q_ref_{index + 1}"].append(q_ref[index])
        else:
            for index in range(3):
                self.data[f"q_ref_{index + 1}"].append(np.nan)

        if w is not None:
            for index in range(3):
                self.data[f"w{index + 1}"].append(w[index])
        else:
            for index in range(3):
                self.data[f"w{index + 1}"].append(np.nan)

        if ee_pos_real is not None:
            self.data["ee_x_real"].append(ee_pos_real[0])
            self.data["ee_y_real"].append(ee_pos_real[1])
        else:
            self.data["ee_x_real"].append(np.nan)
            self.data["ee_y_real"].append(np.nan)

        if ee_pos_des is not None:
            self.data["ee_x_des"].append(ee_pos_des[0])
            self.data["ee_y_des"].append(ee_pos_des[1])
        else:
            self.data["ee_x_des"].append(np.nan)
            self.data["ee_y_des"].append(np.nan)

        if ee_pos_pred is not None:
            self.data["ee_x_pred"].append(ee_pos_pred[0])
            self.data["ee_y_pred"].append(ee_pos_pred[1])
        else:
            self.data["ee_x_pred"].append(np.nan)
            self.data["ee_y_pred"].append(np.nan)

        self.data["local_cbf_active"].append(local_cbf_active if local_cbf_active is not None else False)
        self.data["local_cbf_slack"].append(local_cbf_slack if local_cbf_slack is not None else np.nan)
        self.data["local_cbf_feasible"].append(local_cbf_feasible if local_cbf_feasible is not None else np.nan)
        self.data["local_cbf_solvetime"].append(local_cbf_solvetime if local_cbf_solvetime is not None else np.nan)
        self.data["local_cbf_min"].append(local_cbf_min if local_cbf_min is not None else np.nan)

        self.data["mpc_cbf_active"].append(mpc_cbf_active if mpc_cbf_active is not None else False)
        self.data["mpc_cbf_slack"].append(mpc_cbf_slack if mpc_cbf_slack is not None else np.nan)
        self.data["mpc_feasible"].append(mpc_feasible if mpc_feasible is not None else np.nan)
        self.data["mpc_solvetime"].append(mpc_solvetime if mpc_solvetime is not None else np.nan)
        self.data["mpc_cbf_min"].append(mpc_cbf_min if mpc_cbf_min is not None else np.nan)

        self.step_count += 1

    def save_to_csv(self, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_data_{timestamp}.csv"
        filepath = self.output_dir / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.data).to_csv(filepath, index=False, float_format="%.6f")
        #print(f"\nData saved to: {filepath}")
        #print(f"Total steps recorded: {self.step_count}")
        #print(f"Columns: {len(self.data)}")
        return filepath

    def get_dataframe(self):
        return pd.DataFrame(self.data)

    def clear(self):
        for key in self.data:
            self.data[key] = []
        self.step_count = 0