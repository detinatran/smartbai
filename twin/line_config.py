"""smartbai line definition: a small Dubai beverage-bottling SME line. Shared by the scene builder, the discrete-event
simulator and the tool server, so station IDs and parameters live in one place. Rates are illustrative SME values
(bottles per minute), to be replaced by partner data when available."""

LINE = dict(
    name="smartbai_bottling_line_v1",
    product="500 ml PET bottled water / juice",
    conveyor_speed_mps=0.25,
    bottle_pitch_m=0.30,
)

# x position (m) along the conveyor, nominal rate (bottles/min), sensors exposed by the twin
STATIONS = [
    dict(id="S1", name="Depalletizer", type="robot_ur10", x=1.5, rate_bpm=60, sensors=["motor_current", "joint_temp"]),
    dict(id="S2", name="Filler", type="filler", x=5.0, rate_bpm=45, sensors=["vibration", "motor_temp", "fill_level"]),
    dict(id="S3", name="Capper", type="capper", x=8.0, rate_bpm=50, sensors=["torque", "motor_temp"]),
    dict(id="S4", name="Labeler", type="labeler", x=11.0, rate_bpm=55, sensors=["vibration", "label_tension"]),
    dict(id="S5", name="Vision QC", type="vision_qc", x=14.0, rate_bpm=70, sensors=["camera", "reject_count"]),
    dict(id="S6", name="Case packer", type="robot_ur5e", x=17.0, rate_bpm=48, sensors=["motor_current", "joint_temp"]),
    dict(id="S7", name="Palletizer", type="robot_ur10", x=20.5, rate_bpm=60, sensors=["motor_current", "joint_temp"]),
]
CONVEYOR_LENGTH_M = 21.5
