# Terminal menu (`racerbot`)

> **Who this is for:** anyone running a car at a test session through a OnePortal terminal.
> **Read first:** [launchstart.md](launchstart.md), the commands this menu watches and will later start.
> **You'll be able to:** see on one screen whether the car is ready, run the hardware check, and look through recorded runs.
> **Time:** 5 minutes to install once per car, then one word to start.

`racerbot` is a menu that runs in the terminal. Type it and you get one screen showing the car's state: whether the hardware is up, what's driving, whether LB is held, the battery, the sensors and how busy the Jetson is.

This is **step 1: watch only**. Nothing in the menu can start a driving layer yet, so it is safe to open at any time, including while the car is driving.

The full design, including the driving screens that come next, is the [Racerbot terminal menu design](https://claude.ai/artifact/JsFn9KmNcZSBna2jAnGabJ).

## Highlights

- **One status line for the whole car**, on every screen: car, hardware, which driving layer is running, LB, battery.
- **LB as the car sees it.** It reads `/joy` live, so you can check that the right controller is talking to the right car before anyone holds the button.
- **Catches the combinations that go wrong.** It flags two driving layers at once (teleop always wins, so autonomy sits still). It flags the simulator running beside real drivers. It flags two `/joy` publishers, which means two cars share one network and one car's LB can drive the other.
- **The hardware check in one key.** `c` runs `racerbot_doctor` and colours its PASS, WARN and FAIL lines.
- **Every run on the car in one table**: when, what drove, whether it moved, how long, how big, and whether it's in git yet. Reports on a run are one key away.
- **Light on the Jetson.** It reads `/proc` and `/sys` directly and keeps only the latest value of each topic. It subscribes to six [topics](glossary.md#topic) (named ROS data streams) and publishes nothing.
- **Survives a closed browser tab.** It runs inside `tmux`, so the next login opens the same screen.

## Install (once per car)

Only needed the first time on each car, or after `src/racerbot_console/requirements.txt` changes.

**Any terminal on the car:**

```bash
cd ~/racerbot-ws
tools/console/install.sh
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select racerbot_console
```

**Working when:** the install script prints `Installed. Textual 8.2.8.` and the build finishes with `1 package finished`.

**If it says `python3-venv is missing`:** run `sudo apt install python3-venv`, then run the script again. **If it says `tmux is missing`:** run `sudo apt install tmux`. Without tmux the menu still works, but it stops when the browser tab closes.

The script makes its own Python environment in `~/.local/share/racerbot/console-venv`, which can still see ROS, and puts Textual there. It doesn't change the system Python, uses no `sudo`, and changes nothing about how the workspace builds.

## Start it

**Any terminal on the car** (no `source` lines needed, the launcher does that):

```bash
racerbot
```

**Working when:** the top line shows the car's name (`rb1` or `rb2`) and `HW up` or `HW off`. The bottom line lists the keys.

**If it says `racerbot: command not found`:** open a new terminal, or run `~/.local/bin/racerbot`. **If the panel says `ROS is not importable`:** the workspace isn't built; run the `colcon build` line from [Install](#install-once-per-car).

To leave it running and close the tab, just close the tab. Typing `racerbot` again later goes back to the same menu. `q` quits for real.

## The screens

| Key | Screen | What's on it now |
|---|---|---|
| `1` | Home | Car panel (battery, speed, sensors, IMU, `/joy`, `/drive`, dashboard, Jetson load) and Session (runs today, runs not in git yet) |
| `2` | Drive | The usual launch commands. Starting them from the menu comes in step 2 |
| `3` | Runs | Every run on this car, newest first. `a` summary, `A` SLAM report, `I` IMU report on the selected run, `F5` reload |
| `4` | Check | Output of the hardware check. `c` runs it from any screen |
| `5` | Learn | Slots for the self-learning system, greyed out until each part is built |
| `6` | Logs | Output of everything the menu has run, also saved in `~/.local/state/racerbot/console/` |
| `7`, `8` | ROS, Workspace | Placeholders for step 4 |

`?` shows the keys, `Ctrl+P` opens Textual's command list, and `q` (or `F10`) quits. If a check or report is still running, `q` asks you to press it again and then stops it.

### Reading the status line

```text
 rb2  │ ● HW up │ DRIVING: gap follow │ LB ● HELD │ ● 11.70 V 3.90/cell
```

- **`HW`**: `up` when all six bringup programs run, `partial` with the missing ones named, `off` when none do.
- **`DRIVING`**: the driving layer that's running, ours or Team A's. Red if two run at once.
- **`LB`**: `HELD` only while a fresh `/joy` message has LB pressed. `controller silent` means `/joy` stopped arriving more than 1 s ago.
- **Battery**: the same thresholds as the doctor. Yellow below 3.6 V per cell, red below 3.4 V per cell.

**The status line is information, not a safety device.** Every driving node enforces the LB deadman on its own. To stop the car, let go of LB or use the battery switch, never the menu.

## Safety

- The menu publishes nothing to ROS and has no service or action clients. `src/racerbot_console/test/test_no_drive_paths.py` fails if any is added.
- It can only run the jobs listed in `src/racerbot_console/racerbot_console/jobs.py`: the hardware check and three run reports. None of them can move the car, and the same test fails if one is added.
- Starting driving layers from the menu (step 2) will come with its own checks and a floor test before merge, as the design describes.

## Tests

The logic tests need no ROS and no Textual. The screen tests need Textual, so run them with the menu's own Python:

```bash
cd ~/racerbot-ws
python3 -m pytest src/racerbot_console/test/ -q                                   # logic tests
~/.local/share/racerbot/console-venv/bin/python -m pytest src/racerbot_console/test/ -q   # everything
python3 tools/check_mutations.py tools/mutations/console.json --no-ros            # prove they can fail
```

**Working when:** the second line reports `116 passed`, and the last line ends `32/32 breaks caught`. Run with plain `python3` (without Textual), `test_app.py` is skipped with a message pointing here.

<details>
<summary>How it's built (skip unless you're changing it)</summary>

- `status.py`, `layers.py`, `sysinfo.py`, `runs.py`, `jobs.py`: plain Python, no ROS or Textual, each with its own tests.
- `ros_reader.py`: the only file that imports `rclpy`. One node on a background thread subscribes to `/sensors/core`, `/odom`, `/joy`, `/scan`, `/drive` and `/imu/status`, and counts `/joy` publishers once a second.
- `backend.py`: puts the readings, the process list and the Jetson numbers into one snapshot, at most once a second for the slow parts.
- `app.py`: the Textual screens. They refresh 4 times a second from the snapshot and never wait on ROS.
- `tools/console/racerbot`: the launcher (tmux, sourcing). `tools/console/install.sh`: the one-time install.

</details>
