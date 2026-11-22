// A display limit, not an estimate of how long the actuator held a command.
const COMMAND_MAX_AGE = 0.5;
const ODOM_MAX_AGE = 0.5;
const JOY_MAX_AGE = 0.5;

function startsAfterOdometryGap(row, i, rows) {
  return i > 0 && row[0] - rows[i - 1][0] > ODOM_MAX_AGE;
}

function completeDistance(run) {
  return run.odom.some(startsAfterOdometryGap) ? null : run.distance;
}

function odometryPath(rows, point) {
  return rows.map((row, i) => {
    const restart = i === 0 || startsAfterOdometryGap(row, i, rows);
    return (restart ? "M" : "L") + point(row).map(n => n.toFixed(2)).join(",");
  }).join("");
}

function commandPath(rows, column, x, y) {
  let path = "";
  for (let i = 0; i < rows.length; i++) {
    const row = rows[i], prev = rows[i - 1];
    const px = x(row[0]).toFixed(2), py = y(row[column]).toFixed(2);
    if (prev && row[0] - prev[0] <= COMMAND_MAX_AGE) {
      path += "H" + px + "V" + py;
    } else {
      if (prev) path += "H" + x(prev[0] + COMMAND_MAX_AGE).toFixed(2);
      path += "M" + px + "," + py;
    }
  }
  return path;
}

function commandLabel(cmd, time, column, unit) {
  if (!cmd) return "—" + unit;
  const value = cmd[column].toFixed(2) + unit;
  const age = time - cmd[0];
  return age > COMMAND_MAX_AGE ? "Gap · last " + value + " (" + age.toFixed(1) + " s ago)" : value;
}

function joystickLabel(joy, time) {
  return joy && time - joy[0] <= JOY_MAX_AGE ? "LB " + (joy[1] ? "held" : "released") : "LB not sampled";
}

if (typeof module !== "undefined") module.exports = {commandPath, commandLabel, odometryPath, completeDistance, joystickLabel};
